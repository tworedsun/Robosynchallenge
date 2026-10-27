#!/usr/bin/env python3
"""FoundationStereo 单次推理:立体像对 -> 视差 + 深度(.npz)。

从 ``_stage/Depth_Fetch/common/scripts/fs_infer.py`` 移植,唯一的行为变化是
**路径参数化**:

* ``FS_ROOT`` 从环境变量 ``FS_ROOT`` 读取,默认
  :attr:`robosyn_hybrid.settings.Settings.foundationstereo_root`;
* ``--ckpt`` 默认 :attr:`robosyn_hybrid.settings.Settings.fs_ckpt`;
* 不再出现任何硬编码的绝对路径,也不再有写死的解释器 / venv 位置。

刻意不依赖 open3d,只需要一个轻量 venv(见
``policy/hybrid/robosyn_hybrid/requirements/foundationstereo.txt`` 与 ``policy/hybrid/robosyn_hybrid/scripts/setup_fs_env.sh``)。

CLI(SPEC §2.4)::

    python fs_infer.py --left L.png --right R.png --ckpt CKPT --out OUT.npz \\
                       --fx FX --baseline B --valid-iters N

输出 npz 键:``disparity`` / ``depth`` / ``valid``。

额外提供了一个**常驻 worker** 模式(``--serve``),对应参考实现里的
``fs_server.py``。原先那个独立文件不在提交包结构里,所以它的 JSON-lines
协议被原样搬到这里:一次加载模型,之后逐条请求推理。协议与
``fs_server.py`` 完全一致,因此既有的 worker 脚本仍然可以和本文件对接::

    request : {"left": path, "right": path, "fx": float, "baseline": float,
               "out": path, "valid_iters": int}
    ping    : {"ping": true}
    reply   : {"ok": true, "infer_s": ..., "total_s": ..., "valid_fraction": ...}
              {"ok": false, "error": "..."}

``robosyn_hybrid.stereo.run_fs`` 会优先连 ``FS_STEREO_SOCKET``
(默认路径由 :mod:`robosyn_hybrid.settings` 提供),连不上再回退子进程。

``import torch`` 全部在函数内部延迟导入,所以没有 torch 时模块仍可 import。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["build_model", "infer", "run_once", "serve", "resolve_fs_root", "cli"]

_PKG_DIR = Path(__file__).resolve().parent


def _ensure_package_importable() -> None:
    """让 ``robosyn_hybrid.settings`` 在"当脚本跑"时也能 import。

    直接 ``python policy/hybrid/robosyn_hybrid/fs_infer.py`` 时 ``sys.path[0]`` 是
    ``policy/hybrid/robosyn_hybrid/`` 本身,包根需要手动补上。
    """
    package_root = str(_PKG_DIR.parent)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


def _prepend_sys_path(path: Path) -> None:
    """把 ``path`` 放到 ``sys.path`` 最前面(与参考实现的 ``insert(0, ...)`` 等价)。

    先移除同名条目再插到 0,避免 FS_ROOT 被别处的同名 ``core`` 包抢先解析。
    """
    text = str(path)
    while text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


def resolve_fs_root() -> Path:
    """FoundationStereo 仓库根:``FS_ROOT`` 环境变量优先,否则用 settings。"""
    raw = os.environ.get("FS_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    _ensure_package_importable()
    from robosyn_hybrid.settings import get_settings

    return get_settings().foundationstereo_root


def resolve_ckpt_default() -> str:
    """checkpoint 默认值:``FS_CKPT`` -> ``settings.fs_ckpt``(无硬编码兜底)。"""
    raw = os.environ.get("FS_CKPT", "").strip()
    if raw:
        return str(Path(raw).expanduser())
    _ensure_package_importable()
    from robosyn_hybrid.settings import get_settings

    return str(get_settings().fs_ckpt)


def resolve_socket_path() -> Path:
    """worker socket 路径:``FS_STEREO_SOCKET`` 优先,否则用 settings。

    这里**不提供**硬编码兜底:参考实现把 socket 写死在某个绝对路径下,
    提交包只允许由 :mod:`robosyn_hybrid.settings` 决定。
    """
    raw = os.environ.get("FS_STEREO_SOCKET", "").strip()
    if raw:
        return Path(raw).expanduser()
    _ensure_package_importable()
    from robosyn_hybrid.settings import get_settings

    return Path(get_settings().fs_socket)


def resolve_valid_iters_default() -> int:
    """``valid_iters`` 默认值,来自 settings(``FS_STEREO_VALID_ITERS``,默认 32)。"""
    _ensure_package_importable()
    from robosyn_hybrid.settings import get_settings

    return int(get_settings().fs_valid_iters)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def build_model(ckpt_path: str, valid_iters: int, device: str = "cuda") -> Any:
    """加载 FoundationStereo 权重(全部 torch 相关 import 都在函数内)。"""
    import torch
    from omegaconf import OmegaConf

    ckpt = Path(ckpt_path).expanduser()
    cfg_path = ckpt.parent / "cfg.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"FoundationStereo config not found at {cfg_path}; the checkpoint is "
            "expected to sit next to its cfg.yaml"
        )

    fs_root = resolve_fs_root()
    _prepend_sys_path(fs_root)
    from core.foundation_stereo import FoundationStereo  # noqa: E402

    cfg = OmegaConf.load(str(cfg_path))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = int(valid_iters)
    cfg["scale"] = 1.0
    cfg["hiera"] = 0
    cfg["z_far"] = 10.0
    cfg["get_pc"] = 0
    model = FoundationStereo(cfg)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model


def infer(
    model: Any,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    valid_iters: int,
    device: str = "cuda",
) -> np.ndarray:
    """一对 RGB -> 视差图(float32)。"""
    import torch

    fs_root = resolve_fs_root()
    _prepend_sys_path(fs_root)
    from core.utils.utils import InputPadder  # noqa: E402

    img0 = torch.as_tensor(left_rgb).to(device).float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right_rgb).to(device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    autocast = (
        torch.cuda.amp.autocast(True)
        if str(device).startswith("cuda")
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast:
        disp = model.forward(img0, img1, iters=int(valid_iters), test_mode=True)
    return padder.unpad(disp.float())[0, 0].cpu().numpy()


def depth_from_disparity(
    disparity: np.ndarray, fx: float, baseline: float
) -> tuple[np.ndarray, np.ndarray]:
    """视差 -> ``(depth, valid)``,``depth = fx * baseline / disparity``。"""
    valid = np.asarray(disparity) > 0
    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[valid] = float(fx) * float(baseline) / np.asarray(disparity)[valid]
    return depth, valid


def load_pair(left_path: str, right_path: str) -> tuple[np.ndarray, np.ndarray]:
    """读入一对 RGB 图,尺寸不一致直接报错。"""
    import imageio.v2 as imageio

    left = np.asarray(imageio.imread(left_path))
    right = np.asarray(imageio.imread(right_path))
    if left.shape != right.shape:
        raise SystemExit(f"image size mismatch: {left.shape} vs {right.shape}")
    left = np.ascontiguousarray(left[..., :3])
    right = np.ascontiguousarray(right[..., :3])
    return left, right


def run_once(
    model: Any,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    *,
    fx: float,
    baseline: float,
    out_path: str | Path,
    valid_iters: int,
    device: str = "cuda",
) -> dict:
    """单次推理并写出 npz;返回统计信息。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    disparity = infer(model, left_rgb, right_rgb, valid_iters, device=device)
    depth, valid = depth_from_disparity(disparity, fx, baseline)
    np.savez_compressed(
        str(out), disparity=disparity.astype(np.float32), depth=depth, valid=valid
    )
    return {
        "out": str(out),
        "valid_fraction": float(valid.mean()) if valid.size else 0.0,
        "disparity_median_px": float(np.median(disparity[valid])) if valid.any() else None,
        "depth_median_m": float(np.median(depth[valid])) if valid.any() else None,
    }


# ---------------------------------------------------------------------------
# 常驻 worker(原 fs_server.py 的等价物)
# ---------------------------------------------------------------------------
def _already_running(path: Path) -> bool:
    """socket 上已经有 worker 在应答时为 True。"""
    if not path.exists():
        return False
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(5.0)
    try:
        probe.connect(str(path))
        probe.sendall(b'{"ping": true}\n')
        data = probe.recv(4096)
        return b'"ok"' in data
    except Exception:  # noqa: BLE001
        return False
    finally:
        probe.close()


def _serve(model: Any, valid_iters_default: int, device: str) -> None:
    """JSON-lines 请求循环,一次加载、多次推理。"""
    path = resolve_socket_path()
    if path.exists():
        path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(8)
    print(f"[fs_infer] worker ready on {path}", flush=True)
    try:
        while True:
            conn, _addr = server.accept()
            try:
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                request = json.loads(buf.decode() or "{}")
                if request.get("ping"):
                    conn.sendall(b'{"ok": true, "pong": true}\n')
                    continue
                started = time.perf_counter()
                left, right = load_pair(request["left"], request["right"])
                iters = int(request.get("valid_iters", valid_iters_default))
                t0 = time.perf_counter()
                stats = run_once(
                    model,
                    left,
                    right,
                    fx=float(request["fx"]),
                    baseline=float(request["baseline"]),
                    out_path=request["out"],
                    valid_iters=iters,
                    device=device,
                )
                reply = {
                    "ok": True,
                    "infer_s": round(time.perf_counter() - t0, 3),
                    "total_s": round(time.perf_counter() - started, 3),
                    "valid_fraction": stats["valid_fraction"],
                    "disparity_median_px": stats["disparity_median_px"],
                }
            except Exception as exc:  # noqa: BLE001 - 单个请求失败不能让 worker 退出
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            try:
                conn.sendall((json.dumps(reply) + "\n").encode())
            except Exception:  # noqa: BLE001
                pass
            finally:
                conn.close()
    finally:
        server.close()
        if path.exists():
            path.unlink()


def serve(*, device: str = "cuda", ckpt: str | None = None, valid_iters: int | None = None) -> int:
    """启动常驻 worker(幂等:已有 worker 在跑就直接返回 0)。"""
    path = resolve_socket_path()
    if _already_running(path):
        print(f"[fs_infer] a worker is already serving {path}")
        return 0
    if path.exists():
        path.unlink()

    ckpt = ckpt or os.environ.get("FS_STEREO_CKPT", "").strip() or resolve_ckpt_default()
    iters = int(valid_iters if valid_iters is not None else resolve_valid_iters_default())
    started = time.perf_counter()
    print(f"[fs_infer] loading {ckpt} ...", flush=True)
    model = build_model(ckpt, iters, device=device)
    print(f"[fs_infer] model ready in {time.perf_counter() - started:.1f} s", flush=True)
    _serve(model, iters, device)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FoundationStereo: stereo pair -> disparity/depth npz, "
        "or a persistent worker (--serve)."
    )
    parser.add_argument("--left", default=None, help="left/reference PNG (infer mode)")
    parser.add_argument("--right", default=None, help="right/target PNG (infer mode)")
    parser.add_argument("--ckpt", default=None, help="model_best_bp2.pth path")
    parser.add_argument("--fx", type=float, default=None, help="rectified focal length")
    parser.add_argument("--baseline", type=float, default=None, help="baseline in metres")
    # 同时接受 --valid-iters(SPEC 写法)与 --valid_iters(参考实现写法)。
    parser.add_argument(
        "--valid-iters", "--valid_iters", dest="valid_iters", type=int, default=None
    )
    parser.add_argument("--out", default=None, help="output .npz path (infer mode)")
    parser.add_argument("--device", default=os.environ.get("FS_STEREO_DEVICE", "cuda"))
    parser.add_argument(
        "--serve", action="store_true", help="run as a persistent worker instead"
    )
    parser.add_argument("--fs-root", default=None, help="override FS_ROOT")
    return parser


def cli(argv: list[str] | None = None) -> int:
    """命令行入口。

    命名为 ``cli`` 而不是参考测量脚本里那个同名入口,避免与读真值的脚本
    混淆(那个入口需要构建仿真环境、读物体位姿与渲染深度)。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.fs_root:
        os.environ["FS_ROOT"] = str(args.fs_root)
        _prepend_sys_path(Path(args.fs_root).expanduser().resolve())
    valid_iters = int(
        args.valid_iters if args.valid_iters is not None else resolve_valid_iters_default()
    )
    ckpt = args.ckpt or resolve_ckpt_default()

    if args.serve:
        return serve(device=args.device, ckpt=ckpt, valid_iters=valid_iters)

    missing = [
        name
        for name, value in (
            ("--left", args.left),
            ("--right", args.right),
            ("--fx", args.fx),
            ("--baseline", args.baseline),
            ("--out", args.out),
        )
        if value is None
    ]
    if missing:
        parser.error(
            f"infer mode requires {', '.join(missing)} (or pass --serve for worker mode)"
        )

    left, right = load_pair(args.left, args.right)
    print(f"[fs] pair {left.shape}, ckpt={ckpt}", flush=True)
    model = build_model(ckpt, valid_iters, device=args.device)
    stats = run_once(
        model,
        left,
        right,
        fx=float(args.fx),
        baseline=float(args.baseline),
        out_path=args.out,
        valid_iters=valid_iters,
        device=args.device,
    )
    print(
        f"[fs] disparity valid {100 * stats['valid_fraction']:.1f}%, "
        f"median disp {stats['disparity_median_px']}px, "
        f"median depth {stats['depth_median_m']}m -> {stats['out']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
