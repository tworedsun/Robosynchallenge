#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""click_bell 分层几何诊断工具(Workstream G)。

================================================================================
【醒目的合规声明 —— 必读】
================================================================================
本文件是**诊断工具**,**绝不被策略调用**,也不属于任何策略/部署路径:

* ``policy/hybrid/deploy_policy.py`` / ``robosyn_hybrid/clickbell.py`` /
  ``flow.py`` **不得** import 本模块,也不得复制本模块里的任何函数;
* 策略本体**不读任何物体真值位姿**:真值只允许出现在本文件里;
* **L5 是全脚本唯一读真值的层**(``base.sim.get_articulation("button")``,
  源自官方 env ``click_bell.py::compute_task_state`` 的写法)。L1-L4、L6、L7
  本身不读物体真值;L6/L7 里出现的 "gt_*" 字段只是**引用 L5 的结论**做交叉
  验证(帮助区分"估计点不可达"与"根本不可达"),已在字段名与注释里显式标注。
* 本文件不参与评测计分,是一次性的"上机体检",可以随意读真值。

================================================================================
目的
================================================================================
打包机没有 GPU,click_bell 的混合几何链路(腕相机安装 → 运动双目深度 →
VLM 按钮定位 → 反投影 → IK → 按压)无法在本机运行验证。本脚本让用户拿到
GPU 机器后跑**一条命令**,就把这条链路的每一层都打上可判定的数字:

    L1 腕相机安装常量     stereo.verify_mount(env, "right")
    L2 运动双目深度       stereo.fetch_depth(base, "right", q, ...)
    L3 VLM 按钮定位       vlm.vlm_available / vlm.vlm_chat(按钮 prompt, [RGB])
    L4 按钮反投影         stereo.estimate_point_from_pixel(bbox 内 grid x grid 取中位数)
    L5 几何真值比对       base.sim.get_articulation("button")  ← 唯一读真值处
    L6 按压位姿 IK        grasp.solve_ik_named + grasp.fk_pose_named 回代
    L7 按压可行性         沿 EE -z 逐级下压 8 mm 的 FK/IK 可达性(不 step 环境)

每层都捕获异常并写进该层的 ``error`` 字段,继续往下跑或有序退出;最后聚合
成一份 JSON + 一份人类可读总结。

================================================================================
退出码规则(全部写死在 :func:`_aggregate`)
================================================================================
``pass``        该层判定通过;
``fail``        该层判定不通过  → 退出码 1(**永远阻塞**);
``skip``        该层没能评估  → 是否导致退出码 1 取决于 ``blocking``:

    * ``blocking=True`` : 环境/配置问题(如 ``env.sensors`` 不可用、VLM 未配置
      且用户也没给 ``--button-uv``、深度抓取失败)→ 退出码 1;
    * ``blocking=False``: **用户显式选择**(``--skip-depth`` / ``--skip-vlm``)
      或用户用 ``--button-uv``/``--button-bbox`` 手动给了像素 → 不阻塞。

================================================================================
用法
================================================================================
::

    python robosyn_hybrid/scripts/verify_clickbell_geometry.py \
        --repo-root /path/to/RoboSynChallenge \
        --setting random --seed 0 \
        --out /path/to/diag.json [--skip-depth] [--skip-vlm]

``--repo-root`` 必须以官方仓库根(含 ``configs/click_bell/<setting>/gym_config.json``
与 ``scripts/eval_policy.py``)为准;本脚本会把它写进 ``ROBOSYN_REPO_ROOT``,
供 :mod:`robosyn_hybrid.settings` 解析资产路径。

本文件里所有常量都来自公开资料(``CLICKBELL_SPEC.md`` §1 / 官方 gym_config),
没有任何绝对路径常量,也没有任何物体真值常量。

.. warning::
   **必须在真有 GPU 的评测机上跑。** 环境构造走 dexsim / OptiX,在没有 CUDA 的
   机器上会在**原生层 abort(SIGABRT)**,Python 的 try/except 抓不到。为此本
   脚本:(1) 进 ``main()`` 就打开 :mod:`faulthandler`,崩溃时打印 Python 侧栈;
   (2) 在构造环境**之前**先把一份 ``in_progress`` 的 JSON 存根写到 ``--out``,
   这样即使原生崩溃,``--out`` 也会留下一份可读的证据。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

try:  # numpy 是几何计算的最小依赖;缺失时仍然允许 `--help` 正常工作。
    import numpy as np
except Exception:  # pragma: no cover - 只有极端环境才会走到
    np = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 相对定位(绝不写绝对路径常量)
# ---------------------------------------------------------------------------
SCRIPT_PATH: Path = Path(__file__).resolve()
#: ``<submission>/``,即 ``robosyn_hybrid`` 包的父目录,用来 import 本包。
PACKAGE_PARENT: Path = SCRIPT_PATH.parents[2]

TOOL_NAME = "verify_clickbell_geometry"
TOOL_VERSION = "1.0.0"

TASK_NAME = "click_bell"
DEFAULT_SETTING = "random"
GYM_CONFIG_REL_DIR = Path("configs") / TASK_NAME
EVAL_POLICY_REL = Path("scripts") / "eval_policy.py"
BUTTON_UID = "button"
BUTTON_COVER_LINK = "button_cover"
BUTTON_BASE_LINK = "button_base"

#: 官方成功判据:``press_depth = -button_qpos[:,0] >= 0.0048``(粘滞锁存)。
#: 出处 ``robosynchallenge/tasks/click_bell/click_bell.py::compute_task_state``。
SUCCESS_PRESS_DEPTH_M = 0.0048

#: 官方 URDF ``button.urdf`` 的 prismatic 关节行程``[-0.005, 0.0]``(米)。
#: 这里只作为"读不到 sim 关节限位时的回退值",正常路径从 sim 读。
BUTTON_JOINT_TRAVEL_FALLBACK_M = 0.005

#: ``CLICKBELL_SPEC.md`` §1 照抄的按压/接近常量。若策略模块
#: ``robosyn_hybrid.clickbell`` 已实现并暴露同名常量,本脚本会优先复用它
#: (见 :func:`_resolve_constants`),避免两处数值漂移。
DEFAULT_CONSTANTS: dict[str, float] = {
    "PRESS_AXIS_Z": -1.0,
    "PRESS_FIRST_STEP_M": 0.005,
    "PRESS_STEP_M": 0.0025,
    "PRESS_MAX_M": 0.008,
    "PRESS_MAX_STEPS": 4,
    "APPROACH_LIFT_M": 0.12,
    "APPROACH_LATERAL_TOL_M": 0.008,
    "STAGE_EE_TRANSLATION_TOL_M": 3.0e-3,
    "STAGE_EE_ROTATION_TOL_RAD": 5.0e-3,
    "STAGE_IK_ORIENTATION_MAX_RAD": math.radians(20.0),
    "STAGE_IK_ORIENTATION_RNG_SEED": 0,
}

#: 从策略模块复用的常量白名单(名字 -> 本脚本内部键)。
_REUSABLE_CONSTANTS: dict[str, str] = {
    "PRESS_FIRST_STEP_M": "PRESS_FIRST_STEP_M",
    "PRESS_STEP_M": "PRESS_STEP_M",
    "PRESS_MAX_M": "PRESS_MAX_M",
    "PRESS_MAX_STEPS": "PRESS_MAX_STEPS",
    "APPROACH_LIFT_M": "APPROACH_LIFT_M",
    "APPROACH_LATERAL_TOL_M": "APPROACH_LATERAL_TOL_M",
}

#: ``stereo.fetch_depth`` 契约里的全部键(``CLICKBELL_SPEC.md`` §4 / 该函数
#: docstring)。注意:文档里常被口口相传成 "21 个键",实际返回 **20 个**
#: (本脚本按实际返回校验,并把数量打印出来)。
DEPTH_REQUIRED_KEYS: tuple[str, ...] = (
    "ok",
    "arm",
    "q1",
    "q2",
    "reference_exposure",
    "reference_rgb",
    "reference_rect",
    "rect2",
    "reference_camera_pose",
    "K0",
    "K_rect",
    "Rr1",
    "Rr2",
    "H_ref",
    "baseline_m",
    "motion_camera_m",
    "rectification_deg",
    "disparity",
    "depth_map",
    "valid_map",
)

#: 顶面判定的"顶带"厚度:mesh 顶点落在 ``zmax - TOP_BAND_M`` 内的算顶面。
_TOP_BAND_M = 1.0e-3

# ===========================================================================
# 通用小工具
# ===========================================================================


def _npv(value: Any) -> Any:
    """把 torch tensor / 标量 / 数组统一成 numpy 数组(不 import torch)。"""
    if np is None:  # pragma: no cover - main() 已经拦住了
        raise RuntimeError("numpy is required")
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _jsonable(value: Any) -> Any:
    """把 numpy / torch / Path 等递归转成可 ``json.dumps`` 的对象。

    非有限浮点(``nan`` / ``inf``,诊断里代表"取不到")一律写成 ``null``,
    保证 ``--out`` 是严格合法的 JSON。
    """
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        return _jsonable(_npv(value))
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if np is not None:
        if isinstance(value, np.ndarray):
            return _jsonable(value.tolist())
        if isinstance(value, np.generic):
            return _jsonable(value.item())
    return str(value)


def _f(value: Any, default: float = float("nan")) -> float:
    """安全转 float(诊断输出里宁可给 NaN 也不要抛异常)。"""
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _mm(value_m: float) -> float:
    """米 → 毫米(输出统一用毫米,方便肉眼判定)。"""
    return float(value_m) * 1000.0


def _vec3(value: Any) -> list[float]:
    return [float(x) for x in _npv(value).reshape(-1)[:3]]


def _fmt_mm(vec3_m: Sequence[float]) -> str:
    """``[dx, dy, dz]``(米)→ ``"(dx, dy, dz) mm"``。"""
    return "(" + ", ".join(f"{_mm(x):+.2f}" for x in vec3_m) + ") mm"


def _format_checks(checks: dict[str, dict[str, Any]]) -> str:
    """把 checks 打印成人类可读的多行文本。"""
    lines: list[str] = []
    for label, item in checks.items():
        ok = item.get("ok")
        mark = "ok  " if ok is True else ("FAIL" if ok is False else "?   ")
        detail = item.get("detail") or ""
        lines.append(f"      [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return "\n".join(lines)


def _pose_delta(predicted: Any, truth: Any) -> dict[str, float]:
    """两个 4x4 位姿之间的平移(米)/旋转(度)误差。"""
    pred = np.asarray(predicted, dtype=np.float64).reshape(4, 4)
    true = np.asarray(truth, dtype=np.float64).reshape(4, 4)
    trace = float(np.trace(pred[:3, :3] @ true[:3, :3].T))
    return {
        "pos_err_m": float(np.linalg.norm(pred[:3, 3] - true[:3, 3])),
        "rot_err_deg": float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))),
    }


def _rotation_error_deg(rotation_a: Any, rotation_b: Any) -> float:
    """两个旋转矩阵之间的测地角度差(度)。"""
    a = np.asarray(rotation_a, dtype=np.float64).reshape(3, 3)
    b = np.asarray(rotation_b, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(a @ b.T))
    return float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))


def _axis_angle_matrix(axis: Any, angle_rad: float) -> Any:
    """罗德里格斯公式:单位轴 + 角度 → 3x3 旋转矩阵。"""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> Any:
    """URDF 的 ``rpy``(固定轴 roll-pitch-yaw)→ 旋转矩阵,即 ``Rz(y)Ry(p)Rx(r)``。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


# ===========================================================================
# 层结果容器
# ===========================================================================


@dataclass
class Layer:
    """一层诊断结果:状态 + 判定 + 可判定的数字。"""

    name: str
    title: str
    status: str = "skip"  # pass | fail | skip
    blocking: bool = False
    skip_reason: str | None = None
    verdict: str = ""
    error: str | None = None
    duration_s: float = 0.0
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    # -- 记录 ---------------------------------------------------------------
    def check(self, label: str, ok: bool | None, detail: str = "") -> None:
        """记一条子检查;``ok=None`` 表示"无法判定"。"""
        self.checks[label] = {"ok": None if ok is None else bool(ok), "detail": str(detail)}

    def mark_pass(self, verdict: str) -> None:
        self.status = "pass"
        self.blocking = False
        self.verdict = verdict

    def mark_fail(self, verdict: str) -> None:
        self.status = "fail"
        self.blocking = True
        self.verdict = verdict

    def mark_skip(self, verdict: str, *, blocking: bool, reason: str) -> None:
        self.status = "skip"
        self.blocking = bool(blocking)
        self.skip_reason = reason
        self.verdict = verdict

    # -- 输出 ---------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "status": self.status,
            "blocking": self.blocking,
            "skip_reason": self.skip_reason,
            "verdict": self.verdict,
            "error": self.error,
            "duration_s": round(float(self.duration_s), 3),
            "checks": _jsonable(self.checks),
            "data": _jsonable(self.data),
        }


def _run_layer(layer: Layer, body: Callable[[], None]) -> Layer:
    """跑一层:异常只记录到 ``error``,绝不外抛(单个失败不拖垮整次诊断)。"""
    started = time.perf_counter()
    try:
        body()
    except Exception as exc:  # noqa: BLE001 - 诊断工具必须吞掉一切异常
        layer.error = f"{type(exc).__name__}: {exc}"
        layer.check("层执行", False, layer.error)
        layer.mark_fail(f"层执行失败:{layer.error}")
        traceback.print_exc()
    finally:
        layer.duration_s = time.perf_counter() - started
        if not layer.verdict:
            layer.verdict = "未给出判定" if layer.status == "skip" else layer.verdict
    return layer


# ===========================================================================
# 运行上下文
# ===========================================================================


@dataclass
class Context:
    """层间共享状态(环境、深度结果、按钮像素/世界点、按压解等)。"""

    args: argparse.Namespace
    repo_root: Path
    gym_config: dict[str, Any]
    constants: dict[str, float]
    constants_source: str
    hooks: "ButtonHooks | None" = None
    setup: dict[str, Any] = field(default_factory=dict)
    env: Any = None
    base: Any = None
    stereo: Any = None
    vlm: Any = None
    priors: Any = None
    grasp: Any = None
    settings: Any = None
    depth: dict[str, Any] | None = None
    reference_rgb: Any = None
    reference_source: str = ""
    button_uv: list[float] | None = None
    button_bbox: list[float] | None = None
    #: L3 的 LOOK 视野搜索结果(与策略同一条代码路径)。
    look: dict[str, Any] | None = None
    look_hit: dict[str, Any] | None = None
    look_uv: list[float] | None = None
    look_bbox: list[float] | None = None
    look_qpos: Any = None
    look_camera_pose: Any = None
    look_rgb: Any = None
    button_world_m: Any = None
    button_point_info: dict[str, Any] = field(default_factory=dict)
    gt: dict[str, Any] | None = None
    cover_geometry: dict[str, Any] = field(default_factory=dict)
    press_pose: Any = None
    press_q: Any = None
    press_ik: dict[str, Any] = field(default_factory=dict)
    #: 依赖层未满足时的"是否阻塞"开关(显式 CLI 选择 → 不阻塞)。
    downstream_blocking: bool = True
    warnings: list[str] = field(default_factory=list)


# ===========================================================================
# import 辅助(全部延迟到函数内,保证 --help 干净)
# ===========================================================================


def _ensure_sys_path(args: argparse.Namespace, repo_root: Path) -> list[str]:
    """把官方仓库、其 ``scripts/``、本包父目录、用户额外路径放进 ``sys.path``。"""
    added: list[str] = []
    candidates: list[Path] = [repo_root, repo_root / "scripts", PACKAGE_PARENT]
    if args.pythonpath:
        candidates.extend(Path(p).expanduser() for p in str(args.pythonpath).split(os.pathsep) if p)
    for path in reversed(candidates):
        text = str(path)
        if text and Path(text).exists() and text not in sys.path:
            sys.path.insert(0, text)
    for path in candidates:
        text = str(path)
        if text in sys.path:
            added.append(text)
    return added


def _import_hybrid(ctx: Context) -> None:
    """延迟 import 本包模块(cv2/numpy 会随之进来;torch 仍留在各自函数内)。"""
    from robosyn_hybrid import grasp, priors, settings, stereo, vlm  # noqa: PLC0415

    ctx.stereo = stereo
    ctx.vlm = vlm
    ctx.priors = priors
    ctx.grasp = grasp
    ctx.settings = settings.get_settings()


def _resolve_constants() -> tuple[dict[str, float], str]:
    """按压/接近常量:优先复用策略模块,失败则用本文件的副本。

    兜底时把失败原因一起写进来源字符串 —— 静默退回默认值会掩盖"策略模块坏掉了"
    这种更严重的问题。
    """
    merged = dict(DEFAULT_CONSTANTS)
    try:
        from robosyn_hybrid import clickbell  # type: ignore  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - 策略模块可能还没写完
        return merged, (
            "diagnostic_defaults(CLICKBELL_SPEC.md §1);"
            f"clickbell 导入失败:{type(exc).__name__}: {str(exc)[:120]}"
        )
    reused: list[str] = []
    for attr, key in _REUSABLE_CONSTANTS.items():
        value = getattr(clickbell, attr, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            merged[key] = float(value)
            reused.append(attr)
    if not reused:
        return merged, "diagnostic_defaults(CLICKBELL_SPEC.md §1;clickbell 未暴露同名常量)"
    return merged, "robosyn_hybrid.clickbell: " + ",".join(sorted(reused))


# ===========================================================================
# 环境构造(照抄 handle_basket_flow.py::main 开头 + 官方 eval_policy)
# ===========================================================================


def _unwrap_base(env: Any, max_hops: int = 8) -> tuple[Any, int]:
    """沿 ``.env`` 链解包到最内层 env(官方 env 的 ``sim`` / ``robot`` 在那一层)。"""
    base = env
    hops = 0
    for _ in range(max_hops):
        nxt = getattr(base, "env", None)
        if nxt is None or nxt is base:
            break
        base = nxt
        hops += 1
    return base, hops


def _module_provenance() -> dict[str, Any]:
    """记录实际被 import 的 embodichain / robosynchallenge 来自哪个检出。

    这一格对 L1 的解读至关重要:``EmbodiChain_v023``/``v024`` 的
    ``matrix_from_quat`` 是 **wxyz**,而 ``EmbodiChain_main`` 的同名函数是
    **xyzw**;腕相机安装常量该按哪种顺序读,取决于评测机实际加载了哪个检出。
    """
    out: dict[str, Any] = {}
    for name in ("embodichain", "robosynchallenge"):
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__file__", None))
            version = getattr(module, "__version__", None)
            if version is not None:
                out[f"{name}_version"] = str(version)
        except Exception as exc:  # noqa: BLE001
            out[name] = f"import failed: {type(exc).__name__}: {exc}"
    return out


def _build_context(args: argparse.Namespace) -> Context:
    """构造官方 click_bell 环境并 reset(与官方 ``eval_policy.py`` 完全同路)。"""
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"--repo-root 不存在或不是目录:{repo_root}")
    eval_policy_path = repo_root / EVAL_POLICY_REL
    if not eval_policy_path.is_file():
        raise FileNotFoundError(
            f"官方评测脚本不存在:{eval_policy_path};"
            "--repo-root 必须指向官方 RoboSynChallenge 仓库根"
        )
    gym_config_path = repo_root / GYM_CONFIG_REL_DIR / str(args.setting) / "gym_config.json"
    if not gym_config_path.is_file():
        raise FileNotFoundError(f"gym_config 不存在:{gym_config_path}")

    # settings.repo_root() 读 ROBOSYN_REPO_ROOT,资产路径全部由它派生(不硬编码)。
    os.environ["ROBOSYN_REPO_ROOT"] = str(repo_root)
    if args.embodichain_root:
        os.environ["EMBODICHAIN_ROOT"] = str(Path(args.embodichain_root).expanduser().resolve())
    elif not os.environ.get("EMBODICHAIN_ROOT"):
        sibling = repo_root.parent / "EmbodiChain"
        if sibling.is_dir():
            os.environ["EMBODICHAIN_ROOT"] = str(sibling)

    added_paths = _ensure_sys_path(args, repo_root)
    # 官方 find_action_config() 用的是**相对**路径,必须切到仓库根。
    os.chdir(repo_root)

    with gym_config_path.open("r", encoding="utf-8") as handle:
        gym_config = json.load(handle)

    constants, constants_source = _resolve_constants()
    ctx = Context(
        args=args,
        repo_root=repo_root,
        gym_config=gym_config,
        constants=constants,
        constants_source=constants_source,
        hooks=_resolve_button_hooks(args),
        downstream_blocking=not (bool(args.skip_depth) or bool(args.skip_vlm)),
    )
    _import_hybrid(ctx)

    # 官方 scripts/eval_policy.py 的三个入口(照抄 handle_basket_flow.main)。
    from eval_policy import (  # noqa: PLC0415
        find_action_config,
        make_env_from_configs,
        prepare_episode_reset,
    )

    cfg: dict[str, Any] = {
        "task_name": TASK_NAME,
        "setting": str(args.setting),
        "num_envs": 1,
        "device": str(args.device),
        "headless": bool(args.headless),
        "gpu_id": int(args.gpu_id),
        "renderer": str(args.renderer),
        "filter_dataset_saving": True,
        "eval_video_log": False,
        "eval_reset_sync_steps": 1,
        "max_steps": int(args.max_steps),
    }
    action_config = find_action_config(cfg)
    env, gym_config_used = make_env_from_configs(cfg, gym_config, action_config)
    base, hops = _unwrap_base(env)

    actual_seed = prepare_episode_reset(env, int(args.seed))
    env.reset(seed=int(actual_seed), options={"save_data": False})

    ctx.env = env
    ctx.base = base
    _apply_manual_pixel(ctx)
    ctx.setup = {
        "ok": True,
        "error": None,
        "repo_root": str(repo_root),
        "gym_config_path": str(gym_config_path),
        "eval_policy_path": str(eval_policy_path),
        "gym_id": gym_config_used.get("id"),
        "gym_max_episode_steps": gym_config_used.get("max_episode_steps"),
        "requested_seed": int(args.seed),
        "actual_seed": int(actual_seed),
        "env_chain_hops": int(hops),
        "base_type": type(base).__name__,
        "sys_path_added": added_paths,
        "env_cfg": _jsonable(cfg),
        "embodichain_root": os.environ.get("EMBODICHAIN_ROOT"),
        "module_provenance": _module_provenance(),
        "cwd": str(Path.cwd()),
        "constants_source": constants_source,
        "constants": _jsonable(constants),
        "button_hooks": _jsonable(ctx.hooks.describe()) if ctx.hooks else None,
        "settings": _jsonable(ctx.settings.describe()) if ctx.settings is not None else None,
    }
    return ctx


# ===========================================================================
# L1 —— 腕相机安装常量
# ===========================================================================


def _verify_mount_order(ctx: Context, arm: str) -> dict[str, Any]:
    """额外做一次"四元数顺序开关"对照:直接量测安装矩阵 vs 两种读法。

    这是 L1 里**最可操作**的一格:``priors.camera_mount`` 的顺序由
    ``HYBRID_QUAT_ORDER`` 决定,若实测支持 ``xyzw``,用户只要改一个环境变量。
    """
    stereo, priors = ctx.stereo, ctx.priors
    result: dict[str, Any] = {"ok": False}
    try:
        sensors = getattr(ctx.base, "sensors", None)
        if sensors is None:
            result["error"] = "base 上取不到 env.sensors"
            return result
        sensor = sensors[f"cam_{arm}_wrist"]
        arena = np.asarray(
            _npv(sensor.get_arena_pose(to_matrix=True))[0], dtype=np.float64
        ).reshape(4, 4)
        truth = arena @ priors.SIM_TO_OPTICAL
        link = np.asarray(stereo.link6_pose(ctx.base, arm), dtype=np.float64).reshape(4, 4)
        measured_mount = np.linalg.inv(link) @ truth
        result["measured_mount"] = _jsonable(measured_mount)
        result["measured_mount_pos_m"] = _jsonable(measured_mount[:3, 3])
        orders: dict[str, Any] = {}
        for order in ("wxyz", "xyzw"):
            mount = np.asarray(priors.camera_mount(arm, quat_order=order), dtype=np.float64)
            delta = _pose_delta(link @ mount, truth)
            raw_delta = _pose_delta(link @ (mount @ priors.SIM_TO_OPTICAL), arena)
            orders[order] = {
                "mount_delta": delta,
                "raw_mount_delta": raw_delta,
                "pos_err_mm": _mm(delta["pos_err_m"]),
                "rot_err_deg": delta["rot_err_deg"],
            }
        result["orders"] = orders
        # 生效顺序的"开关"在 stereo.mount_quat_order()(内部读 settings.quat_order);
        # 老版本没有这个函数时退回 settings,再不行就退回 stereo 的模块常量。
        configured: Any = None
        getter = getattr(stereo, "mount_quat_order", None)
        if callable(getter):
            configured = getter()
        elif ctx.settings is not None:
            configured = getattr(ctx.settings, "quat_order", None)
        if configured is None:
            configured = getattr(stereo, "MOUNT_QUAT_ORDER", None)
        result["configured_order"] = configured
        good = [
            order
            for order, item in orders.items()
            if item["pos_err_mm"] < 2.0 and item["rot_err_deg"] < 0.5
        ]
        result["matching_orders"] = good
        result["recommended_order"] = good[0] if good else None
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - 只是附加信息
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _layer1(ctx: Context, layer: Layer) -> None:
    """L1:``stereo.verify_mount(base, arm)`` —— 判定 ``wxyz/post_flip`` 且 <2mm/<0.5°。"""
    arm = str(ctx.args.arm)
    record = ctx.stereo.verify_mount(ctx.base, arm)
    layer.data["verify_mount"] = _jsonable(record)

    sensor_available = bool(record.get("sensor_available"))
    best = record.get("best_variant")
    best_delta = record.get("best_variant_delta") or {}
    pos_mm = _mm(_f(best_delta.get("pos_err_m"), float("nan")))
    rot_deg = _f(best_delta.get("rot_err_deg"), float("nan"))

    order_check = _verify_mount_order(ctx, arm)
    layer.data["order_check"] = _jsonable(order_check)
    # 哪个检出被加载决定了"该按 wxyz 还是 xyzw 读安装常量",必须留证。
    layer.data["module_provenance"] = _jsonable(ctx.setup.get("module_provenance"))

    layer.check("env.sensors 可用", sensor_available, record.get("error") or "")
    if not sensor_available:
        layer.mark_skip(
            "无法判定:env.sensors 不可用,verify_mount 拿不到安装真值("
            f"{record.get('error')});请在评测机上用未包裹的 env 重跑",
            blocking=True,
            reason="env_sensors_unavailable",
        )
        return

    expected = "/".join(record.get("reference_variant") or ["wxyz", "post_flip"])
    layer.check(f"best_variant == {expected}", best == expected, f"实测 best_variant={best}")
    layer.check("位置误差 < 2 mm", pos_mm < 2.0, f"{pos_mm:.3f} mm")
    layer.check("旋转误差 < 0.5 deg", rot_deg < 0.5, f"{rot_deg:.3f} deg")
    recommended = order_check.get("recommended_order")
    if recommended:
        layer.check(
            "HYBRID_QUAT_ORDER 与实测一致",
            recommended == order_check.get("configured_order"),
            f"实测推荐 {recommended},当前配置 {order_check.get('configured_order')}",
        )
    layer.data["pos_err_mm"] = pos_mm
    layer.data["rot_err_deg"] = rot_deg

    if bool(record.get("ok")) and best == expected and pos_mm < 2.0 and rot_deg < 0.5:
        layer.mark_pass(
            f"通过:{best} 位置误差 {pos_mm:.3f} mm / 旋转误差 {rot_deg:.3f} deg"
        )
    else:
        layer.mark_fail(
            f"不通过:best_variant={best} 位置误差 {pos_mm:.3f} mm / 旋转误差 {rot_deg:.3f} deg"
            f"(要求 {expected} 且 <2 mm / <0.5 deg)"
        )


# ===========================================================================
# L2 —— 运动双目深度
# ===========================================================================


def _depth_stats(depth_map: Any, valid_map: Any) -> dict[str, float]:
    """有效像素比例、深度中位数、最小/最大值(全部只看有效像素)。"""
    depth = np.asarray(depth_map, dtype=np.float64)
    valid = np.asarray(valid_map).astype(bool)
    good = valid & np.isfinite(depth) & (depth > 1e-4)
    values = depth[good]
    if values.size == 0:
        return {
            "valid_fraction": float(valid.mean()) if valid.size else float("nan"),
            "valid_depth_fraction": 0.0,
            "depth_median_m": float("nan"),
            "depth_min_m": float("nan"),
            "depth_max_m": float("nan"),
            "valid_pixels": 0,
        }
    return {
        "valid_fraction": float(valid.mean()),
        "valid_depth_fraction": float(good.mean()),
        "depth_median_m": float(np.median(values)),
        "depth_min_m": float(values.min()),
        "depth_max_m": float(values.max()),
        "valid_pixels": int(values.size),
    }


def _layer2(ctx: Context, layer: Layer) -> None:
    """L2:``stereo.fetch_depth`` —— 键齐全、有效比例、深度中位数、baseline、校正角。"""
    if ctx.args.skip_depth:
        ctx.downstream_blocking = False
        layer.mark_skip("跳过:用户指定 --skip-depth", blocking=False, reason="cli_flag")
        return

    stereo = ctx.stereo
    arm = str(ctx.args.arm)
    q, _grip = stereo.arm_joints(ctx.base, arm)
    depth = stereo.fetch_depth(
        ctx.base,
        arm,
        q,
        settle_steps=max(0, int(ctx.args.settle_steps)),
        motion_steps=max(1, int(ctx.args.motion_steps)),
        nudge_candidates=int(ctx.args.nudge_candidates),
        baseline_target=float(ctx.args.baseline_target),
    )
    ctx.depth = depth

    missing = [key for key in DEPTH_REQUIRED_KEYS if key not in depth]
    extra = [key for key in depth if key not in DEPTH_REQUIRED_KEYS]
    layer.data["key_count"] = len(depth)
    layer.data["expected_key_count"] = len(DEPTH_REQUIRED_KEYS)
    layer.data["missing_keys"] = missing
    layer.data["extra_keys"] = extra
    layer.check(
        f"{len(DEPTH_REQUIRED_KEYS)} 个契约键齐全",
        not missing,
        f"缺失 {missing}" if missing else f"齐全(实际返回 {len(depth)} 个键)",
    )
    if missing:
        layer.mark_fail(f"不通过:fetch_depth 返回缺少契约键 {missing}")
        return

    stats = _depth_stats(depth["depth_map"], depth["valid_map"])
    layer.data["depth_stats"] = _jsonable(stats)
    baseline = _f(depth["baseline_m"])
    motion = _vec3(depth["motion_camera_m"])
    motion_norm = float(np.linalg.norm(motion))
    rect_deg = _f(depth["rectification_deg"])
    ref_rgb = np.asarray(depth["reference_rgb"])
    depth_map = np.asarray(depth["depth_map"])
    layer.data["baseline_m"] = baseline
    layer.data["baseline_mm"] = _mm(baseline)
    layer.data["motion_camera_m"] = motion
    layer.data["motion_camera_norm_mm"] = _mm(motion_norm)
    layer.data["rectification_deg"] = rect_deg
    layer.data["reference_exposure"] = int(depth["reference_exposure"])
    layer.data["reference_rgb_shape"] = list(ref_rgb.shape)
    layer.data["depth_map_shape"] = list(depth_map.shape)
    layer.data["reference_camera_pose"] = _jsonable(depth["reference_camera_pose"])
    layer.data["K_rect"] = _jsonable(depth["K_rect"])

    layer.check(
        "深度图与参考 RGB 同尺寸",
        tuple(depth_map.shape[:2]) == tuple(ref_rgb.shape[:2]),
        f"depth={tuple(depth_map.shape[:2])} rgb={tuple(ref_rgb.shape[:2])}",
    )
    min_valid = float(ctx.args.min_valid_fraction)
    layer.check(
        f"有效像素比例 >= {min_valid:.3f}",
        stats["valid_fraction"] >= min_valid,
        f"{stats['valid_fraction']:.4f}",
    )
    layer.check(
        "深度中位数在 [0.05, 3.0] m 内",
        0.05 <= _f(stats["depth_median_m"], -1.0) <= 3.0,
        f"{_f(stats['depth_median_m']):.4f} m",
    )
    layer.check(
        "baseline 在 [5, 70] mm 内",
        0.005 <= baseline <= 0.070,
        f"{_mm(baseline):.2f} mm(运动位移 {_mm(motion_norm):.2f} mm)",
    )
    layer.check(
        "rectification 角度在 [0, 90] deg 内",
        0.0 <= rect_deg <= 90.0,
        f"{rect_deg:.3f} deg",
    )

    ctx.reference_rgb = depth["reference_rgb"]
    ctx.reference_source = "L2 fetch_depth reference_rgb"

    ok = (
        not missing
        and stats["valid_fraction"] >= min_valid
        and 0.05 <= _f(stats["depth_median_m"], -1.0) <= 3.0
        and 0.005 <= baseline <= 0.070
        and 0.0 <= rect_deg <= 90.0
    )
    summary = (
        f"{len(depth)} 键齐全;有效像素 {stats['valid_fraction'] * 100:.1f}%,"
        f"深度中位数 {_f(stats['depth_median_m']):.4f} m,"
        f"baseline {_mm(baseline):.2f} mm,校正角 {rect_deg:.3f} deg"
    )
    if ok:
        layer.mark_pass("通过:" + summary)
    else:
        layer.mark_fail("不通过:" + summary)

    # 抓完深度后把手臂放回 q1(参考曝光位),让后续 IK 的初始种子更自然。
    # 这一步失败不影响本层结论。
    try:
        stereo.move_arm(ctx.base, arm, q, max(1, int(ctx.args.motion_steps)))
        layer.data["arm_restored_to_q1"] = True
    except Exception as exc:  # noqa: BLE001
        layer.data["arm_restored_to_q1"] = False
        layer.data["arm_restore_error"] = f"{type(exc).__name__}: {exc}"


# ===========================================================================
# L3 —— VLM 按钮定位
# ===========================================================================


@dataclass
class ButtonHooks:
    """按钮定位/反投影要用到的一组"策略同款"钩子(全部带兜底)。"""

    prompt: Callable[[int, int], str]
    finish: Callable[[dict, dict | None, int, int], dict]
    grid: int
    min_samples: int
    sample_radius: int
    prompt_source: str
    parse_source: str
    geometry_source: str
    import_error: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "prompt_source": self.prompt_source,
            "parse_source": self.parse_source,
            "geometry_source": self.geometry_source,
            "grid": self.grid,
            "min_samples": self.min_samples,
            "sample_radius": self.sample_radius,
            "import_error": self.import_error,
        }


def _builtin_finish(
    reply: dict, parsed: dict | None, width: int, height: int
) -> dict:
    """内置解析器的适配层:统一成 ``(reply, parsed, width, height)`` 签名。"""
    return _finish_button_reply(parsed, width, height)


def _resolve_button_hooks(args: argparse.Namespace) -> ButtonHooks:
    """优先复用 ``robosyn_hybrid.clickbell`` 的按钮 prompt / 解析 / 网格常量。

    这是本脚本**保真度**的关键:只有用策略同款的 prompt、解析器和 bbox 网格,
    L3/L4 的结论才等价于"策略在评测机上跑得通吗"。策略模块缺失或改名时逐项
    退回内置兜底,并在输出里写明每一项实际来自哪里。
    """
    hooks = ButtonHooks(
        prompt=_button_prompt,
        finish=_builtin_finish,
        grid=5,
        min_samples=3,
        sample_radius=3,
        prompt_source="diagnostic_builtin",
        parse_source="diagnostic_builtin",
        geometry_source="diagnostic_builtin",
    )
    if args.builtin_button_hooks:
        return hooks
    try:
        from robosyn_hybrid import clickbell  # type: ignore  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - 策略模块可能还没写完
        hooks.import_error = f"{type(exc).__name__}: {str(exc)[:160]}"
        return hooks

    prompt_fn = getattr(clickbell, "_button_prompt", None)
    if callable(prompt_fn):
        try:
            probe = prompt_fn(640, 480)
            if isinstance(probe, str) and probe.strip():
                hooks.prompt = prompt_fn
                hooks.prompt_source = "clickbell._button_prompt"
        except Exception:  # noqa: BLE001 - 签名不符就退回内置
            pass
    finish_fn = getattr(clickbell, "_finish_button", None)
    if callable(finish_fn):
        try:
            # 用"无 JSON"的探针验签名:纯函数,不产生副作用。
            probe = finish_fn({"ok": True, "text": ""}, None, 640, 480)
            if isinstance(probe, dict) and "ok" in probe:
                hooks.finish = finish_fn
                hooks.parse_source = "clickbell._finish_button"
        except Exception:  # noqa: BLE001
            pass
    reused: list[str] = []
    for attr, field_name in (
        ("BUTTON_BBOX_GRID", "grid"),
        ("BUTTON_MIN_SAMPLES", "min_samples"),
        ("BUTTON_SAMPLE_RADIUS", "sample_radius"),
    ):
        value = getattr(clickbell, attr, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            setattr(hooks, field_name, int(value))
            reused.append(attr)
    if reused:
        hooks.geometry_source = "clickbell." + ",".join(sorted(reused))

    # 用户显式指定的阈值优先于策略常量。
    if args.button_grid is not None:
        hooks.grid = max(2, int(args.button_grid))
        hooks.geometry_source += "+cli(--button-grid)"
    if args.sample_radius is not None:
        hooks.sample_radius = max(0, int(args.sample_radius))
        hooks.geometry_source += "+cli(--sample-radius)"
    return hooks


def _button_prompt(width: int, height: int) -> str:
    """按钮定位 prompt 的**内置兜底副本**。

    首选是策略模块 ``robosyn_hybrid.clickbell._button_prompt``(见
    :func:`_resolve_button_hooks`):只有用策略同款 prompt,L3/L4 才真正等价于
    "策略在评测机上会不会跑通"。当策略模块还没落地 / 改名时,才退回这一份。
    两份的 schema 完全一致(``visible`` / ``u`` / ``v`` / ``bbox_xyxy`` /
    ``confidence`` / ``note``),所以下游解析无需分支。
    """
    return (
        "Return strict JSON only, with no markdown. Locate the push button visible "
        "in this wrist RGB frame. It is a small round cap/disc standing on top of a "
        "slightly wider cylindrical base; its colour and texture may be randomized. "
        "Report the centre of the visible button top face in pixel coordinates and an "
        "axis-aligned bbox tightly around that top face. "
        "If not visible set visible=false. Schema: "
        '{"visible":true|false,"u":number|null,"v":number|null,'
        '"bbox_xyxy":[left,top,right,bottom]|null,"confidence":0..1,'
        '"note":"short reason"}.'
        f" Image size is {width}x{height}."
    )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """从 VLM 回复里抠出第一个 JSON 对象(容忍 markdown 代码块与前后噪声)。"""
    if not text:
        return None
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        newline = cleaned.find("\n")
        if newline != -1 and cleaned[:newline].strip().lower() in {"json", ""}:
            cleaned = cleaned[newline + 1 :]
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001 - 继续用括号扫描
        pass
    depth = 0
    start = -1
    for index, char in enumerate(cleaned):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(cleaned[start : index + 1])
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(parsed, dict):
                    return parsed
    return None


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return None
    return out if math.isfinite(out) else None


def _finish_button_reply(
    parsed: dict[str, Any] | None, width: int, height: int
) -> dict[str, Any]:
    """把 VLM 的按钮回复裁剪/校验成 ``{visible, uv, bbox_xyxy, ...}``。

    容错优先:``visible`` 缺省时只要给了 bbox/u 就当作可见。
    """
    out: dict[str, Any] = {"ok": False, "error": None, "answer": _jsonable(parsed)}
    if parsed is None:
        out["error"] = "VLM 回复里没有可解析的 JSON"
        return out
    bbox_raw = parsed.get("bbox_xyxy")
    bbox: list[float] | None = None
    if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
        values = [_as_float(item) for item in bbox_raw]
        if all(item is not None for item in values):
            x0, y0, x1, y1 = (float(item) for item in values)  # type: ignore[misc]
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            bbox = [
                max(0.0, min(width - 1.0, x0)),
                max(0.0, min(height - 1.0, y0)),
                max(0.0, min(width - 1.0, x1)),
                max(0.0, min(height - 1.0, y1)),
            ]
    u = _as_float(parsed.get("u"))
    v = _as_float(parsed.get("v"))
    if (u is None or v is None) and bbox is not None:
        u = 0.5 * (bbox[0] + bbox[2])
        v = 0.5 * (bbox[1] + bbox[3])
        out["uv_from"] = "bbox_centre"
    visible = parsed.get("visible")
    if visible is None:
        visible = bbox is not None or (u is not None and v is not None)
        out["visible_from"] = "inferred"
    if not bool(visible):
        out["visible"] = False
        out["error"] = "VLM 报告按钮不可见"
        return out
    if u is None or v is None:
        out["error"] = "VLM 没有给出可用的中心像素"
        return out
    if not (0 <= u < width and 0 <= v < height):
        out["error"] = f"VLM 中心像素越界:({u:.1f}, {v:.1f})"
        return out
    if bbox is None:  # 只给了中心:退化成一个 36x36 的框,够 L4 采样。
        radius = 18.0
        bbox = [
            max(0.0, u - radius),
            max(0.0, v - radius),
            min(width - 1.0, u + radius),
            min(height - 1.0, v + radius),
        ]
        out["bbox_from"] = "centre_fallback"
    out.update(
        {
            "ok": True,
            "visible": True,
            "uv": [float(u), float(v)],
            "bbox_xyxy": [float(item) for item in bbox],
            "confidence": _as_float(parsed.get("confidence")),
            "note": parsed.get("note"),
        }
    )
    return out


def _apply_manual_pixel(ctx: Context) -> None:
    """把 ``--button-uv`` / ``--button-bbox`` 落到上下文(幂等)。

    在构造环境后就先应用一次:这样即使 L3 因为端点不可用而整层失败,用户手动
    给的像素依然能让 L4-L7 跑起来。
    """
    args = ctx.args
    if args.button_bbox is not None:
        ctx.button_bbox = [float(x) for x in args.button_bbox]
    if args.button_uv is not None:
        ctx.button_uv = [float(x) for x in args.button_uv]
    if ctx.button_bbox is None and ctx.button_uv is not None:
        radius = 18.0  # 与 clickbell/vlm 的"只有中心点"兜底半径一致
        ctx.button_bbox = [
            ctx.button_uv[0] - radius,
            ctx.button_uv[1] - radius,
            ctx.button_uv[0] + radius,
            ctx.button_uv[1] + radius,
        ]
    if ctx.button_uv is None and ctx.button_bbox is not None:
        ctx.button_uv = [
            0.5 * (ctx.button_bbox[0] + ctx.button_bbox[2]),
            0.5 * (ctx.button_bbox[1] + ctx.button_bbox[3]),
        ]


def _look_search_flow(ctx: Context) -> tuple[Any, Any, str]:
    """构造一个"最小 ``ClickBellFlow`` + ``ButtonViewSearch``"组合。

    L3 必须走**和策略完全相同**的 LOOK 代码路径,而不是另写一份搜索逻辑,
    所以这里直接复用 ``robosyn_hybrid.clickbell``:

    * ``ClickBellFlow`` 只用来提供已绑定的 ``env/base``、``_servo_qpos`` 与
      ``_ik_from_seeds``(它的 ACT 部分完全用不到,``act_policy`` 传 ``None``);
    * ``ButtonViewSearch`` 就是策略 ``_stage_look`` 里用的那个搜索器。

    这样"诊断能不能看到按钮"与"策略在评测机上能不能看到按钮"度量的是同一条
    路径(候选点生成、反解方向、IK、伺服、VLM 提问与解析全部同源)。

    返回 ``(flow, search, source)``。
    """
    from robosyn_hybrid import clickbell  # type: ignore  # noqa: PLC0415

    debug_dir = None
    if ctx.args.out:
        debug_dir = Path(str(ctx.args.out)).with_suffix("")
    flow = clickbell.ClickBellFlow(
        None,
        ctx.settings,
        max_env_steps=int(ctx.args.max_steps),
        debug_dir=debug_dir,
    )
    flow._bind_env(ctx.env)
    search = clickbell.ButtonViewSearch(flow, arm=str(ctx.args.arm))
    return flow, search, "robosyn_hybrid.clickbell.ButtonViewSearch"


def _look_trace_summary(record: dict[str, Any]) -> str:
    """把 LOOK 搜索记录压成一行人类可读文本(每个执行过的视角一条)。"""
    parts: list[str] = []
    for item in record.get("trace") or []:
        if not item.get("executed"):
            continue
        vision = item.get("vlm") or {}
        parts.append(
            "{label}@{pt} steps={steps} vlm={visible}({lat:.2f}s){uv}".format(
                label=item.get("view"),
                pt=[round(float(v), 2) for v in (item.get("point_m") or [])[:2]],
                steps=(item.get("move") or {}).get("steps_used"),
                visible=bool(vision.get("visible")),
                lat=float(vision.get("latency_s") or 0.0),
                uv=(
                    " uv=" + str([round(float(v), 1) for v in vision["uv"]])
                    if vision.get("uv")
                    else ""
                ),
            )
        )
    return "; ".join(parts) if parts else "(没有任何视角真正执行)"


def _layer3(ctx: Context, layer: Layer) -> None:
    """L3:LOOK 视野搜索 + 按钮定位(与策略同一条代码路径)。

    与旧版的区别:不再直接问初始位姿那一帧,而是先跑策略的 ``LOOK``
    **视野搜索**(在官方公开的按钮区域常量
    ``configs/click_bell/random/gym_config.json:init_button_pose.position_range``
    内由中心向外试视角),命中后再在**命中视角**上重取一路深度并重新问一次
    VLM(对应策略的 ``DEPTH_FETCH`` → ``VLM_LOCATE``)。这样 L4 的反投影用的
    就是"策略真正会用的那一帧"。
    """
    args = ctx.args
    vlm = ctx.vlm
    if args.skip_vlm:
        ctx.downstream_blocking = False
        layer.mark_skip("跳过:用户指定 --skip-vlm", blocking=False, reason="cli_flag")
    else:
        available = bool(vlm.vlm_available())
        layer.data["vlm_available"] = available
        layer.data["vlm_url_configured"] = bool(
            ctx.settings is not None and ctx.settings.vlm_url
        )
        layer.data["vlm_enabled"] = bool(
            ctx.settings is None or ctx.settings.vlm_enabled
        )
        layer.check("VLM 端点已配置(HYBRID_VLM_URL)", available)
        if not available:
            reason = (
                "VLM 不可用:未配置 HYBRID_VLM_URL 或 HYBRID_VLM_ENABLED=0。"
                "L4/L5 没有像素输入就无从检验,等价于整条链路无法验证"
            )
            manual = ctx.args.button_uv is not None or ctx.args.button_bbox is not None
            if manual:
                layer.mark_skip(
                    reason + ";用户已用 --button-uv/--button-bbox 手动给像素,本层不阻塞",
                    blocking=False,
                    reason="vlm_unavailable_manual_pixel",
                )
            else:
                layer.mark_skip(reason, blocking=True, reason="vlm_unavailable")
        else:
            try:
                flow, search, source = _look_search_flow(ctx)
            except Exception as exc:  # noqa: BLE001 - 复用失败就明确失败,不静默换路径
                error = f"{type(exc).__name__}: {exc}"
                layer.check("复用 clickbell 的 LOOK 搜索器", False, error)
                layer.mark_fail(
                    f"不通过:无法构造策略同款 LOOK 搜索器({error});"
                    "诊断必须跑策略真正会走的那条代码路径"
                )
                _apply_manual_pixel(ctx)
                return

            layer.data["look_source"] = source
            layer.data["look_config"] = _jsonable(search.describe())
            hook_info = getattr(ctx.hooks, "describe", None)
            layer.data["button_hooks"] = _jsonable(hook_info() if callable(hook_info) else None)
            started = time.perf_counter()
            try:
                record = search.run()
            except Exception as exc:  # noqa: BLE001 - 搜索自身异常也不外抛
                record = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": [],
                    "hit": None,
                }
            layer.duration_s = time.perf_counter() - started
            if not isinstance(record, dict):
                record = {"ok": False, "error": "look search returned non-dict"}
            layer.data["look"] = _jsonable(record)
            layer.data["look_steps_used"] = int(record.get("steps_used") or 0)
            layer.data["look_views_executed"] = int(record.get("views_executed") or 0)
            layer.data["look_candidate_points"] = _jsonable(
                record.get("candidate_points_m")
            )
            layer.data["look_stop_reason"] = record.get("stop_reason")
            layer.check(
                "LOOK 视野搜索总步数 <= 预算",
                int(record.get("steps_used") or 0)
                <= int((record.get("config") or {}).get("step_budget") or 0),
                f"用了 {record.get('steps_used')} 步 / 预算 "
                f"{(record.get('config') or {}).get('step_budget')} 步",
            )

            hit = record.get("hit") if isinstance(record.get("hit"), dict) else None
            if hit is None:
                layer.check("LOOK 视野搜索命中按钮", False, _look_trace_summary(record))
                layer.mark_fail(
                    "不通过:LOOK 视野搜索没有找到「看得见按钮」的视角"
                    f"({record.get('error')});轨迹:{_look_trace_summary(record)}"
                )
            else:
                ctx.look = _jsonable(record)
                ctx.look_hit = _jsonable(hit)
                ctx.look_uv = [float(v) for v in hit["uv"]]
                ctx.look_bbox = [float(v) for v in hit["bbox_xyxy"]]
                ctx.look_qpos = np.asarray(hit["qpos"], dtype=np.float64).reshape(-1)
                ctx.look_camera_pose = np.asarray(
                    hit["camera_pose"], dtype=np.float64
                ).reshape(4, 4)
                ctx.look_rgb = getattr(search, "hit_rgb", None)
                layer.data["look_hit"] = _jsonable(hit)
                layer.data["look_hit_uv"] = ctx.look_uv
                layer.data["look_hit_bbox_xyxy"] = ctx.look_bbox
                layer.data["look_trace"] = _jsonable(record.get("trace"))
                layer.check(
                    "LOOK 视野搜索命中按钮",
                    True,
                    f"{hit.get('view')} @ 候选点 "
                    f"{[round(float(v), 3) for v in hit['point_m'][:2]]}"
                    f"(第 {int(hit.get('index', 0)) + 1} 个候选),"
                    f"相机位姿偏差 {hit.get('camera_pose_error_mm')} mm",
                )
                layer.check(
                    "命中视角的 bbox 宽高 > 0",
                    float(ctx.look_bbox[2]) - float(ctx.look_bbox[0]) > 0.0
                    and float(ctx.look_bbox[3]) - float(ctx.look_bbox[1]) > 0.0,
                    f"bbox={[round(v, 1) for v in ctx.look_bbox]}",
                )

                # —— 命中视角上重取深度(策略的 DEPTH_FETCH 用的是"看过的那个视角")
                vlm_uv = list(ctx.look_uv)
                vlm_bbox = list(ctx.look_bbox)
                source_note = "look_hit"
                if not args.skip_depth:
                    try:
                        depth2 = ctx.stereo.fetch_depth(
                            ctx.base,
                            str(args.arm),
                            ctx.look_qpos,
                            settle_steps=max(1, int(args.settle_steps)),
                            motion_steps=max(1, int(args.motion_steps)),
                            nudge_candidates=int(args.nudge_candidates),
                            baseline_target=float(args.baseline_target),
                        )
                        ctx.depth = depth2
                        ctx.reference_rgb = depth2["reference_rgb"]
                        ctx.reference_source = "L3 LOOK 命中视角的 fetch_depth"
                        layer.data["look_depth"] = _jsonable(
                            {
                                key: value
                                for key, value in depth2.items()
                                if key in DEPTH_REQUIRED_KEYS
                                and key not in ("reference_rgb", "reference_rect",
                                                "rect2", "disparity", "depth_map",
                                                "valid_map", "q1", "q2", "K0",
                                                "K_rect", "Rr1", "Rr2", "H_ref")
                            }
                        )
                        layer.data["look_depth_stats"] = _jsonable(
                            _depth_stats(depth2["depth_map"], depth2["valid_map"])
                        )
                        # —— 命中视角上再问一次 VLM(策略的 VLM_LOCATE)
                        hooks = ctx.hooks
                        rgb2 = np.asarray(depth2["reference_rgb"])
                        height2, width2 = int(rgb2.shape[0]), int(rgb2.shape[1])
                        prompt2 = (
                            hooks.prompt(width2, height2)
                            if hooks
                            else _button_prompt(width2, height2)
                        )
                        reply2 = vlm.vlm_chat(
                            prompt2, [rgb2], timeout_s=args.vlm_timeout
                        )
                        parsed2 = _parse_json_object(str(reply2.get("text") or ""))
                        finished2 = (
                            hooks.finish(reply2, parsed2, width2, height2)
                            if hooks
                            else _builtin_finish(reply2, parsed2, width2, height2)
                        )
                        if not isinstance(finished2, dict):
                            finished2 = _builtin_finish(
                                reply2, parsed2, width2, height2
                            )
                        layer.data["vlm_locate"] = _jsonable(finished2)
                        layer.data["vlm_locate_latency_s"] = _f(
                            reply2.get("latency_s")
                        )
                        if finished2.get("ok"):
                            vlm_uv = [float(v) for v in finished2["uv"]]
                            vlm_bbox = [float(v) for v in finished2["bbox_xyxy"]]
                            source_note = "look_hit + 命中视角 VLM_LOCATE"
                            layer.check(
                                "命中视角上重问 VLM 仍可见", True,
                                f"uv={[round(v, 1) for v in vlm_uv]},"
                                f"耗时 {_f(reply2.get('latency_s')):.2f} s",
                            )
                        else:
                            layer.check(
                                "命中视角上重问 VLM 仍可见",
                                None,
                                "重问未给出可用 bbox,沿用 LOOK 命中视角的结果:"
                                + str(finished2.get("error")),
                            )
                    except Exception as exc:  # noqa: BLE001 - 重取深度失败不推翻 L3
                        layer.data["look_depth_error"] = f"{type(exc).__name__}: {exc}"
                        layer.check(
                            "命中视角重取深度",
                            None,
                            f"失败({type(exc).__name__}: {exc});沿用 L2 的深度",
                        )
                else:
                    layer.check(
                        "命中视角重取深度", None, "用户指定 --skip-depth,未重取"
                    )

                ctx.button_uv = [float(v) for v in vlm_uv]
                ctx.button_bbox = [float(v) for v in vlm_bbox]
                layer.data["bbox_xyxy"] = ctx.button_bbox
                layer.data["uv"] = ctx.button_uv
                layer.data["pixel_source"] = source_note
                layer.mark_pass(
                    "通过:LOOK 命中视角 {} @ 候选点 {};"
                    "VLM bbox={},uv={};搜索用 {} 步 / {} 个视角;像素来源:{}".format(
                        hit.get("view"),
                        [round(float(v), 3) for v in hit["point_m"][:2]],
                        [round(v, 1) for v in ctx.button_bbox],
                        [round(v, 1) for v in ctx.button_uv],
                        record.get("steps_used"),
                        record.get("views_executed"),
                        source_note,
                    )
                )

    # 手动像素覆盖(VLM 跳过/失败时依然能让 L4-L7 跑起来;幂等)。
    _apply_manual_pixel(ctx)
    layer.data["manual_bbox_xyxy"] = _jsonable(ctx.args.button_bbox)
    layer.data["manual_uv"] = _jsonable(ctx.args.button_uv)
    layer.data["button_uv"] = _jsonable(ctx.button_uv)
    layer.data["button_bbox_xyxy"] = _jsonable(ctx.button_bbox)
    layer.data["reference_rgb_source"] = ctx.reference_source


# ===========================================================================
# L4 —— 按钮反投影
# ===========================================================================


def _bbox_grid_pixels(bbox: Sequence[float], grid: int) -> list[tuple[float, float]]:
    """bbox 内的 ``grid x grid`` 采样像素(逐行照抄 ``clickbell.py`` 的网格)。

    策略做法:按 bbox 宽高的 ``1/(2*grid)`` 向内收缩半个格子,再在收缩后的
    区间上 ``linspace``,避免采样点落到按钮边缘以外的背景上。本函数保持同一
    几何,这样 L4 的中位数与策略 ESTIMATE_POSE 的中位数可比。
    """
    values = [float(item) for item in list(bbox)[:4]]
    left, right = sorted((values[0], values[2]))
    top, bottom = sorted((values[1], values[3]))
    count = max(2, int(grid))
    inset_x = (right - left) / (2.0 * count)
    inset_y = (bottom - top) / (2.0 * count)
    us = np.linspace(left + inset_x, right - inset_x, count)
    vs = np.linspace(top + inset_y, bottom - inset_y, count)
    return [(float(u), float(v)) for v in vs for u in us]


def _layer4(ctx: Context, layer: Layer) -> None:
    """L4:由 bbox 反投影出按钮顶面世界坐标(多点取中位数,不做视线内推)。"""
    if ctx.depth is None:
        layer.mark_skip(
            "跳过:没有可用的深度结果(L2 未跑或失败),无法反投影",
            blocking=ctx.downstream_blocking,
            reason="cli_flag" if not ctx.downstream_blocking else "no_depth_result",
        )
        return
    if ctx.button_bbox is None and ctx.button_uv is None:
        layer.mark_skip(
            "跳过:没有按钮像素(L3 未给出 bbox/uv,也没有 --button-uv)",
            blocking=True,
            reason="no_button_pixel",
        )
        return

    stereo = ctx.stereo
    hooks = ctx.hooks or ButtonHooks(
        prompt=_button_prompt,
        finish=_builtin_finish,
        grid=5,
        min_samples=3,
        sample_radius=3,
        prompt_source="diagnostic_builtin",
        parse_source="diagnostic_builtin",
        geometry_source="diagnostic_builtin",
    )
    bbox = ctx.button_bbox if ctx.button_bbox is not None else [0.0, 0.0, 0.0, 0.0]
    pixels = _bbox_grid_pixels(bbox, hooks.grid)
    radius = max(0, int(hooks.sample_radius))
    layer.data["button_hooks"] = _jsonable(hooks.describe())
    layer.data["grid"] = int(hooks.grid)
    layer.data["min_samples"] = int(hooks.min_samples)
    points: list[list[float]] = []
    depths: list[float] = []
    attempts: list[dict[str, Any]] = []
    for u, v in pixels:
        estimate = stereo.estimate_point_from_pixel(ctx.depth, (u, v), radius)
        attempts.append(
            {
                "pixel": [round(u, 2), round(v, 2)],
                "ok": bool(estimate.get("ok")),
                "depth_m": estimate.get("depth_m"),
                "pixel_rectified": estimate.get("pixel_rectified"),
                "sample_count": estimate.get("sample_count"),
                "error": estimate.get("error"),
            }
        )
        if estimate.get("ok"):
            points.append([float(x) for x in estimate["p_world_m"]])
            depths.append(float(estimate["depth_m"]))

    layer.data["pixels_attempted"] = len(pixels)
    layer.data["pixels_ok"] = len(points)
    layer.data["sample_radius"] = radius
    layer.data["attempts"] = _jsonable(attempts)
    layer.check(
        f"有效采样点 >= {int(hooks.min_samples)}(策略同款下限)",
        len(points) >= int(hooks.min_samples),
        f"{len(points)}/{len(pixels)}",
    )
    if not points:
        layer.mark_fail("不通过:bbox 内所有采样点都没有有效深度")
        return

    arr = np.asarray(points, dtype=np.float64)
    median = np.median(arr, axis=0)
    spread = np.linalg.norm(arr - median[None, :], axis=1)
    ctx.button_world_m = median
    ctx.button_point_info = {
        "pixels_ok": len(points),
        "depth_median_m": float(np.median(depths)),
        "spread_max_mm": _mm(float(spread.max())),
    }
    layer.data["p_world_m"] = _jsonable(median)
    layer.data["depth_median_m"] = float(np.median(depths))
    layer.data["depth_min_m"] = float(np.min(depths))
    layer.data["depth_max_m"] = float(np.max(depths))
    layer.data["spread_max_mm"] = _mm(float(spread.max()))
    layer.data["spread_mean_mm"] = _mm(float(spread.mean()))
    layer.check(
        "反投影点离散度 < 10 mm(中位数可信)",
        _mm(float(spread.max())) < 10.0,
        f"最大偏离 {_mm(float(spread.max())):.2f} mm",
    )
    layer.mark_pass(
        f"通过:{len(points)}/{len(pixels)} 个采样点有效,"
        f"按钮顶面世界坐标 = ({median[0]:.4f}, {median[1]:.4f}, {median[2]:.4f}) m,"
        f"深度中位数 {np.median(depths):.4f} m,离散度 {_mm(float(spread.max())):.2f} mm"
    )


# ===========================================================================
# L5 —— 几何真值比对(全脚本唯一读真值处)
# ===========================================================================


def _mesh_vertices(path: Path) -> Any:
    """读 OBJ 顶点(``v x y z``)。其它格式明确报错,不猜。"""
    if path.suffix.lower() != ".obj":
        raise ValueError(f"只支持 .obj 视觉网格,收到 {path.suffix!r}")
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"{path} 里没有顶点")
    return np.asarray(vertices, dtype=np.float64)


def _urdf_visual_offsets(urdf_path: Path, link_name: str) -> list[dict[str, Any]]:
    """解析 URDF,返回某 link 全部 ``<visual>`` 的 (mesh 路径, origin, scale)。"""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    out: list[dict[str, Any]] = []
    for link in root.findall("link"):
        if link.get("name") != link_name:
            continue
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            origin = visual.find("origin")
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            if origin is not None:
                if origin.get("xyz"):
                    xyz = [float(x) for x in origin.get("xyz").split()]
                if origin.get("rpy"):
                    rpy = [float(x) for x in origin.get("rpy").split()]
            scale = [1.0, 1.0, 1.0]
            if mesh.get("scale"):
                scale = [float(x) for x in mesh.get("scale").split()]
            out.append(
                {
                    "mesh": str(mesh.get("filename")),
                    "xyz": xyz,
                    "rpy": rpy,
                    "scale": scale,
                }
            )
    return out


def _cover_geometry(ctx: Context) -> dict[str, Any]:
    """由官方资产(URDF + 视觉 mesh)推出按钮顶面在 ``button_cover`` link 系下的位置。

    这一步读的是**已公开发布的资产几何**(官方 ``assets/button/button.urdf``),
    不是场景真值位姿;它让 L5 能同时给出两个参照:
    官方 ``validate_depth_fetch.py`` 用的 cover **link 原点**,以及物理上真正被
    按压的 **顶面中心**。两者的 z 差在官方资产里约 29 mm,不区分会得出"假误差"。
    """
    info: dict[str, Any] = {"ok": False, "source": None, "error": None}
    try:
        entries = ctx.gym_config.get("articulation") or []
        entry = None
        for item in entries:
            if str(item.get("uid")) == BUTTON_UID:
                entry = item
                break
        if entry is None:
            raise KeyError(f"gym_config 里找不到 articulation uid={BUTTON_UID!r}")
        fpath = Path(str(entry.get("fpath")))
        urdf_path = fpath if fpath.is_absolute() else (ctx.repo_root / fpath)
        if not urdf_path.is_file():
            raise FileNotFoundError(f"按钮 URDF 不存在:{urdf_path}")
        visuals = _urdf_visual_offsets(urdf_path, BUTTON_COVER_LINK)
        if not visuals:
            raise KeyError(f"{urdf_path} 的 {BUTTON_COVER_LINK} 没有 visual mesh")
        all_vertices: list[Any] = []
        for visual in visuals:
            mesh_path = Path(visual["mesh"])
            if not mesh_path.is_absolute():
                mesh_path = urdf_path.parent / mesh_path
            vertices = _mesh_vertices(mesh_path)
            scale = np.asarray(visual["scale"], dtype=np.float64).reshape(3)
            vertices = vertices * scale[None, :]
            rotation = _rpy_to_matrix(*[float(x) for x in visual["rpy"]])
            translation = np.asarray(visual["xyz"], dtype=np.float64).reshape(3)
            all_vertices.append(vertices @ rotation.T + translation[None, :])
            info.setdefault("meshes", []).append(str(mesh_path))
        vertices_all = np.concatenate(all_vertices, axis=0)
        low = vertices_all.min(axis=0)
        high = vertices_all.max(axis=0)
        top_band = vertices_all[vertices_all[:, 2] >= high[2] - _TOP_BAND_M]
        top_center = top_band.mean(axis=0) if top_band.size else 0.5 * (low + high)
        info.update(
            {
                "ok": True,
                "source": "official_urdf_and_visual_mesh",
                "urdf": str(urdf_path),
                "bbox_min_m": _jsonable(low),
                "bbox_max_m": _jsonable(high),
                "top_offset_local_m": _jsonable(top_center),
                "top_z_m": float(high[2]),
                "top_band_vertices": int(top_band.shape[0]),
                "cover_radius_m": float(max(high[0] - low[0], high[1] - low[1]) * 0.5),
            }
        )
    except Exception as exc:  # noqa: BLE001 - 退化成只比对 link 原点
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _read_button_ground_truth(ctx: Context, cover_geom: dict[str, Any]) -> dict[str, Any]:
    """**唯一读真值的地方**:``base.sim.get_articulation("button")``。

    写法照抄官方 env(``_stage/official_clickbell/click_bell.py``)与官方
    真值校验脚本(``_stage/Depth_Fetch/click_bell/validate_depth_fetch.py``
    的 ``button_points``)。合规红线只约束策略本体,不约束本诊断脚本。
    """
    base = ctx.base
    sim = getattr(base, "sim", None)
    if sim is None:
        raise RuntimeError(
            "base 上取不到 .sim;解包链可能没到最内层 env,无法读按钮真值"
        )
    articulation = sim.get_articulation(BUTTON_UID)
    if articulation is None:
        raise RuntimeError(f"sim 里找不到 articulation uid={BUTTON_UID!r}")

    cover = np.asarray(
        _npv(articulation.get_link_pose(link_name=BUTTON_COVER_LINK, to_matrix=True))[0],
        dtype=np.float64,
    ).reshape(4, 4)
    base_link = np.asarray(
        _npv(articulation.get_link_pose(link_name=BUTTON_BASE_LINK, to_matrix=True))[0],
        dtype=np.float64,
    ).reshape(4, 4)
    qpos = np.asarray(_npv(articulation.get_qpos()), dtype=np.float64).reshape(-1)
    try:
        limits = np.asarray(
            _npv(articulation.get_qpos_limits()), dtype=np.float64
        ).reshape(-1, 2)
        lower, upper = float(limits[0, 0]), float(limits[0, 1])
    except Exception:  # noqa: BLE001 - 拿不到就用官方 URDF 的公开行程
        lower, upper = -BUTTON_JOINT_TRAVEL_FALLBACK_M, 0.0

    top_offset = np.asarray(
        cover_geom.get("top_offset_local_m") or [0.0, 0.0, 0.0], dtype=np.float64
    ).reshape(3)
    press_depth = -float(qpos[0]) if qpos.size else float("nan")
    record: dict[str, Any] = {
        "source": "base.sim.get_articulation('button')",
        "link_names": _jsonable(getattr(articulation, "link_names", None)),
        "joint_names": _jsonable(getattr(articulation, "joint_names", None)),
        "cover_pose": _jsonable(cover),
        "base_pose": _jsonable(base_link),
        "cover_origin_world_m": _jsonable(cover[:3, 3]),
        "base_origin_world_m": _jsonable(base_link[:3, 3]),
        "cover_top_world_m": _jsonable(cover[:3, :3] @ top_offset + cover[:3, 3]),
        "cover_top_offset_local_m": _jsonable(top_offset),
        "press_depth_m": press_depth,
        "joint_limit_m": [lower, upper],
        "required_press_depth_m": float(SUCCESS_PRESS_DEPTH_M),
        "travel_limit_m": float(abs(lower)),
        "travel_margin_m": float(abs(lower) - SUCCESS_PRESS_DEPTH_M),
    }
    return record


def _interpret_xyz_error(error_xyz_mm: Sequence[float]) -> str:
    """按误差主方向给一句可操作的启发式解释(明确标注是启发式)。"""
    dx, dy, dz = (float(x) for x in error_xyz_mm)
    horizontal = math.hypot(dx, dy)
    vertical = abs(dz)
    if vertical >= 2.0 * max(horizontal, 1e-6) and vertical > 2.0:
        return "误差以竖直(z)为主 → 优先怀疑深度尺度/按钮顶面高度,或 L1 腕相机安装"
    if horizontal >= 2.0 * max(vertical, 1e-6) and horizontal > 2.0:
        return "误差以水平(x/y)为主 → 优先怀疑 VLM 像素偏移或 L1 相机外参"
    if max(horizontal, vertical) <= 2.0:
        return "三维误差都在 2 mm 内 → 反投影链路自洽"
    return "误差在水平与竖直上都显著 → 先复核 L1,再看 L2 深度质量"


def _project_world_point(camera_pose: Any, K: Any, point_world: Any) -> dict[str, Any]:
    """世界点投到 (u, v) 像素(用参考曝光的外参,不读模拟器渲染)。"""
    cam = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)
    intrinsics = np.asarray(K, dtype=np.float64).reshape(3, 3)
    point = np.asarray(point_world, dtype=np.float64).reshape(3)
    p_cam = cam[:3, :3].T @ (point - cam[:3, 3])
    if p_cam[2] <= 1e-6:
        return {"ok": False, "error": "点在相机后方"}
    uv = intrinsics @ p_cam
    return {
        "ok": True,
        "uv": [float(uv[0] / uv[2]), float(uv[1] / uv[2])],
        "camera_z_m": float(p_cam[2]),
    }


def _layer5(ctx: Context, layer: Layer) -> None:
    """L5:几何真值比对(本脚本唯一读真值的一层)。"""
    if ctx.button_world_m is None:
        layer.mark_skip(
            "跳过:没有反投影结果(L4 未通过),无法做真值比对",
            blocking=ctx.downstream_blocking,
            reason="cli_flag" if not ctx.downstream_blocking else "no_estimate",
        )
        return

    cover_geom = _cover_geometry(ctx)
    ctx.cover_geometry = cover_geom
    layer.data["cover_geometry"] = _jsonable(cover_geom)
    gt = _read_button_ground_truth(ctx, cover_geom)
    ctx.gt = gt
    layer.data["ground_truth"] = _jsonable(gt)

    estimate = np.asarray(ctx.button_world_m, dtype=np.float64).reshape(3)
    true_top = np.asarray(gt["cover_top_world_m"], dtype=np.float64).reshape(3)
    true_origin = np.asarray(gt["cover_origin_world_m"], dtype=np.float64).reshape(3)

    err_top_m = estimate - true_top
    err_origin_m = estimate - true_origin
    err_top_mm = [_mm(x) for x in err_top_m]
    err_origin_mm = [_mm(x) for x in err_origin_m]
    lateral_mm = math.hypot(err_top_mm[0], err_top_mm[1])
    dz_mm = err_top_mm[2]
    norm_mm = float(np.linalg.norm(err_top_mm))

    layer.data["estimate_world_m"] = _jsonable(estimate)
    layer.data["truth_cover_top_world_m"] = _jsonable(true_top)
    layer.data["truth_cover_origin_world_m"] = _jsonable(true_origin)
    layer.data["error_vs_top_xyz_mm"] = err_top_mm
    layer.data["error_vs_top_xyz_m"] = _jsonable(err_top_m)
    layer.data["error_vs_top_norm_mm"] = norm_mm
    # 照抄官方真值校验脚本的口径:相对 button_cover **link 原点**的误差。
    layer.data["error_vs_cover_origin_xyz_mm"] = err_origin_mm
    layer.data["error_vs_cover_origin_norm_mm"] = float(np.linalg.norm(err_origin_mm))
    layer.data["lateral_error_mm"] = lateral_mm
    layer.data["vertical_error_mm"] = dz_mm
    layer.data["interpretation_heuristic"] = _interpret_xyz_error(err_top_mm)

    # 像素域交叉验证:把真值顶面中心投回参考帧,与 VLM 像素比。
    if ctx.depth is not None:
        projection = _project_world_point(
            ctx.depth["reference_camera_pose"], ctx.depth["K0"], true_top
        )
        layer.data["gt_projection"] = _jsonable(projection)
        if projection.get("ok") and ctx.button_uv is not None:
            du = float(ctx.button_uv[0]) - float(projection["uv"][0])
            dv = float(ctx.button_uv[1]) - float(projection["uv"][1])
            layer.data["gt_uv"] = projection["uv"]
            layer.data["vlm_uv"] = [float(x) for x in ctx.button_uv]
            layer.data["uv_error_px"] = [du, dv]
            layer.data["uv_error_norm_px"] = math.hypot(du, dv)
            layer.check(
                "真值顶面投影 vs VLM 像素 < 40 px",
                math.hypot(du, dv) < 40.0,
                f"du={du:+.1f} px, dv={dv:+.1f} px",
            )

    lateral_tol = float(ctx.args.lateral_tol_mm)
    max_high = float(ctx.args.max_high_mm)
    min_low = float(ctx.args.min_low_mm)
    layer.check(
        f"横向误差 <= {lateral_tol:.1f} mm",
        lateral_mm <= lateral_tol,
        f"{lateral_mm:.2f} mm(分解 dx={err_top_mm[0]:+.2f}, dy={err_top_mm[1]:+.2f} mm)",
    )
    layer.check(
        f"竖直误差 <= +{max_high:.2f} mm(太高会压不到位)",
        dz_mm <= max_high,
        f"{dz_mm:+.2f} mm",
    )
    layer.check(
        f"竖直误差 >= {min_low:.2f} mm(太低会把工具压进按钮/底座)",
        dz_mm >= min_low,
        f"{dz_mm:+.2f} mm",
    )
    travel_margin = _f(gt.get("travel_margin_m"))
    layer.check(
        "按钮行程裕量 >= 0(4.8 mm 成功判据 vs 关节行程)",
        travel_margin >= 0.0,
        f"行程 {_mm(_f(gt.get('travel_limit_m'))):.2f} mm,"
        f"需要 {_mm(_f(gt.get('required_press_depth_m'))):.2f} mm,"
        f"裕量 {_mm(travel_margin):.2f} mm",
    )
    layer.data["top_offset_source"] = cover_geom.get("source") or (
        "link_origin_fallback(mesh 解析失败:"
        + str(cover_geom.get("error"))
        + ")"
    )

    verdict = (
        f"真值误差(顶面中心)XYZ = {_fmt_mm(err_top_m)}(|e|={norm_mm:.2f} mm,"
        f"横向 {lateral_mm:.2f} mm,竖直 {dz_mm:+.2f} mm);"
        f"对 cover 原点 = {_fmt_mm(err_origin_m)};"
        f"{layer.data['interpretation_heuristic']}"
    )
    if (
        lateral_mm <= lateral_tol
        and dz_mm <= max_high
        and dz_mm >= min_low
        and travel_margin >= 0.0
    ):
        layer.mark_pass("通过:" + verdict)
    else:
        layer.mark_fail("不通过:" + verdict)


# ===========================================================================
# L6 —— 按压位姿 IK
# ===========================================================================


def _require_cuda(grasp: Any) -> None:
    """``grasp.solve_ik_named`` 把张量写死 ``.cuda()``:没有 CUDA 直接说清楚。"""
    import torch  # 延迟 import:顶层保持轻量

    if not torch.cuda.is_available():
        raise RuntimeError(
            "本机 torch.cuda.is_available() == False,而 grasp.solve_ik_named 硬编码 "
            ".cuda();IK 无法解算(这本身就是一条必须先解决的环境问题)"
        )


def _solve_named_ik(
    ctx: Context,
    pose: Any,
    seeds: Sequence[tuple[str, Any]],
    *,
    name: str = "right_arm",
) -> dict[str, Any]:
    """多种子 IK + FK 回代,返回误差最小的解(不抛异常,失败返回 ``ok=False``)。"""
    grasp = ctx.grasp
    best: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    for label, seed in seeds:
        seed_array = np.asarray(seed, dtype=np.float64).reshape(-1)[:6]
        try:
            qpos = grasp.solve_ik_named(ctx.base, np.asarray(pose, dtype=np.float64), seed_array, name)
        except Exception as exc:  # noqa: BLE001
            attempts.append({"seed": label, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if qpos is None:
            attempts.append({"seed": label, "ok": False, "reason": "IK 返回 None"})
            continue
        try:
            fk = np.asarray(grasp.fk_pose_named(ctx.base, qpos, name), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            attempts.append({"seed": label, "error": f"FK: {type(exc).__name__}: {exc}"})
            continue
        position_error = float(np.linalg.norm(fk[:3, 3] - np.asarray(pose, dtype=np.float64)[:3, 3]))
        rotation_error = _rotation_error_deg(fk[:3, :3], np.asarray(pose, dtype=np.float64)[:3, :3])
        item = {
            "seed": label,
            "ok": True,
            "qpos": _jsonable(qpos),
            "fk_position_m": _jsonable(fk[:3, 3]),
            "pos_err_mm": _mm(position_error),
            "rot_err_deg": rotation_error,
            "joint_limit_margin_rad": _joint_limit_margin(ctx.base, qpos, name),
        }
        attempts.append(item)
        score = position_error + 1e-4 * rotation_error
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "qpos": np.asarray(qpos, dtype=np.float64).reshape(-1),
                "fk": fk,
                "pos_err_mm": item["pos_err_mm"],
                "rot_err_deg": rotation_error,
                "seed": label,
                "joint_limit_margin_rad": item["joint_limit_margin_rad"],
            }
    return {"ok": best is not None, "best": best, "attempts": attempts}


def _joint_limit_margin(base: Any, qpos: Any, name: str) -> float | None:
    """关节限位余量(弧度):离最近限位还有多远;取不到返回 ``None``。"""
    try:
        limits = np.asarray(
            _npv(base.robot.get_qpos_limits(name=name)), dtype=np.float64
        ).reshape(-1, 2)
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)[: limits.shape[0]]
        return float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
    except Exception:  # noqa: BLE001
        return None


def _pose_from(point: Any, rotation: Any) -> Any:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(point, dtype=np.float64).reshape(3)
    return pose


def _layer6(ctx: Context, layer: Layer) -> None:
    """L6:按压位姿 IK —— 目标位姿 IK 是否成功、解出的关节角、FK 回代误差。"""
    if ctx.button_world_m is None:
        layer.mark_skip(
            "跳过:没有按钮世界坐标(L4 未通过),无法构造按压目标",
            blocking=ctx.downstream_blocking,
            reason="cli_flag" if not ctx.downstream_blocking else "no_button_point",
        )
        return
    try:
        _require_cuda(ctx.grasp)
    except Exception as exc:  # noqa: BLE001
        layer.check("CUDA 可用(grasp IK 硬编码 .cuda())", False, str(exc))
        layer.mark_fail(f"不通过:{exc}")
        return

    arm = str(ctx.args.arm)
    name = f"{arm}_arm"
    stereo, grasp = ctx.stereo, ctx.grasp
    q_now, _grip = stereo.arm_joints(ctx.base, arm)
    ee_now = np.asarray(grasp.fk_pose_named(ctx.base, q_now, name), dtype=np.float64)
    rotation_ref = ee_now[:3, :3]  # 照抄 SPEC:沿用当前 EE 姿态,不读专家 affordance
    point = np.asarray(ctx.button_world_m, dtype=np.float64).reshape(3)
    lift = float(ctx.constants["APPROACH_LIFT_M"])
    layer.data["ee_now_position_m"] = _jsonable(ee_now[:3, 3])
    layer.data["ee_now_rotation"] = _jsonable(rotation_ref)
    layer.data["reference_orientation_source"] = "current EE orientation (SPEC §1)"
    layer.data["staging_lift_m"] = lift
    layer.data["ee_to_button_mm"] = _mm(float(np.linalg.norm(point - ee_now[:3, 3])))

    # 种子照抄"合法运行期只用机器人自身状态"的原则:当前 qpos、零位、
    # 以及官方 gym_config 公开的 init_qpos(不含任何专家 affordance)。
    seeds: list[tuple[str, Any]] = [
        ("current", q_now),
        ("zeros", np.zeros(6, dtype=np.float64)),
    ]
    robot_cfg = ctx.gym_config.get("robot")
    if isinstance(robot_cfg, dict) and robot_cfg.get("init_qpos"):
        init_all = [float(x) for x in robot_cfg["init_qpos"]]
        # init_qpos 是 16 维(左臂6+左夹爪2+右臂6+右夹爪2);只取待用臂的 6 个。
        offset = 0 if arm == "left" else 8
        if len(init_all) >= offset + 6:
            seeds.append(
                ("gym_init_qpos", np.asarray(init_all[offset : offset + 6], dtype=np.float64))
            )

    targets = {
        "press": _pose_from(point, rotation_ref),
        "staging": _pose_from(point + np.array([0.0, 0.0, lift]), rotation_ref),
    }
    pos_tol = float(ctx.args.ik_pos_tol_mm)
    rot_tol = float(ctx.args.ik_rot_tol_deg)
    results: dict[str, Any] = {}
    for label, pose in targets.items():
        solved = _solve_named_ik(ctx, pose, seeds, name=name)
        results[label] = solved
        layer.data[f"ik_{label}"] = _jsonable(solved)

    press = results["press"]
    press_best = press.get("best")
    if press_best is None:
        # 主要目标失败 → 做一次有界的姿态扰动搜索(对应 SPEC 的扰动搜索)。
        samples = max(0, int(ctx.args.ik_orientation_samples))
        max_rad = math.radians(float(ctx.args.ik_orientation_max_deg))
        layer.data["orientation_search"] = {"samples": samples, "max_deg": math.degrees(max_rad)}
        if samples > 0:
            rng = np.random.default_rng(
                int(ctx.constants["STAGE_IK_ORIENTATION_RNG_SEED"])
            )
            best_search: dict[str, Any] | None = None
            for index in range(samples):
                axis = rng.normal(size=3)
                angle = float(rng.uniform(0.0, max_rad))
                rotation = _axis_angle_matrix(axis, angle) @ rotation_ref
                solved = _solve_named_ik(
                    ctx, _pose_from(point, rotation), [("current", q_now)], name=name
                )
                best = solved.get("best")
                if best is not None and (
                    best_search is None or best["score"] < best_search["score"]
                ):
                    best_search = dict(best)
                    best_search["perturbation_deg"] = math.degrees(angle)
                    best_search["perturbation_index"] = index
            layer.data["orientation_search"]["best"] = _jsonable(best_search)
            if best_search is not None:
                layer.data["press_ik_from_orientation_search"] = _jsonable(best_search)
                # 扰动解只作为"存在性"证据,不当作主路径成功(主路径沿用当前 EE 姿态)。

    if press_best is None:
        search_best = layer.data.get("press_ik_from_orientation_search")
        if search_best:
            layer.mark_fail(
                "不通过:沿用当前 EE 姿态时按压位姿 IK 全部失败;"
                f"只靠 ±{float(ctx.args.ik_orientation_max_deg):.0f}° 姿态扰动才找到解"
                f"(位置误差 {search_best.get('pos_err_mm'):.2f} mm),"
                "说明策略必须依赖 PRESS_IK 的扰动搜索分支"
            )
        else:
            layer.mark_fail(
                "不通过:按压位姿 IK 无解(多种子 + 扰动搜索都失败);"
                "可能原因:目标点不可达、solver 未为该控制部件配置、或 CUDA 不可用"
            )
        return

    ctx.press_pose = _pose_from(point, rotation_ref)
    ctx.press_q = np.asarray(press_best["qpos"], dtype=np.float64)
    ctx.press_ik = {
        "qpos": _jsonable(ctx.press_q),
        "pos_err_mm": press_best["pos_err_mm"],
        "rot_err_deg": press_best["rot_err_deg"],
        "seed": press_best["seed"],
        "joint_limit_margin_rad": press_best["joint_limit_margin_rad"],
        "staging": results["staging"].get("best"),
    }
    layer.check(
        f"press IK FK 回代位置误差 <= {pos_tol:.2f} mm",
        press_best["pos_err_mm"] <= pos_tol,
        f"{press_best['pos_err_mm']:.3f} mm",
    )
    layer.check(
        f"press IK FK 回代姿态误差 <= {rot_tol:.2f} deg",
        press_best["rot_err_deg"] <= rot_tol,
        f"{press_best['rot_err_deg']:.4f} deg",
    )
    staging_best = results["staging"].get("best")
    if staging_best is None:
        layer.check("staging(抬升 120 mm)IK 有解", False, "无解")
    else:
        layer.check(
            f"staging IK FK 位置误差 <= {pos_tol:.2f} mm",
            staging_best["pos_err_mm"] <= pos_tol,
            f"{staging_best['pos_err_mm']:.3f} mm(种子 {staging_best['seed']})",
        )
    margin = press_best.get("joint_limit_margin_rad")
    if margin is not None:
        layer.check("关节限位余量 > 0.02 rad", margin > 0.02, f"{margin:.4f} rad")

    # GT 交叉验证:若用**真值**顶面点,IK 是否依然可达(区分"估计点不可达"与"根本不可达")。
    gt_cross: dict[str, Any] = {}
    if ctx.gt is not None:
        truth_point = np.asarray(ctx.gt["cover_top_world_m"], dtype=np.float64).reshape(3)
        solved = _solve_named_ik(
            ctx, _pose_from(truth_point, rotation_ref), seeds, name=name
        )
        gt_cross = {
            "note": "引用 L5(唯一读真值的层)的真值点做交叉验证;策略本体不读真值",
            "target_world_m": _jsonable(truth_point),
            "solved": _jsonable(solved.get("best")),
        }
        layer.data["gt_cross_check"] = _jsonable(gt_cross)
    qpos_text = ", ".join(f"{x:+.4f}" for x in ctx.press_q)
    layer.mark_pass(
        f"通过:press IK 成功(种子 {press_best['seed']}),qpos=[{qpos_text}],"
        f"FK 回代位置误差 {press_best['pos_err_mm']:.3f} mm / "
        f"姿态误差 {press_best['rot_err_deg']:.4f} deg"
        + (";staging 有解" if staging_best is not None else ";staging 无解")
    )


# ===========================================================================
# L7 —— 按压可行性
# ===========================================================================


def _press_depth_schedule(constants: dict[str, float]) -> list[float]:
    """照抄 SPEC §2.2 的下压位移序列:5 mm → +2.5 mm → 补齐到 8 mm。"""
    first = float(constants["PRESS_FIRST_STEP_M"])
    step = float(constants["PRESS_STEP_M"])
    max_m = float(constants["PRESS_MAX_M"])
    iterations = max(1, int(round(max_m / step)))
    depths: list[float] = []
    pressed = 0.0
    for index in range(iterations):
        this_step = first if index == 0 else step
        this_step = min(this_step, max_m - pressed)
        if this_step <= 0.0:
            break
        pressed += this_step
        depths.append(round(pressed, 9))
    return depths


def _layer7(ctx: Context, layer: Layer) -> None:
    """L7:从 press pose 沿 EE -z 下压 8 mm 的可达性预判(纯 FK/IK,不 step 环境)。"""
    if ctx.press_pose is None or ctx.press_q is None:
        layer.mark_skip(
            "跳过:没有 press pose 解(L6 未通过),无法预判下压可达性",
            blocking=ctx.downstream_blocking,
            reason="cli_flag" if not ctx.downstream_blocking else "no_press_pose",
        )
        return

    constants = ctx.constants
    name = f"{ctx.args.arm}_arm"
    stereo = ctx.stereo
    press_pose = np.asarray(ctx.press_pose, dtype=np.float64).reshape(4, 4)
    rotation = press_pose[:3, :3]
    axis_ee = np.array([0.0, 0.0, float(constants.get("PRESS_AXIS_Z", -1.0))])
    axis_world = rotation @ axis_ee  # EE 系 -z 在世界系里的方向
    pos_tol = float(ctx.args.ik_pos_tol_mm)
    rot_tol = float(ctx.args.ik_rot_tol_deg)
    schedule = _press_depth_schedule(constants)
    required = float(ctx.args.success_depth_mm) / 1000.0
    layer.data["press_axis_world"] = _jsonable(axis_world)
    layer.data["press_depth_schedule_m"] = [float(x) for x in schedule]
    layer.data["required_travel_m"] = required
    layer.data["press_max_m"] = float(constants["PRESS_MAX_M"])
    layer.data["note"] = "本层只用 FK/IK 预判可达性,不实际 step 环境、不做接触仿真"

    def probe(depth_m: float, seeds: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        target = press_pose.copy()
        target[:3, 3] = press_pose[:3, 3] + axis_world * float(depth_m)
        solved = _solve_named_ik(ctx, target, seeds, name=name)
        best = solved.get("best")
        item: dict[str, Any] = {
            "depth_m": float(depth_m),
            "target_world_m": _jsonable(target[:3, 3]),
            "ik_ok": best is not None,
        }
        if best is not None:
            item.update(
                {
                    "seed": best["seed"],
                    "pos_err_mm": best["pos_err_mm"],
                    "rot_err_deg": best["rot_err_deg"],
                    "joint_limit_margin_rad": best["joint_limit_margin_rad"],
                    "qpos": _jsonable(best["qpos"]),
                }
            )
        return item

    q_seed = np.asarray(ctx.press_q, dtype=np.float64)
    q_now, _grip = stereo.arm_joints(ctx.base, str(ctx.args.arm))
    steps: list[dict[str, Any]] = []
    deepest_ok = 0.0
    for depth in schedule:
        seeds: list[tuple[str, Any]] = [("previous", q_seed), ("current", q_now)]
        item = probe(depth, seeds)
        steps.append(item)
        if item["ik_ok"] and item["pos_err_mm"] <= pos_tol and item["rot_err_deg"] <= rot_tol:
            q_seed = np.asarray(item["qpos"], dtype=np.float64)
            deepest_ok = float(depth)
    layer.data["steps"] = _jsonable(steps)

    # 额外单独探一次"成功判据所需位移"(4.8 mm 不在 5/7.5/8 的步进序列里)。
    required_probe = probe(required, [("previous", q_seed), ("current", q_now)])
    layer.data["required_probe"] = _jsonable(required_probe)
    required_ok = bool(
        required_probe["ik_ok"]
        and required_probe["pos_err_mm"] <= pos_tol
        and required_probe["rot_err_deg"] <= rot_tol
    )

    full_ok = bool(
        deepest_ok >= float(constants["PRESS_MAX_M"]) - 1e-9
    )
    layer.check(
        f"逐级下压到位序列 {[round(_mm(x), 2) for x in schedule]} mm 全部可解",
        full_ok,
        f"最深可解 {_mm(deepest_ok):.2f} mm",
    )
    layer.check(
        f"下压 {_mm(required):.2f} mm(成功判据所需)可解",
        required_ok,
        ""
        if required_ok
        else "不可解;即使接触成立也可能压不到位",
    )

    # GT 预算(引用 L5 的结论;策略本体不读真值,这里只是把 L5 的数字翻成
    # "8 mm 行程还够不够压到 4.8 mm"这样一句可判定的话)。
    budget: dict[str, Any] = {
        "note": "引用 L5(唯一读真值的层)的数值做预算;策略本体不读真值"
    }
    if ctx.gt is not None and ctx.button_world_m is not None:
        truth_top = np.asarray(ctx.gt["cover_top_world_m"], dtype=np.float64).reshape(3)
        estimate = np.asarray(ctx.button_world_m, dtype=np.float64).reshape(3)
        delta = estimate - truth_top
        dz_mm = _mm(float(delta[2]))
        lateral_mm = _mm(float(math.hypot(delta[0], delta[1])))
        press_max = float(constants["PRESS_MAX_M"])
        effective = press_max - max(0.0, dz_mm / 1000.0)
        cover_radius_m = _f(ctx.cover_geometry.get("cover_radius_m"))
        radius_known = math.isfinite(cover_radius_m) and cover_radius_m > 0.0
        budget.update(
            {
                "vertical_error_mm": dz_mm,
                "lateral_error_mm": lateral_mm,
                "effective_travel_m": effective,
                "effective_travel_mm": _mm(effective),
                "required_travel_mm": _mm(required),
                "budget_ok": bool(effective >= required),
                "cover_radius_mm": _mm(cover_radius_m) if radius_known else None,
                "lateral_margin_mm": (_mm(cover_radius_m) - lateral_mm) if radius_known else None,
            }
        )
        layer.check(
            "GT 预算:8 mm 行程仍够压到 4.8 mm",
            bool(budget["budget_ok"]),
            f"竖直偏差 {dz_mm:+.2f} mm → 有效行程 {_mm(effective):.2f} mm"
            f"(需要 {_mm(required):.2f} mm)",
        )
        # 顶面半径来自 L5 解析的官方 mesh;解析失败时不拿它下结论。
        if radius_known:
            layer.check(
                "GT 预算:横向偏差仍在按钮顶面半径内",
                bool(budget["lateral_margin_mm"] > 0.0),
                f"横向 {lateral_mm:.2f} mm,顶面半径 {_mm(cover_radius_m):.2f} mm,"
                f"余量 {budget['lateral_margin_mm']:.2f} mm",
            )
        else:
            layer.check("GT 预算:按钮顶面半径已知", None, "L5 未能解析官方 mesh,跳过该子检查")
    layer.data["gt_budget"] = _jsonable(budget)

    summary = (
        f"最深可解 {_mm(deepest_ok):.2f}/{_mm(float(constants['PRESS_MAX_M'])):.2f} mm;"
        f"4.8 mm 探针 {'可解' if required_ok else '不可解'}"
    )
    if full_ok and required_ok:
        layer.mark_pass("通过:" + summary + "(纯运动学预判)")
    else:
        layer.mark_fail("不通过:" + summary + "(纯运动学预判;接触/摩擦未建模)")


# ===========================================================================
# 聚合 / 输出
# ===========================================================================


def _next_steps(results: Sequence[Layer]) -> list[str]:
    """按各层结论给出可操作的下一步(纯规则,不猜)。"""
    hints: list[str] = []
    by_name = {item.name: item for item in results}

    l1 = by_name.get("L1_mount")
    if l1 is not None and l1.status != "pass":
        order_check = (l1.data or {}).get("order_check") or {}
        recommended = order_check.get("recommended_order")
        configured = order_check.get("configured_order")
        if recommended and configured and recommended != configured:
            hints.append(
                f"L1:实测安装矩阵与 {recommended} 一致,当前 HYBRID_QUAT_ORDER={configured};"
                f"用 `HYBRID_QUAT_ORDER={recommended}` 重跑(settings 会同时改变 "
                "priors.camera_mount 与 stereo 主路径)"
            )
        else:
            hints.append(
                "L1:先修腕相机安装常量,否则 L4/L5 的误差会被系统性污染"
            )
    l2 = by_name.get("L2_depth")
    if l2 is not None and l2.status != "pass":
        hints.append(
            "L2:检查 FS_CKPT / FS_ROOT / FS_PYTHON / FS_STEREO_SOCKET,"
            "并确认 FoundationStereo 子进程或常驻 worker 能出视差图"
        )
    l3 = by_name.get("L3_vlm")
    if l3 is not None and l3.status != "pass":
        hints.append(
            "L3:配置 HYBRID_VLM_URL(见 robosyn_hybrid/scripts/run_vlm_server.sh)"
            "或用 --button-uv/--button-bbox 手动给像素继续验证 L4-L7"
        )
    l5 = by_name.get("L5_ground_truth")
    if l5 is not None and l5.status == "fail":
        hints.append(
            "L5:对照 error_vs_top_xyz_mm 的分解决定改哪一层"
            "(竖直为主 → 深度/高度;水平为主 → VLM 像素或相机外参)"
        )
    l6 = by_name.get("L6_press_ik")
    if l6 is not None and l6.status == "fail":
        hints.append(
            "L6:确认评测机有 CUDA(策略 IK 硬编码 .cuda())、"
            "robot.compute_ik(name='right_arm') 有 solver,并检查目标点是否在可达空间内"
        )
    l7 = by_name.get("L7_press_feasibility")
    if l7 is not None and l7.status == "fail":
        hints.append(
            "L7:下压路径不可解时优先怀疑估算点偏高/偏低,可先用 --button-uv 换一个像素复测"
        )
    return hints


def _aggregate(results: Sequence[Layer], ctx: Context | None, wall_time_s: float) -> dict[str, Any]:
    passed = [item.name for item in results if item.status == "pass"]
    failed = [item.name for item in results if item.status == "fail"]
    skipped = [item.name for item in results if item.status == "skip"]
    blocking_skips = [item.name for item in results if item.status == "skip" and item.blocking]
    warnings = list(ctx.warnings) if ctx is not None else []
    for item in results:
        if item.status == "skip" and not item.blocking and item.skip_reason == "cli_flag":
            warnings.append(f"{item.name}: 用户显式跳过(--skip-*),该层未被验证")
    exit_code = 1 if (failed or blocking_skips) else 0
    return {
        "exit_code": exit_code,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "blocking_skips": blocking_skips,
        "warnings": warnings,
        "next_steps": _next_steps(results),
        "wall_time_s": round(float(wall_time_s), 3),
    }


def _print_summary(payload: dict[str, Any]) -> None:
    """人类可读总结:每层状态 + 判定 + 子检查。"""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append(f"click_bell 分层几何诊断 —— {TOOL_NAME} v{TOOL_VERSION}")
    meta = payload.get("meta") or {}
    lines.append(
        f"  repo_root={meta.get('repo_root')}  setting={meta.get('setting')}  "
        f"seed={meta.get('seed')}  arm={meta.get('arm')}"
    )
    setup = payload.get("setup") or {}
    if not setup.get("ok", False):
        lines.append(f"  [SETUP FAILED] {setup.get('error')}")
        lines.append("=" * 78)
        print("\n".join(lines), flush=True)
        return
    lines.append(
        f"  gym_config={setup.get('gym_config_path')}  actual_seed={setup.get('actual_seed')}  "
        f"env={setup.get('base_type')}(解包 {setup.get('env_chain_hops')} 层)"
    )
    lines.append(f"  常量来源:{setup.get('constants_source')}")
    hooks = setup.get("button_hooks") or {}
    if hooks:
        lines.append(
            "  按钮钩子:prompt={} parse={} 几何={}(grid={}, min_samples={}, radius={})".format(
                hooks.get("prompt_source"),
                hooks.get("parse_source"),
                hooks.get("geometry_source"),
                hooks.get("grid"),
                hooks.get("min_samples"),
                hooks.get("sample_radius"),
            )
        )
        if hooks.get("import_error"):
            lines.append(
                "  ! clickbell 导入失败,按钮 prompt/解析已退回内置副本:"
                f"{hooks['import_error']}"
            )
    lines.append("-" * 78)
    for item in payload.get("layers", []):
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}.get(item["status"], "????")
        suffix = ""
        if item["status"] == "skip":
            suffix = f" [{'阻塞' if item['blocking'] else '不阻塞'}:{item.get('skip_reason')}]"
        lines.append(f"[{item['name']}] {mark}{suffix}  ({item['duration_s']:.2f}s)")
        lines.append(f"    {item['verdict']}")
        if item.get("error"):
            lines.append(f"    error: {item['error']}")
        checks = item.get("checks") or {}
        if checks:
            lines.append(_format_checks(checks))
    verdict = payload.get("verdict") or {}
    lines.append("-" * 78)
    lines.append(
        f"结果:pass={len(verdict.get('passed', []))} fail={len(verdict.get('failed', []))} "
        f"skip={len(verdict.get('skipped', []))} "
        f"(阻塞性 skip={len(verdict.get('blocking_skips', []))})"
    )
    for text in verdict.get("warnings") or []:
        lines.append(f"  ! {text}")
    steps = verdict.get("next_steps") or []
    if steps:
        lines.append("下一步:")
        for index, text in enumerate(steps, 1):
            lines.append(f"  {index}. {text}")
    lines.append(f"退出码:{verdict.get('exit_code')}")
    lines.append("=" * 78)
    print("\n".join(lines), flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _save_reference_rgb(ctx: Context, out_path: Path) -> str | None:
    """可选:把参考 RGB 落盘,方便用 --button-uv 手挑像素。"""
    try:
        import cv2  # 延迟 import

        rgb = ctx.reference_rgb
        if rgb is None and ctx.depth is not None:
            rgb = ctx.depth["reference_rgb"]
        if rgb is None:
            return None
        image = np.asarray(rgb)
        target = out_path.with_name(out_path.stem + "_reference_rgb.png")
        cv2.imwrite(str(target), cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2BGR))
        return str(target)
    except Exception:  # noqa: BLE001 - 存图失败不影响诊断结论
        return None


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI(顶层只依赖 argparse,``--help`` 必须干净可跑)。"""
    parser = argparse.ArgumentParser(
        prog="verify_clickbell_geometry.py",
        description=(
            "click_bell 分层几何诊断(L1 安装 / L2 深度 / L3 VLM / L4 反投影 / "
            "L5 真值比对 / L6 IK / L7 按压可行性)。"
            "诊断工具,绝不被策略调用;策略本体不读任何物体真值位姿。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        default=os.environ.get("ROBOSYN_REPO_ROOT"),
        help="官方 RoboSynChallenge 仓库根(含 configs/ 与 scripts/eval_policy.py);"
        "缺省时读环境变量 ROBOSYN_REPO_ROOT",
    )
    parser.add_argument("--setting", default=DEFAULT_SETTING, help="configs/click_bell/<setting>/gym_config.json")
    parser.add_argument("--seed", type=int, default=0, help="episode 种子(prepare_episode_reset)")
    parser.add_argument("--out", default=None, help="诊断 JSON 输出路径(缺省只打印总结)")
    parser.add_argument("--arm", default="right", choices=("right", "left"), help="用哪条臂跑几何链路")
    parser.add_argument("--device", default="cpu", help="仿真设备(eval_policy 的 device)")
    parser.add_argument("--renderer", default="hybrid", help="渲染后端(eval_policy 的 renderer)")
    parser.add_argument("--gpu-id", type=int, default=0, help="仿真 GPU id")
    parser.add_argument("--max-steps", type=int, default=361, help="episode 步数上限(click_bell 规格 361)")
    parser.add_argument(
        "--no-headless", dest="headless", action="store_false", help="打开可视化窗口(调试用)"
    )
    parser.set_defaults(headless=True)
    parser.add_argument(
        "--embodichain-root", default=None, help="EmbodiChain 检出根(缺省沿用 EMBODICHAIN_ROOT 或同级推测)"
    )
    parser.add_argument(
        "--pythonpath",
        default=None,
        help=f"额外 sys.path(用 {os.pathsep!r} 分隔),用于指定 EmbodiChain 等依赖",
    )
    # -- 深度(L2)/反投影(L4)--
    parser.add_argument("--skip-depth", action="store_true", help="跳过 L2(以及依赖它的 L4-L7)")
    parser.add_argument("--settle-steps", type=int, default=1, help="depth fetch 前稳定步数")
    parser.add_argument("--motion-steps", type=int, default=5, help="nudge 运动步数")
    parser.add_argument("--nudge-candidates", type=int, default=2000, help="找 nudge 姿态的候选数")
    parser.add_argument("--baseline-target", type=float, default=0.02, help="目标 baseline(米)")
    parser.add_argument(
        "--sample-radius", type=int, default=None, help="反投影采样半径(像素);缺省用 clickbell 常量,否则 3"
    )
    parser.add_argument(
        "--button-grid", type=int, default=None, help="bbox 内 grid x grid 反投影采样;缺省用 clickbell 常量,否则 5"
    )
    parser.add_argument(
        "--builtin-button-hooks",
        action="store_true",
        help="不复用 clickbell 的按钮 prompt/解析/网格,改用本脚本内置副本(用于 A/B 对照)",
    )
    parser.add_argument(
        "--min-valid-fraction", type=float, default=0.05, help="L2 判定用的最小有效像素比例"
    )
    # -- VLM(L3)--
    parser.add_argument("--skip-vlm", action="store_true", help="跳过 L3 的 VLM 调用")
    parser.add_argument("--vlm-timeout", type=int, default=None, help="VLM 超时(秒),缺省用 settings")
    parser.add_argument(
        "--button-uv", nargs=2, type=float, default=None, metavar=("U", "V"), help="手动指定按钮中心像素"
    )
    parser.add_argument(
        "--button-bbox",
        nargs=4,
        type=float,
        default=None,
        metavar=("L", "T", "R", "B"),
        help="手动指定按钮 bbox(像素)",
    )
    # -- 判定阈值 --
    parser.add_argument("--lateral-tol-mm", type=float, default=8.0, help="L5 横向误差门限(mm)")
    parser.add_argument(
        "--max-high-mm",
        type=float,
        default=3.2,
        help="L5 允许「估计点高于真值」的上限(mm),缺省 8.0(PRESS_MAX_M)-4.8(成功判据)=3.2",
    )
    parser.add_argument(
        "--min-low-mm", type=float, default=-10.0, help="L5 允许「估计点低于真值」的下限(mm)"
    )
    parser.add_argument("--ik-pos-tol-mm", type=float, default=2.0, help="L6/L7 IK 回代位置误差门限(mm)")
    parser.add_argument("--ik-rot-tol-deg", type=float, default=2.0, help="L6/L7 IK 回代姿态误差门限(deg)")
    parser.add_argument(
        "--ik-orientation-samples", type=int, default=64, help="press IK 失败时的姿态扰动搜索次数(0=关闭)"
    )
    parser.add_argument(
        "--ik-orientation-max-deg", type=float, default=20.0, help="姿态扰动搜索的最大角度(deg)"
    )
    parser.add_argument(
        "--success-depth-mm", type=float, default=1000.0 * SUCCESS_PRESS_DEPTH_M, help="成功判据所需下压量(mm)"
    )
    parser.add_argument("--save-rgb", action="store_true", help="把参考 RGB 落到 <out>_reference_rgb.png")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """入口:构造环境 → 逐层诊断 → 写 JSON + 打印总结 → 返回退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    started = time.perf_counter()

    # dexsim / OptiX 的原生崩溃(SIGABRT/SIGSEGV)无法被 Python 捕获;打开
    # faulthandler 至少能把崩在哪个 Python 语句打出来。
    try:
        import faulthandler  # noqa: PLC0415

        faulthandler.enable()
    except Exception:  # noqa: BLE001
        pass

    if np is None:
        print(
            "[verify] 本脚本需要 numpy;当前解释器里 import numpy 失败,"
            "请先激活评测环境的 Python。",
            file=sys.stderr,
        )
        return 1

    payload: dict[str, Any] = {
        "tool": {
            "name": TOOL_NAME,
            "version": TOOL_VERSION,
            "task": TASK_NAME,
            "purpose": "click_bell 分层几何诊断(GPU 机器上一次性定位几何链路故障)",
            "warning": (
                "本脚本是诊断工具,绝不被策略调用;L5 是全脚本唯一读物体真值位姿的层,"
                "策略本体不读任何物体真值。"
            ),
        },
        "meta": {
            "argv": [str(item) for item in (argv if argv is not None else sys.argv[1:])],
            "python": sys.version,
            "executable": sys.executable,
            "cwd": str(Path.cwd()),
            "repo_root": args.repo_root,
            "setting": args.setting,
            "seed": int(args.seed),
            "arm": args.arm,
            "skip_depth": bool(args.skip_depth),
            "skip_vlm": bool(args.skip_vlm),
        },
        "setup": {"ok": False, "error": None},
        "layers": [],
        "verdict": {"exit_code": 1, "passed": [], "failed": [], "skipped": [], "next_steps": []},
    }

    ctx: Context | None = None
    # 先写一份"进行中"的存根:即使随后在 dexsim/OptiX 里原生 abort,--out 也会
    # 留下一份能说明"崩在环境构造阶段"的证据(正常路径最后会整体覆盖它)。
    if args.out:
        payload["setup"] = {
            "ok": False,
            "error": "in_progress",
            "note": (
                "环境构造尚未完成;若进程在此阶段被原生 abort(SIGABRT/SIGSEGV),"
                "本文件会停留在这个状态,请检查评测机 GPU / 驱动 / OptiX"
            ),
            "embodichain_root": os.environ.get("EMBODICHAIN_ROOT"),
        }
        try:
            _write_json(Path(args.out), payload)
            print(f"[verify] 已写入 in_progress 存根:{args.out}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[verify] 写存根失败:{type(exc).__name__}: {exc}", file=sys.stderr)
    try:
        if not args.repo_root:
            raise SystemExit(
                "必须提供 --repo-root(或设置环境变量 ROBOSYN_REPO_ROOT)指向官方 "
                "RoboSynChallenge 仓库根"
            )
        ctx = _build_context(args)
        payload["setup"] = _jsonable(ctx.setup)
        payload["meta"]["repo_root"] = str(ctx.repo_root)
    except SystemExit as exc:  # argparse 风格的参数错误
        payload["setup"] = {"ok": False, "error": str(exc)}
        print(str(exc), file=sys.stderr)
        payload["verdict"] = _aggregate([], None, time.perf_counter() - started)
        # 环境都没建起来:JSON 里的退出码必须和进程退出码一致,不能留 0。
        payload["verdict"]["failed"] = ["SETUP"]
        payload["verdict"]["exit_code"] = 2
        if args.out:
            _write_json(Path(args.out), payload)
        _print_summary(payload)
        return 2
    except Exception as exc:  # noqa: BLE001 - 环境构造失败也要留下 JSON
        payload["setup"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "embodichain_root": os.environ.get("EMBODICHAIN_ROOT"),
            "cwd": str(Path.cwd()),
        }
        traceback.print_exc()
        payload["verdict"] = _aggregate([], None, time.perf_counter() - started)
        payload["verdict"]["failed"] = ["SETUP"]
        payload["verdict"]["exit_code"] = 1
        if args.out:
            _write_json(Path(args.out), payload)
        _print_summary(payload)
        return 1

    layers: list[Layer] = [
        Layer("L1_mount", "腕相机安装常量(stereo.verify_mount)"),
        Layer("L2_depth", "运动双目深度(stereo.fetch_depth)"),
        Layer("L3_vlm", "VLM 按钮定位(vlm.vlm_available / vlm_chat)"),
        Layer("L4_backproject", "按钮反投影(estimate_point_from_pixel)"),
        Layer("L5_ground_truth", "几何真值比对(唯一读真值的层)"),
        Layer("L6_press_ik", "按压位姿 IK(grasp.solve_ik_named + FK 回代)"),
        Layer("L7_press_feasibility", "按压可行性(纯 FK/IK 预判)"),
    ]
    bodies: list[Callable[[], None]] = [
        lambda: _layer1(ctx, layers[0]),
        lambda: _layer2(ctx, layers[1]),
        lambda: _layer3(ctx, layers[2]),
        lambda: _layer4(ctx, layers[3]),
        lambda: _layer5(ctx, layers[4]),
        lambda: _layer6(ctx, layers[5]),
        lambda: _layer7(ctx, layers[6]),
    ]
    try:
        for layer, body in zip(layers, bodies):
            print(f"[verify] >>> {layer.name} {layer.title}", flush=True)
            _run_layer(layer, body)
            print(f"[verify] <<< {layer.name} {layer.status.upper()}", flush=True)
    finally:
        payload["layers"] = [item.to_json() for item in layers]
        payload["verdict"] = _aggregate(layers, ctx, time.perf_counter() - started)
        rgb_path = None
        if args.save_rgb and args.out:
            rgb_path = _save_reference_rgb(ctx, Path(args.out))
        if rgb_path:
            payload["reference_rgb_png"] = rgb_path
        if args.out:
            try:
                _write_json(Path(args.out), payload)
                print(f"[verify] JSON 已写入 {args.out}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[verify] 写 JSON 失败:{type(exc).__name__}: {exc}", file=sys.stderr)
        env = ctx.env if ctx is not None else None
        if env is not None:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001
                    print(f"[verify] env.close() 异常:{type(exc).__name__}: {exc}", flush=True)

    _print_summary(payload)
    return int(payload["verdict"]["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
