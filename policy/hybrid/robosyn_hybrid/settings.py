"""Environment-driven settings for the RoboSynChallenge hybrid submission.

Nothing in this package may hard-code an absolute path.  Every filesystem
location is resolved here from an environment variable with a portable
default derived from the repository root.

``HYBRID_FK_FRAME`` —— ``compute_fk`` 返回哪个帧
------------------------------------------------
:attr:`Settings.fk_frame` 是"``robot.compute_fk(name="<arm>_arm",
link_names=["<arm>_link6"], to_matrix=True)`` 到底返回哪个帧"的开关:

* ``"tool"``(默认):返回的是 **tool/TCP 帧**,``link_names`` 参数被忽略。
  此时 :func:`robosyn_hybrid.priors.camera_mount` 必须再左乘一个实测解出的
  常量 ``FK_TO_LINK6`` 才能把结果换算回 ``<arm>_link6`` 连杆帧。
* ``"link6"``:评测机上的检出确实返回 link6 连杆帧,不需要任何补偿。

该常量是 2026-10-10 在真实 RTX 4090 + ``EmbodiChain_main 0.2.4`` 检出上
实测解出的(残差 1.277e-06),不是推测值;换检出后请用
``policy/hybrid/robosyn_hybrid/scripts/verify_clickbell_geometry.py`` 的 L1 层重测。
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["Settings", "get_settings", "asset", "repo_root"]

_PKG_DIR = Path(__file__).resolve().parent

#: VLM 默认模型名。与 ollama 的命名一致:`ollama pull qwen3vl:8b` 后即可直接用。
#: 参考实现的 holder / 按钮定位精度(筒口 3D 误差 3.17 mm)就是在
#: ollama + `qwen3vl:8b` 上测出来的,所以默认跟随它。
#: 用 llama.cpp 的 `llama-server` 时该字段会被忽略(它只服务已加载的那个模型),
#: 因此同一个默认值对两种后端都安全。
_DEFAULT_VLM_MODEL = "qwen3vl:8b"


def repo_root() -> Path:
    """Root of the RoboSynChallenge checkout this package was dropped into."""
    env = os.environ.get("ROBOSYN_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # <repo>/policy/hybrid/robosyn_hybrid/settings.py -> <repo>
    return _PKG_DIR.parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        warnings.warn(f"{name}={raw!r} is not an integer; using {default}.")
        return default


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    assets_dir: Path
    sam2_ckpt: Path
    foundationstereo_root: Path
    fs_ckpt: Path
    fs_python: str
    fs_socket: str
    fs_valid_iters: int
    vlm_url: str
    vlm_model: str
    vlm_timeout_s: int
    vlm_enabled: bool
    quat_order: str
    #: ``compute_fk`` 返回哪个帧的开关,取值 ``"tool"`` / ``"link6"``。
    #: ``"tool"``(默认)表示返回的是 TCP/tool 帧,``priors.camera_mount`` 需要
    #: 再补一个 ``FK_TO_LINK6`` 才能把结果换算回 ``<arm>_link6`` 连杆帧;
    #: ``"link6"`` 表示已经返回 link6 连杆帧,不需要任何补偿。
    fk_frame: str
    debug_dir: Path | None
    strict: bool

    @property
    def sam2_available(self) -> bool:
        return self.sam2_ckpt.is_file()

    @property
    def foundationstereo_available(self) -> bool:
        return self.fs_ckpt.is_file() and self.foundationstereo_root.is_dir()

    @property
    def vlm_available(self) -> bool:
        return bool(self.vlm_enabled and self.vlm_url)

    def describe(self) -> dict:
        return {
            "repo_root": str(self.repo_root),
            "assets_dir": str(self.assets_dir),
            "sam2_ckpt": str(self.sam2_ckpt),
            "sam2_available": self.sam2_available,
            "foundationstereo_root": str(self.foundationstereo_root),
            "fs_ckpt": str(self.fs_ckpt),
            "foundationstereo_available": self.foundationstereo_available,
            "fs_python": self.fs_python,
            "fs_socket": self.fs_socket,
            "fs_valid_iters": self.fs_valid_iters,
            "vlm_url": self.vlm_url,
            "vlm_model": self.vlm_model,
            "vlm_enabled": self.vlm_enabled,
            "vlm_available": self.vlm_available,
            "quat_order": self.quat_order,
            "fk_frame": self.fk_frame,
            "debug_dir": str(self.debug_dir) if self.debug_dir else None,
            "strict": self.strict,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = repo_root()
    assets = _env_path("ROBOSYN_HYBRID_ASSETS", root / "robosyn_hybrid_assets")
    fs_root = _env_path("FS_ROOT", assets / "FoundationStereo")
    quat_order = os.environ.get("HYBRID_QUAT_ORDER", "wxyz").strip().lower()
    if quat_order not in {"wxyz", "xyzw"}:
        warnings.warn(f"HYBRID_QUAT_ORDER={quat_order!r} unknown; using 'wxyz'.")
        quat_order = "wxyz"
    # ``HYBRID_FK_FRAME``:``compute_fk`` 返回的是 tool/TCP 帧还是 link6 连杆帧。
    # 默认 ``"tool"`` —— 2026-10-10 在真实 GPU 服务器上实测确认的官方惯例。
    fk_frame = os.environ.get("HYBRID_FK_FRAME", "tool").strip().lower()
    if fk_frame not in {"tool", "link6"}:
        warnings.warn(f"HYBRID_FK_FRAME={fk_frame!r} unknown; using 'tool'.")
        fk_frame = "tool"
    debug_raw = os.environ.get("HYBRID_DEBUG_DIR")
    return Settings(
        repo_root=root,
        assets_dir=assets,
        sam2_ckpt=_env_path("SAM2_CKPT", assets / "sam2.1_t.pt"),
        foundationstereo_root=fs_root,
        fs_ckpt=_env_path(
            "FS_CKPT",
            fs_root / "pretrained_models" / "11-33-40" / "model_best_bp2.pth",
        ),
        fs_python=os.environ.get("FS_PYTHON", sys.executable),
        fs_socket=os.environ.get("FS_STEREO_SOCKET", "/tmp/fs_stereo.sock"),
        fs_valid_iters=_env_int("FS_STEREO_VALID_ITERS", 32),
        vlm_url=os.environ.get("HYBRID_VLM_URL", "").strip(),
        # 默认模型名与 ollama 的命名一致(`qwen3vl:8b`),因为参考实现的
        # holder/按钮定位精度(筒口 3.17 mm)就是在 ollama + qwen3vl:8b 上测出来的。
        # llama.cpp 的 llama-server 会忽略请求里的 model 字段,所以这个名字对它也无害。
        vlm_model=os.environ.get("HYBRID_VLM_MODEL", _DEFAULT_VLM_MODEL).strip()
        or _DEFAULT_VLM_MODEL,
        vlm_timeout_s=_env_int("HYBRID_VLM_TIMEOUT_S", 120),
        vlm_enabled=_env_bool("HYBRID_VLM_ENABLED", True),
        quat_order=quat_order,
        fk_frame=fk_frame,
        debug_dir=_env_path("HYBRID_DEBUG_DIR", Path("/dev/null")) if debug_raw else None,
        strict=_env_bool("HYBRID_STRICT", False),
    )


def asset(name: str, *, required: bool = True) -> Path:
    """Resolve a file inside the assets directory.

    ``required=True`` raises with an actionable message pointing at the
    downloader; ``required=False`` returns the intended path after warning.
    """
    settings = get_settings()
    path = settings.assets_dir / name
    if path.exists():
        return path
    message = (
        f"Required asset '{name}' not found at {path}. "
        "Run `bash policy/hybrid/robosyn_hybrid/scripts/download_assets.sh` first, or set ROBOSYN_HYBRID_ASSETS "
        "to the directory that holds the downloaded weights."
    )
    if required:
        raise FileNotFoundError(message)
    warnings.warn(message)
    return path
