"""SAM2.1 掩码分割（bbox prompt）。

从 ``_stage/harness/handle_basket_clickbell_20260924/milk_mask.py`` 移植，改动：

* 删除模块级硬编码的权重路径；默认权重改为延迟解析
  :func:`robosyn_hybrid.settings.get_settings` 的 ``sam2_ckpt``
  （可用 ``SAM2_CKPT`` 环境变量覆盖）。
* :func:`load_sam2` 在 ``sam2`` 包/``ultralytics`` 缺失或权重文件不存在时抛
  :class:`RuntimeError`，并给出可操作提示（指向 ``policy/hybrid/robosyn_hybrid/scripts/download_assets.sh``）；
  调用方（状态机）负责降级。

其余函数行为与原始实现逐字一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # 作为包导入（正常路径）
    from .settings import get_settings
except ImportError:  # 直接以脚本/扁平目录方式导入时的兜底
    from settings import get_settings  # type: ignore[no-redef]

__all__ = [
    "default_sam2_model",
    "load_sam2",
    "segment_milk_mask",
    "mask_axis_points",
    "mask_horizontal_pairs",
]

# 权重下载/环境准备的统一提示语（避免各处重复且路径写死）。
_DOWNLOAD_HINT = (
    "请先运行 `bash policy/hybrid/robosyn_hybrid/scripts/download_assets.sh` 下载 SAM2.1 权重 "
    "（默认放到 <assets_dir>/sam2.1_t.pt），"
    "或用 `SAM2_CKPT=/path/to/sam2.1_t.pt` 指定已下载的权重文件。"
)


def default_sam2_model() -> Path:
    """SAM2 权重的默认路径，来自 ``settings.sam2_ckpt``（不硬编码绝对路径）。"""
    return get_settings().sam2_ckpt


def __getattr__(name: str) -> Any:
    """兼容旧代码里的 ``DEFAULT_SAM2_MODEL``：延迟解析，避免模块级硬编码路径。"""
    if name == "DEFAULT_SAM2_MODEL":
        return str(default_sam2_model())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_sam2(model_path: str | None = None, device: str = "cuda") -> object:
    """加载 SAM2 模型（走 ``ultralytics.SAM``）。

    参数
    ----
    model_path:
        权重文件路径；``None`` 时用 :func:`default_sam2_model`（``SAM2_CKPT`` /
        ``assets_dir/sam2.1_t.pt``）。不会读取任何硬编码的绝对路径。
    device:
        目标设备；与原始实现一致，这里只做记录，真正生效在
        :func:`segment_milk_mask` 的 ``predict(device=...)``。

    异常
    ----
    RuntimeError:
        权重文件不存在，或 ``ultralytics``（含 ``sam2`` 后端）未安装时抛出，
        消息里带可操作提示，调用方据此降级到纯 ACT。
    """
    if model_path is None:
        model_path = str(default_sam2_model())
    path = Path(str(model_path)).expanduser()
    if not path.is_file():
        raise RuntimeError(f"SAM2 权重不存在: {path}。{_DOWNLOAD_HINT}")

    try:
        # 延迟导入：只有真正需要分割时才要求 ultralytics/sam2 可用。
        from ultralytics import SAM
    except ImportError as exc:
        raise RuntimeError(
            "未安装 SAM2 推理依赖（`ultralytics`，SAM2 后端在其中）。"
            "请运行 `bash policy/hybrid/robosyn_hybrid/scripts/setup_fs_env.sh` 或 `pip install ultralytics`；"
            f"权重下载见 download_assets.sh。{_DOWNLOAD_HINT}"
        ) from exc

    return SAM(str(path))


def segment_milk_mask(model: Any, image: Any, bbox_xyxy: Any, *, device: str = "cuda") -> dict:
    """用 bbox prompt 分割牛奶掩码（与原始实现一致：取面积最大的那个掩码）。"""
    image = np.asarray(image)
    bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(1, 4)
    results = model.predict(
        source=image,
        bboxes=bbox,
        device=device,
        retina_masks=True,
        verbose=False,
    )
    if not results:
        return {"ok": False, "error": "SAM2 returned no result"}
    result = results[0]
    if result.masks is None or result.masks.data is None:
        return {"ok": False, "error": "SAM2 returned no mask"}
    masks = result.masks.data.detach().cpu().numpy()
    if masks.ndim == 2:
        masks = masks[None, ...]
    masks = masks.astype(bool)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    if areas.size == 0:
        return {"ok": False, "error": "SAM2 returned empty masks"}
    index = int(np.argmax(areas))
    mask = masks[index]
    scores = None
    if result.boxes is not None and result.boxes.conf is not None:
        try:
            scores = result.boxes.conf.detach().cpu().numpy()
        except Exception:
            scores = None
    score = None if scores is None or index >= len(scores) else float(scores[index])
    return {
        "ok": bool(mask.any()),
        "mask": mask,
        "area_px": int(mask.sum()),
        "score": score,
        "bbox_xyxy": bbox[0].tolist(),
        "all_mask_count": int(masks.shape[0]),
    }


def mask_axis_points(
    mask: np.ndarray, low: float = 0.10, high: float = 0.90
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[np.ndarray, np.ndarray] | None]:
    """用 PCA 主轴取掩码两端的代表点（与原始实现一致）。

    返回 ``(p1, p2, (center, axis))``；点太少时返回 ``(None, None, None)``。
    """
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size < 20:
        return None, None, None
    points = np.c_[xs, ys].astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    projections = (points - center) @ axis
    p_low = np.percentile(projections, low * 100.0)
    p_high = np.percentile(projections, high * 100.0)
    low_points = points[np.abs(projections - p_low) <= 2.0]
    high_points = points[np.abs(projections - p_high) <= 2.0]
    if len(low_points) == 0 or len(high_points) == 0:
        return None, None, (center, axis)
    p1 = np.median(low_points, axis=0)
    p2 = np.median(high_points, axis=0)
    return p1, p2, (center, axis)


def mask_horizontal_pairs(
    mask: np.ndarray, y_fractions: Iterable[float] = (0.25, 0.35, 0.50, 0.65)
) -> list[dict]:
    """在若干高度分数处取掩码的左右边缘对，用于估计牛奶盒的横向宽度（与原始实现一致）。"""
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)
    if xs.size < 20:
        return []
    y0, y1 = int(ys.min()), int(ys.max())
    height = max(1, y1 - y0 + 1)
    pairs: list[dict] = []
    for frac in y_fractions:
        yc = int(round(y0 + frac * (height - 1)))
        ylo, yhi = max(0, yc - 3), min(mask.shape[0], yc + 4)
        band_y, band_x = np.where(mask[ylo:yhi, :])
        if band_x.size < 4:
            continue
        left_x = float(np.percentile(band_x, 2))
        right_x = float(np.percentile(band_x, 98))
        left_points = band_x[np.abs(band_x - left_x) <= 3]
        right_points = band_x[np.abs(band_x - right_x) <= 3]
        if left_points.size == 0 or right_points.size == 0:
            continue
        pairs.append(
            {
                "y_frac": float(frac),
                "left_uv": [float(np.median(left_points)), float(yc)],
                "right_uv": [float(np.median(right_points)), float(yc)],
                "width_px": max(0.0, right_x - left_x),
            }
        )
    return pairs
