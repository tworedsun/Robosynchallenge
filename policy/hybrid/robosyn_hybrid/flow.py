"""handle_basket 混合策略状态机(可被 ``eval_policy.py`` 反复调用的 ``eval``)。

本模块把参考实现 ``handle_basket_flow.py`` 的 ``main()``(line 577-986)改造成
有限状态机,供官方评测循环使用::

    while env_steps < max_env_steps:
        obs, info, truncated, new_times = policy_pkg.eval(eval_env, model, obs)
        env_steps = int(info["elapsed_steps"].item())
        if not is_truncated and env.get_wrapper_attr("is_task_success")(): break
        if is_truncated: break

因此 ``eval()`` 的契约是:

* 每次调用推进**一个微步**:运动/ACT 阶段执行 1 个 ``env.step()``;
  纯计算阶段(定位/深度/IK 搜索)算完后立刻继续到下一个会 ``env.step`` 的阶段;
* 每次只 step 1 步 → 外层循环能在成功的那一刻立即 break,不浪费动作步数;
* 一次 ``eval()`` 内连续推进纯计算阶段超过 ``MAX_PURE_ADVANCES``(32)次仍未
  产生 ``env.step()`` → 返回 ``truncated=True`` 防死循环;
* **任何阶段失败都不抛异常、不提前退出**:记录 ``self.failures`` 后进入
  ``FALLBACK_ACT``,把剩余步数交给纯 ACT(不带任何夹爪覆盖),让 episode
  正常跑完并拿到 ACT 基线水平的结果。

合规:本文件只使用 RGB 图像、机器人本体感受(qpos/FK/IK)与官方公开常量;
不读取任何物体真值位姿、不读模拟器深度缓冲、不直接访问模拟器传感器句柄;
腕相机位姿只用 ``robosyn_hybrid.priors.camera_mount`` + FK(在 ``stereo.py``
内);全部临时/调试落盘只在 ``debug_dir`` 非空时发生。
"""

from __future__ import annotations

import importlib
import json
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# 常量:照搬 SPEC §2.8 与原实现 line 45-56(数值一个都不改)
# ---------------------------------------------------------------------------

#: 官方三路 RGB 相机(高视角 + 左右腕视角)。
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")

GRIP_OPEN = 0.05
GRIP_GRASP = 0.0
PREGRASP_OFFSET_M = 0.05
GRASP_TILT_DEG = -20.0
RIGHT_GRIP_TIGHT = -0.005
RIGHT_GRASP_DOWN_M = 0.01
RIGHT_GRASP_CLOSE_STEPS = 20
PRE_GRASP_ACT_CHUNKS = 3
POST_GRASP_ACT_CHUNKS = 2
ACT_OPEN_INTENT_CMD = 0.02
ACT_OPEN_INTENT_DELTA = 0.01

#: 官方公开的牛奶盒尺寸先验(与 ``priors.MILK_PRIOR_SIZE`` 同值)。
MILK_PRIOR_SIZE = np.array([0.043, 0.0995, 0.0444], dtype=np.float64)

# ---------------------------------------------------------------------------
# 流程参数(原实现是 argparse 默认值,这里固化为常量)
# ---------------------------------------------------------------------------

SETTLE_STEPS = 8                 # --settle-steps
MOTION_STEPS = 5                 # --motion-steps
PREGRASP_STEPS = 15              # --pregrasp-steps
PREGRASP_HOLD_STEPS = 2          # 原 hold_q(q_pre, 2, ...)
ADVANCE_STEPS = 8                # --advance-steps
CLOSE_STEPS = 11                 # --close-steps
GRASP_HOLD_STEPS = 2             # 原 hold_q(q_grasp, 2, ...)
LIFT_STEPS = 8                   # --lift-steps
LIFT_HOLD_STEPS = 3              # 原 hold_q(q_lift, 3, ...)
LIFT_OFFSET_M = 0.10             # 原 get_offset_pose(grasp_pose, 0.10, "z", "extrinsic")
RIGHT_GRASP_MOVE_STEPS = 5       # 原 move_right_q(..., 5, ...)
RIGHT_RELEASE_STEPS = 5          # 原 hold_right_q(..., 5, GRIP_OPEN, ...)

#: ``fetch_depth`` 内部的 settle 步数。原实现直接传 8;本状态机把 settle 拆成
#: 独立的 SETTLE 阶段,因此这里只传 1(``stereo.move_arm`` 要求 steps >= 1)。
FETCH_SETTLE_STEPS = 1
NUDGE_CANDIDATES = 2000
BASELINE_TARGET_M = 0.02

#: VLM 定位重试(原实现:2 次尝试,间隔 2 秒,单次超时 180s)。
VLM_ATTEMPTS = 2
VLM_TIMEOUT_S = 180
VLM_RETRY_SLEEP_S = 2.0
VLM_RELEASE_TIMEOUT_S = 60

SAM2_DEVICE = "cuda"

#: 一个 ACT chunk 的步数(原实现 ``act_step=50``);仅当 ``act_policy`` 没有
#: 暴露 ``act_step`` 时使用该默认值。
ACT_CHUNK_STEPS_DEFAULT = 50

#: 一次 ``eval()`` 调用内允许的连续纯计算阶段推进上限(SPEC §2.8 要求 32)。
MAX_PURE_ADVANCES = 32
#: 连续降级次数上限(防止降级路径自身死循环)。
MAX_DEGRADES = 8
#: ``failures`` / ``act_state`` / release gate 检查记录的条数上限。
MAX_FAILURES = 64
MAX_ACT_STATE = 400
MAX_GATE_CHECKS = 50

#: 全部阶段走完后是否把剩余步数交给纯 ACT。
#: ``True``  = DONE 之后继续跑 ACT(与降级路径一致,ACT 还有可能补救);
#: ``False`` = DONE 之后保持当前位姿(更利于“已达成状态”攒满 55 步判定)。
DONE_RUNS_PURE_ACT = True

#: ``fetch_depth`` 返回结果里适合写进 summary 的标量/小数组键。
SAFE_DEPTH_KEYS = (
    "ok",
    "arm",
    "q1",
    "q2",
    "reference_exposure",
    "baseline_m",
    "motion_camera_m",
    "rectification_deg",
)

_MISSING = object()
_MODULE_CACHE: dict[str, Any] = {}
_TIMING_WARNED = False


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def npv(value: Any) -> np.ndarray:
    """把 torch 张量/普通容器统一成 numpy 数组(原实现 line 71-72)。"""
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def jsonable(value: Any) -> Any:
    """递归转成可 JSON 序列化的对象(原实现 line 75-86)。"""
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = npv(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def as_done(value: Any) -> bool:
    """把 terminated/truncated 统一成 bool(等价于 ``RecordingEnvProxy._as_done``)。"""
    if value is None:
        return False
    if hasattr(value, "detach"):
        value = npv(value)
    array = np.asarray(value)
    if array.size == 0:
        return False
    try:
        return bool(array.any())
    except Exception:
        return bool(value)


def _lazy(module_name: str) -> Any:
    """惰性 import 同包模块。

    成功/失败结果都会缓存。失败时抛 ``RuntimeError``,由阶段守卫捕获后降级,
    因此 WS-A/WS-B 的模块缺失不会让 handle_basket 直接崩掉。
    """
    cached = _MODULE_CACHE.get(module_name, _MISSING)
    if cached is not _MISSING:
        if cached is None:
            raise RuntimeError(f"robosyn_hybrid.{module_name} 不可用(import 失败)")
        return cached
    package = __package__ or "robosyn_hybrid"
    try:
        module = importlib.import_module(f".{module_name}", package)
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        _MODULE_CACHE[module_name] = None
        raise RuntimeError(
            f"无法 import robosyn_hybrid.{module_name}: {type(exc).__name__}: {exc}"
        ) from exc
    _MODULE_CACHE[module_name] = module
    return module


def _torch() -> Any:
    """惰性 import torch。"""
    import torch

    return torch


def _to_action_tensor(action: Any) -> Any:
    """把动作数组转成 ``env.step`` 需要的张量(torch 缺失时退回 numpy)。"""
    array = np.asarray(action, dtype=np.float32)
    try:
        torch = _torch()
        tensor = torch.as_tensor(array)
        return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor
    except Exception:
        return array[None, :] if array.ndim == 1 else array


def _action_rows(action: Any) -> list[np.ndarray]:
    """把 ``select_chunk_action`` 的返回值规整成 [step][dim] 的动作行列表。

    兼容 ``[D]`` / ``[B, D]`` / ``[H, D]`` / ``[B, H, D]`` 四种形状。
    """
    array = np.asarray(npv(action), dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return [np.asarray(row, dtype=np.float32).reshape(-1) for row in array]


def _unwrap_env(env: Any) -> Any:
    """沿 ``.env`` 链向下解开环境包装(原 ``main()`` line 615-620 的写法)。

    原实现最多走 6 层;这里放宽到 8 层,链条更短时行为完全一致。
    """
    base = env
    for _ in range(8):
        nxt = getattr(base, "env", None)
        if nxt is None:
            break
        base = nxt
    return base


def full_action(
    base: Any,
    left_q: np.ndarray,
    left_grip: float,
    right_q: np.ndarray,
    right_grip: float,
) -> np.ndarray:
    """组装 14 维动作(原实现 line 108-118)。

    以当前 qpos 为底,覆盖左臂/左夹爪/右臂/右夹爪,再按 ``active_joint_ids``
    取出真正生效的那一维子集。
    """
    target = npv(base.robot.get_qpos())[0].astype(np.float32)
    for name, value in (
        ("left_arm", left_q),
        ("left_eef", [left_grip]),
        ("right_arm", right_q),
        ("right_eef", [right_grip]),
    ):
        ids = base.robot.get_joint_ids(name=name, remove_mimic=True)
        target[ids] = np.asarray(value, dtype=np.float32)
    return target[base.active_joint_ids]


def _safe_rgb_of(obs: Any, camera: str) -> np.ndarray | None:
    """从 obs 里取某路 RGB(原 ``rgb_of``,line 97-101),失败返回 None。"""
    try:
        image = npv(obs["sensor"][camera]["color"])
        while image.ndim > 3:
            image = image[0]
        return np.ascontiguousarray(image[..., :3].astype(np.uint8))
    except Exception:
        return None


def frame3(obs: Any) -> np.ndarray | None:
    """三路相机横向拼接(原 ``frame3``,line 104-105),仅用于调试落盘。"""
    frames = [_safe_rgb_of(obs, camera) for camera in CAMS]
    if any(frame is None for frame in frames):
        return None
    try:
        return np.concatenate([frame for frame in frames if frame is not None], axis=1)
    except Exception:
        return None


def _lift_offset_pose(grasp_pose: np.ndarray, distance: float = LIFT_OFFSET_M) -> np.ndarray:
    """抬升位姿:``get_offset_pose(pose, distance, "z", "extrinsic")`` 的等价实现。

    官方 helper 可用时优先调用;不可用时用已验证等价的平移
    (``offset_matrix @ pose`` 只改变平移分量:平移 += distance * [0, 0, 1])。
    """
    try:
        from embodichain.lab.gym.utils.misc import get_offset_pose

        return np.asarray(
            get_offset_pose(
                np.asarray(grasp_pose, dtype=np.float64).copy(),
                float(distance),
                direction="z",
                mode="extrinsic",
            )
        )
    except Exception:
        out = np.asarray(grasp_pose, dtype=np.float64).copy()
        out[2, 3] += float(distance)
        return out


# ---------------------------------------------------------------------------
# ACT 计时:必须用官方 policy.inference_timing(只计模型前向,不含 env.step)
# ---------------------------------------------------------------------------


def _timing_device(act_policy: Any) -> Any:
    for name in ("device", "act_device", "pytorch_device"):
        value = getattr(act_policy, name, None)
        if value is not None:
            return value
    return None


def _warn_timing_once() -> None:
    global _TIMING_WARNED
    if _TIMING_WARNED:
        return
    _TIMING_WARNED = True
    warnings.warn(
        "policy.inference_timing 不可用,ACT 计时退化为 time.perf_counter"
        "(仍然只计模型前向,不含 env.step)。",
        stacklevel=3,
    )


def start_inference(device: Any = None) -> float:
    """官方 ``inference_timing.start_inference``;失败退化为 ``perf_counter``。"""
    try:
        from policy.inference_timing import start_inference as _start

        return float(_start(device))
    except Exception:
        _warn_timing_once()
        return time.perf_counter()


def finish_inference(started_at: float, samples: list[float], device: Any = None) -> None:
    """官方 ``inference_timing.finish_inference``;失败退化为 ``perf_counter``。"""
    try:
        from policy.inference_timing import finish_inference as _finish

        _finish(started_at, samples, device)
        return
    except Exception:
        _warn_timing_once()
        samples.append(time.perf_counter() - started_at)


# ---------------------------------------------------------------------------
# 牛奶位姿估计(原实现 line 224-411 的移植,GT-free)
# ---------------------------------------------------------------------------


def _estimate_point(
    stereo: Any, depth_result: dict, uv: Iterable[float], sample_radius: int = 3
) -> dict:
    """``stereo.estimate_point_from_pixel`` 的安全封装(异常 → ``ok=False``)。"""
    try:
        estimate = stereo.estimate_point_from_pixel(
            depth_result, list(uv), sample_radius=sample_radius
        )
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(estimate, dict):
        return {"ok": False, "error": "estimate_point_from_pixel 返回非字典"}
    return estimate


def _mask_horizontal_pairs(mask: Any) -> list:
    """``sam2_mask.mask_horizontal_pairs`` 的安全封装(不可用 → 空列表)。"""
    if mask is None:
        return []
    try:
        sam2_mask = _lazy("sam2_mask")
        return list(sam2_mask.mask_horizontal_pairs(np.asarray(mask, dtype=bool)))
    except Exception:
        return []


def estimated_milk_pose_from_bbox(
    depth_result: dict,
    bbox: Iterable[float],
    vlm_yaw_deg: float | None,
    p1_uv: Iterable[float] | None = None,
    p2_uv: Iterable[float] | None = None,
    mask: Any = None,
) -> tuple[np.ndarray | None, dict]:
    """由 bbox(+可选 SAM2 掩码)与立体深度估计牛奶的 6D 位姿。

    逐行对应原实现 ``estimated_milk_pose_from_bbox``(line 224-411):
    只用 RGB-D 几何 + 公开尺寸先验,不读任何真值。

    返回 ``(milk_pose | None, info)``;失败时 ``milk_pose is None``。
    """
    stereo = _lazy("stereo")
    x0, y0, x1, y1 = (float(v) for v in bbox)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    points = []
    # 1) bbox 内的 3x3 网格采样(带深度有效性过滤)。
    for du in (-0.25, 0.0, 0.25):
        for dv in (-0.25, 0.0, 0.25):
            uv = [cx + du * (x1 - x0), cy + dv * (y1 - y0)]
            estimate = _estimate_point(stereo, depth_result, uv, 3)
            if estimate.get("ok"):
                points.append(np.asarray(estimate["p_world_m"], dtype=np.float64))
    if not points:
        return None, {"error": "no valid milk depth points"}
    points = np.asarray(points)
    cam_pose = np.asarray(depth_result["reference_camera_pose"], dtype=np.float64)
    camera_x = cam_pose[:3, 0]
    camera_horizontal_yaw = np.arctan2(camera_x[1], camera_x[0])

    # 掩码给的是可见表面而不是几何中心:在中心邻域采样,并沿相机射线把
    # “最短边的一半”推入物体内部。只用 RGB-D 几何 + 已知尺寸先验。
    center_samples = []
    ray_samples = []
    sample_uvs = [(cx, cy)]
    for du in (-0.12, 0.12):
        sample_uvs.append((cx + du * (x1 - x0), cy))
    for dv in (-0.12, 0.12):
        sample_uvs.append((cx, cy + dv * (y1 - y0)))
    if mask is not None and np.asarray(mask).any():
        ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
        sample_uvs.append((float(xs.mean()), float(ys.mean())))
    for uv in sample_uvs:
        estimate = _estimate_point(stereo, depth_result, uv, 3)
        if not estimate.get("ok"):
            continue
        p_world = np.asarray(estimate["p_world_m"], dtype=np.float64)
        p_camera = np.asarray(estimate["p_camera_m"], dtype=np.float64)
        ray_camera = p_camera / max(1e-9, float(np.linalg.norm(p_camera)))
        ray_world = cam_pose[:3, :3] @ ray_camera
        ray_world = ray_world / max(1e-9, float(np.linalg.norm(ray_world)))
        center_samples.append(p_world)
        ray_samples.append(ray_world)
    if center_samples:
        surface_center = np.median(np.asarray(center_samples), axis=0)
        inward_ray = np.median(np.asarray(ray_samples), axis=0)
        inward_ray = inward_ray / max(1e-9, float(np.linalg.norm(inward_ray)))
        half_short_side = 0.5 * float(np.min(MILK_PRIOR_SIZE))
        center = surface_center + half_short_side * inward_ray
    else:
        surface_center = np.median(points, axis=0)
        inward_ray = np.zeros(3, dtype=np.float64)
        half_short_side = 0.0
        center = surface_center.copy()

    # 官方摆放把牛奶本体原点放在桌面上:不信任顶面深度,而是采样掩码下方
    # 可见桌面,用其平面高度作为牛奶原点 z;X/Y 仍来自上面的中心射线修正。
    table_z_samples = []
    table_uvs = [
        (cx, y1 + 0.03 * (y1 - y0)),
        (cx, y1 + 0.07 * (y1 - y0)),
        (cx, y1 + 0.12 * (y1 - y0)),
        (x0 + 0.20 * (x1 - x0), y1 + 0.06 * (y1 - y0)),
        (x1 - 0.20 * (x1 - x0), y1 + 0.06 * (y1 - y0)),
    ]
    for uv in table_uvs:
        estimate = _estimate_point(stereo, depth_result, uv, 3)
        if estimate.get("ok"):
            table_z_samples.append(float(estimate["p_world_m"][2]))
    table_z_used = None
    if len(table_z_samples) >= 3:
        table_z_used = float(np.median(table_z_samples))
        center[2] = table_z_used

    center_correction = {
        "surface_center_world_m": surface_center.tolist(),
        "inward_ray_world": inward_ray.tolist(),
        "half_short_side_m": half_short_side,
        "sample_count": len(center_samples),
        "table_z_samples_m": table_z_samples,
        "table_z_used_m": table_z_used,
    }
    yaw_source = "camera_horizontal_plus_vlm_yaw"
    p1_world = p2_world = None
    edge_trials = []
    mask_pairs = [] if mask is None else _mask_horizontal_pairs(mask)
    width, height = x1 - x0, y1 - y0
    # 2) bbox 内左右边缘对(用 0.06m 的公开宽度先验打分)。
    for y_frac in (0.25, 0.35, 0.50, 0.65):
        uv_l = [x0 + 0.08 * width, y0 + y_frac * height]
        uv_r = [x1 - 0.08 * width, y0 + y_frac * height]
        e1 = _estimate_point(stereo, depth_result, uv_l, 3)
        e2 = _estimate_point(stereo, depth_result, uv_r, 3)
        trial = {
            "y_frac": y_frac,
            "left_uv": uv_l,
            "right_uv": uv_r,
            "left_ok": bool(e1.get("ok")),
            "right_ok": bool(e2.get("ok")),
        }
        if e1.get("ok"):
            trial["left_world_m"] = e1["p_world_m"]
        else:
            trial["left_error"] = e1.get("error")
        if e2.get("ok"):
            trial["right_world_m"] = e2["p_world_m"]
        else:
            trial["right_error"] = e2.get("error")
        edge_trials.append(trial)
    valid_pairs = []
    for trial in edge_trials:
        if not trial.get("left_ok") or not trial.get("right_ok"):
            continue
        left = np.asarray(trial["left_world_m"], dtype=np.float64)
        right = np.asarray(trial["right_world_m"], dtype=np.float64)
        distance = float(np.linalg.norm((right - left)[:2]))
        dz = float(abs(right[2] - left[2]))
        score = abs(distance - 0.06) + 2.0 * dz
        valid_pairs.append((score, distance, dz, left, right, trial["y_frac"]))
    valid_pairs = [
        pair for pair in valid_pairs if 0.02 <= pair[1] <= 0.15 and pair[2] <= 0.04
    ]
    # 3) SAM2 掩码的左右成对点(更稳的朝向来源)。
    mask_candidates = []
    for pair in mask_pairs:
        e1 = _estimate_point(stereo, depth_result, pair["left_uv"], 3)
        e2 = _estimate_point(stereo, depth_result, pair["right_uv"], 3)
        if not (e1.get("ok") and e2.get("ok")):
            continue
        left = np.asarray(e1["p_world_m"], dtype=np.float64)
        right = np.asarray(e2["p_world_m"], dtype=np.float64)
        distance = float(np.linalg.norm((right - left)[:2]))
        dz = float(abs(right[2] - left[2]))
        if 0.02 <= distance <= 0.15 and dz <= 0.04:
            score = abs(distance - 0.06) + 2.0 * dz - 0.01 * pair["width_px"]
            mask_candidates.append((score, distance, dz, left, right, pair))
    if mask_candidates:
        _, distance, dz, p1_world, p2_world, pair = min(
            mask_candidates, key=lambda item: item[0]
        )
        direction = p2_world - p1_world
        yaw = float(np.arctan2(direction[1], direction[0]))
        yaw_source = "sam2_mask_left_right_depth"
        points = np.vstack([points, p1_world[None, :], p2_world[None, :]])
    if yaw_source == "sam2_mask_left_right_depth":
        pass
    elif valid_pairs:
        _, distance, dz, p1_world, p2_world, y_frac = min(valid_pairs, key=lambda x: x[0])
        direction = p2_world - p1_world
        yaw = float(np.arctan2(direction[1], direction[0]))
        yaw_source = "bbox_left_right_depth"
        points = np.vstack([points, p1_world[None, :], p2_world[None, :]])
    elif p1_uv is not None and p2_uv is not None:
        e1 = _estimate_point(stereo, depth_result, p1_uv, 3)
        e2 = _estimate_point(stereo, depth_result, p2_uv, 3)
        if e1.get("ok") and e2.get("ok"):
            p1_world = np.asarray(e1["p_world_m"], dtype=np.float64)
            p2_world = np.asarray(e2["p_world_m"], dtype=np.float64)
            direction = p2_world - p1_world
            if float(np.linalg.norm(direction[:2])) > 1e-4:
                yaw = float(np.arctan2(direction[1], direction[0]))
                yaw_source = "vlm_two_point_depth"
                points = np.vstack([points, p1_world[None, :], p2_world[None, :]])
    if yaw_source not in {
        "sam2_mask_left_right_depth",
        "bbox_left_right_depth",
        "vlm_two_point_depth",
    }:
        yaw = camera_horizontal_yaw + np.radians(float(vlm_yaw_deg or 0.0))
    from scipy.spatial.transform import Rotation

    image_h, image_w = depth_result["reference_rgb"].shape[:2]
    border_touch = bool(
        x0 <= 1.0 or y0 <= 1.0 or x1 >= image_w - 2.0 or y1 >= image_h - 2.0
    )
    if border_touch:
        return None, {
            "error": "milk mask touches image border",
            "border_touch": border_touch,
            "bbox_xyxy": [float(v) for v in (x0, y0, x1, y1)],
            "image_size": [int(image_w), int(image_h)],
            "table_z_samples_m": table_z_samples,
            "mask_area_px": None if mask is None else int(np.asarray(mask, dtype=bool).sum()),
            "surface_center_world_m": surface_center.tolist(),
            "inward_ray_world": inward_ray.tolist(),
        }
    milk_pose = np.eye(4, dtype=np.float64)
    milk_pose[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    milk_pose[:3, 3] = center
    return milk_pose, {
        "center_world_m": center.tolist(),
        "center_correction": center_correction,
        "points_world_m": points.tolist(),
        "camera_horizontal_yaw_deg": float(np.degrees(camera_horizontal_yaw)),
        "vlm_yaw_deg": vlm_yaw_deg,
        "estimated_world_yaw_deg": float(np.degrees(yaw)),
        "yaw_source": yaw_source,
        "bbox_edge_trials": edge_trials,
        "p1_world_m": None if p1_world is None else p1_world.tolist(),
        "p2_world_m": None if p2_world is None else p2_world.tolist(),
    }


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------


class HandleBasketFlow:
    """handle_basket 的有限状态机(依赖注入版 ``main()``)。

    ``act_policy`` 需要提供 ``select_chunk_action(obs) -> (action, ran_inference)``
    与 ``action_dim``;本类**不**加载任何 ACT 权重。
    """

    # ---- 状态名(与 SPEC §2.8 的状态序列一一对应)----
    SETTLE = "SETTLE"
    DEPTH_FETCH = "DEPTH_FETCH"
    VLM_LOCATE = "VLM_LOCATE"
    SAM2 = "SAM2"
    ESTIMATE_POSE = "ESTIMATE_POSE"
    GRASP_SEARCH = "GRASP_SEARCH"
    APPROACH = "APPROACH"
    ADVANCE = "ADVANCE"
    CLOSE = "CLOSE"
    LIFT = "LIFT"
    ACT_TAKEOVER = "ACT_TAKEOVER"
    RIGHT_GRASP = "RIGHT_GRASP"
    ACT_POST = "ACT_POST"
    RIGHT_RELEASE = "RIGHT_RELEASE"
    DONE = "DONE"
    FALLBACK_ACT = "FALLBACK_ACT"
    ACT_FILL = "ACT_FILL"

    # ------------------------------------------------------------------
    # 构造 / reset
    # ------------------------------------------------------------------

    def __init__(
        self,
        act_policy: Any,
        settings: Any,
        *,
        act_chunks: int = 6,
        max_env_steps: int = 500,
        debug_dir: Path | None = None,
    ) -> None:
        self.act_policy = act_policy
        self.settings = settings

        try:
            requested_chunks = max(0, int(act_chunks))
        except Exception:
            requested_chunks = 0
        self.act_chunks = requested_chunks
        # 原实现:pre = min(PRE_GRASP_ACT_CHUNKS, requested);
        # total = pre + (POST if pre >= PRE else 0)。
        self._pre_chunks = min(PRE_GRASP_ACT_CHUNKS, requested_chunks)
        self._total_chunks = self._pre_chunks + (
            POST_GRASP_ACT_CHUNKS if self._pre_chunks >= PRE_GRASP_ACT_CHUNKS else 0
        )

        try:
            self.max_env_steps = int(max_env_steps)
        except Exception:
            self.max_env_steps = 500
        if self.max_env_steps <= 0:
            self.max_env_steps = 500

        candidate_dir = debug_dir if debug_dir else getattr(settings, "debug_dir", None)
        self.debug_dir: Path | None = Path(candidate_dir) if candidate_dir else None

        # 一个 ACT chunk 的步数:优先取 wrapper 暴露的 act_step。
        budget = getattr(act_policy, "act_step", None)
        if not isinstance(budget, (int, np.integer)) or int(budget) <= 0:
            budget = ACT_CHUNK_STEPS_DEFAULT
        self._act_chunk_budget = int(budget)

        # 全 episode 累计(不随 reset 清空),便于 summary 统计。
        self.failure_history: list[dict] = []
        self._inference_times_all: list[float] = []

        self._env: Any = None
        self._base: Any = None
        self._init_episode_state()

    def reset(self, env: Any = None) -> None:
        """开始新 episode:清空全部运行时状态,并(可选)重新绑定 env。

        ``env=None`` 时也会清掉 env 绑定,让下一次 ``eval()`` 自动重新绑定,
        从而恢复动作通道/动作维度等 env 相关信息。
        """
        self._init_episode_state()
        self._env = None
        if env is not None:
            self._bind_env(env)

    # ------------------------------------------------------------------
    # 对外主循环
    # ------------------------------------------------------------------

    def eval(
        self, env: Any, obs: Any
    ) -> tuple[Any, dict, Any, list[float]]:
        """推进一个微步,返回 ``(obs, info, truncated, inference_times)``。

        * 至少产生 1 个 ``env.step()``,或把阶段指针向前推进;
        * 纯计算阶段连续推进超过 32 次 → ``truncated=True``;
        * 任何异常都在内部降级,绝不向外抛。
        """
        inference_times: list[float] = []
        self._ensure_bound(env)
        info = self._probe_info(env)
        self._maybe_new_episode(env, info)
        if obs is not None:
            # 放在绑定/新 episode 检测之后,保证用的是最新一帧 obs。
            self._obs = obs

        # 上一轮已经看到 terminated/truncated:本轮直接收尾,避免继续 step 死环境。
        if self._env_done:
            self._force_truncated = True

        advances = 0
        while True:
            if self._force_truncated:
                break
            if self._steps_taken >= self.max_env_steps:
                self._force_truncated = True
                self._record_failure(
                    self._state,
                    f"已达 max_env_steps={self.max_env_steps},强制截断",
                    dedup=False,
                    fatal=False,
                )
                break
            advances += 1
            if advances > MAX_PURE_ADVANCES:
                self._force_truncated = True
                self._record_failure(
                    self._state,
                    f"单次 eval() 内连续推进纯计算阶段超过 {MAX_PURE_ADVANCES} 次,"
                    "强制截断以防死循环",
                    dedup=False,
                )
                break

            before_steps = int(self._steps_taken)
            self._produced_internal_step = False
            try:
                self._advance(env, inference_times)
            except Exception as exc:  # 阶段失败 → 降级,绝不外抛
                self._degrade(self._state, exc)
            if int(self._steps_taken) > before_steps or self._produced_internal_step:
                break

        info = self._probe_info(env)
        elapsed = self._info_elapsed(info)
        if self._last_seen_elapsed is None:
            self._last_seen_elapsed = elapsed
        else:
            self._last_seen_elapsed = max(self._last_seen_elapsed, elapsed)
        self._steps_taken = max(int(self._steps_taken), elapsed)
        if self._steps_taken >= self.max_env_steps:
            self._force_truncated = True

        if self._force_truncated:
            truncated_out: Any = True
        else:
            truncated_out = self._last_truncated if self._last_truncated is not None else False
        return self._obs, info, truncated_out, inference_times

    def summary(self) -> dict:
        """返回本 episode 的可序列化摘要(降级理由写入 ``failures``)。"""
        timings = list(self._inference_times_all)
        return jsonable(
            {
                "task": "handle_basket",
                "success": self._query_success(),
                "final_state": self._state,
                "failed": bool(self.failed),
                "degraded_to_pure_act": bool(self._state in (self.FALLBACK_ACT, self.ACT_FILL)),
                "env_steps": int(self._steps_taken),
                "max_env_steps": int(self.max_env_steps),
                "stages": list(self._history),
                "failures": list(self.failures),
                "degradations": list(self.degradations),
                "failure_count_total": len(self.failure_history),
                "depth": self._depth_record,
                "depth_error": self._depth_error,
                "vlm_attempts": self._vlm_attempts,
                "vlm": self._vision,
                "sam2": self._sam2_record,
                "milk_estimation": self._point_info,
                "target_poses": self._target_record,
                "ik": {
                    "q_pre": None if self._q_pre is None else np.asarray(self._q_pre).tolist(),
                    "q_grasp": None
                    if self._q_grasp is None
                    else np.asarray(self._q_grasp).tolist(),
                    "q_lift": None if self._q_lift is None else np.asarray(self._q_lift).tolist(),
                    "search_candidate_count": None
                    if not isinstance(self._search, dict)
                    else self._search.get("candidate_count"),
                    "rejection": None
                    if not isinstance(self._search, dict)
                    else self._search.get("rejection"),
                },
                "act": {
                    "chunks_requested": self.act_chunks,
                    "chunks_pre": self._pre_chunks,
                    "chunks_total": self._total_chunks,
                    "chunk_steps": self._act_chunk_budget,
                    "chunks_started": self._act_chunks_started,
                    "select_calls": self._act_select_calls,
                    "steps": self._act_steps,
                    "ran_inference": self._act_ran_inference,
                    "right_open_hold_steps": self.right_open_hold_steps,
                    "policy_broken": bool(self._act_broken),
                },
                "release_gate": self._gate_summary(),
                "handle_grasp": self._grasp_state,
                "inference": {
                    "calls": len(timings),
                    "total_s": float(sum(timings)) if timings else 0.0,
                    "mean_s": float(sum(timings) / len(timings)) if timings else 0.0,
                },
                "environment": {
                    "vlm_available": bool(getattr(self.settings, "vlm_available", False)),
                    "sam2_available": bool(getattr(self.settings, "sam2_available", False)),
                    "foundationstereo_available": bool(
                        getattr(self.settings, "foundationstereo_available", False)
                    ),
                    "debug_dir": str(self.debug_dir) if self.debug_dir else None,
                },
            }
        )

    # ------------------------------------------------------------------
    # episode / env 绑定
    # ------------------------------------------------------------------

    def _init_episode_state(self) -> None:
        """清空一个 episode 的全部运行时状态。"""
        self._state: str = self.SETTLE
        self._stage_i: int = 0
        self._history: list[dict] = []
        self._obs: Any = None
        self._last_info: dict | None = None
        self._last_truncated: Any = None
        self._force_truncated: bool = False
        self._env_done: bool = False
        self._steps_taken: int = 0
        self._last_seen_elapsed: int | None = None
        self._degrade_count: int = 0
        self._produced_internal_step: bool = False

        # 失败记录
        self.failed: bool = False
        self.failures: list[dict] = []
        self.degradations: list[dict] = []
        self._failure_keys: set[tuple[str, str]] = set()

        # 关节基线
        self._left_hold_q: np.ndarray | None = None
        self._left_hold_grip: float = GRIP_OPEN
        self._right_hold_q: np.ndarray | None = None
        self._right_hold_grip: float = GRIP_OPEN

        # 感知 / 位姿结果
        self._depth: dict | None = None
        self._depth_record: dict | None = None
        self._depth_error: dict | None = None
        self._vlm_attempts: list[dict] = []
        self._vision: dict | None = None
        self._mask_result: dict | None = None
        self._sam2_record: dict | None = None
        self._mask: Any = None
        self._milk_pose: np.ndarray | None = None
        self._point_info: dict | None = None
        self._search: dict | None = None
        self._best: dict | None = None
        self._pregrasp_pose: np.ndarray | None = None
        self._grasp_pose: np.ndarray | None = None
        self._q_pre: np.ndarray | None = None
        self._q_grasp: np.ndarray | None = None
        self._target_record: dict | None = None
        self._q_lift: np.ndarray | None = None
        self._lift_ik_done: bool = False
        self._approach_from: np.ndarray | None = None

        # 右臂抓篮柄
        self._right_grasp_ready: bool = False
        self._right_grasp_ref: dict | None = None
        self._right_release_ref: dict | None = None
        self._grasp_state: dict = {"locked": False, "vlm": None, "down_ik": None}

        # ACT 执行状态
        self._act_chunk_index: int = 0
        self._act_remaining: int = self._act_chunk_budget
        self._pending_rows: list[np.ndarray] = []
        self._act_steps: int = 0
        self._act_select_calls: int = 0
        self._act_chunks_started: int = 0
        self._act_ran_inference: int = 0
        self._act_broken: bool = False
        self.right_open_hold_steps: int = 0
        self._act_state: list[dict] = []

        # 动作通道 / release gate
        self._env_dim: int | None = None
        self._dim_mismatch_recorded: bool = False
        self._gate: dict = {
            "left_eef_channel": None,
            "right_eef_channel": None,
            "allowed": False,
            "last_open": False,
            "last_cmd": None,
            "last_check_step": -10**9,
            "edge_poll_remaining": 0,
            "edge_streak": 0,
            "next_check_step": None,
            "blocked_count": 0,
            "locked_steps": 0,
            "checks": [],
            "step": 0,
        }

    def _bind_env(self, env: Any) -> None:
        """绑定 env:解包 base、探测动作通道、初始化 ACT chunk 计数。"""
        self._env = env
        self._base = _unwrap_env(env)
        self._env_dim = self._probe_action_dim(env)
        left_channel, right_channel = self._probe_eef_channels()
        self._gate["left_eef_channel"] = left_channel
        self._gate["right_eef_channel"] = right_channel
        self._act_chunk_index = 0
        # 0 表示“还没有开始任何 chunk”:第一个 ACT 微步会开启 chunk 0。
        self._act_remaining = 0
        self._pending_rows = []

    def _ensure_bound(self, env: Any) -> None:
        """首次调用/更换 env 对象时自动完成绑定(WS-D 忘记 reset 也不会崩)。"""
        if self._env is env:
            return
        if self._steps_taken > 0:
            # 换了一个 env 实例:按新 episode 处理。
            self._init_episode_state()
        self._bind_env(env)

    def _maybe_new_episode(self, env: Any, info: dict) -> None:
        """``elapsed_steps`` 归零 → 判定外层重新 reset 了环境,重建状态。"""
        elapsed = self._info_elapsed(info)
        if (
            self._last_seen_elapsed is not None
            and elapsed < self._last_seen_elapsed
            and self._steps_taken > 5
            and elapsed <= 1
        ):
            self._init_episode_state()
            self._bind_env(env)
            self._last_seen_elapsed = elapsed

    def _probe_action_dim(self, env: Any) -> int | None:
        try:
            space = getattr(getattr(env, "unwrapped", env), "single_action_space", None)
            if space is None:
                return None
            return int(np.prod(space.shape))
        except Exception:
            return None

    def _probe_eef_channels(self) -> tuple[int | None, int | None]:
        """找出左右夹爪在“生效动作向量”中的下标(原实现 line 772-782)。"""
        left_channel: int | None = None
        right_channel: int | None = None
        try:
            active_ids = np.asarray(npv(self._base.active_joint_ids), dtype=int).reshape(-1)
            left_ids = np.asarray(
                self._base.robot.get_joint_ids(name="left_eef", remove_mimic=True), dtype=int
            ).reshape(-1)
            matches = np.where(active_ids == left_ids[0])[0]
            left_channel = int(matches[0]) if matches.size else None
        except Exception:
            left_channel = None
        try:
            active_ids = np.asarray(npv(self._base.active_joint_ids), dtype=int).reshape(-1)
            right_ids = np.asarray(
                self._base.robot.get_joint_ids(name="right_eef", remove_mimic=True),
                dtype=int,
            ).reshape(-1)
            matches = np.where(active_ids == right_ids[0])[0]
            right_channel = int(matches[0]) if matches.size else None
        except Exception:
            right_channel = None
        return left_channel, right_channel

    # ------------------------------------------------------------------
    # env 步进 / 信息探测
    # ------------------------------------------------------------------

    def _step_env_step(self, env: Any, action: Any) -> tuple[Any, bool]:
        """执行 1 次 ``env.step()`` 并同步内部计数/终止标志。

        返回 ``(obs, env_truncated)``。异常由调用方(``eval``)统一降级。
        """
        obs, _reward, terminated, truncated, info = env.step(_to_action_tensor(action))
        self._steps_taken += 1
        if isinstance(info, dict):
            self._last_info = info
        self._last_truncated = truncated
        self._obs = obs
        if as_done(truncated):
            self._force_truncated = True
        if as_done(terminated):
            # terminated 可能是成功信号:不强制截断,让外层先做成功判定,
            # 下一次 eval() 调用再收尾(见 eval 顶部的 _env_done 分支)。
            self._env_done = True
        return obs, as_done(truncated)

    def _probe_info(self, env: Any) -> dict:
        """取当前 info(优先 ``env.get_info()``,否则用最近一次 step 的 info)。"""
        info: Any = None
        getter = getattr(env, "get_info", None)
        if callable(getter):
            try:
                info = getter()
            except Exception:
                info = None
        if not isinstance(info, dict):
            info = dict(self._last_info) if isinstance(self._last_info, dict) else {}
        else:
            info = dict(info)
        if "elapsed_steps" not in info:
            info["elapsed_steps"] = np.int64(self._steps_taken)
        return info

    def _info_elapsed(self, info: dict) -> int:
        """从 info 里安全读取 ``elapsed_steps``(必须是可 ``.item()`` 的对象)。"""
        value = info.get("elapsed_steps", None) if isinstance(info, dict) else None
        if value is None:
            return int(self._steps_taken)
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return int(item())
            except Exception:
                pass
        try:
            return int(value)
        except Exception:
            return int(self._steps_taken)

    def _hold_action(self) -> np.ndarray:
        """保持当前关节位姿的动作(最后兜底:任何策略失败时使用)。"""
        q_left, grip_left = self._arm_joints(self._base, "left")
        q_right, grip_right = self._arm_joints(self._base, "right")
        return full_action(self._base, q_left, grip_left, q_right, grip_right)

    def _fit_action(self, row: np.ndarray) -> np.ndarray:
        """把 ACT 动作对齐到环境动作维度(不一致时裁剪/补齐并记录一次)。"""
        env_dim = self._env_dim
        if env_dim is None or int(row.size) == int(env_dim):
            return row
        if not self._dim_mismatch_recorded:
            self._dim_mismatch_recorded = True
            self.failed = True
            self._record_failure(
                self._state,
                f"ACT 动作维度 {int(row.size)} != 环境动作维度 {int(env_dim)},"
                "已裁剪/补齐(夹爪覆盖同时禁用)",
            )
            # 维度不一致时,active_joint_ids 推出的通道下标不可信 → 禁用夹爪覆盖。
            self._gate["left_eef_channel"] = None
            self._gate["right_eef_channel"] = None
        if int(row.size) > int(env_dim):
            return np.asarray(row, dtype=np.float32)[: int(env_dim)]
        try:
            hold = self._hold_action()
            if int(hold.size) == int(env_dim):
                padded = np.asarray(hold, dtype=np.float32).copy()
                padded[: int(row.size)] = np.asarray(row, dtype=np.float32)
                return padded
        except Exception:
            pass
        return np.pad(np.asarray(row, dtype=np.float32), (0, int(env_dim) - int(row.size)))

    def _arm_joints(self, base: Any, arm: str) -> tuple[np.ndarray, float]:
        """读取机械臂关节角与夹爪开度(stereo.arm_joints 的安全封装)。

        优先使用 ``stereo.arm_joints``;不可用时退化为与之逐行等价的本地实现。
        """
        try:
            stereo = _lazy("stereo")
            q, grip = stereo.arm_joints(base, arm)
            return np.asarray(q, dtype=np.float64).reshape(-1), float(grip)
        except Exception:
            q_all = np.asarray(npv(base.robot.get_qpos(name=f"{arm}_arm")))
            q = q_all[0] if q_all.ndim > 1 else q_all
            grip_all = np.asarray(npv(base.robot.get_qpos(name=f"{arm}_eef")))
            return np.asarray(q, dtype=np.float64).reshape(-1), float(grip_all.reshape(-1)[0])

    # ------------------------------------------------------------------
    # 失败记录 / 降级
    # ------------------------------------------------------------------

    def _record_failure(
        self, stage: str, reason: str, *, dedup: bool = True, fatal: bool = True
    ) -> None:
        """记录失败原因。

        ``dedup=True`` 时相同 (stage, reason) 只记一次;``fatal=True`` 时把
        episode 标记为 ``failed``(阶段失败并触发 ACT 降级);``fatal=False``
        只做记录(例如正常跑到 max_env_steps 的截断)。
        """
        key = (str(stage), str(reason))
        if dedup and key in self._failure_keys:
            return
        if len(self.failures) >= MAX_FAILURES:
            return
        self._failure_keys.add(key)
        entry = {
            "stage": str(stage),
            "reason": str(reason),
            "fatal": bool(fatal),
            "env_step": int(self._steps_taken),
            "wall_time_s": float(time.time()),
        }
        self.failures.append(entry)
        self.failure_history.append(entry)
        if fatal:
            self.failed = True

    def _record_degradation(self, stage: str, reason: str) -> None:
        """记录“非致命降级”(例如掩码不可用但仍用 bbox 继续)。"""
        key = (f"soft:{stage}", str(reason))
        if key in self._failure_keys:
            return
        self._failure_keys.add(key)
        self.degradations.append(
            {
                "stage": str(stage),
                "reason": str(reason),
                "env_step": int(self._steps_taken),
            }
        )

    def _degrade(self, stage: str, exc: Exception | str) -> None:
        """任一阶段失败 → 记录原因并回退到纯 ACT(绝不抛异常/提前退出)。"""
        reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
        self.failed = True
        self._degrade_count += 1
        self._record_failure(stage, reason)
        if self._state not in (self.FALLBACK_ACT, self.ACT_FILL):
            self._enter(self.FALLBACK_ACT)
        if self._degrade_count > MAX_DEGRADES:
            # 降级路径自身持续失败(例如 env.step 已经不可用):强制收尾。
            self._force_truncated = True

    # ------------------------------------------------------------------
    # 状态推进
    # ------------------------------------------------------------------

    def _enter(self, state: str) -> None:
        """切换状态并重置该阶段的微步计数。"""
        self._stage_i = 0
        self._state = state
        self._history.append({"state": state, "env_step": int(self._steps_taken)})

    def _advance(self, env: Any, inference_times: list[float]) -> None:
        """调用当前状态对应的 ``_stage_*`` 方法(每个阶段一个方法)。"""
        handler: Callable[..., None] | None = getattr(
            self, f"_stage_{self._state.lower()}", None
        )
        if handler is None:
            raise RuntimeError(f"未知状态 {self._state!r}")
        handler(env, inference_times)

    def _mark_internal_steps(self, count: int) -> None:
        """登记由 stereo 等内部函数偷偷执行的 env.step 次数。"""
        self._steps_taken += max(0, int(count))
        self._produced_internal_step = True

    @staticmethod
    def _interp(
        q_from: np.ndarray, q_to: np.ndarray, index: int, steps: int
    ) -> np.ndarray:
        """插值到第 ``index`` 步。

        等价于原实现 ``np.linspace(1/steps, 1, steps)[index]``(端点式 linspace
        的步长恰为 1/steps,因此第 index 个 alpha == (index+1)/steps)。
        """
        alpha = float((index + 1) / max(1, int(steps)))
        return (1.0 - alpha) * np.asarray(q_from, dtype=np.float64) + alpha * np.asarray(
            q_to, dtype=np.float64
        )

    # ------------------------------------------------------------------
    # 阶段 0:SETTLE
    # ------------------------------------------------------------------

    def _capture_holds(self) -> None:
        """记录左右臂的初始关节角(后续运动阶段用它们保持另一条臂不动)。"""
        self._left_hold_q, self._left_hold_grip = self._arm_joints(self._base, "left")
        self._right_hold_q, self._right_hold_grip = self._arm_joints(self._base, "right")

    def _stage_settle(self, env: Any, inference_times: list[float]) -> None:
        """SETTLE:在原位保持 ``SETTLE_STEPS`` 步(原 fetch_depth 的 settle 动作)。"""
        if self._stage_i == 0:
            self._capture_holds()
            self._record_debug_three_cam("settle_three_cam.png")
        if self._left_hold_q is None or self._right_hold_q is None:
            raise RuntimeError("无法读取初始关节角(robot.get_qpos 失败)")
        action = full_action(
            self._base,
            self._left_hold_q,
            self._left_hold_grip,
            self._right_hold_q,
            self._right_hold_grip,
        )
        self._step_env_step(env, action)
        self._stage_i += 1
        if self._stage_i >= SETTLE_STEPS:
            self._enter(self.DEPTH_FETCH)

    # ------------------------------------------------------------------
    # 阶段 1:DEPTH_FETCH(内部会自己走若干 env.step)
    # ------------------------------------------------------------------

    def _stage_depth_fetch(self, env: Any, inference_times: list[float]) -> None:
        """DEPTH_FETCH:调用 ``stereo.fetch_depth`` 取一路 GT-free 运动立体深度。"""
        stereo = _lazy("stereo")
        if self._left_hold_q is None:
            self._capture_holds()
        depth = stereo.fetch_depth(
            self._base,
            "left",
            self._left_hold_q,
            settle_steps=FETCH_SETTLE_STEPS,
            motion_steps=MOTION_STEPS,
            nudge_candidates=NUDGE_CANDIDATES,
            baseline_target=BASELINE_TARGET_M,
        )
        if not isinstance(depth, dict):
            raise RuntimeError("fetch_depth 未返回字典")
        if depth.get("ok") is False:
            raise RuntimeError(f"fetch_depth 失败: {depth.get('error')}")
        if "reference_rgb" not in depth or "reference_camera_pose" not in depth:
            raise RuntimeError("fetch_depth 结果缺少 reference_rgb/reference_camera_pose")
        self._depth = depth
        self._depth_record = {
            key: jsonable(value) for key, value in depth.items() if key in SAFE_DEPTH_KEYS
        }
        # fetch_depth 内部:settle(FETCH_SETTLE_STEPS) + 运动(MOTION_STEPS)步。
        self._mark_internal_steps(FETCH_SETTLE_STEPS + MOTION_STEPS)
        self._debug_write_png("stereo_reference_rgb.png", depth.get("reference_rgb"))
        self._enter(self.VLM_LOCATE)

    # ------------------------------------------------------------------
    # 阶段 2:VLM_LOCATE
    # ------------------------------------------------------------------

    def _stage_vlm_locate(self, env: Any, inference_times: list[float]) -> None:
        """VLM_LOCATE:用 OpenAI 兼容 VLM 在腕相机 RGB 上定位牛奶(最多 2 次)。"""
        vlm = _lazy("vlm")
        if not isinstance(self._depth, dict):
            raise RuntimeError("缺少深度参考帧,无法做 VLM 定位")
        vision: Any = None
        attempts: list[Any] = []
        for attempt in range(VLM_ATTEMPTS):
            attempt_dir = self._debug_subdir(f"vlm/attempt_{attempt}")
            vision = vlm.locate_milk(
                self._depth["reference_rgb"],
                camera="cam_left_wrist",
                output_dir=attempt_dir,
                timeout_s=VLM_TIMEOUT_S,
            )
            attempts.append(vision)
            if isinstance(vision, dict) and vision.get("ok"):
                break
            if attempt + 1 < VLM_ATTEMPTS:
                time.sleep(VLM_RETRY_SLEEP_S)
        self._vlm_attempts = [jsonable(item) for item in attempts]
        self._vision = vision if isinstance(vision, dict) else {"ok": False, "error": "VLM 返回非字典"}
        self._debug_write_json("vlm.json", self._vision)
        if not self._vision.get("ok"):
            raise RuntimeError(
                f"VLM 牛奶定位失败: {self._vision.get('error') or 'unknown'}"
            )
        bbox = self._vision.get("bbox_xyxy")
        if not bbox or len(list(bbox)) != 4:
            raise RuntimeError(f"VLM 返回的 bbox_xyxy 非法: {bbox!r}")
        self._enter(self.SAM2)

    # ------------------------------------------------------------------
    # 阶段 3:SAM2
    # ------------------------------------------------------------------

    def _stage_sam2(self, env: Any, inference_times: list[float]) -> None:
        """SAM2:SAM2.1 掩码。模型不可用属致命失败(降级);掩码为空只算软降级。"""
        sam2_mask = _lazy("sam2_mask")
        if not isinstance(self._depth, dict) or not isinstance(self._vision, dict):
            raise RuntimeError("SAM2 阶段缺少深度帧或 VLM 结果")
        model = sam2_mask.load_sam2()
        mask_result = sam2_mask.segment_milk_mask(
            model,
            self._depth["reference_rgb"],
            self._vision["bbox_xyxy"],
            device=SAM2_DEVICE,
        )
        if not isinstance(mask_result, dict):
            raise RuntimeError("segment_milk_mask 未返回字典")
        self._mask_result = mask_result
        self._sam2_record = {
            key: jsonable(value) for key, value in mask_result.items() if key != "mask"
        }
        if mask_result.get("ok") and mask_result.get("mask") is not None:
            self._mask = mask_result["mask"]
            self._debug_write_png(
                "milk_sam2_mask.png",
                (np.asarray(self._mask).astype(np.uint8) * 255),
            )
            self._debug_write_json("milk_sam2.json", self._sam2_record)
        else:
            # 与原实现一致:掩码不可用时允许 mask=None 继续(只用 bbox 边缘对)。
            self._mask = None
            self._record_degradation(
                self.SAM2, f"掩码不可用,退化为仅用 bbox: {mask_result.get('error')}"
            )
        self._enter(self.ESTIMATE_POSE)

    # ------------------------------------------------------------------
    # 阶段 4:ESTIMATE_POSE
    # ------------------------------------------------------------------

    def _stage_estimate_pose(self, env: Any, inference_times: list[float]) -> None:
        """ESTIMATE_POSE:由 bbox(+掩码)与深度估计牛奶 6D 位姿。"""
        if not isinstance(self._depth, dict) or not isinstance(self._vision, dict):
            raise RuntimeError("位姿估计阶段缺少深度帧或 VLM 结果")
        milk_pose, point_info = estimated_milk_pose_from_bbox(
            self._depth,
            self._vision["bbox_xyxy"],
            self._vision.get("yaw_deg"),
            p1_uv=self._vision.get("p1_uv"),
            p2_uv=self._vision.get("p2_uv"),
            mask=self._mask,
        )
        if milk_pose is None:
            self._depth_error = jsonable(point_info)
            raise RuntimeError(
                f"牛奶位姿估计失败: {(point_info or {}).get('error') or 'unknown'}"
            )
        self._milk_pose = np.asarray(milk_pose, dtype=np.float64)
        self._point_info = jsonable(point_info)
        self._debug_write_json("milk_estimation.json", self._point_info)
        self._enter(self.GRASP_SEARCH)

    # ------------------------------------------------------------------
    # 阶段 5:GRASP_SEARCH
    # ------------------------------------------------------------------

    def _stage_grasp_search(self, env: Any, inference_times: list[float]) -> None:
        """GRASP_SEARCH:在 scale/yaw/z 网格上搜索可达抓取位姿与 IK 解。"""
        grasp = _lazy("grasp")
        if self._milk_pose is None:
            raise RuntimeError("抓取搜索阶段缺少牛奶位姿")
        if self._left_hold_q is None:
            self._capture_holds()
        q_seed = self._arm_joints(self._base, "left")[0]
        yaw_hint = (self._point_info or {}).get("estimated_world_yaw_deg")
        search = grasp.search_grasp_poses(
            self._base,
            self._milk_pose[:3, 3],
            q_seed,
            yaw_hint_deg=yaw_hint,
        )
        if not isinstance(search, dict) or search.get("best") is None:
            rejection = None if not isinstance(search, dict) else search.get("rejection")
            raise RuntimeError(f"抓取位姿搜索无解 (rejection={rejection})")
        best = search["best"]
        # 只保留摘要字段,不长期持有全部候选位姿数组。
        self._search = {
            "candidate_count": search.get("candidate_count"),
            "rejection": search.get("rejection"),
        }
        self._best = best
        self._pregrasp_pose = np.asarray(best["pregrasp"], dtype=np.float64)
        self._grasp_pose = np.asarray(best["grasp"], dtype=np.float64)
        self._q_pre = np.asarray(best["q_pre"], dtype=np.float64).reshape(-1)
        self._q_grasp = np.asarray(best["q_grasp"], dtype=np.float64).reshape(-1)
        self._target_record = jsonable(
            {
                "pregrasp": self._pregrasp_pose.tolist(),
                "grasp": self._grasp_pose.tolist(),
                **best.get("pose_info", {}),
                "searched_yaw_deg": best.get("yaw_deg"),
                "searched_scale": best.get("scale"),
                "searched_z_offset": best.get("z_offset"),
                "reachable_candidate_count": search.get("candidate_count"),
                "rejection": search.get("rejection"),
            }
        )
        self._debug_write_json("target_poses.json", self._target_record)
        self._enter(self.APPROACH)

    # ------------------------------------------------------------------
    # 阶段 6:APPROACH(移到位姿 + hold)
    # ------------------------------------------------------------------

    def _stage_approach(self, env: Any, inference_times: list[float]) -> None:
        """APPROACH:左臂从当前位姿插值到预抓取位姿 ``PREGRASP_STEPS`` 步,再 hold。"""
        if self._q_pre is None:
            raise RuntimeError("接近阶段缺少预抓取关节解")
        if self._stage_i == 0:
            if self._left_hold_q is None:
                self._capture_holds()
            self._approach_from = self._arm_joints(self._base, "left")[0]
        total = PREGRASP_STEPS + PREGRASP_HOLD_STEPS
        if self._stage_i < PREGRASP_STEPS:
            q = self._interp(self._approach_from, self._q_pre, self._stage_i, PREGRASP_STEPS)
        else:
            q = self._q_pre
        action = full_action(
            self._base, q, GRIP_OPEN, self._right_hold_q, self._right_hold_grip
        )
        self._step_env_step(env, action)
        self._stage_i += 1
        if self._stage_i >= total:
            self._enter(self.ADVANCE)

    # ------------------------------------------------------------------
    # 阶段 7:ADVANCE(预抓取 → 抓取)
    # ------------------------------------------------------------------

    def _stage_advance(self, env: Any, inference_times: list[float]) -> None:
        """ADVANCE:左臂从预抓取插值到抓取位姿(夹爪保持张开)。"""
        if self._q_pre is None or self._q_grasp is None:
            raise RuntimeError("推进阶段缺少抓取关节解")
        q = self._interp(self._q_pre, self._q_grasp, self._stage_i, ADVANCE_STEPS)
        action = full_action(
            self._base, q, GRIP_OPEN, self._right_hold_q, self._right_hold_grip
        )
        self._step_env_step(env, action)
        self._stage_i += 1
        if self._stage_i >= ADVANCE_STEPS:
            self._enter(self.CLOSE)

    # ------------------------------------------------------------------
    # 阶段 8:CLOSE(合拢夹爪 + hold)
    # ------------------------------------------------------------------

    def _stage_close(self, env: Any, inference_times: list[float]) -> None:
        """CLOSE:在抓取位姿保持 ``CLOSE_STEPS`` 步并把左夹爪合到 ``GRIP_GRASP``。"""
        if self._q_grasp is None:
            raise RuntimeError("合拢阶段缺少抓取关节解")
        total = CLOSE_STEPS + GRASP_HOLD_STEPS
        action = full_action(
            self._base, self._q_grasp, GRIP_GRASP, self._right_hold_q, self._right_hold_grip
        )
        self._step_env_step(env, action)
        self._stage_i += 1
        if self._stage_i >= total:
            self._enter(self.LIFT)

    # ------------------------------------------------------------------
    # 阶段 9:LIFT(先算 IK 再抬起 + hold)
    # ------------------------------------------------------------------

    def _stage_lift(self, env: Any, inference_times: list[float]) -> None:
        """LIFT:沿世界 z 抬升 0.10m 并保持。IK 无解 → 抛异常交给 ``eval`` 降级。"""
        if not self._lift_ik_done:
            self._lift_ik_done = True
            if self._grasp_pose is None or self._q_grasp is None:
                raise RuntimeError("抬起阶段缺少抓取位姿")
            grasp = _lazy("grasp")
            lift_pose = _lift_offset_pose(self._grasp_pose, LIFT_OFFSET_M)
            q_lift = grasp.solve_ik(self._base, lift_pose, self._q_grasp)
            if q_lift is None:
                raise RuntimeError("抬升 IK 无解 (q_lift is None)")
            self._q_lift = np.asarray(q_lift, dtype=np.float64).reshape(-1)
            # 本微步只做纯计算,不 step;下一次 _advance 立即从 LIFT_STEPS 开始。
            return
        if self._q_lift is None:
            raise RuntimeError("抬起阶段缺少抬升关节解")
        total = LIFT_STEPS + LIFT_HOLD_STEPS
        if self._stage_i < LIFT_STEPS:
            q = self._interp(self._q_grasp, self._q_lift, self._stage_i, LIFT_STEPS)
        else:
            q = self._q_lift
        action = full_action(
            self._base, q, GRIP_GRASP, self._right_hold_q, self._right_hold_grip
        )
        self._step_env_step(env, action)
        self._stage_i += 1
        if self._stage_i >= total:
            if self._total_chunks > 0:
                self._enter(self.ACT_TAKEOVER)
            else:
                self._enter(self.DONE)

    # ------------------------------------------------------------------
    # ACT 执行(所有 ACT 阶段共用的微步)
    # ------------------------------------------------------------------

    def _select_chunk_action(self, obs: Any) -> list[np.ndarray]:
        """调用 ``act_policy.select_chunk_action`` 并计时(只计模型前向)。"""
        policy = self.act_policy
        device = _timing_device(policy)
        started_at = start_inference(device)
        completed = False
        ran_inference = True
        action: Any = None
        try:
            out = policy.select_chunk_action(obs)
            if isinstance(out, tuple) and len(out) == 2:
                action, ran_inference = out[0], bool(out[1])
            else:
                action = out
            completed = True
        finally:
            if completed and ran_inference:
                finish_inference(started_at, self._inference_times_all, device)
        self._act_select_calls += 1
        if ran_inference:
            self._act_ran_inference += 1
        return _action_rows(action)

    def _next_act_action(self, inference_times: list[float]) -> np.ndarray:
        """取下一个待执行动作:缓存用尽时重新调一次策略(一个 chunk)。"""
        if not self._pending_rows:
            rows = self._select_chunk_action(self._obs)
            if not rows:
                raise RuntimeError("select_chunk_action 返回了空动作序列")
            self._pending_rows = rows
        return self._pending_rows.pop(0)

    def _judge_place_release(self, obs: Any) -> dict:
        """release gate 的 VLM 判定;任何异常都返回 ``ok=False``(保持硬锁)。"""
        rgb_high = _safe_rgb_of(obs, "cam_high")
        rgb_wrist = _safe_rgb_of(obs, "cam_left_wrist")
        if rgb_high is None or rgb_wrist is None:
            return {"ok": False, "error": "缺少 cam_high/cam_left_wrist 图像"}
        try:
            vlm = _lazy("vlm")
            output_dir = self._debug_subdir(
                f"release_vlm/chunk_{self._act_chunk_index}_step_{self._gate['step']}"
            )
            result = vlm.judge_place_release(
                rgb_high,
                rgb_wrist,
                output_dir=output_dir,
                timeout_s=VLM_RELEASE_TIMEOUT_S,
            )
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(result, dict):
            return {"ok": False, "error": "judge_place_release 返回非字典"}
        return result

    def _act_guard(self, action: np.ndarray, obs: Any) -> tuple[np.ndarray, bool, float | None]:
        """ACT 动作覆盖守卫(逐行对应原实现 line 807-900 的 step 包装前半段)。

        1. release gate:未获准放料时,左夹爪每一步都被硬写成 ``GRIP_GRASP``;
        2. ``chunk_idx == 2``:右夹爪强制张开(为随后的篮柄抓取做准备);
        3. ``chunk_idx >= PRE_GRASP_ACT_CHUNKS`` 且已锁定篮柄:右夹爪覆盖为
           ``RIGHT_GRIP_TIGHT``。

        返回 ``(覆盖后的动作, open_request, cmd_value)``。
        """
        gate = self._gate
        array = np.asarray(npv(action), dtype=np.float32).copy().reshape(-1)
        left_channel = gate.get("left_eef_channel")
        right_channel = gate.get("right_eef_channel")
        if left_channel is not None and not (0 <= int(left_channel) < array.size):
            left_channel = None
        if right_channel is not None and not (0 <= int(right_channel) < array.size):
            right_channel = None

        cmd_value: float | None = None
        cmd_delta = 0.0
        open_request = False
        if left_channel is not None:
            cmd_value = float(array[int(left_channel)])
            if gate["last_cmd"] is not None:
                cmd_delta = cmd_value - float(gate["last_cmd"])
            # 夹爪物理张开之前就检测“想张开”的意图(旧的 >0.2 阈值太晚)。
            open_request = bool(
                cmd_value > ACT_OPEN_INTENT_CMD or cmd_delta > ACT_OPEN_INTENT_DELTA
            )
        if not gate["allowed"] and left_channel is not None:
            if open_request:
                # 新的放料意图,或上一次 edge 判定后的复检时刻。
                is_new_intent = not gate["last_open"]
                is_edge_poll = (
                    gate["edge_poll_remaining"] > 0
                    and gate["next_check_step"] is not None
                    and gate["step"] >= gate["next_check_step"]
                )
                if is_new_intent or is_edge_poll:
                    release_result = self._judge_place_release(obs)
                    gate["last_check_step"] = gate["step"]
                    if len(gate["checks"]) < MAX_GATE_CHECKS:
                        gate["checks"].append(
                            {
                                "chunk": self._act_chunk_index,
                                "step": gate["step"],
                                "act_left_eef": cmd_value,
                                "result": jsonable(release_result),
                            }
                        )
                    place_ready = bool(
                        release_result.get("ok") and release_result.get("place_ready")
                    )
                    relation = release_result.get("place_relation")
                    if place_ready:
                        gate["allowed"] = True
                        gate["edge_streak"] = 0
                        gate["edge_poll_remaining"] = 0
                        gate["next_check_step"] = None
                    elif relation == "edge":
                        gate["edge_streak"] += 1
                        if gate["edge_streak"] >= 2:
                            gate["allowed"] = True
                            gate["edge_poll_remaining"] = 0
                            gate["next_check_step"] = None
                        else:
                            gate["edge_poll_remaining"] = 3
                            gate["next_check_step"] = gate["step"] + 5
                    else:
                        gate["edge_streak"] = 0
                        if is_edge_poll:
                            gate["edge_poll_remaining"] = max(
                                0, gate["edge_poll_remaining"] - 1
                            )
                            gate["next_check_step"] = (
                                gate["step"] + 5
                                if gate["edge_poll_remaining"] > 0
                                else None
                            )
                        else:
                            gate["edge_poll_remaining"] = 0
                            gate["next_check_step"] = None
                if not gate["allowed"]:
                    gate["blocked_count"] += 1
            if not gate["allowed"]:
                # 硬锁:在放料 VLM 放行之前,无论 ACT 想不想张开,左夹爪指令
                # 每个控制步都被覆盖为 GRIP_GRASP。
                array[int(left_channel)] = GRIP_GRASP
                gate["locked_steps"] += 1
        if self._act_chunk_index == 2 and right_channel is not None:
            array[int(right_channel)] = GRIP_OPEN
            self.right_open_hold_steps += 1
        if (
            self._act_chunk_index >= PRE_GRASP_ACT_CHUNKS
            and self._grasp_state.get("locked")
            and right_channel is not None
        ):
            array[int(right_channel)] = RIGHT_GRIP_TIGHT
        return array, open_request, cmd_value

    def _act_guard_finish(self, open_request: bool, cmd_value: float | None) -> None:
        """step 之后的 gate 状态更新(原实现 line 902-905)。"""
        self._gate["last_open"] = bool(open_request)
        self._gate["last_cmd"] = cmd_value
        self._gate["step"] += 1

    def _act_micro_step(
        self, env: Any, inference_times: list[float], *, guarded: bool
    ) -> tuple[Any, bool]:
        """执行 1 个 ACT 控制步,返回 ``(obs, 当前 chunk 是否结束)``。

        策略前向失败时把后续执行降级为“保持当前位姿”,保证 episode 仍能跑完。
        """
        if self._act_remaining <= 0:
            self._act_remaining = self._act_chunk_budget
            self._pending_rows = []
            self._act_chunks_started += 1

        open_request = False
        cmd_value: float | None = None
        if self._act_broken:
            action = self._hold_action()
            guard_active = False
        else:
            try:
                action = self._next_act_action(inference_times)
                guard_active = guarded
            except Exception as exc:
                self._act_broken = True
                self.failed = True
                self._record_failure(
                    self._state,
                    "ACT 前向失败,后续退化为保持当前位姿: "
                    f"{type(exc).__name__}: {exc}",
                )
                action = self._hold_action()
                guard_active = False

        row = self._fit_action(np.asarray(action, dtype=np.float32).reshape(-1))
        if guard_active:
            row, open_request, cmd_value = self._act_guard(row, self._obs)
        obs, _done = self._step_env_step(env, row)
        if guard_active:
            self._act_guard_finish(open_request, cmd_value)
        self._act_remaining -= 1
        self._act_steps += 1
        self._record_act_state()
        return obs, self._act_remaining <= 0

    def _stage_act_takeover(self, env: Any, inference_times: list[float]) -> None:
        """ACT_TAKEOVER:抓取前的 ``PRE_GRASP_ACT_CHUNKS`` 个 ACT chunk。"""
        if self._pre_chunks <= 0:
            self._enter(self.DONE)
            return
        if self._act_chunk_index >= self._pre_chunks:
            self._enter(self.RIGHT_GRASP)
            return
        _obs, finished = self._act_micro_step(env, inference_times, guarded=True)
        if finished:
            self._act_chunk_index += 1
            if self._act_chunk_index >= self._pre_chunks:
                self._enter(self.RIGHT_GRASP)

    def _stage_right_grasp(self, env: Any, inference_times: list[float]) -> None:
        """RIGHT_GRASP:右臂下压 0.01m 并合拢抓篮柄(原实现 line 933-952)。"""
        if not self._right_grasp_ready:
            self._right_grasp_ready = True
            grasp = _lazy("grasp")
            q_left_now, grip_left_now = self._arm_joints(self._base, "left")
            q_right_now, _grip_right_now = self._arm_joints(self._base, "right")
            tcp_right = grasp.fk_pose_named(self._base, q_right_now, "right_arm")
            target_right = np.asarray(tcp_right, dtype=np.float64).copy()
            target_right[2, 3] -= RIGHT_GRASP_DOWN_M
            q_down = grasp.solve_ik_named(
                self._base, target_right, q_right_now, "right_arm"
            )
            self._grasp_state["down_ik"] = q_down is not None
            if q_down is None:
                # 原实现在这里 break(放弃剩余 ACT chunk);改造后只记录失败并
                # 直接继续后续 ACT,以免白白丢掉剩余步数。
                self._grasp_state["error"] = "right grasp down IK failed"
                self.failed = True
                self._record_failure(
                    self.RIGHT_GRASP,
                    "右臂下压抓篮柄 IK 无解,跳过右臂抓取,直接继续后续 ACT",
                )
                self._enter(self.ACT_POST)
                return
            self._right_grasp_ref = {
                "q_left": q_left_now,
                "grip_left": grip_left_now,
                "q_right": q_right_now,
                "q_down": np.asarray(q_down, dtype=np.float64).reshape(-1),
            }
            return  # 纯计算微步:IK 不算 env.step,下一次 _advance 开始运动。
        ref = self._right_grasp_ref
        if ref is None:
            self._enter(self.ACT_POST)
            return
        total = RIGHT_GRASP_MOVE_STEPS + RIGHT_GRASP_CLOSE_STEPS
        if self._stage_i < RIGHT_GRASP_MOVE_STEPS:
            q_right = self._interp(
                ref["q_right"], ref["q_down"], self._stage_i, RIGHT_GRASP_MOVE_STEPS
            )
            right_grip = GRIP_OPEN
        else:
            q_right = ref["q_down"]
            right_grip = RIGHT_GRIP_TIGHT
        action = full_action(
            self._base, ref["q_left"], ref["grip_left"], q_right, right_grip
        )
        self._step_env_step(env, action)
        self._stage_i += 1
        if self._stage_i >= total:
            _q_right, closed_grip = self._arm_joints(self._base, "right")
            self._grasp_state["locked"] = True
            self._grasp_state["closed_grip"] = float(closed_grip)
            self._enter(self.ACT_POST)

    def _stage_act_post(self, env: Any, inference_times: list[float]) -> None:
        """ACT_POST:抓篮柄之后的 ``POST_GRASP_ACT_CHUNKS`` 个 ACT chunk。"""
        if self._act_chunk_index >= self._total_chunks:
            self._enter(
                self.RIGHT_RELEASE if self._grasp_state.get("locked") else self.DONE
            )
            return
        _obs, finished = self._act_micro_step(env, inference_times, guarded=True)
        if finished:
            self._act_chunk_index += 1

    def _stage_right_release(self, env: Any, inference_times: list[float]) -> None:
        """RIGHT_RELEASE:右夹爪张开、松开篮柄(原实现 line 955-958)。"""
        if self._stage_i == 0:
            q_left_rel, grip_left_rel = self._arm_joints(self._base, "left")
            q_right_rel, _grip_right_rel = self._arm_joints(self._base, "right")
            self._right_release_ref = {
                "q_left": q_left_rel,
                "grip_left": grip_left_rel,
                "q_right": q_right_rel,
            }
        ref = self._right_release_ref
        if ref is None:
            self._enter(self.DONE)
            return
        action = full_action(
            self._base, ref["q_left"], ref["grip_left"], ref["q_right"], GRIP_OPEN
        )
        self._step_env_step(env, action)
        self._stage_i += 1
        if self._stage_i >= RIGHT_RELEASE_STEPS:
            self._enter(self.DONE)

    def _stage_done(self, env: Any, inference_times: list[float]) -> None:
        """DONE:流程正常走完,剩余步数按 ``DONE_RUNS_PURE_ACT`` 处理。"""
        self._enter(self.ACT_FILL)

    def _stage_act_fill(self, env: Any, inference_times: list[float]) -> None:
        """ACT_FILL:流程走完后用剩余步数继续(默认纯 ACT,可选保持位姿)。"""
        if DONE_RUNS_PURE_ACT:
            self._act_micro_step(env, inference_times, guarded=False)
        else:
            self._step_env_step(env, self._hold_action())

    def _stage_fallback_act(self, env: Any, inference_times: list[float]) -> None:
        """FALLBACK_ACT:任一阶段失败后的纯 ACT 执行(不做任何夹爪覆盖)。"""
        self._act_micro_step(env, inference_times, guarded=False)

    # ------------------------------------------------------------------
    # 调试落盘(debug_dir 为 None 时一律不落盘)
    # ------------------------------------------------------------------

    def _debug_subdir(self, relative: str) -> Path | None:
        if self.debug_dir is None:
            return None
        try:
            path = Path(self.debug_dir) / relative
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            return None

    def _debug_write_png(self, relative: str, image: Any) -> None:
        if self.debug_dir is None or image is None:
            return
        try:
            import imageio.v2 as imageio

            path = Path(self.debug_dir) / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(path, np.asarray(image))
        except Exception:
            # 调试落盘失败绝不影响评测流程。
            pass

    def _debug_write_json(self, relative: str, payload: Any) -> None:
        if self.debug_dir is None or payload is None:
            return
        try:
            path = Path(self.debug_dir) / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _record_debug_three_cam(self, relative: str) -> None:
        """仅在 debug_dir 非空时保存一帧三路相机拼图。"""
        if self.debug_dir is None or self._obs is None:
            return
        frame = frame3(self._obs)
        if frame is not None:
            self._debug_write_png(relative, frame)

    def _record_act_state(self) -> None:
        """记录 ACT 每步的关节状态(仅 debug 模式;原实现 line 906-918)。"""
        if self.debug_dir is None or len(self._act_state) >= MAX_ACT_STATE:
            return
        try:
            grasp = _lazy("grasp")
            q_left, grip_left = self._arm_joints(self._base, "left")
            tcp_left = grasp.fk_pose(self._base, q_left)
            self._act_state.append(
                {
                    "chunk": self._act_chunk_index,
                    "step": self._act_steps,
                    "left_q": np.asarray(q_left).tolist(),
                    "left_grip": float(grip_left),
                    "left_tcp": np.asarray(tcp_left).tolist(),
                    "release_allowed": bool(self._gate["allowed"]),
                }
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 收尾查询
    # ------------------------------------------------------------------

    def _gate_summary(self) -> dict:
        """release gate 摘要(去掉不需要的字段)。"""
        gate = self._gate
        return {
            "left_eef_channel": gate.get("left_eef_channel"),
            "right_eef_channel": gate.get("right_eef_channel"),
            "allowed": bool(gate.get("allowed")),
            "last_check_step": gate.get("last_check_step"),
            "edge_poll_remaining": gate.get("edge_poll_remaining"),
            "edge_streak": gate.get("edge_streak"),
            "next_check_step": gate.get("next_check_step"),
            "blocked_count": gate.get("blocked_count"),
            "locked_steps": gate.get("locked_steps"),
            "step": gate.get("step"),
            "checks": gate.get("checks", []),
        }

    def _query_success(self) -> bool | None:
        """查询 ``env.get_wrapper_attr("is_task_success")()``(失败返回 None)。"""
        env = self._env
        if env is None:
            return None
        try:
            getter = getattr(env, "get_wrapper_attr", None)
            if not callable(getter):
                return None
            func = getter("is_task_success")
            if not callable(func):
                return None
            return bool(func())
        except Exception:
            return None
