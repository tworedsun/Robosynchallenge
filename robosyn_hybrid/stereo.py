"""运动立体深度(GT-free 子集)。

本模块是下列两份只读参考源码的**合规子集**移植:

* ``_stage/Depth_Fetch/common/scripts/measure_motion_stereo_depth.py``
* ``_stage/Depth_Fetch/click_bell/depth_fetch.py``

只保留不依赖真值的部分:RGB 观测、机器人本体感受(``get_qpos`` /
``get_qpos_limits``)、正运动学(``compute_fk``)、官方公开配置里的常量
(相机内参、腕相机安装位姿)。参考实现里凡是需要读物体真值位姿、模拟器
深度缓冲或"先建环境再挑视角"的那些辅助函数**一律未移植**,包括它们的
入口点;``capture`` 也**不再读取**观测里的 ``"depth"`` 字段(模拟器深度
缓冲是合规红线)。

绝对路径全部消除
----------------
FS_ROOT / FS_INFER / venv / socket 全部来自 :mod:`robosyn_hybrid.settings`
(``FS_ROOT`` / ``FS_CKPT`` / ``FS_PYTHON`` / ``FS_STEREO_SOCKET`` /
``FS_STEREO_VALID_ITERS``)。中间文件一律走 :mod:`tempfile` 创建的目录,
socket 路径也只由 settings 决定,参考实现里那几个写死的路径常量全部删除。
``import torch`` 全部保持在函数内部延迟导入,因此没有 torch 时本模块
仍然可以被 import。

``verify_mount`` 的结论(腕相机安装关系)—— **已由实测结案**
----------------------------------------------------------
参考实现的相机光学位姿是
``camera_optical_pose = sensor.get_arena_pose(to_matrix=True) @ SIM_TO_OPTICAL``,
而 parented 相机在 EmbodiChain 里的 local pose 就是 ``[R(quat) | pos]``,
于是

    cam_from_link6 = inv(fk_link6(q)) @ camera_optical_pose
                   = [R(quat) | pos] @ SIM_TO_OPTICAL

这正是 :func:`robosyn_hybrid.priors.camera_mount` 的返回值(它已经把
``SIM_TO_OPTICAL`` 乘进去了)。

**但关键前提必须补上**:2026-10-10 在真实 RTX 4090 + ``EmbodiChain_main
0.2.4`` 检出上实测确认,``robot.compute_fk(name="<arm>_arm",
link_names=["<arm>_link6"], to_matrix=True)`` 的 ``link_names`` **实际被
忽略**,返回的是 **tool/TCP 帧**而不是 ``<arm>_link6`` 连杆帧。因此主路径是

    wrist_camera_pose(q) = fk_returned(q) @ FK_TO_LINK6 @ camera_mount_raw(arm)

其中 ``FK_TO_LINK6 = inv(TOOL_FROM_LINK6)`` 就是
:data:`robosyn_hybrid.priors.FK_TO_LINK6`,由实测反解得来
(``TOOL_FROM_LINK6`` = 绕 z 轴 180° 再沿 z 平移 +0.143 m,残差 1.277e-06)。
该偏置已内置在 :func:`robosyn_hybrid.priors.camera_mount` 内部,由
``HYBRID_FK_FRAME`` 开关控制(``"tool"`` 默认补偏置 / ``"link6"`` 不补)。

**四元数顺序结论 = ``wxyz``**(``[w, x, y, z]``),证据如下(均为公开源码,
非运行时真值):

1. ``embodichain.utils.math.matrix_from_quat`` 的 docstring 明确写
   *"The quaternion orientation in (w, x, y, z)"*;
2. ``SensorCfg.OffsetCfg.quat`` 的 docstring 同样是 ``(w, x, y, z)``,而
   ``OffsetCfg.transformation`` 就是用 ``matrix_from_quat`` 构造 ``[R|pos]``;
3. 官方 harness 把同一组常量直接命名为 ``WRIST_MOUNT_QUAT_WXYZ`` 并以
   ``quat_order="wxyz"`` 装载
   (``Offical_Harness/rpent_ext/harness_tasks/click_bell.py``)。

组合顺序与四元数顺序:**旧解析预测表已作废**
--------------------------------------------
下表是修复**之前**的纯解析预测(以 ``wxyz + post_flip`` 为真值、只用公开
常量与推导出的 FK 关系算出,未使用仿真器)。**它已被真实 GPU 上的实测
推翻**:修复前实测位置误差 **0.280 m**、旋转误差 **180°**(右臂
``best_variant`` 落到 ``xyzw/pre_flip``),根因就是上面那条被忽略的
``link_names``(FK 返回 tool 帧)。补上 ``FK_TO_LINK6`` 后,右臂与左臂的
L1 实测均回到 ``wxyz/post_flip`` 且位置误差 < 2 mm、旋转误差 < 0.5°。
**下表只作为"纯解析预测有多不可靠"的反面记录保留,不要再据此推断运行时
行为。**

    =====================  ==================  ==================  ==========================
    变体(旧解析预测)       位置误差            旋转误差            说明
    =====================  ==================  ==================  ==========================
    wxyz + post_flip       0.00 mm             0.00°               **主路径;补 FK_TO_LINK6 后实测成立**
    xyzw + post_flip       0.00 mm             152.81°             四元数顺序读反(平移同值)
    wxyz + no_flip         0.00 mm             180.00°             漏乘 SIM_TO_OPTICAL
    xyzw + no_flip         0.00 mm             129.82°             两个错误叠加
    wxyz + pre_flip        63.25 mm            180.00°            SIM_TO_OPTICAL 乘错一侧
    xyzw + pre_flip        63.25 mm            80.19°             两个错误叠加
    =====================  ==================  ==================  ==========================

.. warning::
   **上表是解析预测,不是实测**,并且已被实测证伪:旧表说 ``wxyz/post_flip``
   误差为零,实测是 **0.280 m / 180°**。真实数字请以评测机上 L1 层的 JSON
   输出为准:

   .. code-block:: python

       from robosyn_hybrid.stereo import verify_mount
       print(verify_mount(env, "left"))

   重测入口是 ``python robosyn_hybrid/scripts/verify_clickbell_geometry.py``
   的 L1 层;返回的 ``candidates`` / ``measured_mount`` / ``warnings`` 应与
   ``docs/COMPLIANCE.md`` §2.5 的记录一致。``verify_mount`` 只记录差异、
   从不抛错,并且在合规模式下(不访问 ``env.sensors``)会直接回报
   ``sensor_available=False``。

合规说明
--------
:func:`verify_mount` 会读取 ``env.sensors[...]`` 的相机位姿,这**只是开发期
诊断**,不属于合规主路径。``fetch_depth`` / ``find_nudge`` /
``wrist_camera_pose`` 全程不触碰 ``env.sensors``,也不读模拟器深度缓冲。
"""

from __future__ import annotations

import json
import socket
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import priors
from .settings import get_settings

__all__ = [
    "LATERAL_MIN",
    "LATERAL_MAX",
    "VERTICAL_FRACTION",
    "AXIAL_FRACTION",
    "MOUNT_QUAT_ORDER",
    "MOUNT_VARIANTS",
    "REFERENCE_MOUNT_VARIANT",
    "arm_joints",
    "full_action",
    "move_arm",
    "capture",
    "fk_link6",
    "link6_pose",
    "wrist_camera_pose",
    "mount_quat_order",
    "verify_mount",
    "rectification",
    "find_nudge",
    "run_fs",
    "fetch_depth",
    "sample_depth",
    "estimate_point_from_pixel",
    "fs_infer_path",
]

# ---------------------------------------------------------------------------
# 常量:与原实现完全一致(公开配置里的运动约束,不含任何真值)
# ---------------------------------------------------------------------------
LATERAL_MIN, LATERAL_MAX = 0.020, 0.070
VERTICAL_FRACTION, AXIAL_FRACTION = 0.35, 0.30

#: 本模块采用的四元数顺序。运行时由 ``HYBRID_QUAT_ORDER`` 覆盖,默认 ``wxyz``。
MOUNT_QUAT_ORDER: str = "wxyz"

#: ``verify_mount`` 枚举的全部候选组合:``(四元数顺序, SIM_TO_OPTICAL 位置)``。
MOUNT_VARIANTS: tuple[tuple[str, str], ...] = (
    ("wxyz", "post_flip"),
    ("xyzw", "post_flip"),
    ("wxyz", "no_flip"),
    ("xyzw", "no_flip"),
    ("wxyz", "pre_flip"),
    ("xyzw", "pre_flip"),
)

#: ``verify_mount`` 认为"正确"的变体;主路径与它必须一致。
REFERENCE_MOUNT_VARIANT: tuple[str, str] = ("wxyz", "post_flip")

_LINK6_SUFFIX = "_link6"


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def _to_numpy(value: Any) -> np.ndarray:
    """把 torch tensor / 标量 / 数组统一转成 numpy 数组(不 import torch)。"""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _env_attr(env: Any, name: str) -> Any:
    """在可能被 Gymnasium 包了一层的 ``env`` 上取属性。"""
    unwrapped = getattr(env, "unwrapped", None)
    for candidate in (env, unwrapped):
        if candidate is not None and hasattr(candidate, name):
            return getattr(candidate, name)
    getter = getattr(env, "get_wrapper_attr", None)
    if callable(getter):
        try:
            return getter(name)
        except Exception as exc:  # noqa: BLE001 - 只是把它变成更清楚的错误
            raise AttributeError(f"env exposes no attribute {name!r}: {exc}") from exc
    raise AttributeError(f"env exposes neither attribute {name!r} nor get_wrapper_attr()")


def _robot(env: Any) -> Any:
    """取底层 robot 句柄。只用于本体感受 / FK,不读任何物体真值。"""
    return _env_attr(env, "robot")


def _active_joint_ids(env: Any) -> np.ndarray:
    ids = _to_numpy(_env_attr(env, "active_joint_ids")).astype(np.int64).reshape(-1)
    return ids


def _get_obs(env: Any) -> dict:
    """调用 ``env.get_obs()``,兼容 wrapper。"""
    getter = getattr(env, "get_obs", None)
    if getter is None and hasattr(env, "get_wrapper_attr"):
        try:
            getter = env.get_wrapper_attr("get_obs")
        except Exception:  # noqa: BLE001
            getter = None
    if getter is None:
        raise AttributeError("env exposes neither get_obs() nor get_wrapper_attr('get_obs')")
    return getter()


def _to_torch_device(tensor: Any, env: Any) -> Any:
    """把 tensor 放到 robot 所在设备上(原实现写死 ``.cuda()``)。"""
    import torch

    device = getattr(_robot(env), "device", None)
    if device is not None:
        try:
            return tensor.to(device)
        except Exception:  # noqa: BLE001 - 退回到下面的探测
            pass
    if torch.cuda.is_available():
        try:
            return tensor.cuda()
        except Exception:  # noqa: BLE001
            pass
    return tensor


def _rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    """两个旋转矩阵之间的测地角度差(度)。"""
    cos_angle = (np.trace(rotation_a @ rotation_b.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _pose_delta(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """平移(m)与旋转(度)误差。"""
    return {
        "pos_err_m": float(np.linalg.norm(predicted[:3, 3] - truth[:3, 3])),
        "rot_err_deg": _rotation_error_deg(predicted[:3, :3], truth[:3, :3]),
    }


def fs_infer_path() -> Path:
    """本包内 ``fs_infer.py`` 的路径(相对定位,无绝对路径常量)。"""
    return Path(__file__).resolve().parent / "fs_infer.py"


def mount_quat_order() -> str:
    """当前生效的四元数顺序(来自 ``HYBRID_QUAT_ORDER``,默认 ``wxyz``)。"""
    return get_settings().quat_order or MOUNT_QUAT_ORDER


# ---------------------------------------------------------------------------
# 机器人本体感受 / 动作
# ---------------------------------------------------------------------------
def arm_joints(env: Any, arm: str) -> tuple[np.ndarray, float]:
    """返回 ``(q6, grip)``:某条臂的 6 个关节角与夹爪开度。"""
    robot = _robot(env)
    q = _to_numpy(robot.get_qpos(name=f"{arm}_arm"))[0].astype(np.float64)
    grip = float(_to_numpy(robot.get_qpos(name=f"{arm}_eef"))[0, 0])
    return q, grip


def full_action(
    env: Any,
    left_q: Any,
    left_grip: float,
    right_q: Any,
    right_grip: float,
) -> np.ndarray:
    """把两臂 + 两夹爪拼成 ``active_joint_ids`` 顺序的完整 14 维动作。"""
    robot = _robot(env)
    target = _to_numpy(robot.get_qpos())[0].astype(np.float32)
    for name, value in (
        ("left_arm", left_q),
        ("left_eef", [left_grip]),
        ("right_arm", right_q),
        ("right_eef", [right_grip]),
    ):
        ids = _to_numpy(robot.get_joint_ids(name=name, remove_mimic=True)).astype(np.int64).reshape(-1)
        target[ids] = np.asarray(value, dtype=np.float32)
    return target[_active_joint_ids(env)]


def move_arm(env: Any, arm: str, q_target: Any, steps: int) -> None:
    """把 ``arm`` 线性插值移动到 ``q_target``;另一条臂保持不动。

    与原实现一致:若过程中 episode 被 terminated / truncated,抛
    :class:`RuntimeError`,由上层状态机降级到纯 ACT。
    """
    import torch

    steps = max(int(steps), 1)
    other = "right" if arm == "left" else "left"
    q_start, this_grip = arm_joints(env, arm)
    other_q, other_grip = arm_joints(env, other)
    for alpha in np.linspace(1.0 / steps, 1.0, steps):
        q = (1.0 - alpha) * q_start + alpha * np.asarray(q_target, dtype=np.float64)
        action = (
            full_action(env, q, this_grip, other_q, other_grip)
            if arm == "left"
            else full_action(env, other_q, other_grip, q, this_grip)
        )
        _obs, _reward, terminated, truncated, _info = env.step(
            torch.as_tensor(action).unsqueeze(0)
        )
        if bool(_to_numpy(terminated).reshape(-1)[0]) or bool(
            _to_numpy(truncated).reshape(-1)[0]
        ):
            raise RuntimeError(f"episode ended while moving the {arm} arm")
    return None


def capture(env: Any, arm: str) -> tuple[np.ndarray, dict]:
    """抓一帧腕相机 RGB,返回 ``(rgb uint8 HxWx3, obs)``。

    与原 ``measure_motion_stereo_depth.capture`` 的差别:这里**只取 RGB**,
    不再返回 ``cam["depth"]``——模拟器深度缓冲属于合规红线。
    """
    obs = _get_obs(env)
    cam = obs["sensor"][f"cam_{arm}_wrist"]
    rgb = np.asarray(_to_numpy(cam["color"]))
    while rgb.ndim > 3:
        rgb = rgb[0]
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    return rgb, obs


# ---------------------------------------------------------------------------
# 相机位姿:FK + 公开常量(不访问 env.sensors)
# ---------------------------------------------------------------------------
def fk_link6(env: Any, arm: str, qs: Any) -> np.ndarray:
    """批量正运动学,返回 ``(N, 4, 4)`` 的 ``<arm>_link6`` 世界位姿。

    注意:在实测的官方检出上,``compute_fk`` 的 ``link_names`` 会被忽略、
    实际返回 tool/TCP 帧(见模块 docstring),所以**本函数的返回值语义上仍
    是 tool 帧**;由 ``priors.camera_mount`` 内部左乘 ``FK_TO_LINK6`` 完成
    到 link6 的换算。若 ``compute_fk`` 连 ``link_names`` 参数都不接受,
    直接抛 :class:`RuntimeError`,不做任何静默退回。
    """
    import torch

    robot = _robot(env)
    link_name = f"{arm}{_LINK6_SUFFIX}"
    poses: list[np.ndarray] = []
    for q in np.atleast_2d(np.asarray(qs, dtype=np.float32)):
        tensor = _to_torch_device(torch.as_tensor(q), env)
        try:
            raw = robot.compute_fk(
                tensor, name=f"{arm}_arm", link_names=[link_name], to_matrix=True
            )
        except (TypeError, AttributeError) as exc:
            raise RuntimeError(
                "robot.compute_fk() did not accept link_names=[...]; refusing to "
                f"fall back to the end-effector frame for {link_name} "
                "(the fixed tool->link6 offset would have to be composed "
                f"explicitly): {exc}"
            ) from exc
        out = np.asarray(_to_numpy(raw)).reshape(-1, 4, 4)
        if out.shape[0] == 1:
            poses.append(out[0])
        else:
            poses.append(out[list(robot.link_names).index(link_name)])
    return np.stack(poses).astype(np.float64)


def link6_pose(env: Any, arm: str, q: Any | None = None) -> np.ndarray:
    """``<arm>_link6`` 的世界位姿;``q`` 为 ``None`` 时用当前 ``qpos``。"""
    if q is None:
        q, _grip = arm_joints(env, arm)
    return fk_link6(env, arm, np.asarray(q, dtype=np.float64)[None])[0]


def wrist_camera_pose(env: Any, arm: str) -> np.ndarray:
    """腕相机世界 -> 光学的 4x4 位姿。

    主路径 = ``fk_returned(qpos) @ FK_TO_LINK6 @ camera_mount_raw(arm)``。
    ``priors.camera_mount`` 已包含 ``SIM_TO_OPTICAL``,并在
    ``HYBRID_FK_FRAME="tool"``(默认)时额外左乘实测解出的 ``FK_TO_LINK6``
    ——因为 ``compute_fk(link_names=["<arm>_link6"])`` 实际返回的是 tool 帧。
    **不访问 ``env.sensors``**,不读任何深度缓冲。
    """
    q, _grip = arm_joints(env, arm)
    return link6_pose(env, arm, q) @ priors.camera_mount(arm)


def _mount_variant(arm: str, order: str, flip: str) -> np.ndarray:
    """按 ``(四元数顺序, SIM_TO_OPTICAL 位置)`` 组合出候选安装矩阵。"""
    mount = priors.camera_mount(arm, quat_order=order)
    if flip == "post_flip":
        return mount
    if flip == "no_flip":
        # camera_mount = T_mount @ F,且 F = F^-1(diag(1,-1,-1,1) 自逆)
        return mount @ priors.SIM_TO_OPTICAL
    if flip == "pre_flip":
        return priors.SIM_TO_OPTICAL @ mount
    raise ValueError(f"unknown flip variant {flip!r}")


def verify_mount(env: Any, arm: str) -> dict:
    """把常量路径与 ``env.sensors`` 真值比对;**只记录差异,从不抛错**。

    返回值包含:

    * ``sensor_available`` / ``error``:``env.sensors`` 是否可用;
    * ``measured_mount``:``inv(fk_link6(q)) @ sensor_optical_pose``,
      应当等于 ``priors.camera_mount(arm)``;
    * ``candidates``:``MOUNT_VARIANTS`` 每个变体相对真值的位置 / 旋转误差;
    * ``best_variant``:误差最小的变体;``matches_reference_path`` 表示它是否
      就是主路径 ``REFERENCE_MOUNT_VARIANT``;
    * ``warnings``:不一致时的人类可读提示(FK 坐标系、四元数顺序等)。

    这是**开发期诊断**,不属于合规主路径;合规模式下 ``env.sensors`` 通常
    不可用,函数会平静地返回 ``sensor_available=False``。
    """
    record: dict[str, Any] = {
        "ok": False,
        "arm": arm,
        "sensor_available": False,
        "reference_variant": list(REFERENCE_MOUNT_VARIANT),
        "configured_quat_order": mount_quat_order(),
        "error": None,
        "candidates": {},
        "warnings": [],
    }
    try:
        sensors = _env_attr(env, "sensors")
        uid = f"cam_{arm}_wrist"
        try:
            sensor = sensors[uid]
        except Exception:  # noqa: BLE001 - 也接受 .get() 风格的容器
            sensor = sensors.get(uid)  # type: ignore[union-attr]
        if sensor is None:
            raise KeyError(f"no sensor named {uid!r} in env.sensors")
        arena = np.asarray(
            _to_numpy(sensor.get_arena_pose(to_matrix=True))[0], dtype=np.float64
        )
        record["sensor_available"] = True
        truth = arena @ priors.SIM_TO_OPTICAL

        q, _grip = arm_joints(env, arm)
        link = link6_pose(env, arm, q)
        link_inv = np.linalg.inv(link)

        measured_mount = link_inv @ truth
        expected_mount = priors.camera_mount(arm)
        measured_raw = link_inv @ arena
        expected_raw = expected_mount @ priors.SIM_TO_OPTICAL

        record["measured_mount"] = measured_mount.tolist()
        record["expected_mount"] = expected_mount.tolist()
        record["measured_mount_pos_m"] = measured_mount[:3, 3].tolist()
        record["mount_delta"] = _pose_delta(measured_mount, expected_mount)
        record["raw_mount_delta"] = _pose_delta(measured_raw, expected_raw)
        record["sensor_translation_m"] = arena[:3, 3].tolist()
        record["fk_translation_m"] = link[:3, 3].tolist()
        record["fk_to_sensor_offset_m"] = float(np.linalg.norm(arena[:3, 3] - link[:3, 3]))

        candidates: dict[str, dict[str, Any]] = {}
        for order, flip in MOUNT_VARIANTS:
            name = f"{order}/{flip}"
            try:
                predicted = link @ _mount_variant(arm, order, flip)
            except Exception as exc:  # noqa: BLE001 - 只记录
                candidates[name] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            candidates[name] = _pose_delta(predicted, truth)
        record["candidates"] = candidates

        scored = [
            (name, value)
            for name, value in candidates.items()
            if "pos_err_m" in value and "rot_err_deg" in value
        ]
        if scored:
            best_name, best_value = min(
                scored, key=lambda item: (item[1]["pos_err_m"], item[1]["rot_err_deg"])
            )
            record["best_variant"] = best_name
            record["best_variant_delta"] = best_value
            record["matches_reference_path"] = best_name == "/".join(REFERENCE_MOUNT_VARIANT)
            record["ok"] = bool(
                record["matches_reference_path"]
                and best_value["pos_err_m"] < 2e-3
                and best_value["rot_err_deg"] < 0.5
            )
            if best_name != "/".join(REFERENCE_MOUNT_VARIANT):
                record["warnings"].append(
                    f"best mount variant is {best_name!r}, but the module hard-codes "
                    f"{'/'.join(REFERENCE_MOUNT_VARIANT)!r}; check HYBRID_QUAT_ORDER / "
                    "the published mount constants"
                )
            if best_value["pos_err_m"] >= 2e-3:
                record["warnings"].append(
                    "no candidate matches the sensor within 2 mm; compute_fk() may be "
                    "returning the end-effector frame instead of "
                    f"{arm}{_LINK6_SUFFIX}"
                )
        # FK 坐标系自检:link6 -> sensor 的距离应当等于公开常量 pos 的模长。
        measured_norm = float(np.linalg.norm(measured_mount[:3, 3]))
        expected_norm = float(np.linalg.norm(expected_mount[:3, 3]))
        record["mount_offset_norm_measured_m"] = measured_norm
        record["mount_offset_norm_expected_m"] = expected_norm
        if abs(measured_norm - expected_norm) >= 2e-3:
            record["warnings"].append(
                "measured link6->camera offset length differs from the published "
                f"constant by {abs(measured_norm - expected_norm) * 1000:.1f} mm"
            )
    except Exception as exc:  # noqa: BLE001 - 只记录差异,绝不抛错
        record["error"] = f"{type(exc).__name__}: {exc}"

    print(
        "[verify_mount] "
        + json.dumps(
            {
                "arm": arm,
                "sensor_available": record["sensor_available"],
                "error": record["error"],
                "best_variant": record.get("best_variant"),
                "matches_reference_path": record.get("matches_reference_path"),
                "mount_delta": record.get("mount_delta"),
                "warnings": record["warnings"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return record


# ---------------------------------------------------------------------------
# 几何
# ---------------------------------------------------------------------------
def rectification(cam1: np.ndarray, cam2: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Bouguet 极线校正,与原实现逐行一致。"""
    R1, p1 = cam1[:3, :3], cam1[:3, 3]
    R2, p2 = cam2[:3, :3], cam2[:3, 3]
    t = R1.T @ (p2 - p1)
    baseline = float(np.linalg.norm(t))
    if baseline < 1e-6:
        raise ValueError(
            "motion-stereo baseline is degenerate (two exposures share a camera centre); "
            "the nudge did not translate the wrist camera"
        )
    e1 = t / baseline
    e2 = np.cross(np.array([0.0, 0.0, 1.0]), e1)
    e2 /= max(np.linalg.norm(e2), 1e-9)
    e3 = np.cross(e1, e2)
    Rr = np.array([e1, e2, e3])
    return Rr, Rr @ (R1.T @ R2), baseline


def find_nudge(
    env: Any,
    arm: str,
    q1: np.ndarray,
    cam1: np.ndarray,
    n_candidates: int,
    baseline_target: float,
) -> tuple[np.ndarray, np.ndarray]:
    """在关节空间随机搜索一个"图像水平方向"的小位移 ``q2``。

    只用 FK + ``qpos`` 限制(与参考实现一致),不使用任何物体真值:
    相机相对 link6 的安装矩阵由 ``inv(fk(q1)) @ cam1`` 反推,因此即使公开
    常量有微小偏差,搜索方向依然自洽。
    """
    q1 = np.asarray(q1, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(0)
    limits = _to_numpy(_robot(env).get_qpos_limits(name=f"{arm}_arm"))
    limits = limits[0] if limits.ndim == 3 else limits
    limits = np.asarray(limits, dtype=np.float64).reshape(-1, 2)
    lo, hi = limits[:, 0], limits[:, 1]

    n_candidates = max(int(n_candidates), 1)
    cand = np.clip(q1 + rng.normal(0.0, 0.06, size=(n_candidates, q1.shape[0])), lo, hi)
    cam_from_link6 = np.linalg.inv(fk_link6(env, arm, q1[None]))[0] @ cam1
    cams = fk_link6(env, arm, cand) @ cam_from_link6
    d_cam = (cams[:, :3, 3] - cam1[:3, 3]).astype(np.float64) @ cam1[:3, :3]
    base_x, vert, axial = np.abs(d_cam[:, 0]), np.abs(d_cam[:, 1]), np.abs(d_cam[:, 2])
    ok = (
        (base_x >= LATERAL_MIN)
        & (base_x <= LATERAL_MAX)
        & (vert <= VERTICAL_FRACTION * np.maximum(base_x, 1e-9))
        & (axial <= AXIAL_FRACTION * np.maximum(base_x, 1e-9))
    )
    if not ok.any():
        raise RuntimeError("no image-horizontal nudge found for this pose")
    score = axial / np.maximum(base_x, 1e-9) + vert / np.maximum(base_x, 1e-9)
    if baseline_target:
        order = np.lexsort((score, np.abs(base_x - baseline_target)))
    else:
        order = np.lexsort((-base_x, score))
    pick = int(order[np.flatnonzero(ok[order])[0]])
    return cand[pick].astype(np.float64), d_cam[pick]


# ---------------------------------------------------------------------------
# FoundationStereo:常驻 worker 优先,子进程回退
# ---------------------------------------------------------------------------
def _fs_payload(
    left_png: Path,
    right_png: Path,
    fx: float,
    baseline: float,
    out_npz: Path,
    valid_iters: int,
) -> dict:
    """worker 协议负载(与 ``fs_server.py`` 的 JSON-lines 协议完全一致)。"""
    return {
        "left": str(left_png),
        "right": str(right_png),
        "fx": float(fx),
        "baseline": float(baseline),
        "out": str(out_npz),
        "valid_iters": int(valid_iters),
    }


def _fs_worker_call(payload: dict, socket_path: Path, timeout_s: float) -> dict | None:
    """向常驻 worker 要一张视差图;不可用或失败时返回 ``None``。"""
    if not socket_path.exists():
        return None
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(float(timeout_s))
    try:
        sock.connect(str(socket_path))
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        buffer = b""
        while not buffer.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            buffer += chunk
        return json.loads(buffer.decode("utf-8") or "{}")
    except Exception as exc:  # noqa: BLE001 - worker 掉线不是致命错误
        print(
            f"[fs] persistent worker unavailable ({type(exc).__name__}: {exc}); "
            "falling back to a per-call subprocess",
            flush=True,
        )
        return None
    finally:
        sock.close()


def _fs_subprocess(payload: dict, ckpt: str, timeout_s: float) -> None:
    """子进程回退:``settings.fs_python`` 跑 ``robosyn_hybrid/fs_infer.py``。"""
    settings = get_settings()
    cmd = [
        str(settings.fs_python),
        str(fs_infer_path()),
        "--left",
        payload["left"],
        "--right",
        payload["right"],
        "--ckpt",
        str(ckpt),
        "--fx",
        f"{float(payload['fx']):.6f}",
        "--baseline",
        f"{float(payload['baseline']):.8f}",
        "--valid-iters",
        str(int(payload["valid_iters"])),
        "--out",
        payload["out"],
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=float(timeout_s), check=False
    )
    if proc.stdout:
        print(proc.stdout.strip()[-1500:], flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FoundationStereo failed:\n{proc.stderr[-3000:]}")


def _load_fs_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"FoundationStereo produced no output at {path}")
    with np.load(path) as data:
        missing = [key for key in ("disparity", "depth", "valid") if key not in data]
        if missing:
            raise RuntimeError(f"FoundationStereo output {path} is missing keys {missing}")
        # 显式拷贝,避免数组与已关闭的 npz 文件产生生命周期耦合。
        return (
            np.array(data["disparity"], dtype=np.float32),
            np.array(data["depth"], dtype=np.float32),
            np.array(data["valid"]).astype(bool),
        )


def run_fs(
    rect1: np.ndarray,
    rect2: np.ndarray,
    ckpt: str,
    fx: float,
    baseline: float,
    valid_iters: int = 32,
    *,
    out_npz: Path | str | None = None,
    socket_path: Path | str | None = None,
    timeout_s: float = 1800.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对校正后的立体像对跑 FoundationStereo。

    优先走 ``settings.fs_socket`` 常驻 worker(见 ``fs_infer.py --serve``),
    失败则回退到 ``settings.fs_python`` + ``robosyn_hybrid/fs_infer.py`` 子进程。
    中间文件全部落在 :mod:`tempfile` 创建的目录里。

    返回 ``(disparity, depth, valid)``。
    """
    settings = get_settings()
    left = np.asarray(rect1)
    right = np.asarray(rect2)
    if left.shape != right.shape:
        raise ValueError(f"image size mismatch: {left.shape} vs {right.shape}")
    sock = Path(socket_path) if socket_path is not None else Path(settings.fs_socket)

    with tempfile.TemporaryDirectory(prefix="robosyn_fs_") as tmp:
        tmp_dir = Path(tmp)
        left_png = tmp_dir / "left.png"
        right_png = tmp_dir / "right.png"
        cv2.imwrite(str(left_png), cv2.cvtColor(left[..., :3], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(right_png), cv2.cvtColor(right[..., :3], cv2.COLOR_RGB2BGR))
        npz_path = Path(out_npz) if out_npz is not None else tmp_dir / "fs_raw.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _fs_payload(
            left_png, right_png, float(fx), float(baseline), npz_path, int(valid_iters)
        )

        reply = _fs_worker_call(payload, sock, timeout_s)
        if reply is not None:
            if reply.get("ok"):
                print(
                    f"[fs] persistent worker: infer={reply.get('infer_s')}s "
                    f"total={reply.get('total_s')}s valid={reply.get('valid_fraction')}",
                    flush=True,
                )
                try:
                    return _load_fs_npz(npz_path)
                except Exception as exc:  # noqa: BLE001 - 回退到子进程
                    print(f"[fs] worker reply unusable ({exc}); using a subprocess", flush=True)
            else:
                print(
                    f"[fs] persistent worker failed: {reply.get('error')}; "
                    "falling back to a per-call subprocess",
                    flush=True,
                )

        _fs_subprocess(payload, str(ckpt), timeout_s)
        return _load_fs_npz(npz_path)


# ---------------------------------------------------------------------------
# 单次深度获取 + 像素反投影
# ---------------------------------------------------------------------------
def _warn_if_fs_missing() -> None:
    settings = get_settings()
    if not settings.fs_ckpt.is_file():
        warnings.warn(
            f"FoundationStereo checkpoint not found at {settings.fs_ckpt}; set FS_CKPT "
            "or FS_ROOT (and run robosyn_hybrid/scripts/setup_fs_env.sh). A running worker may still "
            "serve requests with its own FS_STEREO_CKPT.",
            stacklevel=2,
        )


def fetch_depth(
    env: Any,
    arm: str,
    q1: np.ndarray,
    *,
    settle_steps: int = 8,
    motion_steps: int = 5,
    nudge_candidates: int = 2000,
    baseline_target: float = 0.02,
) -> dict:
    """移动立体深度:两次曝光 + 极线校正 + FoundationStereo。

    返回 dict 的键与 ``depth_fetch.fetch_depth`` 完全一致(下游 ``vlm.py`` /
    ``flow.py`` 依赖它们):

    ``ok, arm, q1, q2, reference_exposure, reference_rgb, reference_rect,
    rect2, reference_camera_pose, K0, K_rect, Rr1, Rr2, H_ref, baseline_m,
    motion_camera_m, rectification_deg, disparity, depth_map, valid_map``

    ``depth_map`` 与 ``reference_rect``(以及经 ``H_ref`` 映射后的
    ``reference_rgb``)对齐。全程不读物体真值、不读模拟器深度缓冲。
    """
    settings = get_settings()
    _warn_if_fs_missing()
    q1 = np.asarray(q1, dtype=np.float64).reshape(-1)

    move_arm(env, arm, q1, settle_steps)
    rgb1, _obs = capture(env, arm)
    cam1 = wrist_camera_pose(env, arm)
    K0 = priors.camera_K(f"cam_{arm}_wrist")

    q2, d_cam = find_nudge(env, arm, q1, cam1, nudge_candidates, baseline_target)
    move_arm(env, arm, q2, motion_steps)
    rgb2, _obs = capture(env, arm)
    cam2 = wrist_camera_pose(env, arm)

    t1 = cam1[:3, :3].T @ (cam2[:3, 3] - cam1[:3, 3])
    reference_exposure = 1
    if t1[0] < 0:
        rgb1, rgb2 = rgb2, rgb1
        cam1, cam2 = cam2, cam1
        reference_exposure = 2

    Rr1, Rr2, baseline = rectification(cam1, cam2)
    h, w = rgb1.shape[:2]
    corners = np.array(
        [[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]], dtype=np.float64
    ).T
    cc = (K0 @ Rr1 @ np.linalg.inv(K0)) @ corners
    cc = cc[:2] / cc[2]
    offset = np.array([(w - 1) / 2.0, (h - 1) / 2.0]) - cc.mean(axis=1)
    K_rect = np.array(K0, dtype=np.float32, copy=True)
    K_rect[0, 2] += float(offset[0])
    K_rect[1, 2] += float(offset[1])
    H1 = K_rect @ Rr1 @ np.linalg.inv(K0)
    rect1 = cv2.warpPerspective(rgb1, H1, (w, h))
    rect2 = cv2.warpPerspective(rgb2, K_rect @ Rr2 @ np.linalg.inv(K0), (w, h))

    disp, depth_fs, valid_fs = run_fs(
        rect1,
        rect2,
        str(settings.fs_ckpt),
        float(K_rect[0, 0]),
        float(baseline),
        int(settings.fs_valid_iters),
    )
    return {
        "ok": True,
        "arm": arm,
        "q1": np.asarray(q1, dtype=np.float64),
        "q2": np.asarray(q2, dtype=np.float64),
        "reference_exposure": reference_exposure,
        "reference_rgb": rgb1,
        "reference_rect": rect1,
        "rect2": rect2,
        "reference_camera_pose": cam1,
        "K0": K0,
        "K_rect": K_rect,
        "Rr1": Rr1,
        "Rr2": Rr2,
        "H_ref": H1,
        "baseline_m": float(baseline),
        "motion_camera_m": np.asarray(d_cam, dtype=np.float64),
        "rectification_deg": float(
            np.degrees(np.arccos(np.clip((np.trace(Rr1) - 1) / 2, -1, 1)))
        ),
        "disparity": disp,
        "depth_map": depth_fs,
        "valid_map": valid_fs,
    }


def sample_depth(
    depth: np.ndarray,
    valid: np.ndarray,
    u: float,
    v: float,
    radius: int = 3,
) -> tuple[float | None, int, tuple[int, int]]:
    """在 ``(u, v)`` 周围取有效深度的中位数,返回 ``(depth, count, pixel)``。"""
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None, 0, (ui, vi)
    x0, x1 = max(0, ui - radius), min(w, ui + radius + 1)
    y0, y1 = max(0, vi - radius), min(h, vi + radius + 1)
    patch = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64)
    vpatch = np.asarray(valid[y0:y1, x0:x1], dtype=bool)
    vals = patch[vpatch & np.isfinite(patch) & (patch > 1e-4)]
    if vals.size == 0:
        return None, 0, (ui, vi)
    return float(np.median(vals)), int(vals.size), (ui, vi)


def estimate_point_from_pixel(
    result: dict, pixel: Any, sample_radius: int = 3
) -> dict:
    """把参考 RGB 上的一个像素反投影成世界坐标点。"""
    if not result.get("ok", False):
        raise ValueError("depth fetch result is not valid")
    u0, v0 = float(pixel[0]), float(pixel[1])
    H = result["H_ref"]
    p_h = H @ np.array([u0, v0, 1.0])
    u_r, v_r = float(p_h[0] / p_h[2]), float(p_h[1] / p_h[2])
    depth, count, px = sample_depth(
        result["depth_map"], result["valid_map"], u_r, v_r, sample_radius
    )
    if depth is None:
        return {
            "ok": False,
            "error": "no valid depth at mapped pixel",
            "pixel_rectified": [u_r, v_r],
        }
    K_rect = result["K_rect"]
    Rr1 = result["Rr1"]
    cam = result["reference_camera_pose"]
    p_rect = depth * np.linalg.inv(K_rect) @ np.array([u_r, v_r, 1.0])
    p_original = Rr1.T @ p_rect
    p_world = cam[:3, :3] @ p_original + cam[:3, 3]
    return {
        "ok": True,
        "pixel_original": [u0, v0],
        "pixel_rectified": [u_r, v_r],
        "pixel_rounded": list(px),
        "depth_m": float(depth),
        "sample_count": int(count),
        "p_camera_m": [float(x) for x in p_original],
        "p_world_m": [float(x) for x in p_world],
        "reference_camera_pose": cam,
    }
