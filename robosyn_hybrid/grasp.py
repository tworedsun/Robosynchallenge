"""handle_basket 抓取位姿搜索与逆运动学(移植自参考实现 line 414-576)。

本模块只做**几何计算**,不读取任何物体真值:

* 输入是 env 的最内层实例(只使用其 ``robot`` 的 FK/IK 与关节状态);
* 输入位姿 ``milk_pose`` 由 ``flow.py`` 用 RGB-D(运动立体)+VLM 估计得到;
* 抓取先验 ``MILK_GRASP_OBJECT`` / ``MILK_PRIOR_SIZE`` 是官方
  ``configs/handle_basket/*/gym_config.json`` 已公开的常量。

对应关系(与 SPEC §2.7 一致)::

    build_grasp_poses(base, milk_pose, grasp_object=None)
    scaled_grasp_object(scale)
    fk_pose(base, q)
    ik_with_seeds(base, pose, seeds)
    search_grasp_poses(base, milk_position, q_seed, yaw_hint_deg=None)

另外把参考实现里的内部工具 ``solve_ik`` / ``solve_ik_named`` /
``fk_pose_named`` 也放在这里,供 ``flow.py``(右臂抓篮柄)复用,
避免同一段 IK 代码出现两份实现。
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

# 官方仓库自带的位姿工具(与参考实现完全相同的 import)。
from embodichain.lab.gym.utils.misc import (
    get_frame_changed_pose,
    get_offset_pose,
    get_rotation_replaced_pose,
    get_rotated_pose,
)

__all__ = [
    # SPEC §2.7 规定的公开接口
    "build_grasp_poses",
    "scaled_grasp_object",
    "fk_pose",
    "ik_with_seeds",
    "search_grasp_poses",
    # 参考实现内部工具(供 flow.py 复用)
    "solve_ik",
    "solve_ik_named",
    "fk_pose_named",
    "npv",
    # 常量
    "MILK_GRASP_OBJECT",
    "MILK_PRIOR_SIZE",
    "GRASP_TILT_DEG",
    "PREGRASP_OFFSET_M",
]

# ---------------------------------------------------------------------------
# 常量(与 priors.py 同值;此处内联以保持与原实现逐行一致)
# ---------------------------------------------------------------------------

#: 官方 handle_basket gym_config 里公开的“物体坐标系 -> 抓取坐标系”变换。
MILK_GRASP_OBJECT = np.array(
    [
        [0.0, 0.0, 1.0, 0.02],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.03],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

#: 官方公开的牛奶盒尺寸先验(米)。
MILK_PRIOR_SIZE = np.array([0.043, 0.0995, 0.0444], dtype=np.float64)

#: 抓取姿态绕自身 y 轴的俯仰角(度),原实现 line 49。
GRASP_TILT_DEG = -20.0

#: 预抓取点相对抓取点沿自身 z 轴的退让距离(米),原实现 line 48。
PREGRASP_OFFSET_M = 0.05


def _torch() -> Any:
    """惰性 import torch:本模块只在真实评测环境里被调用。"""
    import torch

    return torch


def npv(value: Any) -> np.ndarray:
    """把 torch 张量/普通容器统一成 numpy 数组(原实现 line 71-72)。"""
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


# ---------------------------------------------------------------------------
# IK / FK
#
# 注意:下面 4 个函数与参考实现一样把张量写死放到 ``.cuda()``(官方
# ``deploy_policy.yml`` 的 ``pytorch_device: cuda``)。若在没有 CUDA 的机器上
# 调用,torch 会抛错 —— 这不是崩溃路径:``flow.py`` 的每个阶段都有 try/except
# 降级,会让 episode 退回纯 ACT 正常跑完。
# ---------------------------------------------------------------------------


def solve_ik(base: Any, pose: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
    """左臂单次 IK(原实现 ``solve_ik``,line 121-135)。

    ``base`` 必须是 env 的最内层实例(含 ``robot``)。失败返回 ``None``,
    绝不抛异常以外的“隐式失败”。
    """
    torch = _torch()
    pose_t = torch.as_tensor(pose, dtype=torch.float32).cuda()
    if pose_t.ndim == 2:
        pose_t = pose_t.unsqueeze(0)
    seed_t = torch.as_tensor(seed, dtype=torch.float32).cuda().unsqueeze(0)
    result = base.robot.compute_ik(pose_t, joint_seed=seed_t, name="left_arm")
    if result is None:
        return None
    success, qpos = result
    if not bool(torch.as_tensor(success).all().item()):
        return None
    q = npv(qpos).reshape(-1)[:6]
    return q.astype(np.float64)


def solve_ik_named(
    base: Any, pose: np.ndarray, seed: np.ndarray, name: str
) -> np.ndarray | None:
    """任意机械臂的 IK(原实现 ``solve_ik_named``,line 167-181)。"""
    torch = _torch()
    pose_t = torch.as_tensor(pose, dtype=torch.float32).cuda()
    if pose_t.ndim == 2:
        pose_t = pose_t.unsqueeze(0)
    seed_t = torch.as_tensor(seed, dtype=torch.float32).cuda().unsqueeze(0)
    result = base.robot.compute_ik(pose_t, joint_seed=seed_t, name=name)
    if result is None:
        return None
    success, qpos = result
    if not bool(torch.as_tensor(success).all().item()):
        return None
    q = npv(qpos).reshape(-1)[:6]
    return np.asarray(q, dtype=np.float64)


def fk_pose(base: Any, q: np.ndarray) -> np.ndarray:
    """左臂末端(``left_arm``)正运动学,返回 4x4(原实现 ``fk_pose``,line 454-463)。"""
    torch = _torch()
    result = base.robot.compute_fk(
        torch.as_tensor(q, dtype=torch.float32).cuda(),
        name="left_arm",
        to_matrix=True,
    )
    matrix = np.asarray(npv(result)).reshape(-1, 4, 4)[0]
    return matrix


def fk_pose_named(base: Any, q: np.ndarray, name: str) -> np.ndarray:
    """任意机械臂 FK,返回 4x4(原实现 ``fk_pose_named``,line 184-192)。"""
    torch = _torch()
    result = base.robot.compute_fk(
        torch.as_tensor(q, dtype=torch.float32).cuda(),
        name=name,
        to_matrix=True,
    )
    return np.asarray(npv(result)).reshape(-1, 4, 4)[0]


# ---------------------------------------------------------------------------
# 抓取位姿构造 / 搜索(原实现 line 414-576)
# ---------------------------------------------------------------------------


def build_grasp_poses(
    base: Any, milk_pose: np.ndarray, grasp_object: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """由估计出的牛奶位姿构造 (预抓取, 抓取, 信息) 三个返回(原实现 line 414-445)。

    注意:SPEC §2.7 把返回类型宽松地写成 ``list[np.ndarray]``,但参考实现
    返回的是 ``(pregrasp, grasp, {"aim_yaw_deg": ...})`` 三元组——此处严格
    照搬参考实现的行为。
    """
    base_pose = npv(base.robot.get_link_pose(link_name="left_arm_base", to_matrix=True))
    base_pose = np.asarray(base_pose).reshape(-1, 4, 4)[0]
    # 让左手从底座指向牛奶的水平偏航角。
    aim_yaw = float(
        np.arctan2(
            *(
                (milk_pose[:2, 3] - base_pose[:2, 3])[1::-1]
            )
        )
    )
    grasp = get_rotation_replaced_pose(
        milk_pose.copy(), aim_yaw, rot_axis="z", mode="intrinsic"
    )
    grasp = get_frame_changed_pose(
        grasp, milk_pose, mode="intrinsic", inverse=True
    )
    if grasp_object is None:
        grasp_object = MILK_GRASP_OBJECT
    # 官方动作库先把“物体局部抓取变换”复合进当前牛奶位姿,再做 intrinsic
    # 换系。直接使用原始 object-local 矩阵会把候选点放到场地原点附近。
    milk_grasp_pose = get_frame_changed_pose(
        milk_pose.copy(), grasp_object, mode="intrinsic"
    )
    grasp = get_frame_changed_pose(grasp, milk_grasp_pose, mode="intrinsic")
    grasp = get_rotated_pose(
        grasp, GRASP_TILT_DEG, rot_axis="y", mode="intrinsic", degrees=True
    )
    pregrasp = get_offset_pose(
        grasp.copy(), -PREGRASP_OFFSET_M, direction="z", mode="intrinsic"
    )
    return pregrasp, grasp, {"aim_yaw_deg": float(np.degrees(aim_yaw))}


def scaled_grasp_object(scale: float) -> np.ndarray:
    """按比例缩放抓取先验的平移部分(原实现 line 448-451)。"""
    matrix = MILK_GRASP_OBJECT.copy()
    matrix[:3, 3] *= float(scale)
    return matrix


def ik_with_seeds(
    base: Any, pose: np.ndarray, seeds: Iterable[np.ndarray]
) -> dict[str, Any] | None:
    """多初值 IK + FK 复核,返回误差最小的解(原实现 line 466-488)。

    返回 ``dict``(含 ``qpos`` / ``position_error`` / ``rotation_error_deg`` /
    ``joint_distance`` / ``score``);全部初值失败时返回 ``None``。
    SPEC §2.7 把返回类型写成 ``np.ndarray | None``,参考实现实际返回指标字典,
    此处以参考实现为准。
    """
    best: dict[str, Any] | None = None
    for seed in seeds:
        q = solve_ik(base, pose, seed)
        if q is None:
            continue
        actual = fk_pose(base, q)
        position_error = float(np.linalg.norm(actual[:3, 3] - pose[:3, 3]))
        trace = float(np.trace(actual[:3, :3].T @ pose[:3, :3]))
        rotation_error = float(
            np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
        )
        joint_distance = float(np.linalg.norm(q - np.asarray(seed, dtype=np.float64)))
        score = position_error + 0.001 * rotation_error + 0.0005 * joint_distance
        if best is None or score < best["score"]:
            best = {
                "qpos": q,
                "position_error": position_error,
                "rotation_error_deg": rotation_error,
                "joint_distance": joint_distance,
                "score": score,
            }
    return best


def search_grasp_poses(
    base: Any,
    milk_position: np.ndarray,
    q_seed: np.ndarray,
    yaw_hint_deg: float | None = None,
) -> dict[str, Any]:
    """在 scale / z-offset / yaw 网格上搜索可达抓取(原实现 line 491-574)。

    返回 ``{"best": dict | None, "candidate_count": int, "rejection": dict}``。
    ``best is None`` 表示 IK 无解,调用方应降级(不得抛异常)。
    """
    from scipy.spatial.transform import Rotation

    hint = 0.0 if yaw_hint_deg is None else float(yaw_hint_deg)
    try:
        init_q = np.asarray(
            npv(base.robot.get_qpos(name="left_arm")).reshape(-1), dtype=np.float64
        )
    except Exception:
        init_q = np.zeros(6, dtype=np.float64)
    seeds: list[np.ndarray] = [
        np.asarray(q_seed, dtype=np.float64),
        init_q,
        np.zeros(6, dtype=np.float64),
    ]
    scales = (0.90, 0.95, 1.00, 1.05, 1.10)
    z_offsets = (-0.01, 0.0, 0.01)
    candidates: list[dict[str, Any]] = []
    rejection = {"no_ik": 0, "far_from_center": 0, "bad_pose": 0}
    for scale in scales:
        grasp_object = scaled_grasp_object(scale)
        for z_offset in z_offsets:
            center = np.asarray(milk_position, dtype=np.float64).copy()
            center[2] += z_offset
            for yaw_deg in np.arange(0.0, 360.0, 15.0):
                milk_pose = np.eye(4)
                milk_pose[:3, :3] = Rotation.from_euler(
                    "z", np.radians(yaw_deg)
                ).as_matrix()
                milk_pose[:3, 3] = center
                pregrasp, grasp, pose_info = build_grasp_poses(
                    base, milk_pose, grasp_object=grasp_object
                )
                position_error = float(
                    np.linalg.norm(pregrasp[:3, 3] - center)
                    + np.linalg.norm(grasp[:3, 3] - center)
                )
                if position_error > 0.28:
                    rejection["far_from_center"] += 1
                    continue
                # 先解可达的抓取点,再用该关节角作为退让点的初值。某些摆放
                # 下精确的预抓取位姿刚好在 IK 工作空间之外,而抓取点可达。
                grasp_ik = ik_with_seeds(base, grasp, seeds)
                if grasp_ik is None:
                    rejection["no_ik"] += 1
                    continue
                pre_ik = ik_with_seeds(base, pregrasp, [grasp_ik["qpos"], *seeds])
                if pre_ik is None:
                    rejection["no_ik"] += 1
                    continue
                if pre_ik["position_error"] > 0.025 or pre_ik["rotation_error_deg"] > 18.0:
                    rejection["bad_pose"] += 1
                    continue
                if (
                    grasp_ik["position_error"] > 0.025
                    or grasp_ik["rotation_error_deg"] > 18.0
                ):
                    rejection["bad_pose"] += 1
                    continue
                delta = abs((float(yaw_deg) - hint + 180.0) % 360.0 - 180.0)
                score = (
                    position_error
                    + 0.08 * (pre_ik["position_error"] + grasp_ik["position_error"])
                    + 0.01 * (pre_ik["rotation_error_deg"] + grasp_ik["rotation_error_deg"])
                    + 0.0003 * delta
                    + 0.20 * abs(scale - 1.0)
                    + 0.50 * abs(z_offset)
                )
                candidates.append(
                    {
                        "score": score,
                        "yaw_deg": float(yaw_deg),
                        "scale": float(scale),
                        "z_offset": float(z_offset),
                        "position_error": position_error,
                        "pregrasp": pregrasp,
                        "grasp": grasp,
                        "pose_info": pose_info,
                        "q_pre": pre_ik["qpos"],
                        "q_grasp": grasp_ik["qpos"],
                        "pre_ik": pre_ik,
                        "grasp_ik": grasp_ik,
                    }
                )
    if not candidates:
        return {"best": None, "candidate_count": 0, "rejection": rejection}
    return {
        "best": min(candidates, key=lambda item: item["score"]),
        "candidate_count": len(candidates),
        "rejection": rejection,
    }
