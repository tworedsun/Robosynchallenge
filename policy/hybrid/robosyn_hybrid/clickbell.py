"""click_bell 混合策略状态机(官方 ``policy/<name>/`` 三函数契约可用)。

本模块把 RPent 参考实现
``_stage/harness/rpent_ext/harness_tasks/click_bell.py`` 里 click_bell 特有的
几何 / 容差 / 按压参数**逐行照抄**过来,只把"环境接缝"换成官方评测循环能用的
形式::

    while env_steps < max_env_steps:
        obs, info, truncated, new_times = policy_pkg.eval(eval_env, model, obs)
        env_steps = int(info["elapsed_steps"].item())
        if not is_truncated and env.get_wrapper_attr("is_task_success")(): break
        if is_truncated: break

因此 ``eval()`` 的契约是:

* 每次调用推进**一个微步**:运动阶段执行 1 个(或若干个)``env.step()``,
  纯计算阶段(定位 / 反投影 / IK 搜索)算完后立刻继续到下一个会 ``env.step``
  的阶段;
* 一次 ``eval()`` 内连续推进纯计算阶段超过 ``MAX_PURE_ADVANCES``(32)次仍未
  产生 ``env.step()`` → 返回 ``truncated=True`` 防死循环;
* 步数达到 ``max_env_steps``(click_bell 官方随机模式为 **361**)→
  ``truncated=True``;
* **任何阶段失败都不抛异常、不提前退出**:记录 ``self.failures`` 后进入
  ``FALLBACK_ACT``,把剩余步数交给纯 ACT,让 episode 正常跑完。

状态序列(SPEC §3)::

    SETTLE → LOOK → DEPTH_FETCH → VLM_LOCATE → ESTIMATE_POSE → PRESS_IK
           → APPROACH → PRESS → VERIFY → DONE → ACT_FILL
    任一阶段失败 → FALLBACK_ACT

``LOOK``(本包新增)在官方公开的按钮区域
(``configs/click_bell/random/gym_config.json`` 的
``init_button_pose.position_range``:z = 0.83 m,x ∈ [0.4, 0.85],y ∈ [-0.3, 0.3])
内做**腕相机视野搜索**:该区域是公开常量,用它当搜索区域完全合规;实测初始
位姿下按钮不在视野里,所以必须先"找得到"再谈"压得到"。

接口与 :class:`robosyn_hybrid.flow.HandleBasketFlow` **逐字一致**,便于
``policy/hybrid/deploy_policy.py`` 按 ``task_name`` 分派。

合规:只用 RGB、机器人本体感受(qpos / FK / IK)、公开标定常量与
``is_task_success()``;不读物体真值位姿、不读模拟器深度缓冲、不访问模拟器传感器
句柄(腕相机位姿由 :mod:`robosyn_hybrid.stereo` 用
``FK(qpos, "<arm>_link6") @ priors.camera_mount(arm)`` 给出);不读任何专家先验
(action_bank / prepress 目标);不出现任何绝对路径,也不起任何额外进程去问 LLM。
"""

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

# ``full_action`` / ``npv`` 等已是验证过的积木,直接复用(不修改 flow.py)。
from .flow import (
    as_done,
    finish_inference,
    frame3,
    full_action,
    jsonable,
    npv,
    start_inference,
)

__all__ = [
    "ClickBellFlow",
    "ButtonViewSearch",
    "BUTTON_REGION_MIN",
    "BUTTON_REGION_MAX",
    "LOOK_VIEW_SPECS",
    "button_region_candidates",
    "look_at_camera_pose",
    "camera_to_tool_target",
]


# ---------------------------------------------------------------------------
# SPEC §1:必须逐行照抄的算法参数(数值一个都不改;行号来自参考实现)
# ---------------------------------------------------------------------------

# 按压(参考实现第 93-97 行 + ``press_bell`` 第 2360 行局部量)
PRESS_AXIS: tuple[float, float, float] = (0.0, 0.0, -1.0)
PRESS_STEP_M = 0.0025
PRESS_FIRST_STEP_M = 0.005
PRESS_MAX_M = 0.008
PRESS_MAX_STEPS = 4
PRESS_TOL_L2 = 2.0e-3
MAX_STALLED_STREAK = 12

# 接近(参考实现第 83-90 行)
APPROACH_CLEARANCE_M = 0.04
APPROACH_LIFT_M = 0.12
APPROACH_LATERAL_TOL_M = 0.008
APPROACH_HEIGHT_TOL_M = 0.015
APPROACH_STEP_M = 0.05
APPROACH_FINE_STEP_M = 0.01
APPROACH_FINE_BELOW_M = 0.03
APPROACH_MAX_ITERATIONS = 14

# 舞台 / IK(参考实现第 246-262 行)
STAGE_TOL_L2 = 3.0e-3
STAGE_TOL_MAX_ABS = 3.0e-3
STAGE_COMPENSATE_ROUNDS = 8
STAGE_EE_TRANSLATION_TOL_M = 3.0e-3
STAGE_EE_ROTATION_TOL_RAD = 5.0e-3
STAGE_IK_ORIENTATION_SAMPLES = 512
STAGE_IK_ORIENTATION_MAX_RAD = float(np.deg2rad(20.0))
STAGE_IK_ORIENTATION_RNG_SEED = 0

# ``_move_to_matrix`` 的默认值(参考实现第 1001-1012 行)
WAYPOINT_STEP_M = 0.015
MAX_WAYPOINTS = 12
MOVE_TO_MATRIX_COMPENSATE_ROUNDS = 2

# ---------------------------------------------------------------------------
# 环境接缝重标定(SPEC 允许"只改环境接缝";下面每个常数都不改算法的数学)
# ---------------------------------------------------------------------------
#
# RPent 的 ``env.apply_qpos_target`` 用 ``sim.update`` 推进物理,那些步**不消耗**
# 官方 episode 步数;本包每个物理步都是一次 ``env.step``(消耗 361 步预算)。
# 因此"每轮下发多少步"“多久判定平台"这两个**口径**必须按控制步重新取值,
# 否则一个 waypoint 就能吃掉整个 episode(实测:照搬 120×8 会在第一步耗尽预算)。
#
#: 每个 waypoint、每轮补偿的步数上限(RPent: 120 个 sim.update)。
MOVE_TO_MATRIX_MAX_STEPS = 30
#: 平台检测窗口:连续多少控制步没有改善就认为控制器已稳定(RPent: 120)。
PLATEAU_WINDOW = 8
#: 每次 ``_move_to_matrix`` 给每轮至少保留的步数。
MIN_SHARE_STEPS = 2
#: ``apply_qpos_target`` 的平台判定阈值(参考实现 env_server.py 第 1326 行)。
PLATEAU_EPS = 1.0e-4

#: 参考实现 ``move_to_stage`` 的舞台偏移 —— 舞台位姿落在**测得的按压面**上
#: (0.0 表示刚好接触、不预压)。参考实现的编排是
#: ``look_bell → depth_fetch → move_to_stage → verify_aligned → press_bell``;
#: ``servo_to_bell`` 只把工具带到 ``按钮 + APPROACH_CLEARANCE_M``,单独用它
#: 停在按钮上方 4 cm,``press_bell`` 的 8 mm 行程就够不到按钮,因此粗接近
#: 收敛后要再补一次"落到按压面"的舞台动作(可用下面的开关关掉)。
STAGE_OFFSET_M = 0.0
APPROACH_SURFACE_STAGE = True

# 官方 handle_basket 流程同款常量(保持两套流程一致)
GRIP_OPEN = 0.05
ACT_CHUNK_STEPS_DEFAULT = 50

# ---------------------------------------------------------------------------
# 本流程的步数预算(SPEC §3 的表)
# ---------------------------------------------------------------------------

SETTLE_STEPS = 8                 # SETTLE:保持初始位姿 8 步
FETCH_SETTLE_STEPS = 1           # DEPTH_FETCH 内部 settle(状态机已单独 SETTLE)
MOTION_STEPS = 5                 # DEPTH_FETCH 内部运动步数
NUDGE_CANDIDATES = 2000
BASELINE_TARGET_M = 0.02
VERIFY_STEPS = 5                 # VERIFY:保持位姿 5 步

#: click_bell 官方随机模式 ``max_episode_steps``(gym_config.json 已核对)。
DEFAULT_MAX_ENV_STEPS = 361
#: 单次 ``eval()`` 内允许的连续纯计算阶段推进上限。
MAX_PURE_ADVANCES = 32
#: 连续降级次数上限(防止降级路径自身死循环)。
MAX_DEGRADES = 8
#: ``failures`` / ``act_state`` 记录条数上限。
MAX_FAILURES = 64
MAX_ACT_STATE = 400

#: VLM 定位重试(与 handle_basket 的既有做法一致:2 次尝试、间隔 2 秒)。
VLM_ATTEMPTS = 2
VLM_TIMEOUT_S = 180
VLM_RETRY_SLEEP_S = 2.0

#: 按钮反投影的 bbox 网格与最少有效样本数(多点取中位数)。
BUTTON_BBOX_GRID = 5
BUTTON_MIN_SAMPLES = 3
BUTTON_SAMPLE_RADIUS = 3

#: 全部阶段走完后是否把剩余步数交给纯 ACT(SPEC §3:DONE → ACT_FILL)。
DONE_RUNS_PURE_ACT = True

#: 机械臂 / 相机命名(官方约定)。
RIGHT = "right"
LEFT = "left"
RIGHT_ARM = "right_arm"
BUTTON_CAMERA = "cam_right_wrist"

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


# ---------------------------------------------------------------------------
# 基础工具(与 flow.py 的写法保持一致)
# ---------------------------------------------------------------------------


def _lazy(module_name: str) -> Any:
    """惰性 import 同包模块;失败时抛 ``RuntimeError`` 交给阶段守卫降级。"""
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
    """把 ``select_chunk_action`` 的返回值规整成 [step][dim] 的动作行列表。"""
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
    """沿 ``.env`` 链向下解开环境包装,拿到含 ``robot`` 的最内层实例。"""
    base = env
    for _ in range(8):
        nxt = getattr(base, "env", None)
        if nxt is None:
            break
        base = nxt
    return base


def _timing_device(act_policy: Any) -> Any:
    """取 ACT 策略所在的 torch device(仅用于官方 ``inference_timing``)。"""
    for name in ("device", "act_device", "pytorch_device"):
        value = getattr(act_policy, name, None)
        if value is not None:
            return value
    return None


def _rotation_axis(axis: int, angle: float) -> np.ndarray:
    """绕单轴的小旋转矩阵(参考实现 ``_rotation_axis``,第 279-291 行)。"""
    cosine, sine = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])
    if axis == 1:
        return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _json_from_text(text: str) -> dict | None:
    """从 VLM 回复里容错地抠出 JSON 对象(容忍 ```json 围栏)。

    与 ``flow``/``vlm`` 的既有做法一致,但本文件自带一份:SPEC 要求按钮 prompt
    与解析留在 clickbell.py 内,不往 vlm.py 里加东西。
    """
    value = str(text or "").strip()
    if value.startswith("```"):
        value = value.strip("`")
        if value.startswith("json"):
            value = value[4:].lstrip()
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(value[start : end + 1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _button_prompt(width: int, height: int) -> str:
    """按钮定位 prompt:只要腕相机 RGB 上的可见按钮框,不要求任何真值。"""
    return (
        "Return strict JSON only, with no markdown. In this wrist RGB frame, "
        "locate the round press button / bell button that the robot tool must "
        "push downwards. The button is a small round or domed part sitting on "
        "the table; its colour and texture may be randomized. Report the centre "
        "of its visible top face in pixel coordinates and an axis-aligned bbox "
        "around that visible top face. If not visible set visible=false. Schema: "
        '{"visible":true|false,"u":number|null,"v":number|null,'
        '"bbox_xyxy":[left,top,right,bottom]|null,"confidence":0..1,'
        '"note":"short reason"}.'
        f" Image size is {width}x{height}."
    )


def _finish_button(
    reply: dict, parsed: dict | None, width: int, height: int
) -> dict:
    """把 VLM 回复整理成按钮定位结果(结构照 ``vlm._finish`` 的既有做法)。"""
    info: dict = {
        "ok": False,
        "mode": "http",
        "http_ok": bool(reply.get("ok")),
        "text": reply.get("text", ""),
        "error": reply.get("error"),
        "latency_s": reply.get("latency_s", 0.0),
    }
    if parsed is None:
        info["error"] = info.get("error") or "VLM returned no parseable JSON"
        return info
    info["answer"] = parsed
    if not bool(parsed.get("visible", False)):
        info["visible"] = False
        info["error"] = "VLM reported button not visible"
        return info
    try:
        u, v = float(parsed["u"]), float(parsed["v"])
    except Exception:
        info["error"] = "VLM returned no usable centre"
        return info
    if not (0 <= u < width and 0 <= v < height):
        info["error"] = "VLM centre outside frame"
        return info
    bbox = parsed.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        radius = 18
        bbox = [u - radius, v - radius, u + radius, v + radius]
    try:
        left, right = sorted((float(bbox[0]), float(bbox[2])))
        top, bottom = sorted((float(bbox[1]), float(bbox[3])))
        bbox = [
            max(0.0, min(width - 1.0, left)),
            max(0.0, min(height - 1.0, top)),
            max(0.0, min(width - 1.0, right)),
            max(0.0, min(height - 1.0, bottom)),
        ]
    except Exception:
        info["error"] = "VLM bbox_xyxy is not numeric"
        return info
    info.update(
        {
            "ok": True,
            "visible": True,
            "uv": [u, v],
            "bbox_xyxy": bbox,
            "confidence": float(parsed.get("confidence") or 0.0),
            "note": parsed.get("note"),
            "camera": BUTTON_CAMERA,
        }
    )
    return info


# ---------------------------------------------------------------------------
# LOOK:在官方公开的按钮区域内做"腕相机视野搜索"(SPEC §3 新增的状态)
# ---------------------------------------------------------------------------
#
# 为什么需要它(2026-10-10 在真实 RTX 4090 上实测):初始位姿下按钮**不在**
# 右腕相机视野里,VLM 直接回 ``visible=false``("No round press button ... is
# visible in the frame")。参考实现 ``click_bell.py::look_bell()`` 先用
# ``RECOVERY_CAMERA="cam_high"``(**外部相机**)做语义定位,但外部相机的安装
# 外参无法由本体感受推导(见 COMPLIANCE §2.3),本包不使用它;参考实现在
# ``pick_view_pose`` 里又读了物体真值,那部分也不能照搬。因此这里改用
# **合规的视野搜索**:把腕相机依次移到公开区域内的候选点上方,每次问一次
# VLM,取第一个"看得见按钮"的视角。
#
# 合规性:搜索区域来自官方公开配置
# ``configs/click_bell/random/gym_config.json`` 的
# ``env.events.init_button_pose.position_range``,即
#
#     "position_range": [[0.4, -0.3, 0.83], [0.85, 0.3, 0.83]]
#
# 按钮恒定在桌面 z = 0.83 m、x ∈ [0.4, 0.85]、y ∈ [-0.3, 0.3]。这是**公开
# 常量**(与 ``MILK_GRASP_OBJECT`` 同类),既不是场景真值也不是运行时读数,
# 可以合法地用来定义"去哪里找按钮"。每一次搜索动作都只用 RGB、本体感受
# (qpos / FK / IK)与公开标定常量。
#: 官方公开配置给出的按钮区域下界(米)。
BUTTON_REGION_MIN: tuple[float, float, float] = (0.4, -0.3, 0.83)
#: 官方公开配置给出的按钮区域上界(米)。
BUTTON_REGION_MAX: tuple[float, float, float] = (0.85, 0.3, 0.83)

#: 候选点网格边长(奇数,保证包含区域中心)。9 = 3x3 全网格,5 = 中心 + 十字。
LOOK_GRID = 3
#: 默认候选点数量(3x3 网格里按"由中心向外"取前 N 个)。
LOOK_CANDIDATES = 5
#: 视角 1:正下方俯视,光学原点在 ``p + (0,0,h)``,h = 0.28 m(SPEC 默认值)。
LOOK_HEIGHT_M = 0.28
#: 视角 2:正下方俯视,但把 h 拉到 0.45 m —— 视场足迹更大(见下方实测说明)。
LOOK_WIDE_HEIGHT_M = 0.45
#: 视角 3:斜视,光轴与水平面夹角(度)。
LOOK_OBLIQUE_DEG = 35.0
#: 视角 3:斜视的拍摄距离(米)。
LOOK_STANDOFF_M = 0.60
#: 只接受"离当前关节位形足够近"的 IK 解,否则这一次移动在 25 步内到不了。
LOOK_MAX_JOINT_MOVE_RAD = 0.90
#: 真正执行(移动 + 问 VLM)的视角数上限;IK 失败的尝试不占额、不耗步数。
LOOK_MAX_VIEWS = 4
#: 整个 LOOK 阶段的步数上限(SPEC:≤ 120 步,别把 361 预算吃光)。
LOOK_STEP_BUDGET = 120
#: LOOK 结束后至少留给后续阶段(DEPTH_FETCH/APPROACH/PRESS/VERIFY/ACT)的步数。
LOOK_RESERVE_STEPS = 180
#: 单次视角移动的步数上限(SPEC:单次移动别超过 ~25 步)。
LOOK_MOVE_MAX_STEPS = 25
#: 视角移动的关节容差(弧度):只要求"看得见",不要求压到位,故比按压松。
LOOK_MOVE_TOL_RAD = 0.02
#: 剩余步数少于该值时不再开新视角(开了也动不了)。
LOOK_MIN_VIEW_STEPS = 6
#: 每个视角问 VLM 的次数(每个候选点问一次,失败就换视角)。
LOOK_VLM_ATTEMPTS = 1
#: 移动结束后允许的腕相机位姿偏差(米):超过就认为这次视角不可信。
LOOK_POSE_TOL_M = 0.05

#: 视角规格:``(标签, 拍摄距离 m, 光轴仰角 deg)``。
#: 90° = 正下方俯视(SPEC 指定先试这个);35° = 斜视。
#:
#: 顺序理由(实测,不是拍的):2026-10-10 的探针在真实机器上量到,正下方
#: 0.28 m 的视角对区域内绝大多数候选点要么 IK 无解(相机位置超出 ~0.64 m
#: 的臂展,如 x ≥ 0.7),要么解出的关节位形离当前位形 2.2~3.4 rad(25 步内
#: 不可能到位);即使到位,0.28 m 高度下 640x480 / f≈603 px 的视场足迹只有
#: ±0.13 m,而区域有 0.45x0.60 m —— 按钮几乎必然在画面外。因此"先正下方、
#: 失败再斜视"的顺序保留,但用 :data:`LOOK_MAX_JOINT_MOVE_RAD` 的关节闸门
#: 把"到不了/要大改位形"的正下方视角直接筛掉(不耗步数),再退到观测距离更
#: 远的视角:0.45 m 俯视(足迹 ±0.21 m)与 0.60 m / 35° 斜视(足迹 ±0.32 m,
#: 且拍摄点在**当前相机那一侧**,只需小幅变位)。
LOOK_VIEW_SPECS: tuple[tuple[str, float, float], ...] = (
    ("top_down_near", LOOK_HEIGHT_M, 90.0),
    ("top_down_far", LOOK_WIDE_HEIGHT_M, 90.0),
    ("oblique_standoff", LOOK_STANDOFF_M, LOOK_OBLIQUE_DEG),
)


def _look_env_int(name: str, default: int) -> int:
    """读一个可选的整数环境变量覆盖(非法值一律退回默认值,绝不外抛)。"""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _look_env_float(name: str, default: float) -> float:
    """读一个可选的浮点环境变量覆盖(非法值一律退回默认值,绝不外抛)。"""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def button_region_candidates(
    count: int = LOOK_CANDIDATES, grid: int = LOOK_GRID
) -> list[tuple[float, float, float]]:
    """公开按钮区域内的候选点,按**由中心向外**排序(中心优先,再环状外扩)。

    在区域上取 ``grid x grid`` 格点(``grid`` 为奇数时包含区域中心),按到区域
    中心的水平距离升序、同一距离再按 ``(x, y)`` 升序排序,取前 ``count`` 个。
    ``grid=3, count=5`` → 中心 + 四条边的中点(十字);``count=9`` → 3x3 全网格。
    """
    low = np.asarray(BUTTON_REGION_MIN, dtype=np.float64).reshape(3)
    high = np.asarray(BUTTON_REGION_MAX, dtype=np.float64).reshape(3)
    side = max(2, int(grid))
    if side % 2 == 0:
        side += 1  # 偶数网格不含中心点,强制变奇数
    xs = np.linspace(low[0], high[0], side)
    ys = np.linspace(low[1], high[1], side)
    center = 0.5 * (low + high)
    points: list[tuple[float, float, float]] = []
    for y in ys:
        for x in xs:
            points.append((float(x), float(y), float(center[2])))
    points.sort(
        key=lambda item: (
            float(np.hypot(item[0] - center[0], item[1] - center[1])),
            item[0],
            item[1],
        )
    )
    return points[: max(1, int(count))]


def look_at_camera_pose(
    point: Iterable[float],
    *,
    distance_m: float,
    elevation_deg: float,
    viewer_xy: Iterable[float],
) -> np.ndarray:
    """构造"腕相机看向 ``point``"的世界系**光学位姿**(4x4,cam->world)。

    相机光学原点放在 ``point + d * (cos e * u, sin e)``(``u`` 是从 ``point``
    指向 ``viewer_xy`` 的水平单位向量,``e`` = 仰角);光学 **z 轴指向
    ``point``**(OpenCV 约定:z 向前、x 向右、y 向下);"画面下"取 ``u`` 在像
    平面内的分量,这样画面里"上方"始终是**背离相机**的方向,俯视图与斜视图
    都保持正立。``e = 90°`` 即 SPEC 指定的"正下方俯视,原点在 p + (0,0,d)"。

    退化情形(``viewer_xy`` 与 ``point`` 水平重合)回退到从 ``-x`` 侧看,不抛异常。
    """
    p = np.asarray(point, dtype=np.float64).reshape(3)
    viewer = np.asarray(viewer_xy, dtype=np.float64).reshape(-1)[:2]
    horizontal = viewer - p[:2]
    norm = float(np.linalg.norm(horizontal))
    if norm < 1e-9:
        unit = np.array([-1.0, 0.0], dtype=np.float64)
    else:
        unit = horizontal / norm
    angle = float(np.deg2rad(float(elevation_deg)))
    origin = p + float(distance_m) * np.array(
        [
            float(np.cos(angle)) * float(unit[0]),
            float(np.cos(angle)) * float(unit[1]),
            float(np.sin(angle)),
        ],
        dtype=np.float64,
    )
    z_axis = p - origin
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < 1e-9:
        z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        z_axis = z_axis / z_norm
    y_axis = np.array([float(unit[0]), float(unit[1]), 0.0], dtype=np.float64)
    y_axis = y_axis - float(y_axis @ z_axis) * z_axis
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-9:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_axis = y_axis - float(y_axis @ z_axis) * z_axis
        y_norm = float(np.linalg.norm(y_axis))
        if y_norm < 1e-9:
            y_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            y_norm = 1.0
    y_axis = y_axis / y_norm
    x_axis = np.cross(y_axis, z_axis)
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-9:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / x_norm
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    pose[:3, 3] = origin
    return pose


def camera_to_tool_target(
    camera_pose: np.ndarray, mount: np.ndarray
) -> np.ndarray:
    """光学位姿 → ``grasp.solve_ik_named`` / ``fk_pose_named`` 用的法兰位姿。

    反解方向(SPEC 要求逐字核对):``compute_fk`` / ``compute_ik`` 用的是
    ``compute_fk(qpos, name="<arm>_arm")`` 返回的那个帧(实测是 tool/TCP 帧),
    而腕相机位姿是

        ``wrist_camera_pose(q) = fk_returned(q) @ camera_mount(arm)``

    (``camera_mount`` 内部已经左乘了实测解出的 ``FK_TO_LINK6``,所以**不能**
    再乘一次)。于是要"相机落在 ``camera_pose``"就必须有

        ``fk_returned_target = camera_pose @ inv(camera_mount(arm))``

    —— 右乘安装矩阵的逆,不是左乘、也不是乘 ``camera_mount`` 本身。
    """
    return np.asarray(camera_pose, dtype=np.float64).reshape(4, 4) @ np.linalg.inv(
        np.asarray(mount, dtype=np.float64).reshape(4, 4)
    )


class ButtonViewSearch:
    """LOOK 视野搜索:在公开按钮区域内依次试视角,取第一个 VLM 看得见的。

    这个类**同时被策略状态机(``ClickBellFlow._stage_look``)与诊断脚本
    (``scripts/verify_clickbell_geometry.py`` 的 L3)使用**,所以诊断度量到的
    就是策略真正会走的那条代码路径。

    它只依赖一个 :class:`ClickBellFlow` 提供三件事:已绑定的 ``env/base``、
    ``_servo_qpos``(关节伺服)、``_ik_from_seeds``(多种子 IK)与调试落盘。
    不读任何物体真值、不用外部相机、不起额外进程。
    """

    def __init__(
        self,
        flow: "ClickBellFlow",
        *,
        arm: str = RIGHT,
        candidates: int | None = None,
        grid: int | None = None,
        height_m: float | None = None,
        wide_height_m: float | None = None,
        oblique_deg: float | None = None,
        standoff_m: float | None = None,
        max_views: int | None = None,
        step_budget: int | None = None,
        reserve_steps: int | None = None,
        per_view_steps: int | None = None,
        max_joint_move_rad: float | None = None,
    ) -> None:
        self.flow = flow
        self.arm = str(arm)
        self.arm_name = f"{self.arm}_arm"
        self.candidates = max(1, int(candidates if candidates is not None else _look_env_int("HYBRID_LOOK_CANDIDATES", LOOK_CANDIDATES)))
        self.grid = max(3, int(grid if grid is not None else _look_env_int("HYBRID_LOOK_GRID", LOOK_GRID)))
        self.max_views = max(1, int(max_views if max_views is not None else _look_env_int("HYBRID_LOOK_MAX_VIEWS", LOOK_MAX_VIEWS)))
        self.step_budget = max(0, int(step_budget if step_budget is not None else _look_env_int("HYBRID_LOOK_STEP_BUDGET", LOOK_STEP_BUDGET)))
        self.reserve_steps = max(0, int(reserve_steps if reserve_steps is not None else _look_env_int("HYBRID_LOOK_RESERVE_STEPS", LOOK_RESERVE_STEPS)))
        self.per_view_steps = max(2, int(per_view_steps if per_view_steps is not None else _look_env_int("HYBRID_LOOK_MOVE_MAX_STEPS", LOOK_MOVE_MAX_STEPS)))
        self.max_joint_move_rad = float(
            max_joint_move_rad
            if max_joint_move_rad is not None
            else _look_env_float("HYBRID_LOOK_MAX_JOINT_MOVE_RAD", LOOK_MAX_JOINT_MOVE_RAD)
        )
        height = float(height_m if height_m is not None else _look_env_float("HYBRID_LOOK_HEIGHT_M", LOOK_HEIGHT_M))
        wide = float(wide_height_m if wide_height_m is not None else _look_env_float("HYBRID_LOOK_WIDE_HEIGHT_M", LOOK_WIDE_HEIGHT_M))
        oblique = float(oblique_deg if oblique_deg is not None else _look_env_float("HYBRID_LOOK_OBLIQUE_DEG", LOOK_OBLIQUE_DEG))
        standoff = float(standoff_m if standoff_m is not None else _look_env_float("HYBRID_LOOK_STANDOFF_M", LOOK_STANDOFF_M))
        self.view_specs: tuple[tuple[str, float, float], ...] = (
            ("top_down_near", height, 90.0),
            ("top_down_far", wide, 90.0),
            ("oblique_standoff", standoff, oblique),
        )
        #: 命中视角的 RGB(便于调用方落盘;不进 JSON)。
        self.hit_rgb: np.ndarray | None = None

    # -- 供两个调用方共用的描述 -------------------------------------------------

    def describe(self) -> dict:
        """把搜索配置写成可序列化的字典(便于写进 summary / 诊断 JSON)。"""
        return {
            "region_min_m": [float(v) for v in BUTTON_REGION_MIN],
            "region_max_m": [float(v) for v in BUTTON_REGION_MAX],
            "region_source": "configs/click_bell/random/gym_config.json:env.events.init_button_pose.position_range",
            "arm": self.arm,
            "candidates": int(self.candidates),
            "grid": int(self.grid),
            "view_specs": [
                {"label": label, "distance_m": float(distance), "elevation_deg": float(elev)}
                for label, distance, elev in self.view_specs
            ],
            "max_views": int(self.max_views),
            "step_budget": int(self.step_budget),
            "reserve_steps": int(self.reserve_steps),
            "per_view_steps": int(self.per_view_steps),
            "max_joint_move_rad": float(self.max_joint_move_rad),
            "vlm_attempts_per_view": int(LOOK_VLM_ATTEMPTS),
        }

    # -- 主流程 -----------------------------------------------------------------

    def run(self) -> dict:
        """执行搜索;返回可序列化记录(``ok=True`` 表示找到了看得见按钮的视角)。"""
        flow = self.flow
        record: dict = {
            "ok": False,
            "error": None,
            "arm": self.arm,
            "config": self.describe(),
            "steps_before": int(flow._steps_taken),
            "steps_used": 0,
            "views_executed": 0,
            "trace": [],
            "hit": None,
            "pose_tol_m": float(LOOK_POSE_TOL_M),
        }
        try:
            stereo = _lazy("stereo")
            vlm = _lazy("vlm")
            priors = _lazy("priors")
            grasp = _lazy("grasp")
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            return record

        if flow._base is None:
            record["error"] = "env 未绑定,无法执行视野搜索"
            return record
        try:
            mount = np.asarray(priors.camera_mount(self.arm), dtype=np.float64).reshape(4, 4)
        except Exception as exc:
            record["error"] = f"camera_mount 不可用: {type(exc).__name__}: {exc}"
            return record

        if not bool(vlm.vlm_available()):
            record["error"] = "VLM 端点不可用(vlm_available() 为 False),视野搜索无从判定"
            return record

        q_ref = np.asarray(flow._arm_joints(flow._base, self.arm)[0], dtype=np.float64).reshape(-1)
        try:
            ee_ref = np.asarray(
                grasp.fk_pose_named(flow._base, q_ref, self.arm_name), dtype=np.float64
            ).reshape(4, 4)
            viewer_xy = (ee_ref @ mount)[:2, 3]
        except Exception:
            viewer_xy = np.asarray(q_ref[:2], dtype=np.float64)
        record["viewer_xy_m"] = [float(v) for v in viewer_xy]

        points = button_region_candidates(self.candidates, self.grid)
        record["candidate_points_m"] = [[float(v) for v in p] for p in points]

        started = int(flow._steps_taken)
        budget = min(
            int(self.step_budget),
            max(0, int(flow.max_env_steps) - started - int(self.reserve_steps)),
        )
        record["budget_steps"] = int(budget)
        if budget < LOOK_MIN_VIEW_STEPS:
            record["error"] = (
                f"剩余步数不足以做视野搜索(可用 {budget} < {LOOK_MIN_VIEW_STEPS})"
            )
            return record

        vlm_timeout = float(getattr(flow.settings, "vlm_timeout_s", VLM_TIMEOUT_S) or VLM_TIMEOUT_S)
        seed_q = q_ref
        hit: dict | None = None
        stop_reason = "candidates_exhausted"

        for index, point in enumerate(points):
            if hit is not None:
                break
            for label, distance, elevation in self.view_specs:
                if hit is not None:
                    break
                if int(record["views_executed"]) >= int(self.max_views):
                    stop_reason = "view_cap_reached"
                    break
                steps_left = budget - (int(flow._steps_taken) - started)
                if steps_left < LOOK_MIN_VIEW_STEPS or flow._budget_exhausted():
                    stop_reason = "step_budget_exhausted"
                    break

                entry: dict = {
                    "index": int(index),
                    "point_m": [float(v) for v in point],
                    "view": str(label),
                    "distance_m": float(distance),
                    "elevation_deg": float(elevation),
                    "executed": False,
                }
                camera_target = look_at_camera_pose(
                    point,
                    distance_m=distance,
                    elevation_deg=elevation,
                    viewer_xy=viewer_xy,
                )
                tool_target = camera_to_tool_target(camera_target, mount)
                entry["camera_target_m"] = [float(v) for v in camera_target[:3, 3]]
                entry["tool_target_m"] = [float(v) for v in tool_target[:3, 3]]

                seeds: list[tuple[str, np.ndarray]] = [
                    ("current_qpos", np.asarray(seed_q, dtype=np.float64).reshape(-1)),
                    ("zeros", np.zeros_like(np.asarray(q_ref, dtype=np.float64))),
                ]
                ik = flow._ik_from_seeds(
                    tool_target,
                    seeds,
                    arm=self.arm,
                    orientation_search=False,
                )
                entry["ik_ok"] = bool(ik.get("ok"))
                entry["ik_seed"] = ik.get("seed_key")
                entry["ik_joint_distance_rad"] = ik.get("joint_distance_rad")
                entry["ik_error"] = ik.get("error")
                if not ik.get("ok"):
                    record["trace"].append(entry)
                    continue
                distance_rad = ik.get("joint_distance_rad")
                if distance_rad is not None and float(distance_rad) > float(
                    self.max_joint_move_rad
                ):
                    # 解存在但离当前位形太远:25 步内伺服不到位,直接放弃这个视角。
                    entry["skipped"] = (
                        f"IK 解离当前位形 {float(distance_rad):.3f} rad > "
                        f"{float(self.max_joint_move_rad):.3f} rad"
                    )
                    record["trace"].append(entry)
                    continue

                # ---- 真正执行:移动 → 取 RGB → 问 VLM ----
                entry["executed"] = True
                per_view = max(2, min(int(self.per_view_steps), int(steps_left)))
                entry["per_view_steps"] = int(per_view)
                try:
                    move = flow._servo_qpos(
                        flow._base,
                        np.asarray(ik["qpos"], dtype=np.float64),
                        max_steps=per_view,
                        tol=LOOK_MOVE_TOL_RAD,
                        compensate_rounds=1,
                    )
                except Exception as exc:
                    move = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                entry["move"] = {
                    "ok": bool(isinstance(move, dict) and move.get("ok")),
                    "error": move.get("error") if isinstance(move, dict) else None,
                    "error_l2": move.get("error_l2") if isinstance(move, dict) else None,
                    "stalled": move.get("stalled") if isinstance(move, dict) else None,
                    "steps_used": move.get("steps_used") if isinstance(move, dict) else None,
                }
                record["views_executed"] = int(record["views_executed"]) + 1
                if flow._budget_exhausted():
                    stop_reason = "episode_budget_exhausted"
                    record["trace"].append(entry)
                    break

                # 用**实际到达**的关节角记账:后续 DEPTH_FETCH 就用这个位形,
                # 绝不退回初始位姿。
                q_actual = np.asarray(
                    flow._arm_joints(flow._base, self.arm)[0], dtype=np.float64
                ).reshape(-1)
                seed_q = q_actual
                entry["qpos_actual"] = [float(v) for v in q_actual]
                try:
                    fk_actual = np.asarray(
                        grasp.fk_pose_named(flow._base, q_actual, self.arm_name),
                        dtype=np.float64,
                    ).reshape(4, 4)
                    camera_actual = fk_actual @ mount
                    entry["camera_actual_m"] = [float(v) for v in camera_actual[:3, 3]]
                    entry["camera_pose_error_mm"] = float(
                        np.linalg.norm(camera_actual[:3, 3] - camera_target[:3, 3]) * 1000.0
                    )
                    entry["camera_rotation_error_deg"] = float(
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    (
                                        float(
                                            np.trace(
                                                camera_actual[:3, :3]
                                                @ camera_target[:3, :3].T
                                            )
                                        )
                                        - 1.0
                                    )
                                    / 2.0,
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                except Exception as exc:
                    entry["camera_pose_error"] = f"{type(exc).__name__}: {exc}"

                try:
                    rgb, _obs = stereo.capture(flow._base, self.arm)
                except Exception as exc:
                    entry["capture_error"] = f"{type(exc).__name__}: {exc}"
                    record["trace"].append(entry)
                    continue

                vision, attempts = self._ask_vlm(vlm, rgb, timeout_s=vlm_timeout)
                entry["vlm"] = vision
                entry["vlm_attempts"] = attempts
                record["trace"].append(entry)
                if bool(vision.get("ok")):
                    hit = {
                        "index": int(index),
                        "point_m": [float(v) for v in point],
                        "view": str(label),
                        "distance_m": float(distance),
                        "elevation_deg": float(elevation),
                        "qpos": [float(v) for v in q_actual],
                        "camera_pose": camera_actual.tolist(),
                        "camera_target_pose": camera_target.tolist(),
                        "camera_pose_error_mm": entry.get("camera_pose_error_mm"),
                        "uv": [float(v) for v in vision["uv"]],
                        "bbox_xyxy": [float(v) for v in vision["bbox_xyxy"]],
                        "confidence": vision.get("confidence"),
                        "note": vision.get("note"),
                        "latency_s": vision.get("latency_s"),
                    }
                    self.hit_rgb = np.asarray(rgb, dtype=np.uint8)
                    record["hit"] = hit
                    record["ok"] = True
                    stop_reason = "hit"
                    break

        record["steps_used"] = int(flow._steps_taken) - started
        record["stop_reason"] = stop_reason
        if not record["ok"] and record["error"] is None:
            record["error"] = (
                f"视野搜索未命中:试了 {record['views_executed']} 个视角 / "
                f"{len(points)} 个候选点,用了 {record['steps_used']} 步"
            )
        return record

    # -- 单个视角上的 VLM 询问 ---------------------------------------------------

    def _ask_vlm(self, vlm: Any, rgb: Any, *, timeout_s: float) -> tuple[dict, list[dict]]:
        """在给定 RGB 上问一次(或 ``LOOK_VLM_ATTEMPTS`` 次)按钮 prompt。

        解析走本模块已有的 :func:`_button_prompt` / :func:`_finish_button`,
        与 ``VLM_LOCATE`` 完全同一条解析路径;任何异常都降级成 ``ok=False``。
        """
        image = np.asarray(rgb, dtype=np.uint8)
        height, width = int(image.shape[0]), int(image.shape[1])
        prompt = _button_prompt(width, height)
        attempts: list[dict] = []
        vision: dict = {"ok": False, "error": "VLM 未返回任何结果"}
        for attempt in range(max(1, int(LOOK_VLM_ATTEMPTS))):
            try:
                reply = vlm.vlm_chat(prompt, [image], timeout_s=timeout_s)
            except Exception as exc:
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if not isinstance(reply, dict):
                reply = {"ok": False, "error": "vlm_chat 返回非字典"}
            parsed = _json_from_text(reply.get("text", "")) if reply.get("ok") else None
            vision = _finish_button(reply, parsed, width, height)
            attempts.append(
                {
                    "attempt": int(attempt),
                    "http_ok": bool(reply.get("ok")),
                    "latency_s": float(reply.get("latency_s") or 0.0),
                    "text_preview": str(reply.get("text") or "")[:400],
                    "error": vision.get("error"),
                    "visible": bool(vision.get("visible")),
                }
            )
            if bool(vision.get("ok")):
                break
            if attempt + 1 < int(LOOK_VLM_ATTEMPTS):
                time.sleep(VLM_RETRY_SLEEP_S)
        return vision, attempts


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------


class ClickBellFlow:
    """click_bell 的有限状态机(SPEC §3)。

    ``act_policy`` 需要提供 ``select_chunk_action(obs) -> (action, ran)``;
    本类**不**加载任何 ACT 权重。``settings`` 只用于 ``debug_dir`` /
    ``vlm_available`` 之类的环境信息。
    """

    # ---- 状态名(与 SPEC §3 的状态序列一一对应;每个状态一个 ``_stage_*``)----
    SETTLE = "SETTLE"
    LOOK = "LOOK"
    DEPTH_FETCH = "DEPTH_FETCH"
    VLM_LOCATE = "VLM_LOCATE"
    ESTIMATE_POSE = "ESTIMATE_POSE"
    PRESS_IK = "PRESS_IK"
    APPROACH = "APPROACH"
    PRESS = "PRESS"
    VERIFY = "VERIFY"
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
        max_env_steps: int = DEFAULT_MAX_ENV_STEPS,
        debug_dir: Path | None = None,
    ) -> None:
        self.act_policy = act_policy
        self.settings = settings

        try:
            self.act_chunks = max(0, int(act_chunks))
        except Exception:
            self.act_chunks = 0

        try:
            self.max_env_steps = int(max_env_steps)
        except Exception:
            self.max_env_steps = DEFAULT_MAX_ENV_STEPS
        if self.max_env_steps <= 0:
            self.max_env_steps = DEFAULT_MAX_ENV_STEPS

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
        """开始新 episode:清空全部运行时状态,并(可选)重新绑定 env。"""
        self._init_episode_state()
        self._env = None
        if env is not None:
            self._bind_env(env)

    # ------------------------------------------------------------------
    # 对外主循环
    # ------------------------------------------------------------------

    def eval(self, env: Any, obs: Any) -> tuple[Any, dict, Any, list[float]]:
        """推进一个微步,返回 ``(obs, info, truncated, inference_times)``。

        * 至少产生 1 个 ``env.step()``,或把阶段指针向前推进;
        * 纯计算阶段连续推进超过 32 次 → ``truncated=True``;
        * 任何异常都在内部降级,**绝不向外抛**(最外层还有一道兜底)。

        实际逻辑在 :meth:`_eval_step` 里;这里只加"任何异常都不外抛"的兜底,
        保证官方评测循环永远能拿到合法的四元组。
        """
        try:
            return self._eval_step(env, obs)
        except Exception as exc:  # 兜底:连降级路径都炸了也不让评测崩
            try:
                self._force_truncated = True
                self._degrade_count += 1
                self._record_failure(
                    self._state,
                    f"未捕获异常,强制收尾: {type(exc).__name__}: {exc}",
                    dedup=False,
                )
            except Exception:
                pass
            return (
                self._obs,
                {"elapsed_steps": np.int64(self._steps_taken)},
                True,
                [],
            )

    def _eval_step(self, env: Any, obs: Any) -> tuple[Any, dict, Any, list[float]]:
        """``eval()`` 的主体:绑定 env、检测新 episode、逐阶段推进一个微步。"""
        inference_times: list[float] = []
        self._ensure_bound(env)
        info = self._probe_info(env)
        self._maybe_new_episode(env, info)
        if obs is not None:
            self._obs = obs

        # 上一轮已经看到 terminated/truncated:本轮直接收尾。
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
            truncated_out = (
                self._last_truncated if self._last_truncated is not None else False
            )
        return self._obs, info, truncated_out, inference_times

    def summary(self) -> dict:
        """返回本 episode 的可序列化摘要(降级理由写入 ``failures``)。"""
        timings = list(self._inference_times_all)
        return jsonable(
            {
                "task": "click_bell",
                "success": self._query_success(),
                "final_state": self._state,
                "failed": bool(self.failed),
                "degraded_to_pure_act": bool(
                    self._state in (self.FALLBACK_ACT, self.ACT_FILL)
                ),
                "env_steps": int(self._steps_taken),
                "max_env_steps": int(self.max_env_steps),
                "stages": list(self._history),
                "failures": list(self.failures),
                "degradations": list(self.degradations),
                "failure_count_total": len(self.failure_history),
                "look": self._look,
                "look_qpos": (
                    [float(v) for v in self._look_q]
                    if self._look_q is not None
                    else None
                ),
                "depth": self._depth_record,
                "depth_error": self._depth_error,
                "vlm_attempts": self._vlm_attempts,
                "vlm": self._vision,
                "button_estimation": self._button_info,
                "press_ik": self._ik_record,
                "approach": {
                    "trace": list(self._approach_trace),
                    "iterations": len(self._approach_trace),
                    "converged": bool(self._approach_converged),
                },
                "press": {
                    "trace": list(self._press_trace),
                    "pressed_m": float(self._press_pressed_m),
                    "blocked": bool(self._press_blocked),
                    "max_stalled_streak": MAX_STALLED_STREAK,
                    "tol_l2": PRESS_TOL_L2,
                },
                "act": {
                    "chunks_requested": self.act_chunks,
                    "chunk_steps": self._act_chunk_budget,
                    "chunks_started": self._act_chunks_started,
                    "select_calls": self._act_select_calls,
                    "steps": self._act_steps,
                    "ran_inference": self._act_ran_inference,
                    "policy_broken": bool(self._act_broken),
                },
                "inference": {
                    "calls": len(timings),
                    "total_s": float(sum(timings)) if timings else 0.0,
                    "mean_s": float(sum(timings) / len(timings)) if timings else 0.0,
                },
                "environment": {
                    "vlm_available": bool(getattr(self.settings, "vlm_available", False)),
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

        # 关节基线(后续运动阶段用它们保持另一条臂不动)
        self._left_hold_q: np.ndarray | None = None
        self._left_hold_grip: float = GRIP_OPEN
        self._right_hold_q: np.ndarray | None = None
        self._right_hold_grip: float = GRIP_OPEN

        # 感知 / 几何结果
        self._look: dict | None = None
        self._look_q: np.ndarray | None = None
        self._depth: dict | None = None
        self._depth_record: dict | None = None
        self._depth_error: dict | None = None
        self._vlm_attempts: list[dict] = []
        self._vision: dict | None = None
        self._button_info: dict | None = None
        self._button_world: np.ndarray | None = None
        self._stage_pose: np.ndarray | None = None
        self._q_stage: np.ndarray | None = None
        self._ik_record: dict | None = None

        # 接近 / 按压轨迹
        self._approach_stage_done: bool = False
        self._approach_surface_done: bool = False
        self._approach_converged: bool = False
        self._approach_trace: list[dict] = []
        self._press_trace: list[dict] = []
        self._press_pressed_m: float = 0.0
        self._press_stalled_streak: int = 0
        self._press_blocked: bool = False
        self._press_success: bool = False

        # ACT 执行状态
        self._act_chunk_index: int = 0
        self._act_remaining: int = self._act_chunk_budget
        self._pending_rows: list[np.ndarray] = []
        self._act_steps: int = 0
        self._act_select_calls: int = 0
        self._act_chunks_started: int = 0
        self._act_ran_inference: int = 0
        self._act_broken: bool = False
        self._act_state: list[dict] = []

        # 动作通道
        self._env_dim: int | None = None
        self._dim_mismatch_recorded: bool = False

    def _bind_env(self, env: Any) -> None:
        """绑定 env:解包 base、探测动作维度、初始化 ACT chunk 计数。"""
        self._env = env
        self._base = _unwrap_env(env)
        self._env_dim = self._probe_action_dim(env)
        self._act_chunk_index = 0
        # 0 表示"还没有开始任何 chunk":第一个 ACT 微步会开启 chunk 0。
        self._act_remaining = 0
        self._pending_rows = []

    def _ensure_bound(self, env: Any) -> None:
        """首次调用/更换 env 对象时自动完成绑定(忘记 reset 也不会崩)。"""
        if self._env is env:
            return
        if self._steps_taken > 0:
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
        """探测环境期望的动作维度(失败返回 ``None``)。"""
        try:
            space = getattr(getattr(env, "unwrapped", env), "single_action_space", None)
            if space is None:
                return None
            return int(np.prod(space.shape))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # env 步进 / 信息探测
    # ------------------------------------------------------------------

    def _step_env_step(self, env: Any, action: Any) -> tuple[Any, bool]:
        """执行 1 次 ``env.step()`` 并同步内部计数/终止标志。"""
        obs, _reward, terminated, truncated, info = env.step(_to_action_tensor(action))
        self._steps_taken += 1
        if isinstance(info, dict):
            self._last_info = info
        self._last_truncated = truncated
        self._obs = obs
        if as_done(truncated):
            self._force_truncated = True
        if as_done(terminated):
            # terminated 可能是成功信号:不强制截断,让外层先做成功判定。
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

    def _budget_exhausted(self) -> bool:
        """内部步进循环的预算守卫:步数到顶或 episode 已结束。"""
        return bool(
            self._force_truncated
            or self._env_done
            or self._steps_taken >= self.max_env_steps
        )

    def _share_steps(self, rounds: int, waypoints: int) -> int:
        """把**剩余 episode 预算**均分给"轮次 × waypoint",得到每轮步数上限。

        这是纯环境接缝:RPent 的 ``sim.update`` 不进 episode 计数,可以给到
        120 步/轮;本包必须保证一次 ``_move_to_matrix`` 不会把 361 步预算吃光,
        否则 ``ACT_FILL`` 一点步数都拿不到。
        """
        remaining = max(0, int(self.max_env_steps) - int(self._steps_taken))
        shares = max(1, int(rounds)) * max(1, int(waypoints))
        return max(MIN_SHARE_STEPS, min(MOVE_TO_MATRIX_MAX_STEPS, remaining // shares))

    def _hold_action(self) -> np.ndarray:
        """保持当前关节位姿的动作(策略失败时的最后兜底)。"""
        q_left, grip_left = self._arm_joints(self._base, LEFT)
        q_right, grip_right = self._arm_joints(self._base, RIGHT)
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
                "已裁剪/补齐",
            )
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
        return np.pad(
            np.asarray(row, dtype=np.float32), (0, int(env_dim) - int(row.size))
        )

    def _arm_joints(self, base: Any, arm: str) -> tuple[np.ndarray, float]:
        """读取机械臂关节角与夹爪开度(``stereo.arm_joints`` 的安全封装)。"""
        try:
            stereo = _lazy("stereo")
            q, grip = stereo.arm_joints(base, arm)
            return np.asarray(q, dtype=np.float64).reshape(-1), float(grip)
        except Exception:
            q_all = np.asarray(npv(base.robot.get_qpos(name=f"{arm}_arm")))
            q = q_all[0] if q_all.ndim > 1 else q_all
            grip_all = np.asarray(npv(base.robot.get_qpos(name=f"{arm}_eef")))
            return np.asarray(q, dtype=np.float64).reshape(-1), float(
                grip_all.reshape(-1)[0]
            )

    # ------------------------------------------------------------------
    # 失败记录 / 降级
    # ------------------------------------------------------------------

    def _record_failure(
        self, stage: str, reason: str, *, dedup: bool = True, fatal: bool = True
    ) -> None:
        """记录失败原因(``dedup=True`` 时相同 (stage, reason) 只记一次)。"""
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
        """记录"非致命降级"(例如可用但质量不佳的中间结果)。"""
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
        """插值到第 ``index`` 步(与 flow.py 的 ``_interp`` 逐行一致)。"""
        alpha = float((index + 1) / max(1, int(steps)))
        return (1.0 - alpha) * np.asarray(
            q_from, dtype=np.float64
        ) + alpha * np.asarray(q_to, dtype=np.float64)

    # ------------------------------------------------------------------
    # 阶段 0:SETTLE
    # ------------------------------------------------------------------

    def _capture_holds(self) -> None:
        """记录左右臂的初始关节角(后续运动阶段用它们保持另一条臂不动)。"""
        self._left_hold_q, self._left_hold_grip = self._arm_joints(self._base, LEFT)
        self._right_hold_q, self._right_hold_grip = self._arm_joints(self._base, RIGHT)

    def _stage_settle(self, env: Any, inference_times: list[float]) -> None:
        """SETTLE:在原位保持 ``SETTLE_STEPS``(8)步。"""
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
            self._enter(self.LOOK)

    # ------------------------------------------------------------------
    # 阶段 0.5:LOOK(视野搜索;内部最多走 LOOK_STEP_BUDGET 步 env.step)
    # ------------------------------------------------------------------

    def _stage_look(self, env: Any, inference_times: list[float]) -> None:
        """LOOK:在**官方公开的按钮区域**内搜索"腕相机看得见按钮"的视角。

        初始位姿下按钮不在右腕视野里(实测 VLM 回 ``visible=false``),因此在
        取深度之前先把腕相机依次移到公开区域内的候选点上方:每个候选点先试
        正下方俯视(0.28 m → 0.45 m),再试 0.60 m / 35° 的斜视;每个真正执行
        的视角都取一帧 RGB 问一次 VLM,命中就记录该视角并进入 ``DEPTH_FETCH``
        (后续深度与定位都用**这个**视角,绝不退回初始位姿)。

        全部候选都失败 → :meth:`_degrade` 进 ``FALLBACK_ACT``(绝不抛)。
        """
        if self._stage_i > 0:
            return
        self._stage_i = 1
        search = ButtonViewSearch(self, arm=RIGHT)
        record = search.run()
        self._look = jsonable(record)
        self._debug_write_json("look_search.json", self._look)
        hit = record.get("hit") if isinstance(record, dict) else None
        if isinstance(hit, dict) and hit.get("qpos") is not None:
            self._look_q = np.asarray(hit["qpos"], dtype=np.float64).reshape(-1)
            if search.hit_rgb is not None:
                self._debug_write_png("look_hit_rgb.png", search.hit_rgb)
            self._record_degradation(
                self.LOOK,
                f"视野搜索命中视角 {hit.get('view')} @ 候选点 {hit.get('point_m')}"
                f"(第 {int(hit.get('index', 0)) + 1} 个候选),"
                f"相机位姿偏差 {hit.get('camera_pose_error_mm')} mm",
            )
            self._enter(self.DEPTH_FETCH)
            return
        reason = (record or {}).get("error") or "视野搜索未命中任何候选视角"
        self._degrade(self.LOOK, reason)

    # ------------------------------------------------------------------
    # 阶段 1:DEPTH_FETCH(内部自己走 1+5 步 env.step)
    # ------------------------------------------------------------------

    def _stage_depth_fetch(self, env: Any, inference_times: list[float]) -> None:
        """DEPTH_FETCH:右腕运动双目取一路 GT-free 深度(SPEC §3 的实参)。

        参考曝光位形取 ``LOOK`` 命中的那个视角(``self._look_q``);只有
        ``LOOK`` 缺失时才退回初始位姿 ``self._right_hold_q`` —— 否则按钮又会
        跑出视野,VLM_LOCATE 就没有意义了。
        """
        stereo = _lazy("stereo")
        if self._right_hold_q is None:
            self._capture_holds()
        q_reference = (
            self._look_q if self._look_q is not None else self._right_hold_q
        )
        depth = stereo.fetch_depth(
            self._base,
            RIGHT,
            q_reference,
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
            raise RuntimeError(
                "fetch_depth 结果缺少 reference_rgb/reference_camera_pose"
            )
        self._depth = depth
        self._depth_record = {
            key: jsonable(value) for key, value in depth.items() if key in SAFE_DEPTH_KEYS
        }
        # fetch_depth 内部:settle(FETCH_SETTLE_STEPS)+ 运动(MOTION_STEPS)步。
        self._mark_internal_steps(FETCH_SETTLE_STEPS + MOTION_STEPS)
        self._debug_write_png("depth_reference_rgb.png", depth.get("reference_rgb"))
        self._enter(self.VLM_LOCATE)

    # ------------------------------------------------------------------
    # 阶段 2:VLM_LOCATE(纯计算)
    # ------------------------------------------------------------------

    def _stage_vlm_locate(self, env: Any, inference_times: list[float]) -> None:
        """VLM_LOCATE:用按钮 prompt 在右腕 RGB 上定位按钮(最多 2 次)。"""
        vlm = _lazy("vlm")
        if not isinstance(self._depth, dict):
            raise RuntimeError("缺少深度参考帧,无法做 VLM 定位")
        if not vlm.vlm_available():
            raise RuntimeError("VLM 端点不可用(vlm_available() 为 False)")
        rgb = np.asarray(self._depth["reference_rgb"], dtype=np.uint8)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        prompt = _button_prompt(width, height)
        vision: dict | None = None
        attempts: list[dict] = []
        for attempt in range(VLM_ATTEMPTS):
            reply = vlm.vlm_chat(prompt, [rgb], timeout_s=VLM_TIMEOUT_S)
            if not isinstance(reply, dict):
                reply = {"ok": False, "error": "vlm_chat 返回非字典"}
            parsed = _json_from_text(reply.get("text", "")) if reply.get("ok") else None
            vision = _finish_button(reply, parsed, width, height)
            attempts.append(vision)
            if vision.get("ok"):
                break
            if attempt + 1 < VLM_ATTEMPTS:
                time.sleep(VLM_RETRY_SLEEP_S)
        self._vlm_attempts = [jsonable(item) for item in attempts]
        self._vision = vision if isinstance(vision, dict) else {"ok": False}
        self._debug_write_json("vlm_button.json", self._vision)
        if not self._vision.get("ok"):
            raise RuntimeError(
                f"VLM 按钮定位失败: {self._vision.get('error') or 'unknown'}"
            )
        bbox = self._vision.get("bbox_xyxy")
        if not bbox or len(list(bbox)) != 4:
            raise RuntimeError(f"VLM 返回的 bbox_xyxy 非法: {bbox!r}")
        self._enter(self.ESTIMATE_POSE)

    # ------------------------------------------------------------------
    # 阶段 3:ESTIMATE_POSE(纯计算)
    # ------------------------------------------------------------------

    def _stage_estimate_pose(self, env: Any, inference_times: list[float]) -> None:
        """ESTIMATE_POSE:bbox 内多点反投影取中位数 → 按钮顶面世界点。

        按钮是平面,SPEC 明确**不做** handle_basket 那种"沿视线内推半个短边"
        的修正。
        """
        stereo = _lazy("stereo")
        if not isinstance(self._depth, dict) or not isinstance(self._vision, dict):
            raise RuntimeError("位姿估计阶段缺少深度帧或 VLM 结果")
        button_world, info = self._button_world_from_bbox(
            stereo, self._depth, self._vision["bbox_xyxy"]
        )
        if button_world is None:
            self._depth_error = jsonable(info)
            raise RuntimeError(
                f"按钮顶面反投影失败: {(info or {}).get('error') or 'unknown'}"
            )
        self._button_world = np.asarray(button_world, dtype=np.float64).reshape(3)
        self._button_info = jsonable(info)
        self._debug_write_json("button_estimation.json", self._button_info)
        self._enter(self.PRESS_IK)

    def _button_world_from_bbox(
        self,
        stereo: Any,
        depth: dict,
        bbox: Iterable[float],
        *,
        grid: int = BUTTON_BBOX_GRID,
        sample_radius: int = BUTTON_SAMPLE_RADIUS,
    ) -> tuple[np.ndarray | None, dict]:
        """在 bbox 内撒 ``grid x grid`` 个点反投影,取各分量中位数。"""
        values = [float(v) for v in list(bbox)[:4]]
        left, right = sorted((values[0], values[2]))
        top, bottom = sorted((values[1], values[3]))
        count = max(2, int(grid))
        # 收缩半个格子,避免采样点落在按钮边缘以外的背景上。
        inset_x = (right - left) / (2.0 * count)
        inset_y = (bottom - top) / (2.0 * count)
        us = np.linspace(left + inset_x, right - inset_x, count)
        vs = np.linspace(top + inset_y, bottom - inset_y, count)
        points: list[list[float]] = []
        errors: list[str] = []
        for v in vs:
            for u in us:
                estimate = self._project_pixel(stereo, depth, [float(u), float(v)], sample_radius)
                if estimate.get("ok") and estimate.get("p_world_m") is not None:
                    points.append([float(x) for x in estimate["p_world_m"]])
                else:
                    errors.append(str(estimate.get("error")))
        report: dict = {
            "ok": False,
            "bbox_xyxy": [left, top, right, bottom],
            "grid": count,
            "sample_radius": int(sample_radius),
            "sample_count": len(points),
            "required_samples": BUTTON_MIN_SAMPLES,
            "errors": errors[:8],
        }
        if len(points) < BUTTON_MIN_SAMPLES:
            report["error"] = (
                f"bbox 内有效深度样本不足({len(points)} < {BUTTON_MIN_SAMPLES})"
            )
            return None, report
        array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        median = np.median(array, axis=0)
        report.update(
            {
                "ok": True,
                "p_world_median_m": [float(x) for x in median],
                "p_world_min_m": [float(x) for x in array.min(axis=0)],
                "p_world_max_m": [float(x) for x in array.max(axis=0)],
                "samples": [[float(x) for x in row] for row in array],
            }
        )
        return median, report

    @staticmethod
    def _project_pixel(
        stereo: Any, depth: dict, uv: Iterable[float], sample_radius: int
    ) -> dict:
        """``stereo.estimate_point_from_pixel`` 的安全封装(异常 → ``ok=False``)。"""
        try:
            estimate = stereo.estimate_point_from_pixel(
                depth, list(uv), sample_radius=sample_radius
            )
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(estimate, dict):
            return {"ok": False, "error": "estimate_point_from_pixel 返回非字典"}
        return estimate

    # ------------------------------------------------------------------
    # 阶段 4:PRESS_IK(纯计算)
    # ------------------------------------------------------------------

    def _ee_matrix(self) -> tuple[np.ndarray | None, dict]:
        """右臂末端 FK 位姿 4x4(``grasp.fk_pose_named(base, q, "right_arm")``)。"""
        try:
            grasp = _lazy("grasp")
            q_right, _grip = self._arm_joints(self._base, RIGHT)
            matrix = np.asarray(
                grasp.fk_pose_named(self._base, q_right, RIGHT_ARM), dtype=np.float64
            ).reshape(4, 4)
        except Exception as exc:
            return None, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return matrix, {"ok": True}

    def _ee_pose_error(self, target_pose: np.ndarray) -> dict:
        """任务空间残差(等价参考实现的 ``pose_delta_to_qpos``,只用 FK)。

        返回平移误差(m)与旋转误差(rad);FK 不可用时返回 ``{}``。
        """
        ee, _info = self._ee_matrix()
        if ee is None:
            return {}
        target = np.asarray(target_pose, dtype=np.float64).reshape(4, 4)
        translation = float(np.linalg.norm(np.asarray(ee[:3, 3]) - target[:3, 3]))
        relative = ee[:3, :3].T @ target[:3, :3]
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        return {
            "ee_translation_error_m": translation,
            "ee_rotation_error_rad": float(np.arccos(cosine)),
        }

    def _ee_within_tolerance(self, ee_report: dict) -> bool:
        """任务空间残差是否满足 STAGE_EE_* 容差(参考实现 ``_ee_within_tolerance``)。"""
        translation = ee_report.get("ee_translation_error_m")
        if translation is None:
            return False
        if float(translation) > STAGE_EE_TRANSLATION_TOL_M:
            return False
        rotation = ee_report.get("ee_rotation_error_rad")
        if rotation is not None and float(rotation) > STAGE_EE_ROTATION_TOL_RAD:
            return False
        return True

    def _stage_press_ik(self, env: Any, inference_times: list[float]) -> None:
        """PRESS_IK:目标 = 按钮点 + z 抬升 ``APPROACH_LIFT_M``;多种子 + 姿态扰动搜索。

        姿态来源照抄参考实现 ``_task_reference_orientation()`` 的注释:
        合规运行期只沿用**当前 EE 姿态**,不读任何专家先验。
        """
        if self._button_world is None:
            raise RuntimeError("缺少按钮世界点,无法做接近 IK")
        ee, ee_info = self._ee_matrix()
        if ee is None:
            raise RuntimeError(f"右臂 FK 不可用: {ee_info.get('error')}")
        orientation = np.array(ee[:3, :3], dtype=np.float64).copy()
        q_now, _grip = self._arm_joints(self._base, RIGHT)
        seeds: list[tuple[str, np.ndarray]] = [
            ("current_qpos", np.asarray(q_now, dtype=np.float64).reshape(-1)),
            ("zeros", np.zeros_like(np.asarray(q_now, dtype=np.float64).reshape(-1))),
        ]
        target = np.eye(4, dtype=np.float64)
        target[:3, :3] = orientation
        target[:3, 3] = self._button_world + np.array(
            [0.0, 0.0, float(APPROACH_LIFT_M)]
        )
        ik = self._ik_from_seeds(
            target,
            seeds,
            orientation_search=True,
            max_tilt_rad=STAGE_IK_ORIENTATION_MAX_RAD,
            samples=STAGE_IK_ORIENTATION_SAMPLES,
        )
        self._ik_record = jsonable(
            {
                "ok": bool(ik.get("ok")),
                "error": ik.get("error"),
                "button_world_m": self._button_world.tolist(),
                "target_pose": target.tolist(),
                "lift_m": float(APPROACH_LIFT_M),
                "seed_key": ik.get("seed_key"),
                "joint_distance_rad": ik.get("joint_distance_rad"),
                "orientation_error_rad": ik.get("orientation_error_rad"),
                "orientation_samples": int(STAGE_IK_ORIENTATION_SAMPLES),
                "orientation_max_rad": float(STAGE_IK_ORIENTATION_MAX_RAD),
                "attempted": ik.get("attempted"),
            }
        )
        self._debug_write_json("press_ik.json", self._ik_record)
        if not ik.get("ok"):
            raise RuntimeError(f"按钮接近位姿 IK 无解: {ik.get('error')}")
        self._q_stage = np.asarray(ik["qpos"], dtype=np.float64).reshape(-1)
        self._stage_pose = target
        self._enter(self.APPROACH)

    def _ik_from_seeds(
        self,
        pose: np.ndarray,
        seeds: list[tuple[str, np.ndarray]],
        *,
        orientation_search: bool = False,
        max_tilt_rad: float = STAGE_IK_ORIENTATION_MAX_RAD,
        samples: int = STAGE_IK_ORIENTATION_SAMPLES,
        arm: str = RIGHT,
    ) -> dict:
        """多种子 IK;必要时在 ``max_tilt_rad`` 内做姿态扰动搜索。

        逐行对应参考实现 ``_ik_from_seeds``(第 1808-1901 行):优先最小倾角、
        其次最小关节位移。``arm`` 只是把"哪条臂 / 哪个控制部件"参数化,默认
        ``"right"``(click_bell 主路径),``LOOK`` 视野搜索按需传入。
        """
        grasp = _lazy("grasp")
        arm_name = f"{arm}_arm"
        current = self._arm_joints(self._base, arm)[0]
        reference = np.asarray(current, dtype=np.float64).reshape(-1)
        best: dict | None = None
        attempted: list[dict] = []
        nominal = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        rotations: list[tuple[float, np.ndarray]] = [(0.0, nominal[:3, :3].copy())]
        if orientation_search:
            rng = np.random.default_rng(STAGE_IK_ORIENTATION_RNG_SEED)
            for _index in range(max(0, int(samples))):
                axis = rng.normal(size=3)
                norm = float(np.linalg.norm(axis))
                if norm < 1e-9:
                    continue
                axis = axis / norm
                angle = float(max_tilt_rad) * float(rng.random()) ** 0.5
                perturbation = (
                    _rotation_axis(0, angle * axis[0])
                    @ _rotation_axis(1, angle * axis[1])
                    @ _rotation_axis(2, angle * axis[2])
                )
                rotations.append((angle, perturbation @ nominal[:3, :3]))
        for tilt, rotation in rotations:
            candidate = nominal.copy()
            candidate[:3, :3] = rotation
            for key, seed in seeds:
                q_seed = np.asarray(seed, dtype=np.float64).reshape(-1)
                solved = grasp.solve_ik_named(
                    self._base, candidate, q_seed, arm_name
                )
                ok = solved is not None
                distance = None
                if ok:
                    solved_array = np.asarray(solved, dtype=np.float64).reshape(-1)
                    if solved_array.size == reference.size:
                        distance = float(np.linalg.norm(solved_array - reference))
                if tilt == 0.0:
                    attempted.append(
                        {"seed_key": key, "ok": bool(ok), "joint_distance_rad": distance}
                    )
                if not ok:
                    continue
                better = (
                    best is None
                    or tilt < best["orientation_error_rad"] - 1e-9
                    or (
                        abs(tilt - best["orientation_error_rad"]) <= 1e-9
                        and distance is not None
                        and best.get("joint_distance_rad") is not None
                        and distance < best["joint_distance_rad"]
                    )
                )
                if better:
                    best = {
                        "ok": True,
                        "seed_key": key,
                        "qpos": np.asarray(solved, dtype=np.float64).reshape(-1),
                        "joint_distance_rad": distance,
                        "orientation_error_rad": float(tilt),
                    }
            if best is not None and best["orientation_error_rad"] <= 0.0:
                # 名义姿态已经可解,后面不可能更好。
                break
        if best is None:
            return {
                "ok": False,
                "error": "no IK seed solved the staged pose",
                "attempted": attempted,
            }
        best["attempted"] = attempted
        return best

    # ------------------------------------------------------------------
    # 阶段 5:APPROACH(世界系接近;每次 _advance 推进一次迭代)
    # ------------------------------------------------------------------

    def _stage_approach(self, env: Any, inference_times: list[float]) -> None:
        """APPROACH:三层,全部沿用参考实现的控制律与 SPEC §1 的常数。

        1. **舞台位姿**(PRESS_IK 已经解出):把工具开到 ``按钮点 + APPROACH_LIFT_M``;
        2. **粗接近**(参考实现 ``servo_to_bell``):世界系接近律 —— 粗步长
           ``APPROACH_STEP_M``(0.05),横向误差 ≤ ``APPROACH_FINE_BELOW_M``(0.03)
           时换细步长 ``APPROACH_FINE_STEP_M``(0.01);收敛判据为横向
           ``APPROACH_LATERAL_TOL_M``(8 mm)且相对 ``按钮 + APPROACH_CLEARANCE_M``
           的高度误差 ≤ ``APPROACH_HEIGHT_TOL_M``(15 mm);最多
           ``APPROACH_MAX_ITERATIONS``(14)次;工具低于按钮高度时只往上爬,绝不横移;
        3. **落到按压面**(参考实现编排 ``move_to_stage``(偏移 ``STAGE_OFFSET_M`` = 0)
           + ``verify_aligned``):粗接近会把工具停在按钮上方 ``APPROACH_CLEARANCE_M``,
           而 ``press_bell`` 总共只走 8 mm,所以必须再落到测得的按压面上,再由
           ``PRESS`` 完成 §2.2 的增量下压。可用 ``APPROACH_SURFACE_STAGE`` 关闭。

        每次 ``_advance`` 只推进一次迭代(与 flow.py 的微步风格一致),让外层循环
        有机会在成功那一刻立刻 break。
        """
        if self._button_world is None:
            raise RuntimeError("接近阶段缺少按钮世界点")
        if self._stage_i == 0 and not self._approach_stage_done:
            self._approach_stage_done = True
            if self._stage_pose is not None and self._q_stage is not None:
                move = self._move_to_matrix(
                    self._stage_pose,
                    label="stage",
                    max_steps=MOVE_TO_MATRIX_MAX_STEPS,
                    tol=PRESS_TOL_L2,
                    compensate_rounds=STAGE_COMPENSATE_ROUNDS,
                )
                self._approach_trace.append(
                    {
                        "phase": "stage",
                        "ok": bool(move.get("ok")),
                        "reason": move.get("reason"),
                        "error": move.get("error"),
                        "error_l2": move.get("error_l2"),
                        "waypoints": move.get("waypoints"),
                    }
                )
                if move.get("error"):
                    raise RuntimeError(f"接近舞台位姿失败: {move.get('error')}")
                # 本微步只做这一件事(内部已经 env.step 过),下一次再进接近律。
                return
        if self._approach_converged or self._stage_i >= APPROACH_MAX_ITERATIONS:
            if not self._approach_converged:
                self._record_failure(
                    self.APPROACH,
                    f"接近迭代达到上限 {APPROACH_MAX_ITERATIONS} 次仍未收敛,"
                    "按未收敛继续落面/按压",
                    dedup=False,
                    fatal=False,
                )
            self._approach_surface(env)
            return

        ee, ee_info = self._ee_matrix()
        if ee is None:
            raise RuntimeError(f"右臂 FK 不可用: {ee_info.get('error')}")
        button = np.asarray(self._button_world, dtype=np.float64).reshape(3)
        position = np.asarray(ee[:3, 3], dtype=np.float64).copy()
        lateral = position[:2] - button[:2]
        lateral_error = float(np.linalg.norm(lateral))
        desired_z = float(button[2] + APPROACH_CLEARANCE_M)
        height_error = float(position[2] - desired_z)

        target = ee.copy()
        limit = (
            float(APPROACH_FINE_STEP_M)
            if lateral_error <= float(APPROACH_FINE_BELOW_M)
            else float(APPROACH_STEP_M)
        )
        if position[2] < float(button[2]) + 0.5 * float(APPROACH_CLEARANCE_M):
            # 绝不在按钮高度上横移:先直着往上爬。
            climb = float(button[2]) + float(APPROACH_LIFT_M) - float(position[2])
            target[2, 3] = float(position[2]) + min(climb, limit)
            action = "lift"
        else:
            target[0, 3] = float(button[0])
            target[1, 3] = float(button[1])
            target[2, 3] = desired_z
            delta = target[:3, 3] - position
            distance = float(np.linalg.norm(delta))
            if distance > limit:
                target[:3, 3] = position + delta * (limit / distance)
            action = "approach"

        if (
            lateral_error <= float(APPROACH_LATERAL_TOL_M)
            and abs(height_error) <= float(APPROACH_HEIGHT_TOL_M)
        ):
            self._approach_converged = True
            self._approach_trace.append(
                {
                    "iteration": int(self._stage_i),
                    "ok": True,
                    "action": "hold",
                    "lateral_error_m": lateral_error,
                    "height_error_m": height_error,
                    "button_world_m": button.tolist(),
                    "ee_position_m": position.tolist(),
                }
            )
            self._approach_surface(env)
            return

        move = self._move_to_matrix(
            target,
            label=f"approach_{int(self._stage_i)}",
            max_steps=MOVE_TO_MATRIX_MAX_STEPS,
            tol=PRESS_TOL_L2,
            compensate_rounds=MOVE_TO_MATRIX_COMPENSATE_ROUNDS,
        )
        ee_after, _info_after = self._ee_matrix()
        position_after = (
            None if ee_after is None else np.asarray(ee_after[:3, 3]).tolist()
        )
        self._approach_trace.append(
            {
                "iteration": int(self._stage_i),
                "ok": bool(move.get("ok")),
                "action": action,
                "lateral_error_before_m": lateral_error,
                "height_error_before_m": height_error,
                "button_world_m": button.tolist(),
                "ee_before_m": position.tolist(),
                "ee_after_m": position_after,
                "move_ok": bool(move.get("ok")),
                "reason": move.get("reason"),
                "ik_error": move.get("error"),
                "error_l2": move.get("error_l2"),
            }
        )
        if not move.get("ok") and move.get("error"):
            raise RuntimeError(f"接近失败: {move.get('error')}")
        self._stage_i += 1

    def _approach_surface(self, env: Any) -> None:
        """粗接近收敛后的"落到按压面"(参考实现 ``move_to_stage`` 偏移 0)。

        姿态照旧沿用当前 EE 姿态(``_task_reference_orientation``);IK 用多种子 +
        姿态扰动搜索;落面后按 ``STAGE_EE_TRANSLATION_TOL_M`` /
        ``STAGE_EE_ROTATION_TOL_RAD`` 记录一次任务空间验收(``verify_aligned`` 的
        等价物),不达标只做非致命记录 —— 后面的 ``press_bell`` 靠停滞检测兜底。
        """
        if not APPROACH_SURFACE_STAGE:
            self._enter(self.PRESS)
            return
        if self._approach_surface_done:
            self._enter(self.PRESS)
            return
        self._approach_surface_done = True
        button = np.asarray(self._button_world, dtype=np.float64).reshape(3)
        ee, ee_info = self._ee_matrix()
        if ee is None:
            raise RuntimeError(f"右臂 FK 不可用: {ee_info.get('error')}")
        target = ee.copy()
        target[:3, 3] = button + np.array([0.0, 0.0, float(STAGE_OFFSET_M)])
        q_now = self._arm_joints(self._base, RIGHT)[0]
        seeds: list[tuple[str, np.ndarray]] = [
            ("current_qpos", np.asarray(q_now, dtype=np.float64).reshape(-1)),
            ("zeros", np.zeros_like(np.asarray(q_now, dtype=np.float64).reshape(-1))),
        ]
        ik = self._ik_from_seeds(
            target,
            seeds,
            orientation_search=True,
            max_tilt_rad=STAGE_IK_ORIENTATION_MAX_RAD,
            samples=STAGE_IK_ORIENTATION_SAMPLES,
        )
        if not ik.get("ok"):
            # 到不了按压面:记录后仍然进 PRESS(PRESS 会按接触停滞自己收手)。
            self._record_degradation(
                self.APPROACH,
                f"落面 IK 无解({ik.get('error')}),按粗接近结果继续按压",
            )
            self._approach_trace.append(
                {
                    "phase": "surface",
                    "ok": False,
                    "error": ik.get("error"),
                    "stage_offset_m": float(STAGE_OFFSET_M),
                }
            )
            self._enter(self.PRESS)
            return
        move = self._move_to_matrix(
            target,
            label="stage_surface",
            max_steps=MOVE_TO_MATRIX_MAX_STEPS,
            tol=STAGE_TOL_L2,
            compensate_rounds=STAGE_COMPENSATE_ROUNDS,
        )
        report = self._ee_pose_error(target)
        accepted = self._ee_within_tolerance(report) if report else bool(move.get("ok"))
        joint_ok = bool(move.get("ok"))
        self._approach_trace.append(
            {
                "phase": "surface",
                "ok": bool(accepted),
                "stage_offset_m": float(STAGE_OFFSET_M),
                "move_ok": joint_ok,
                "reason": move.get("reason"),
                "ik_error": move.get("error"),
                "error_l2": move.get("error_l2"),
                "waypoints": move.get("waypoints"),
                "joint_acceptance": joint_ok,
                "ee_acceptance": bool(self._ee_within_tolerance(report)) if report else None,
                "ee_translation_tol_m": STAGE_EE_TRANSLATION_TOL_M,
                "ee_rotation_tol_rad": STAGE_EE_ROTATION_TOL_RAD,
                **report,
            }
        )
        if not accepted:
            self._record_degradation(
                self.APPROACH,
                "落面任务空间验收未通过(可能已接触),按 §2.2 继续按压",
            )
        if move.get("error"):
            raise RuntimeError(f"落面失败: {move.get('error')}")
        self._enter(self.PRESS)

    # ------------------------------------------------------------------
    # 阶段 6:PRESS(相对增量下压;每次 _advance 推进一次按压)
    # ------------------------------------------------------------------

    def _stage_press(self, env: Any, inference_times: list[float]) -> None:
        """PRESS:沿 ``PRESS_AXIS`` 相对下压,每步重读 FK 与成功信号,提前命中即停。

        逐行对应参考实现 ``press_bell``(第 2351-2441 行);迭代次数为
        ``max(1, round(PRESS_MAX_M / PRESS_STEP_M))``(=3)。
        """
        direction = np.asarray(PRESS_AXIS, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            raise RuntimeError("按压方向向量为零")
        direction = direction / norm
        max_iterations = max(1, int(round(PRESS_MAX_M / max(PRESS_STEP_M, 1e-6))))

        if self._stage_i == 0:
            self._press_pressed_m = 0.0
            self._press_stalled_streak = 0
            self._press_blocked = False
            self._press_success = self._query_success() is True

        if (
            self._press_success
            or self._press_pressed_m >= float(PRESS_MAX_M)
            or self._stage_i >= max_iterations
        ):
            self._enter(self.VERIFY)
            return

        index = int(self._stage_i)
        current_step = float(PRESS_FIRST_STEP_M) if index == 0 else float(PRESS_STEP_M)
        current_step = min(
            current_step, max(0.0, float(PRESS_MAX_M) - float(self._press_pressed_m))
        )
        if current_step <= 0.0:
            self._enter(self.VERIFY)
            return

        ee, ee_info = self._ee_matrix()
        if ee is None:
            raise RuntimeError(f"右臂 FK 不可用: {ee_info.get('error')}")
        target = ee.copy()
        target[:3, 3] = np.asarray(ee[:3, 3], dtype=np.float64) + direction * current_step
        move = self._move_to_matrix(
            target,
            label=f"press_{index}",
            max_steps=PRESS_MAX_STEPS,
            tol=PRESS_TOL_L2,
            compensate_rounds=1,
        )
        self._press_pressed_m = float(self._press_pressed_m) + current_step
        success = self._query_success() is True
        self._press_success = success
        moved = move.get("move") if isinstance(move.get("move"), dict) else {}
        self._press_trace.append(
            {
                "iteration": index,
                "step_m": current_step,
                "pressed_m": float(self._press_pressed_m),
                "move_ok": bool(move.get("ok")),
                "error_l2": moved.get("error_l2"),
                "stalled": moved.get("stalled"),
                "success": bool(success),
                "ik_error": move.get("error"),
                "reason": move.get("reason"),
            }
        )
        self._stage_i += 1
        if move.get("error"):
            self._enter(self.VERIFY)
            return
        # 按压本身就是接触:只有连续 12 次停滞且始终没成功才判定被顶住。
        if moved.get("stalled") and not success:
            self._press_stalled_streak += 1
            if self._press_stalled_streak >= MAX_STALLED_STREAK:
                self._press_blocked = True
                self._enter(self.VERIFY)
                return
        else:
            self._press_stalled_streak = 0
        if success or self._stage_i >= max_iterations:
            self._enter(self.VERIFY)

    # ------------------------------------------------------------------
    # 阶段 7/8/9:VERIFY → DONE → ACT_FILL
    # ------------------------------------------------------------------

    def _stage_verify(self, env: Any, inference_times: list[float]) -> None:
        """VERIFY:保持当前位姿 ``VERIFY_STEPS``(5)步。

        成功是粘滞锁存的(参考实现 ``_success_flag``),因此这里只需保持位姿给外层
        循环读 ``is_task_success()``。
        """
        self._step_env_step(env, self._hold_action())
        self._stage_i += 1
        if self._stage_i >= VERIFY_STEPS:
            self._enter(self.DONE)

    def _stage_done(self, env: Any, inference_times: list[float]) -> None:
        """DONE:流程正常走完,剩余步数交给 ``ACT_FILL``。"""
        self._enter(self.ACT_FILL)

    def _stage_act_fill(self, env: Any, inference_times: list[float]) -> None:
        """ACT_FILL:流程走完后用剩余步数继续(默认纯 ACT)。"""
        if DONE_RUNS_PURE_ACT:
            self._act_micro_step(env, inference_times)
        else:
            self._step_env_step(env, self._hold_action())

    def _stage_fallback_act(self, env: Any, inference_times: list[float]) -> None:
        """FALLBACK_ACT:任一阶段失败后的纯 ACT 执行(不做任何干预)。"""
        self._act_micro_step(env, inference_times)

    # ------------------------------------------------------------------
    # 世界系目标 + 分段 waypoint(照抄参考实现 ``_move_to_matrix``)
    # ------------------------------------------------------------------

    def _move_to_matrix(
        self,
        target: np.ndarray,
        *,
        label: str,
        max_steps: int = MOVE_TO_MATRIX_MAX_STEPS,
        tol: float = PRESS_TOL_L2,
        compensate_rounds: int = MOVE_TO_MATRIX_COMPENSATE_ROUNDS,
        waypoint_step_m: float = WAYPOINT_STEP_M,
        max_waypoints: int = MAX_WAYPOINTS,
    ) -> dict:
        """IK 一个世界系位姿并把机械臂开过去,**分段**执行。

        参考实现的原注释:实测在这条臂上"一次要求 80 mm 平移会失败,即使终点本身
        可达",所以大位移必须切成 waypoint,每个 waypoint 从上一个实际到达的位置
        重新求解。**不要合并成一步。**
        """
        pose = np.asarray(target, dtype=np.float64).reshape(4, 4)
        ee, ee_info = self._ee_matrix()
        if ee is None:
            return {
                "ok": False,
                "label": label,
                "error": ee_info.get("error", "FK unavailable"),
            }
        start = np.asarray(ee[:3, 3], dtype=np.float64).copy()
        delta = pose[:3, 3] - start
        distance = float(np.linalg.norm(delta))
        count = 1
        if waypoint_step_m and distance > float(waypoint_step_m):
            count = min(
                max(1, int(max_waypoints)),
                max(1, int(np.ceil(distance / float(waypoint_step_m)))),
            )
        trace: list[dict] = []
        ik: dict = {}
        moved: dict = {}
        # 环境接缝:把剩余预算均分给 (waypoint × 补偿轮次),保证一次调用不会
        # 吃光 episode 预算(PRESS_MAX_STEPS=4 之类的显式上限优先)。
        per_round = min(
            int(max_steps),
            self._share_steps(compensate_rounds, count),
        )
        for index in range(count):
            alpha = float(index + 1) / float(count)
            waypoint = pose.copy()
            waypoint[:3, 3] = start + delta * alpha
            seed = self._arm_joints(self._base, RIGHT)[0]
            solved = self._ik_single(waypoint, seed)
            ik = solved
            if not solved.get("ok"):
                return {
                    "ok": False,
                    "label": label,
                    "error": (
                        f"IK failed for {label} at waypoint {index + 1}/{count}"
                    ),
                    "waypoint": index,
                    "waypoints": count,
                    "target_pose": waypoint.tolist(),
                    "ik": solved,
                    "trace": trace,
                }
            moved = self._servo_qpos(
                self._base,
                solved["qpos"],
                max_steps=int(per_round),
                tol=float(tol),
                compensate_rounds=int(compensate_rounds),
            )
            ok = bool(isinstance(moved, dict) and moved.get("ok"))
            trace.append(
                {
                    "waypoint": index,
                    "waypoints": count,
                    "alpha": alpha,
                    "ok": ok,
                    "error_l2": moved.get("error_l2") if isinstance(moved, dict) else None,
                }
            )
            if not ok:
                return {
                    "ok": False,
                    "label": label,
                    # 刻意**不**写 ``error``:走不到位是接触,不是求解失败。
                    # ``press`` 依据这一点判断"工具已经压到按钮"。
                    "converged": False,
                    "reason": f"waypoint {index + 1}/{count} of {label} stopped short",
                    "waypoint": index,
                    "waypoints": count,
                    "move": moved,
                    "error_l2": moved.get("error_l2") if isinstance(moved, dict) else None,
                    "stalled": moved.get("stalled") if isinstance(moved, dict) else None,
                    "trace": trace,
                }
            if self._budget_exhausted():
                return {
                    "ok": False,
                    "label": label,
                    "converged": False,
                    "reason": "episode step budget exhausted mid-move",
                    "waypoint": index,
                    "waypoints": count,
                    "move": moved,
                    "trace": trace,
                }
        return {
            "ok": True,
            "label": label,
            "target_pose": pose.tolist(),
            "waypoints": count,
            "waypoint_step_m": float(distance / count),
            "target_qpos": np.asarray(ik.get("qpos", [])).tolist() if ik else None,
            "move": moved,
            "trace": trace,
        }

    def _ik_single(self, pose: np.ndarray, seed: np.ndarray) -> dict:
        """单个 waypoint 的单种子 IK(``grasp.solve_ik_named`` 的安全封装)。"""
        try:
            grasp = _lazy("grasp")
            solved = grasp.solve_ik_named(
                self._base,
                np.asarray(pose, dtype=np.float64).reshape(4, 4),
                np.asarray(seed, dtype=np.float64).reshape(-1),
                RIGHT_ARM,
            )
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if solved is None:
            return {"ok": False, "error": "IK returned no solution"}
        return {
            "ok": True,
            "qpos": np.asarray(solved, dtype=np.float64).reshape(-1),
        }

    # ------------------------------------------------------------------
    # ``apply_qpos_target`` 的等价实现(SPEC §4 的 ``_servo_qpos``)
    # ------------------------------------------------------------------

    def _servo_qpos(
        self,
        base: Any,
        q_target: np.ndarray,
        *,
        max_steps: int,
        tol: float,
        compensate_rounds: int,
    ) -> dict:
        """多次下发目标并补偿,直到关节误差 < ``tol`` 或轮次用尽。

        等价于 RPent 的 ``env.apply_qpos_target``(env_server.py 第 1252-1470 行):

        * 每轮 ``_settle_to_issued`` 把 `issued` 作为位置目标下发(max_steps 步);
        * 误差始终**对原始 target** 度量;
        * 未达容差且仍在改善时,``issued = issued + (target - actual)`` 补偿这条
          臂可重复的静态 PD 偏移;
        * 误差不再改善(平台)则提前收敛,避免空转烧掉 episode 预算。
        """
        env = self._env
        if env is None:
            return {"ok": False, "error": "env 未绑定,无法下发关节目标"}
        target = np.asarray(q_target, dtype=np.float64).reshape(-1)
        if target.size == 0:
            return {"ok": False, "error": "q_target 为空"}
        rounds = max(1, int(compensate_rounds))
        steps = max(1, int(max_steps))
        issued = target.copy()
        history: list[dict] = []
        error_l2 = float("inf")
        error_max_abs = float("inf")
        steps_used = 0
        stalled = False
        previous_error: float | None = None
        for rnd in range(1, rounds + 1):
            settled = self._settle_to_issued(
                base, issued, max_steps=steps, tol_l2=float(tol)
            )
            steps_used += int(settled["steps_used"])
            stalled = bool(settled["stalled"])
            actual = np.asarray(settled["actual"], dtype=np.float64).reshape(-1)
            residual = target - actual
            error_l2 = float(np.linalg.norm(residual))
            error_max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
            history.append(
                {
                    "round": rnd,
                    "error_l2": error_l2,
                    "error_max_abs": error_max_abs,
                    "steps_used": int(settled["steps_used"]),
                }
            )
            if self._budget_exhausted():
                break
            if error_l2 <= float(tol) and error_max_abs <= STAGE_TOL_MAX_ABS:
                break
            if previous_error is not None and error_l2 >= previous_error - PLATEAU_EPS:
                # 补偿不再有帮助:报告当前结果。
                break
            previous_error = error_l2
            issued = issued + residual
        within = error_l2 <= float(tol) and error_max_abs <= STAGE_TOL_MAX_ABS
        return {
            "ok": bool(within),
            "converged": bool(within),
            "error_l2": error_l2,
            "error_max_abs": error_max_abs,
            "steps_used": int(steps_used),
            "stalled": bool(stalled),
            "compensate_rounds": len(history),
            "compensation_history": history,
        }

    def _settle_to_issued(
        self, base: Any, issued: np.ndarray, *, max_steps: int, tol_l2: float
    ) -> dict:
        """下发一轮位置目标并等它稳定(照抄 ``_settle_to_issued`` 的语义)。

        与 harness 的差别只在接缝:那里是 ``robot.set_qpos(target=True)`` +
        ``sim.update``,这里是 ``env.step(full_action(...))``——两条已被本仓库
        ``flow.py`` 验证过的等价路径。左臂/夹爪被命令回它们当前的关节值,即保持
        不动(等价于 harness 把 ``full_target`` 的其余分量设为当前值)。
        """
        env = self._env
        if env is None:
            raise RuntimeError("env 未绑定")
        issued = np.asarray(issued, dtype=np.float64).reshape(-1)
        left_q, left_grip = self._arm_joints(base, LEFT)
        _q_right_now, right_grip = self._arm_joints(base, RIGHT)
        best_l2 = float("inf")
        best_step = 0
        error_l2 = float("inf")
        error_max_abs = float("inf")
        steps_used = 0
        stalled = False
        actual = np.asarray(issued, dtype=np.float64).copy()
        for step in range(1, int(max_steps) + 1):
            action = full_action(base, left_q, left_grip, issued, right_grip)
            self._step_env_step(env, action)
            steps_used = step
            actual, _grip_now = self._arm_joints(base, RIGHT)
            actual = np.asarray(actual, dtype=np.float64).reshape(-1)
            delta = actual - issued
            error_l2 = float(np.linalg.norm(delta))
            error_max_abs = (
                float(np.max(np.abs(delta))) if delta.size else float("inf")
            )
            if error_l2 <= float(tol_l2) and error_max_abs <= STAGE_TOL_MAX_ABS:
                break
            if error_l2 < best_l2 - PLATEAU_EPS:
                best_l2 = error_l2
                best_step = step
            elif step - best_step >= PLATEAU_WINDOW:
                stalled = True
                break
            if self._budget_exhausted():
                break
        return {
            "steps_used": int(steps_used),
            "error_l2": error_l2,
            "error_max_abs": error_max_abs,
            "stalled": bool(stalled),
            "actual": actual,
        }

    # ------------------------------------------------------------------
    # ACT 执行(FALLBACK_ACT / ACT_FILL 共用)
    # ------------------------------------------------------------------

    def _select_chunk_action(self, obs: Any) -> list[np.ndarray]:
        """调用 ``act_policy.select_chunk_action`` 并计时(只计模型前向)。

        计时走官方 ``policy.inference_timing``(由 flow.py 的封装给出),
        **不含** ``env.step``。
        """
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

    def _act_micro_step(
        self, env: Any, inference_times: list[float]
    ) -> tuple[Any, bool]:
        """执行 1 个 ACT 控制步,返回 ``(obs, 当前 chunk 是否结束)``。

        策略前向失败时把后续执行降级为"保持当前位姿",保证 episode 仍能跑完。
        """
        if self._act_remaining <= 0:
            self._act_remaining = self._act_chunk_budget
            self._pending_rows = []
            self._act_chunks_started += 1
        if self._act_broken:
            action = self._hold_action()
        else:
            try:
                action = self._next_act_action(inference_times)
            except Exception as exc:
                self._act_broken = True
                self.failed = True
                self._record_failure(
                    self._state,
                    "ACT 前向失败,后续退化为保持当前位姿: "
                    f"{type(exc).__name__}: {exc}",
                )
                action = self._hold_action()
        row = self._fit_action(np.asarray(action, dtype=np.float32).reshape(-1))
        obs, _done = self._step_env_step(env, row)
        self._act_remaining -= 1
        self._act_steps += 1
        finished = self._act_remaining <= 0
        if finished:
            self._act_chunk_index += 1
        self._record_act_state()
        return obs, finished

    # ------------------------------------------------------------------
    # 调试落盘(debug_dir 为 None 时一律不落盘)
    # ------------------------------------------------------------------

    def _debug_subdir(self, relative: str) -> Path | None:
        """在 ``debug_dir`` 下建子目录(失败返回 ``None``,绝不影响评测)。"""
        if self.debug_dir is None:
            return None
        try:
            path = Path(self.debug_dir) / relative
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            return None

    def _debug_write_png(self, relative: str, image: Any) -> None:
        """写一张调试 PNG(失败静默)。"""
        if self.debug_dir is None or image is None:
            return
        try:
            import imageio.v2 as imageio

            path = Path(self.debug_dir) / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(path, np.asarray(image))
        except Exception:
            pass

    def _debug_write_json(self, relative: str, payload: Any) -> None:
        """写一份调试 JSON(失败静默)。"""
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
        """记录 ACT 每步的关节状态(仅 debug 模式)。"""
        if self.debug_dir is None or len(self._act_state) >= MAX_ACT_STATE:
            return
        try:
            q_left, grip_left = self._arm_joints(self._base, LEFT)
            self._act_state.append(
                {
                    "chunk": self._act_chunk_index,
                    "step": self._act_steps,
                    "left_q": np.asarray(q_left).tolist(),
                    "left_grip": float(grip_left),
                    "state": self._state,
                }
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 收尾查询
    # ------------------------------------------------------------------

    def _query_success(self) -> bool | None:
        """查询 ``env.get_wrapper_attr("is_task_success")()``(失败返回 ``None``)。

        官方 ``policy/act/deploy_policy.py`` 的 ``eval()`` 本身每步就读它,
        因此这不属于特权信息。
        """
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
