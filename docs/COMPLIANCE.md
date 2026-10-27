# 合规声明与待确认事项

本文件记录提交包**实际使用了哪些信号**,以及官方规则中**没有明确、需要向组织方书面确认**的部分。
结论一律附原文出处;查不到的一律写"未找到",不做推测。

---

## 1. 官方明文规则(仅两条输入限制)

### 1.1 评测只提供 RGB

来源:官网 FAQ(`https://robosyn-bench.net/static/js/pages-app.js`)

> Q: "Can participants use depth data, camera intrinsics, or point clouds as model inputs?"
> A: "**No. Official evaluation provides RGB images only, without depth data, camera intrinsics, or point clouds.** Participants may estimate depth or 3D information from a single RGB image using their own models."

配套:

> Q: "Do the organizers provide tools for generating depth data?"
> A: "Yes. We provide data-collection examples and scripts for generating simulated training data with depth information. **This data may be used for training, but official evaluation remains RGB-only.**"

### 1.2 只能基于视觉 + 本体感受

来源:竞赛论文 arXiv:2608.12416v1,§1.3 Tasks

> "All tasks are performed **based solely on visual observations and the robot's proprioceptive feedback**."

### 1.3 官方**没有**的条款(检索负结果)

对官网全文 bundle、官方仓库 README/docs/scripts/launch/policy、论文全文做不区分大小写检索,以下关键词**命中数全为 0**:

`privileged`、`forward kinematics`、`inverse kinematics`、`FK`、`hand-crafted`/`handcrafted`、`analytic`、`state machine`、`not allowed`、`must not`、`prohibit`、`forbidden`、`disallow`、`pure neural`、`scripted`。

官网也**没有独立规则页**(导航仅 Home / Data / FAQ / Evaluation / Leaderboard / My Submissions)。

→ **是否允许策略侧调用 FK/IK、使用固定物体先验、使用解析式流程,官方未规定。**

---

## 2. 本提交包实际使用的信号

### 2.1 允许并使用的

| 信号 | 来源 | 说明 |
| --- | --- | --- |
| 三路 RGB 图像 | `obs["sensor"][cam_high / cam_left_wrist / cam_right_wrist]["color"]` | 官方观测 |
| 机器人关节角 `qpos`(14 维,含夹爪) | `robot.get_qpos()` | 本体感受,§1.2 明文允许 |
| 正运动学 FK | `robot.compute_fk(qpos, name="<arm>_arm", link_names=["<arm>_link6"])` | 由本体感受推导,不含场景信息 |
| 逆运动学 IK | `robot.compute_ik(pose, joint_seed=..., name="left_arm")` | 运动学求解,不含场景真值 |
| 关节限位 | `robot.get_qpos_limits(name="<arm>_arm")` | 机器人描述 |
| 相机内参 | `policy/hybrid/robosyn_hybrid/calibration.json` | **公开常量**,抄自官方 `configs/handle_basket/random/gym_config.json`,运行时不读模拟器 |
| 腕相机安装位姿 | `policy/hybrid/robosyn_hybrid/calibration.json` 的 `pos` / `quat` | 同上,公开常量 |
| 抓取先验 | `policy/hybrid/robosyn_hybrid/priors.py` 的 `MILK_GRASP_OBJECT` | 抄自官方公开配置 |
| 物体尺寸先验 | `MILK_PRIOR_SIZE` | 公开物体描述 |
| 成功判据 `is_task_success()` | `env.get_wrapper_attr("is_task_success")()` | 官方 `policy/act/deploy_policy.py` 的 `eval()` 本身每步就读它 |
| 按钮位置随机范围 | `clickbell.py` 的 `BUTTON_REGION_MIN/MAX` = `(0.4,-0.3,0.83)…(0.85,0.3,0.83)` | **公开常量**,抄自官方 `configs/click_bell/random/gym_config.json` 的 `env.events.init_button_pose.position_range`;仅用作**腕相机视野搜索的区域**(按钮恒定在桌面 z=0.83),不代入任何具体 seed 的真值 |

| holder 位置(VLM 语义) | `handover.py` 的 `HOLDER_PROMPT` + `vlm.vlm_chat` | **只要求 0..1 归一化坐标**,再由代码按图像宽高换算回像素(见下方"归一化坐标约定") |
| 内圈椭圆拟合 | `handover.py::_fit_inner_rim`(参考包 `holder_inner_ellipse.py` 的移植) | 只吃 RGB;`top_frac` / 长短轴比 / 内缩 0.95 都是移植常量 |
| 筒口 3D 位置 | 双腕内圈椭圆中心 + `P = K[R^T \| -R^T t]` 线性三角化 | 相机位姿 = FK + 公开安装常量,`K` = `priors.camera_K`;不读深度缓冲、不读物体真值 |
| 固定交接路点 | `handover.py` 的 `MANUAL_TARGET_QPOS`(14 通道) | 参考包内**与场景无关**的离线遥操作路点(全场次同一个文件),不是任何一局的真值 |

三条混合流程(`handle_basket` = `policy/hybrid/robosyn_hybrid/flow.py`,`click_bell` =
`policy/hybrid/robosyn_hybrid/clickbell.py`,`items_handover` = `policy/hybrid/robosyn_hybrid/handover.py`)
使用的是**同一组信号**,没有新增任何一类;
`clickbell.py` 另外明确不读官方 action bank / 专家先验,按压姿态沿用当前 EE 姿态,
按钮位置由右腕 RGB + 运动立体深度自行估计。

**按钮定位方式(合规要点)**:参考实现的 `look_bell()` 先用 `cam_high`(外部相机)定位,
但 `cam_high` 的外参每回合会被 `random_camera_high_extrinsics` 随机化,公开常量不可用。
本包改用**腕相机视野搜索**:在官方公开的按钮区域内取若干候选点,逐个把腕相机对准并各问一次
VLM,取第一个"看得见按钮"的视角 —— 全程只依赖本体感受(FK/IK)+ RGB + 公开常量。


### 2.2 明确**不使用**的

| 禁用项 | 保证方式 |
| --- | --- |
| 物体真值位姿 `sim.get_rigid_object(...).get_local_pose()` | 参考实现 `measure_motion_stereo_depth.py` 中的 `object_centers()` / `objects_in_view()` / `pick_view_pose()` / `frames_objects()` **未移植**;`items_handover` 参考流程里的 `pen_pose()` 与 `_pose(env, uid)` 两处真值读取也**已删除**(见 §2.4);`grep -rnE "get_rigid_object\|affordance_datas\|compute_task_state" submission/` 必须零命中 |
| 模拟器深度缓冲 | 只用运动立体(两次曝光 RGB)+ FoundationStereo 估计深度 |
| 相机真值外参(运行时读取) | `wrist_camera_pose()` 主路径 = `FK(qpos, link6) @ camera_mount(常量)`;**不访问 `env.sensors`**。`verify_mount()` 仅在显式调用时与真值比对,且只记录不参与决策 |
| 任务成功判定的内部状态 | 不调用 `compute_task_state`;成功判据只读官方 `is_task_success()`(外层 `eval_policy.py` 每步读;混合流程在 PRESS 阶段也每步读它做提前停止 —— 官方 `policy/act/deploy_policy.py` 的 `eval()` 本身就这么做) |

### 2.3 关于"腕相机位姿算不算特权信息"

腕相机刚性安装在 `<arm>_link6` 上,安装位姿 `pos=[-0.12, 0.01, 0.03]`,
`quat=[0.1949, 0.6797, -0.6797, -0.1949]` 是官方仓库公开常量。因此

```
T_world_camera(q) = FK(q, "<arm>_link6") @ T_link6_camera
```

只用到**本体感受 + 公开机器人描述**,与 §1.2 允许的"proprioceptive feedback"一致。

⚠️ 注意:`cam_high` 是外部相机,其外参**无法**由本体感受推导,属于场景配置。
本包在三条混合流程(`handle_basket` / `click_bell` / `items_handover`)中
**都不使用 `cam_high` 做几何计算** —— `handle_basket` 只把 `cam_high` 作为 VLM 的
辅助视图(与腕相机图并排)参与语义判断,不参与深度估计、位姿反投影或 IK;
`click_bell` 完全不用 `cam_high`(只用右腕相机);`items_handover` 也完全不用
`cam_high`(holder 定位用**左+右腕**两张图 + 内圈椭圆三角化)。

---

## 2.4 移植时**移除**的合规违规(诚实记录)

参考实现 `_stage/Depth_Fetch/common/scripts/measure_motion_stereo_depth.py` 里有三处
不符合 §1.1 的"评测只提供 RGB"限制,本提交包**已全部移除**:

| 参考实现的做法 | 位置 | 本包的处理 |
| --- | --- | --- |
| `capture()` 读取 `obs["sensor"][cam]["depth"]` —— **模拟器深度缓冲** | `measure_motion_stereo_depth.py::capture` | `policy/hybrid/robosyn_hybrid/stereo.py::capture()` 只返回 RGB,不再触碰 `depth` 键 |
| `camera_optical_pose()` 读取 `env.sensors[uid].get_arena_pose()` —— **运行时相机真值外参** | 同上 | 改为 `FK(qpos, "<arm>_link6") @ camera_mount(公开常量)` |
| `object_centers()` 读取 `sim.get_rigid_object(uid).get_local_pose()` —— **物体真值位姿** | 同上,被 `pick_view_pose` / `objects_in_view` / `frames_objects` 使用 | 这些函数**未移植**;`env.sim` 在本包中零出现 |

`env.sensors` 在本包中只出现在 `stereo.verify_mount()`(开发期诊断),
而 `verify_mount` **在全代码库中没有任何调用点** —— 它不会在评测时执行。

`items_handover` 的参考流程里另有两处真值读取,同样**已删除**(SPEC §3):

| 参考实现的做法 | 位置(参考包) | 本包的处理 |
| --- | --- | --- |
| `env.sim.get_rigid_object("pen").get_local_pose(...)` —— **笔的真值位姿**,只用于"笔还在爪里吗"的复核 | 主流程脚本的 `pen_pose()` | **删除该检查并如实记录**。移植版只保留原语自身的本体感受信号(垂直对齐的下降量/倾角改善、接触触发下压、夹爪在接触位姿闭合并保持、LIFT 阶段 FK 测得的 TCP 抬升量);`summary()["grasp"]["pen_in_gripper_check"] == "omitted"` 明确标注。**无法在不读真值的前提下完成该检查,故省略** |
| `env.sim.get_rigid_object(uid).get_local_pose(...)` —— **任意物体真值位姿** | 放置模块的 `_pose(env, uid)` | **删除,且不传估计值**:该函数在放置候选族里用来算"笔在 TCP 系里的相对位姿",而该相对位姿正是参考实现自己记录的**物理缺陷**(笔在爪内滑转 ~63°),用一个固定先验去替代只会掩盖它。移植版因此**整条候选族 + fallback 链都没有移植**,放置改成"筒口正上方 5 cm + 保持当前 TCP 姿态 + 多起点 IK + 释放"(参考包内批处理实际调用的那条路径),如实保留倾斜入筒的已知行为 |

**归一化坐标约定(必须保留)**:参考实现实测"VL 模型回答的像素坐标在它自己视觉
编码器的空间里,bbox 会越界且错位 100+px",换算回像素后与基线差仅 ~5px,腕相机
双视图内圈三角化的筒口 3D 误差 **3.17 mm**。本包 `items_handover` 的 holder / pen
提示词**一律要求 0..1 归一化坐标**(`bbox_xyxy_norm` / `center_uv_norm` /
`pen_uv_norm`),再由 `handover.py::_norm_to_pixels` 按实际图像宽高换算回像素;
若模型仍回像素(值 >1)则原样透传(兼容分支),但归一化路径才是主路径。
`summary()["holder_locate"]["coordinate_convention"] == "vlm_normalized_0_1"`。

---

## 2.5 ✅ 已结案:FK 返回的是 tool/EE 帧(不是 link6),修正矩阵已内置

**结论:官方检出上 `robot.compute_fk(name="<arm>_arm", link_names=["<arm>_link6"],
to_matrix=True)` 的 `link_names` 参数被忽略,返回的是 tool/TCP 帧。**
因此腕相机位姿必须补一个常量才能换算回 `<arm>_link6` 连杆帧,否则安装矩阵
整体错位 **0.280 m / 180°**。

### 实测方法(不是解析推测)

2026-10-10 在真实 RTX 4090 服务器 + `EmbodiChain_main 0.2.4` 检出上跑 L1
分层几何诊断 `python policy/hybrid/robosyn_hybrid/scripts/verify_clickbell_geometry.py`,
取 `layers[0].data.verify_mount`:

* `measured_mount = inv(fk_returned(qpos)) @ (sensor.get_arena_pose(to_matrix=True) @ SIM_TO_OPTICAL)`
  —— 腕相机刚性装在 `<arm>_link6` 上,所以这就是"FK 实际返回的帧 -> 相机"
  的真实相对位姿;
* `expected_mount = priors.camera_mount(arm)` —— 公开安装常量算出的
  link6 -> 相机 位姿;
* 两者联立反解出两个帧之间的精确刚体变换:
  `inv(TOOL_FROM_LINK6) = measured_mount @ inv(expected_mount)`。

修复前的实测数字(右臂,证据文件 `clickbell_L1.json`):

```text
mount_delta.pos_err_m    = 0.2800876513224117     (280.1 mm)
mount_delta.rot_err_deg  = 180.0
best_variant             = "xyzw/pre_flip"        (纯解析预测表被推翻)
warnings = ["best mount variant is 'xyzw/pre_flip', but the module hard-codes
             'wxyz/post_flip'", "no candidate matches the sensor within 2 mm;
             compute_fk() may be returning the end-effector frame ..."]
```

### 实测解出的修正矩阵

解出的关系是:**绕 z 轴 180°,再沿 z 平移 +0.143 m**:

```text
TOOL_FROM_LINK6 (link6 -> fk 返回的帧) =
    [[-1, 0, 0, 0.000],
     [ 0,-1, 0, 0.000],
     [ 0, 0, 1, 0.143],
     [ 0, 0, 0, 1.000]]

FK_TO_LINK6 = inv(TOOL_FROM_LINK6) =
    [[-1, 0, 0, 0.000],
     [ 0,-1, 0, 0.000],
     [ 0, 0, 1,-0.143],
     [ 0, 0, 0, 1.000]]
```

代入验证残差 `max|pred - measured| = 1.277e-06`(纯 numpy 复算,反过来证明
该矩阵是数值解出的,不是拍的)。两个常量已作为模块级常量写进
`policy/hybrid/robosyn_hybrid/priors.py`(`TOOL_FROM_LINK6` / `FK_TO_LINK6`,均在 `__all__`),
修正后的主路径是

```text
wrist_camera_pose(q) = fk_returned(q) @ FK_TO_LINK6 @ (T_mount @ SIM_TO_OPTICAL)
```

`FK_TO_LINK6` 由 `priors.camera_mount()` 内部左乘,调用方不需要改。

### `HYBRID_FK_FRAME` 开关

`policy/hybrid/robosyn_hybrid/settings.py` 新增 `Settings.fk_frame`,从环境变量
**`HYBRID_FK_FRAME`** 读取:

| 取值 | 含义 | 行为 |
| --- | --- | --- |
| `tool`(默认) | `compute_fk` 返回 tool/TCP 帧 | `camera_mount()` 左乘 `FK_TO_LINK6` |
| `link6` | 该检出确实返回 link6 连杆帧 | `camera_mount()` 原样返回 |

非法值会 `warnings.warn` 并回落 `"tool"`;该字段也出现在 `Settings.describe()`
里,便于日志留证。

### 修复后的实测结果(2026-10-10,真实 RTX 4090,同一 checkout)

| 运行 | `--arm` | `HYBRID_FK_FRAME` | L1 status | best_variant | 位置误差 | 旋转误差 |
| --- | --- | --- | --- | --- | ---: | ---: |
| `L1_after_right.json` | right | `tool`(默认) | **pass** | `wxyz/post_flip` | 0.000496 mm | 0.000° |
| `L1_after_left.json` | left | `tool`(默认) | **pass** | `wxyz/post_flip` | 0.000674 mm | 0.022° |
| `L1_after_right_link6.json`(对照) | right | `link6` | **fail** | `xyzw/pre_flip` | 280.088 mm | 180.000° |

判定阈值是 `< 2 mm` / `< 0.5°`,两次主路径都通过;`link6` 对照组精确复现了
修复前的错误数字(280.088 mm / 180.000°,旧证据为 280.088 mm / 180.000°),
说明修复确实来自 `FK_TO_LINK6`,不是别的副作用。左右臂用的是**同一个**
`TOOL_FROM_LINK6`,不需要按臂区分。

### 如果评测机是别的 EmbodiChain 检出

请重跑一次(约 1 分钟):

```bash
python policy/hybrid/robosyn_hybrid/scripts/verify_clickbell_geometry.py \
    --repo-root <官方仓库根> --setting random --seed 0 --arm right \
    --skip-depth --skip-vlm --out /tmp/L1.json
# 左臂再跑一次:--arm left --out /tmp/L1_left.json
```

* L1 `status = pass` ⇔ `best_variant == "wxyz/post_flip"` 且位置误差 < 2 mm、
  旋转误差 < 0.5° → 当前实现正确,无需改动;
* 若 `best_variant` 不是 `wxyz/post_flip`,或误差明显不为零 → 该检出返回的帧
  不同,按上表切 `HYBRID_FK_FRAME=link6` 再测;仍不通过则重新反解
  `TOOL_FROM_LINK6`(该常量只在实测的 `EmbodiChain_main 0.2.4` 检出上成立,
  官方仓库并未提供任何显式常量记录这个帧约定)。

四元数顺序仍由三条独立官方源码证据确定为 **`wxyz`**(与上面的帧问题无关,
是两件独立的事):

1. `embodichain/utils/math.py::matrix_from_quat` 的 docstring 明写 "(w, x, y, z)";
2. `SensorCfg.OffsetCfg.quat` 同约定,且 parented 相机(腕相机)正走这一支;
3. 官方 harness 直接把同一常量命名为 `WRIST_MOUNT_QUAT_WXYZ`。

### 附:旧的纯解析预测表(已作废,仅作反面记录)

修复前只用公开常量推导、未上机的预测如下。**它已被实测推翻**——旧表预测
`wxyz + post_flip` 误差为 0,实测是 **0.280 m / 180°**,根因就是本轮结案的
FK 帧问题。不要再据此推断运行时行为。

| 变体 | 位置误差(旧解析预测) | 旋转误差(旧解析预测) |
| --- | ---: | ---: |
| `wxyz` + post_flip | 0 mm | 0° |
| `xyzw` + post_flip | 0 mm | 152.81° |
| `wxyz` + no_flip | 0 mm | 180° |
| `xyzw` + no_flip | — | 129.82° |
| `wxyz` + pre_flip | 63.25 mm | 180° |
| `xyzw` + pre_flip | 63.25 mm | 80.19° |

---

## 3. 需要向组织方书面确认的问题

建议在提交的同时发邮件到 `robosynchallenge@gmail.com`(建议主题
`RoboSynChallenge participant access request` 或
`Clarification on allowed observations and analytic components`):

1. 策略在评测时读取 `robot.get_qpos()` 并调用 `robot.compute_fk()` /
   `robot.compute_ik()` 是否被允许?论文写的是 "based solely on visual
   observations and the robot's proprioceptive feedback",我们理解 FK/IK 是对
   本体感受的确定性变换,而非额外观测,希望确认。
2. 使用官方仓库**公开配置中**的相机内参与腕相机安装位姿常量(而非运行时读取
   模拟器对象)是否被允许?FAQ 说评测"不提供 camera intrinsics",我们不确定这
   是指"不在观测里给",还是"完全不能使用公开的标定值"。
3. 策略中使用**解析式状态机 + 逆运动学求解**的组件(而非端到端神经网络)是否
   被允许?FAQ 与论文均未规定方法学限制。
4. 使用**固定物体尺寸先验**(如牛奶盒 4.3cm × 9.95cm × 4.44cm,来自公开物体
   描述)是否被允许?
5. `data_source_text` 中 "any additional allowed data" 的具体界定是什么?

---

## 3.5 `items_handover`:一次抓取 bug 的定位与修复(真机记录)

`items_handover` 走混合流程。它曾经连续 11 轮真机运行 **0 成功**,一度被判为"混合不可行、
改走纯 ACT";后来定位到是**抓取原语的 bug**,修复后真机首测即成功。

### 3.5.1 根因:`_step_right_q` 的夹爪默认值是"闭合"

```python
def _step_right_q(self, env, q_right, *, grip: float = GRIP_RIGHT_CLOSED):
```

而两个接近阶段调用点**都没有显式覆盖** `grip`:

| 调用点 | 传入 | 后果 |
| --- | --- | --- |
| `_prim_vertical_align`(垂直对齐) | 未传 → 默认 | 夹爪**闭合** |
| `_prim_press`(下压接触) | 未传 → 默认 | 夹爪**闭合** |
| `_prim_lift`(抬升) | 显式 `CLOSED` | 正确 |

于是右夹爪从 PRIMITIVE 第一步起就被命令闭合,**整段接近与下压都没张开过**,
爪子无法从两侧跨住笔 → 笔从未被夹住(`LIFT rise_m` 仅 **≈4.4 mm**)。

这与 `CLOSE` 子阶段的定义(把右夹爪 **1.0 → 0.0** 插值闭合)直接矛盾 ——
进 `CLOSE` 之前夹爪本应是 1.0(张开)。

### 3.5.2 修复

- `_prim_vertical_align` / `_prim_press` 改为 `grip=GRIP_RIGHT_OPEN`;
- `_step_right_q` 的**默认值改为 `GRIP_RIGHT_OPEN`** 并写明教训 ——
  这样"忘记传 `grip`"的后果从"夹不住笔"退化为"多张开一会儿",不再致命。

### 3.5.3 修复前后的真机对照

| 指标 | 修复前(11 轮) | **修复后(seed 39)** |
| --- | --- | --- |
| 官方 `task_success` | **0 / 11** | **1 / 1 = 100%** |
| `LIFT rise_m` | ≈4.4 mm(没夹住) | 正常抓取 |
| 录像文件名 | `episode_000_seed_39_fail.mp4` | **`episode_000_seed_39_success.mp4`** |
| 混合流程消耗步数 | 341 / 350 | 正常 |
| `constraint_ok`(TCP 的 X 轴垂直桌面) | — | **True**,`tcp_x_axis_deviation_deg = 12.958°` |
| `holder_locate` 筒口中心 | — | `[0.64310, 0.10279, 0.89382]`(双腕 VLM 置信 0.98) |

> ⚠️ 修复后只跑了 **1 集**(seed 39)。**1/1 ≠ 100% 的成功率估计**,
> 真实成功率需要多集统计。此前 10 集无偏随机种子的运行(0/10)是在**修复前**的
> 代码上跑的,不能用来描述修复后的版本。

---

## 4. 与官方基线的对照

官方发布的 `evaluation_results/released_checkpoint_results.json` 只包含 5 个任务
(click_bell / drawer_open_place / mixer_operating / table_rearrangement /
water_pouring)的 100 episode 结果,ACT 五任务宏平均 55.6%。
**`handle_basket` 没有官方公布的基线数字。**

`click_bell` 的官方 ACT 基线是 **37%(100 episode,平均 259.43/361 步)**。
本提交包对这个任务新增了一条混合流程(解析下压 + ACT 填充),**没有**该混合流程
的任何成功率数据:它在本机从未跑过(见 §5.2)。

我们在自有环境(单卡 4090,`--act-chunks 6`)测得的 `handle_basket` 结果是
**12/93 = 12.9%**,成功 seed 为 100/124/130/139/140/157/160/163/172/175/177/198。
该数字是在 **Codex CLI 作为 VLM** 的条件下取得的;本提交包改用自托管 VLM 端点,
**该成功率不可直接外推**。

---

## 5. 验证状态(已做到哪一步 / 还差什么)

打包与验证所在的机器**没有可用的 GPU**:`/dev/nvidia*` 不存在,
`torch.cuda.is_available() == False`,`nvidia-smi` 无法与驱动通信。
因此**没有任何一个 episode 在封装后的代码上跑过**。

### 5.1 已经完成的验证(有真实执行证据)

在一个本机的独立 venv 中安装了**真实的 `lerobot==0.4.4`**(复用已有 torch 2.7.0),
用 `submission_hf/ACT_sim_handle_basket/` 里那份**即将上传到 Hugging Face 的权重**
做了端到端加载与前向:

| # | 验证项 | 结果 |
| --- | --- | --- |
| 1 | `ACTPolicy.from_pretrained(<迁移后 checkpoint>)` | **成功**,51.59 M 参数,`chunk_size=50`、`n_action_steps=50` |
| 2 | `make_pre_post_processors(pretrained_path=...)` | **成功**,7 个特征(6 输入 + 1 动作)**统计量一个不缺** |
| 3 | 键名修复是否生效(迁移的最大坑) | **生效**:0.5 灰度图归一化后为 **0.2266**,不是 0.5 |
| 4 | `select_action` + postprocessor | 输出 **shape `(1, 14)` 且全部有限**,动作值量级合理 |
| 5 | 本包 `ACTPolicyWrapper` 加载同一 checkpoint | **成功**,`normalization_path == "external_0.4.4"`、`action_dim == 14`、`act_step == 50` |
| 6 | `select_chunk_action` 的 action queue 语义 | **正确**:第 1 次 `ran_inference=True`,第 2 次 `False`(队列复用),`reset()` 后恢复 `True` |
| 7 | `drain_inference_times()` | 返回真实样本(CPU 单次前向 ≈0.49 s) |
| 8 | `policy/hybrid/robosyn_hybrid/scripts/smoke_imports.py` 预检 | **0 failures**:适配器契约、settings、相机安装矩阵正交且 det=+1、10 任务映射与 HF 链接、固定 revision 全部通过 |
| 9 | 静态集成校验(AST) | 跨模块 import 11/11、懒加载属性 13/13、调用签名 13/13、17 状态 ↔ 17 个 `_stage_*` 一一对应 |
| 10 | `items_handover` 混合流程的静态/逻辑校验 | `py_compile` 通过;临时 AST 校验脚本逐条核对 `handover.py` 对 `stereo/vlm/grasp/priors/act_wrapper/flow` 的 **45 个调用点**实参个数与关键字名、**13 状态 ↔ 13 个 `_stage_*`** 与 **7 子阶段 ↔ 7 个 `_prim_*`** 一一对应、**42 个 `self._*` 调用**全部可解析(其中 12 个继承自 `HandleBasketFlow`)、四方法签名与父类逐字一致、步数预算合计 **345 ≤ 350** 且 `TAIL_RESERVE` 与逐状态预算逐项一致;用**纯 numpy 假环境**(假 robot FK/IK + 假 env + 假 ACT + 假 VLM,含桌面接触模型)跑 **5 个场景**:完整链路 12 相位全部走通(流程自用 332 步 + ACT fill 18 = 350)、VLM 不可用 / IK 无解 / 预算 120 / 官方 clear 300 四种情况下都提前降级为纯 ACT 且**从不超步**;校验脚本已删除 |
| 11 | `click_bell` 混合流程的静态/逻辑校验 | `py_compile` 通过;AST 校验 `clickbell.py` 对 stereo/vlm/grasp/flow/act_wrapper 的 **27 个调用点**实参个数与关键字名逐条匹配、**11 状态 ↔ 11 个 `_stage_*`** 一一对应、`_servo_qpos` 签名与 SPEC 一致;用**纯 numpy 假环境**(假 FK/IK/立体/VLM)干跑,验证状态迁移、`_servo_qpos` 补偿、press 早停、`max_env_steps` 截断与全部降级路径 |

第 1–4 项直接验证了**即将提交的 HF 产物的可用性**;第 5–7 项验证了**七个非混合任务
所依赖的那一半代码**;第 10、11 项只覆盖 `items_handover` / `click_bell` 两条混合流程的
**Python 层逻辑**,不涉及几何量。
第 1–7、10、11 项有真实执行证据,不再是"仅语法检查"。

### 5.2 仍然没有验证的

| 未验证项 | 原因 | 怎么结案 |
| --- | --- | --- |
| `handle_basket` 混合流程任何一个 episode | 需要 GPU(仿真渲染 + `grasp.py` 的 FK/IK 走 `.cuda()`) | `docs/REPRODUCE.md` §6 冒烟 |
| `click_bell` 混合流程的**几何**:按钮反投影精度、APPROACH 停位、PRESS 是否真能压到 4.8 mm | 需要 GPU + 仿真器(渲染/深度/FK/IK 全走 `.cuda()`) | 有 GPU 的机器上跑 `bash policy/hybrid/eval.sh click_bell random smoke_clickbell 0 --max_episodes 1 --eval_expert_check false`,看 `HYBRID_DEBUG_DIR` 下 `button_estimation.json` / `press_ik.json` / summary 里的 `approach`+`press` |
| `items_handover` 混合流程的**几何与放置**:垂直对齐/接触判据在真机上的表现、双腕内圈椭圆的筒口 3D 精度(参考实现实测 3.17 mm)、放置时笔的入筒姿态 | 需要 GPU + 仿真器(渲染 + FK/IK + 真实接触物理) | 有 GPU 的机器上跑 `bash policy/hybrid/eval.sh items_handover random smoke_handover 0 --max_episodes 1 --eval_expert_check false`,看 `HYBRID_DEBUG_DIR` 下 `handover_holder_locate.json` / `handover_final_place.json` 与 summary 里的 `primitive` / `holder_locate` / `step_budget` |
| `items_handover` 的成功率 | 需要 GPU + 仿真器。**已有的参考量级**:10-02 版在 `harness/logs/20261007_oct2_test` 上 5 局 2 成功 = **40%**;10-06 ikfix 版在 batch50 上 6 局 2 成功 ≈ 33%(两者可用日志里 `prior_failures` 是否出现来区分版本)。**样本都很小,只作量级参考** | 同上,跑多局统计;注意参考实现已记录的已知物理缺陷(笔在爪内滑转 ~63° → 倾斜入筒口磕壁弹出),本次封装**不试图修复** |
| `verify_mount` 对真实仿真器的判别 | 同上 | 已在 RTX 4090 上完成,见 §2.5(**已结案**:FK 返回 tool 帧,`FK_TO_LINK6` 已内置);换检出时需重跑 L1 |
| `stereo.fetch_depth` 的深度精度 | 需要 FoundationStereo + GPU | 冒烟时看 `HYBRID_DEBUG_DIR` 下的 depth 产物 |
| `vlm.py` 对真实 VLM 端点的请求/解析 | 需要起一个 OpenAI 兼容端点 | `docs/REPRODUCE.md` §5 的 curl 验证 + 冒烟 |
| SAM2 掩码(`ultralytics` 未安装在验证环境) | 依赖未装 | `pip install -r policy/hybrid/robosyn_hybrid/requirements/robosyn-extra.txt` |
| 九个官方 checkpoint 的实际下载 | 验证机无法访问 `huggingface.co` | 用 `HF_ENDPOINT` 镜像 |

**结论:这份包在"HF 权重 + ACT 适配器"这条链路上已被真实验证;
三条混合流程(handle_basket / click_bell / items_handover)仍是未在真机上运行过的
代码(items_handover 与 click_bell 只有 Python 层假环境干跑)。** 提交前请在有 GPU 的机器上按
`docs/REPRODUCE.md` §6 各跑一次冒烟。§2.5 的 FK 帧判定已在真实 GPU 上结案
(实测 0.280 m / 180°,修正矩阵已内置),但**任何新的 EmbodiChain 检出都应用 L1
重测一次**(见 §2.5)。

