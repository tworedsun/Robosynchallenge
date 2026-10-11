"""items_handover 混合策略状态机(WS-H)。

把已验证的 handover 方案(参考包 ``handover_backup_20261010/`` 的工作目录版)
封装成官方 ``policy/<name>/`` 三函数契约能跑的第三个混合流程,接口与
:class:`robosyn_hybrid.flow.HandleBasketFlow` / ``clickbell.ClickBellFlow``
**逐字一致**::

    __init__(self, act_policy, settings, *, act_chunks=6, max_env_steps=350,
             debug_dir=None)
    reset(self, env=None) -> None
    eval(self, env, obs) -> (obs, info, truncated, list[float])
    summary(self) -> dict

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
基线溯源(**用户 2026-10-10 最终确认:基线 = 10-02 版**)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

移植时实际阅读的参考文件,全部在
``handover_backup_20261010/src/ours_place_binocular_20260922/`` 工作目录下:

==============================  =======================================  ==========================================
文件                            md5 / 大小                               移植时用到的东西
==============================  =======================================  ==========================================
``test_grasp_primitive.py``     ``574d3be9aa8af0c9``(51809 B,10-07)       相位顺序、各相位步数、continuation/takeover/release/manual_target 的动作覆盖语义
``place_above_release.py``      ``e6b7acf56dee3553``(22729 B,10-02 版)    放置原语的**接口语义**(见下面的"已知限制")
``place_above_release2.py``     同目录,批处理实际调用的简版                 放置的**实际执行路径**:筒口正上方 5 cm + 多起点 IK(姿态见 §约束)
``my_place_flow.py``            同目录                                     batch 实际跑的编排(步数账 346 的来源)
``holder_vlm.py``               ``067337df95b35d3a``(9336 B)             holder VLM 的**归一化坐标约定**与提示词
``holder_inner_ellipse.py``     —                                        内圈椭圆拟合(top_frac / 长短轴比 / 内缩 0.95)
``holder_stereo_inner_eval.py`` —                                        双腕内圈三角化(P = K[R^T | -R^T t])
``manual_target_primitive.py``  —                                        固定路点的 smoothstep 插值语义
``manual_targets/*.json``       —                                        固定 14 通道路点(见 ``MANUAL_TARGET_QPOS``)
==============================  =======================================  ==========================================

**版本考证(用户已核实,写在这里便于日后追溯)**

* **本移植的基线 = 09-28 版**(用户指定):
  ``test_grasp_primitive.py`` 取 **09-28 13:35** 那份(46375 B,md5 ``5e51a0dc``,
  与 ``backups/backups_20260929/before_next_edit/`` 里那份逐字节相同);
  ``place_above_release.py`` 取 09-28 当日在用的那份(**15730 B**,md5 ``1348155e``,
  即 ``backups/backups_20260927/place_above_release_before_fixed_prior.py``)。
  判定依据:09-28 13:57 启动的 ``20260928_seed20_29`` 批次结果里
  ``IK_FREE_FALLBACK`` **0 命中**、也没有 ``fixed_wrist_prior`` 那套,
  说明当时还没有 10-02 的 freeik fallback。
* ⚠️ **那次批次的成功率是带特权信号测出来的,不能当作本移植的成绩。**
  该批次是 10 个 seed(20–29)**8 局成功 / 10 = 80%**(逐 seed、允许重试),
  首次尝试 8/15 ≈ 53%;而它的结果字段(``target_tilt_deg`` / ``target_roll_deg`` /
  ``target_offset_m``)全部来自 ``_vertical_target_candidates`` —— **那个函数读
  ``pen = _pose(<env>, "pen")`` 笔真值**(该真值辅助函数已按 SPEC §3 移除,
  本文件没有对应实现)。参考实现的另一条路
  ``_place_with_official`` / ``_place_with_joint_constraints`` 用的是
  ``expert_actions = base`` 的**官方专家轨迹生成接口**(``create_demo_`` 开头的
  那个公开 API,完整名字见参考实现)。
  **也就是说参考的三条放置路径全都依赖特权信息,任何合规移植都无法复用它们,
  其成功率也无法携带过来。** 本文件因此走纯几何的
  ``place_above_release2`` 路径(见下)。
* 其余历史副本(09-27 / 09-29 / 10-02 / 10-06)与本移植**行为无关**:候选族与
  fallback 链整条都没有移植(它们的函数都要读 ``pen`` 真值),所以换任何一份早期
  副本都不会改变本文件的行为;记录在此只为溯源。
* 10-02 版另有一份实测数据(``harness/logs/20261007_oct2_test``,5 局 2 成功 = 40%,
  失败全部停在 ``stage=constraints``),同样属于**带特权信号的参考数字**。
* **已知限制(如实保留,不要"顺手修好")**:参考实现的
  ``place_above_and_release`` 在 ``expert_actions is not None`` 时会从
  ``_place_with_official`` / ``_place_with_joint_constraints`` **直接 return**,
  永远走不到后面的竖直接近候选扫描(10-06 addendum 自己记录过这一点)。本移植
  **同样不做候选族兜底**:筒口正上方的多起点 IK 无解就直接记 failure → 降级为
  纯 ACT,不尝试其它姿态族。
* 参考 ``test_grasp_primitive.py`` 会尝试读 ``final_place_result[
  "post_release_task_state"]`` 快照,而 10-02 版不产生该键,于是落到 ``post_hoc``
  分支。**这对提交无影响**:官方 ``scripts/eval_policy.py`` 自己调
  ``is_task_success()`` 判定成功,我们的适配器**不做任何自评估**,因此移植版
  **不实现**该快照/自评估逻辑。

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
版本相关行为(**这些常量是唯一的"换基线时要复核"的开关**)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

参考实现存在多个时间点的工作副本,且部分改动被回退过;因此把"行为可能随基线
变化"的地方全部收敛成下面这些模块级常量,换基线时只改这一处、并按注释复核:

============================  =========================================  ==========================
常量                           当前取值(工作目录版执行时的行为)             换基线时需要复核的点
============================  =========================================  ==========================
``CHUNK1_STEPS``              50(``CHUNK_STEPS``/``act_step``)            两个版本一致
``PRIMITIVE_SUBSTAGE_BUDGET`` 4/20/6/9/3/3/10(接触触发的 PRESS ≤6)     原语步数由 contact 决定,版本间一致
``LEFT_EXTEND_*``             线性插值 8 步                                有的副本用 smoothstep 插值
``MANUAL_TARGET_QPOS``        固定 14 通道路点 35+8 步                    路点来源见该常量注释
``PLACE_*``                   48 步 = 移动 20 + 保持 10 + 释放 9 + 稳定 9  有的副本是 25+9(34 步)或 20+9+15(44 步)
``PLACE_CLEARANCE_M``         0.11 m(筒口正上方),姿态由下一条常量决定        **WS-K**:原版是 0.05 m 且"保持当前 TCP 姿态";高度族见 §WS-K
``PLACE_CLEARANCE_LADDER_M``  0.17 / 0.05 m(11 cm 之后的后备高度)           **WS-K**:200 万样本 FK 扫描定的优先级
``PLACE_TCP_X_WORLD``         ``(0,0,-1)``,由 ``HYBRID_PLACE_TCP_X`` 覆盖   **WS-I 新增**:见 §放置姿态约束
============================  =========================================  ==========================

**移植选的是"批处理实测那一版"的行为**:``place_above_release2.py`` 的
"到筒口正上方、多起点 IK、直接释放"那条路径
(它不读物体真值,也不需要候选族)。**有意替换的两处**:原版的"保持当前
TCP 姿态"在 WS-I 里换成了**显式姿态约束**(TCP 的 X 轴垂直于桌面,见下面
"放置姿态约束"一节);原版的 5 cm 接近高度在 **WS-K** 里换成 11 cm(见下面
"WS-K:接近高度根因"一节);其余部分(多起点 IK、步数账、释放时序)逐行不变。
参考包里另一条 ``place_above_release.py``
的**候选族 + fallback 链**只在 ``IK_FREE_FALLBACK`` 打开时才会跑,而且它的
两个函数都要读 ``pen`` 真值,所以**整条链都没有移植**;换基线到更早的副本
(没有 freeik fallback)对本文件**没有影响** —— 这一点已按版本差异逐条核对过。

``eval()`` 语义、降级语义(任何阶段失败 → ``FALLBACK_ACT`` 纯 ACT 跑完剩余
步数,绝不外抛)、ACT 计时(官方 ``policy.inference_timing``,只计前向)全部
继承自 ``flow.HandleBasketFlow`` —— 本类只替换状态与阶段实现,因此两个流程
的对外行为逐条同构。

相位顺序(照 SPEC §2 的 ``phase["name"]`` 顺序)::

    SETTLE → CHUNK1 → PRIMITIVE → ALIGNMENT → CONTINUATION → LEFT_TAKEOVER
    → RIGHT_RELEASE → MANUAL_TARGET → HOLDER_LOCATE → FINAL_PLACE → DONE
    → ACT_FILL

.. warning::

   **步数预算与合计**(官方 ``configs/items_handover/random/gym_config.json``
   的 ``max_episode_steps`` 是 **350**,``clear`` 是 **300**;参考流程真实用量
   346,只剩 4 步余量。本移植**不新增任何步数**,并且把每个状态的步数写成
   硬上限,合计 345):

   ==============  ====  =====================================================
   状态             步数  说明
   ==============  ====  =====================================================
   SETTLE             0  纯计算:记录左右臂关节基线
   CHUNK1            50  1 个 ACT chunk(``act_step``=50,与参考 ``CHUNK_STEPS`` 同值)
   PRIMITIVE         55  LOOK_PEN ≤4 + VERTICAL_ALIGN ≤20 + PRESS ≤6
                         + CLOSE 9 + SQUEEZE 3 + PAUSE 3 + LIFT 10
   ALIGNMENT         20  回到 CHUNK1 末态(右夹爪保持闭合,右夹爪通道强制 0.0)
   CONTINUATION     100  2 个 ACT chunk;右夹爪通道锁死 0.0
   LEFT_TAKEOVER     20  EXTEND 8 + CLOSE 9 + HOLD 3
   RIGHT_RELEASE      9  左夹爪 0.0(握住)、右夹爪 1.0(张开)
   MANUAL_TARGET     43  35 步 smoothstep 到固定路点 + 8 步保持
   HOLDER_LOCATE      0  纯计算:VLM(归一化坐标)+ 内圈椭圆拟合 + 双目三角化
   FINAL_PLACE       48  移动 20(到筒口正上方 11 cm,**TCP X 轴 ⊥ 桌面**;
                          位置容差先 10 mm 后 20 mm,如实记录)
                         + 释放前保持 10 + 释放 9 + 稳定 9
   DONE               0  收尾
   ACT_FILL           —  剩余步数交给纯 ACT(不额外新增步数)
   ==============  ====  =====================================================

   **合计 = 0+50+55+20+100+20+9+43+0+48+0 = 345 ≤ 350**(余量 5)。

   参考实现子阶段实测值(``AI_HANDOFF_ADDENDUM_20261006.md``:chunk 50 +
   primitive ≈53 + align 20 + continuation 100 + takeover 20 + release 9 +
   manual target 43 + placement 48 ≈ 346)。本移植把 PRIMITIVE 的硬上限放宽
   到 55(给接触触发的 PRESS 多留 2 步),其余逐项相同,因此总量 345 仍
   **少于**参考的 346,没有新增步数。

   每个相位的让步数由 :func:`HandoverFlow._state_cap` 动态夹紧:允许量 =
   ``max_env_steps - 已用步数 - TAIL_RESERVE[该相位]``,其中 ``TAIL_RESERVE``
   是"该相位之后所有相位还需要的步数"(见 :data:`TAIL_RESERVE`,静态校验脚本
   逐项核对了它等于逐状态预算之和)。因此:

   * 绑定环境时只读地探测环境自带的 ``max_episode_steps`` 并**只降不升**地收紧
     上限(``random`` 350 / ``clear`` 300),**不做任何写入/补丁**;
   * 预算不足时**从最早的相位开始压缩**并记一条 ``degradation``;若某相位可用量
     降到 0,则**立刻降级为纯 ACT** 跑完剩余步数 —— 宁可少做动作,也绝不撞官方
     gym 的 TimeLimit(撞上会触发环境重置,该任务直接判 0);
   * ``eval()`` 在 ``_steps_taken >= max_env_steps`` 时立即返回
     ``truncated=True``,并在 ``summary()["step_budget"]`` 里留下
     ``exhausted / exhausted_reason / compressed_states`` 三条记录。

   **任何情况下总步数都不会超过 ``max_env_steps``(默认 350)。**

.. note::

   **两处真值读取已移除(SPEC §3)**

   1. 参考脚本里名为 ``pen_pose()`` 的辅助函数(读 ``pen`` 刚体真值位姿,用于
      "笔还在爪里吗"的复核;SPEC §3 表格第 1 行)。
      **无法在不读真值的前提下做该检查,已省略**。移植版只保留原语自身的
      本体感受信号:垂直对齐的角速度/下降量、接触触发下压是否成功、夹爪是否
      在接触位姿闭合并保持、以及 LIFT 阶段 FK 测得的 TCP 抬升量。这些信号
      都不包含被夹物体的位姿,因此"笔是否随爪上升"这个判据被**删除**;
      ``summary()["grasp"]["pen_in_gripper_check"]`` 明确记录 ``"omitted"``。
      两个参考副本里这个函数的**定义与调用点完全相同**(逐字节同位置),
      所以该省略与基线选择无关。
   2. 参考放置模块里名为 ``_pose(<env>, uid)`` 的辅助函数(读任意刚体真值;
      SPEC §3 表格第 2 行)。移植版**删除**;放置目标只由
      :meth:`HandoverFlow._stage_holder_locate` 的双目三角化估计给出
      (公开相机内参 + FK 腕相机位姿 + VLM 归一化坐标 + 内圈椭圆拟合),
      不使用任何物体真值。该函数在较新的副本里还被 "free_orientation"
      fallback 候选族复用,因此移植版**连候选族一起不移植**(见上面的版本表)。

.. note::

   **holder VLM 的归一化坐标约定(必须保留)**

   参考实现实测(``AI_HANDOFF_ADDENDUM_20261006.md`` §"2026-10-07 补充"):
   "VL 模型回答的像素坐标在它自己视觉编码器的空间里,bbox 会越界且错位
   100+px —— 必须要求归一化 0..1 坐标再由代码换算回像素"。换算回像素后与
   deepseek 基线只差 ~5px,腕相机双视图内圈三角化的筒口 3D 误差 **3.17 mm**。

   因此本文件的 holder / pen 提示词**一律要求 0..1 归一化坐标**
   (:data:`HOLDER_PROMPT` / :data:`PEN_PROMPT` 里的 ``bbox_xyxy_norm`` /
   ``center_uv_norm`` / ``pen_uv_norm``),再由 :func:`_norm_to_pixels` 用
   实际图像宽高换算回像素。兼容分支也照抄参考实现:如果模型仍然回像素
   (值 >1),原样使用,但归一化路径才是主路径。

.. note::

   **几何的复用与替换**

   * 腕相机世界位姿 = ``stereo.wrist_camera_pose``(FK + ``priors.camera_mount``,
     已含实测解出的 ``FK_TO_LINK6`` tool→link6 帧补偿);
     相机内参 = ``priors.camera_K``(公开 ``calibration.json``,
     与参考 ``holder_stereo_inner_eval.py`` 的 K 逐值相同)。
   * 参考 ``camera_projection.py`` 自己拼的 ``MOUNT_POS/MOUNT_QUAT_WXYZ`` 与
     本仓库 ``calibration.json`` 的 ``cam_*_wrist`` 条目是同一组公开常量,
     移植版统一走 ``priors``/``stereo``,不再重复硬编码。
   * 双臂 FK/IK 用本文件的 :meth:`_fk` / :meth:`_ik`(device-aware,逐行对应
     参考 ``test_grasp_primitive.left_ik`` 与 ``place_above_release2.solve_ik``)。
     没有直接调用 ``grasp.py`` 的 ``.cuda()`` 硬编码 helper,这样同一段几何
     在无 CUDA 的静态检查环境里也能跑通(参考实现本身也是 device-aware)。

.. note::

   **放置姿态约束(WS-I)与其假设风险(必须如实读完再改动)**

   用户 2026-10 的要求原话:"因为我夹笔的时候基本上交接手之后是笔是垂直于
   那个夹爪的,所以我只需要保证夹爪平行于桌面就行了,也就是 TCP 的 x 轴垂直
   于桌面。你如果想要直接看笔的位置的话,那肯定要读真值啊,这肯定是有问题的"。

   实现:``FINAL_PLACE`` 不再沿用参考 ``place_above_release2`` 的"保持当前 TCP
   姿态",而是由 :meth:`HandoverFlow._place_target_pose` 显式构造目标姿态 ——
   **TCP 的 X 轴对齐** :data:`PLACE_TCP_X_WORLD`(默认 ``(0,0,-1)``,垂直于
   桌面;可用环境变量 ``HYBRID_PLACE_TCP_X`` 覆盖),**TCP 的 Z 轴落在桌面平面
   内**(由当前 TCP 的接近方向投影得到,尽量少改手腕位形)。整个过程只用到
   ``center``(VLM + 双目估计)与**自己的 TCP 位姿**(``compute_fk``,本体感受),
   **不读任何物体真值**;自检字段
   ``summary()["final_place"]["tcp_x_axis_deviation_deg"]`` /
   ``["tcp_x_angle_to_world_z_deg"]`` 同样只读自己的 FK,与官方判据
   ``items_handover.py::_is_fall_x``(看笔的 X 轴)无关。

   **两个"夹角"字段的符号约定(别读成 bug)**:SPEC §2.4 的有符号公式
   ``angle = arccos(tcp_x · +Z)`` 对默认的 ``PLACE_TCP_X_WORLD = (0,0,-1)``
   会读到 **180°**(X 轴朝下),而物理要求"垂直于桌面"只要求 X 轴与世界 Z 轴
   **共线**,朝上朝下都算,所以判定用的是**轴偏差**
   ``tcp_x_axis_deviation_deg = min(angle, 180 - angle)``(0° = 完全垂直);
   ``constraint_ok`` 就是这条轴偏差 < **15°**,不是有符号角 < 15°
   (否则默认朝向下会永远判 False,并刷出无意义的 degradation)。
   官方 ``_is_fall_x`` 用的是**同一个有符号公式**,阈值 **1.309 rad ≈ 75°**,
   即它要求物体的 X 轴**朝上**;是否与"用户假设"一致取决于下面那条未验证的
   几何关系。

   ⚠️ **假设(尚未在 GPU / 仿真器上验证)**

   > **假设**:笔在交接后**垂直于夹爪**,因此 TCP 的 X 轴垂直于桌面
   > ⟹ 笔的长轴接近竖直。
   >
   > **实测数据对此有保留**:日志 ``manual_target_pose_debug.json`` 的
   > ``pen_in_tcp`` 显示(4 组实测:09-27 seed_0/seed_1、10-06 seed_38、
   > 09-29 seed_38),笔在 TCP 系里的长轴更接近 **TCP 的 Z 轴**
   > (``0.976`` / ``0.914`` / ``0.860`` / ``0.867``),而沿 **TCP X** 的分量
   > 只有 ``0.216`` / ``0.406`` / ``0.511`` / ``0.498``;反倒是笔的**横轴**
   > ≈ TCP 的 X 轴。也就是说:夹爪平行于桌面确实会让笔朝竖直方向附近靠,
   > 但"TCP X ⊥ 桌面 ⟹ 笔竖直"严格来讲是**用户的几何假设**,不是本仓库
   > 已实测的结论。若该 seed 上笔长轴≈TCP Z 的关系成立,那么"TCP 平行于
   > 桌面"反而会让笔接近**水平**,官方 ``_is_fall_x``(阈值 75°)会判"倒下"。
   > 另外默认的 X 轴方向 ``(0,0,-1)`` 让 TCP 的 X 轴**朝下**;若笔长轴≈TCP X,
   > 则笔的 X 轴也朝下,官方有符号判据同样会判"倒下" —— 这正是需要在 GPU 上
   > 用真值诊断复核的那一条(可用 ``HYBRID_PLACE_TCP_X=0,0,1`` 试反朝向)。
   >
   > **验证方法**:冒烟时先看 ``summary()["final_place"]
   > ["tcp_x_axis_deviation_deg"]``(以及有符号的
   > ``["tcp_x_angle_to_world_z_deg"]``) —— 它只能证明**我们自己的约束达成
   > 了**(``constraint_ok`` 阈值 **15°**),以及多起点 IK 是否真的满足了姿态
   > (``tcp_x_alignment_to_place_axis_deg`` 应接近 0;注意
   > :meth:`HandoverFlow._solve_left_pose` 只按**位置**残差选解);笔的**实际**
   > 姿态必须另用**允许读真值的诊断脚本**复核。两者一致则假设成立,不一致则以
   > 笔的真值诊断为准,**不要用本文件的 ``constraint_ok`` 冒充任务成功**。
   >
   > **如实报告**:``constraint_ok == False`` 时本文件记一条 ``degradation``
   > (``summary()["degradations"]``),**不谎报成功**;此外本流程从不做
   > 自评估,任务成败由官方 ``scripts/eval_policy.py::is_task_success()``
   > 判定。

.. note::

   **WS-K:接近高度根因(2026-10-11,只改接近高度与位置容差)**

   之前一直解不出 IK,并沿"放宽俯仰(±8/15/25/35°)"调 —— **方向错了**。
   200 万关节角批量 FK 扫描(左臂,固定 seed 39 的筒口目标点邻域)给出的判决
   (列 = 该邻域内能达到的**最小** TCP X 轴轴偏差,<15° 即达标):

   ==================  ==========  ========  =========  =========
   目标变体              最近可达     10 mm     20 mm      50 mm
   ==================  ==========  ========  =========  =========
   ``as_is_5cm``(原)     3.93 mm     26.06°   **26.06°**  9.209°
   ``raise_60mm``(11 cm) 8.92 mm     26.50°   **9.209°**  2.378°
   ``raise_120mm``(17 cm) 3.33 mm    17.89°   **11.88°**  2.061°
   ``shift_+y_100mm``     6.42 mm     41.76°    36.72°     9.215°
   ==================  ==========  ========  =========  =========

   结论:姿态本身不缺,**缺的是够得着的高度**。所以:
   :data:`PLACE_CLEARANCE_M` = **11 cm**(第一优先),
   :data:`PLACE_CLEARANCE_LADDER_M` = (17 cm, 5 cm) 依次后备;
   :data:`PLACE_POSITION_TOL_LADDER_M` = (10 mm, 20 mm) 先严后松;
   实际用的 clearance / 容差**如实**写进 ``final_place.accepted_variant`` 与
   ``final_place.position_tol_mm``。姿态门槛
   (:data:`PLACE_TCP_X_TOL_DEG` = 15°)不删,``constraint_ok`` 照旧如实记录。

   **代价(必须如实知道,不要只写好处)**:

   * 笔从 **11 cm**(而不是 5 cm)高处落下:空中时间更长、落地速度更大,
     落点姿态更容易翻/跳/弹出筒外 —— 这是为"TCP X 轴垂直桌面"付的代价,
     用户 2026-10-11 明确接受("只要笔能放进去就行,管他有多高")。
   * 11 cm 上只能换到 **~9° 轴偏差 + 10~20 mm 位置残差**,不是 0°/5 mm;
     参考实现的 5 cm/姿态不限在几何上"更稳",但姿态做不到。
   * 20 mm 容差会让落点横向偏差比原实现大;若笔仍进不去,先查
     ``final_place.residual_mm`` / ``constraint_ok`` 与落点录像,不要改判据。

   用户给出的优先级(2026-10-11):**第一优先 = 官方成功判据**(笔没倒 &
   筒没倒 & 笔筒 xy 距离 ≤ 30 mm & 笔筒 AABB 的 Z 重叠 > 80 mm);
   **第二优先 = TCP X 轴垂直桌面(<15°)** —— 后者是手段,前者是目的。
   因此万一"姿态达标但笔进不去",说明"TCP X 竖直 ⟹ 笔竖直"这个几何假设在该次
   抓取上不成立(实测有约 40% 的样本笔长轴更靠近 TCP Z 而非 TCP X),
   **如实记录,不要为了 console 好看而谎报**。

合规:只用 RGB、机器人本体感受(``robot.get_qpos`` / FK / IK)与官方公开
常量;轨迹与候选搜索只在 ``debug_dir`` 非空时落盘。
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .flow import (
    HandleBasketFlow,
    _lazy,
    _safe_rgb_of,
    _timing_device,
    full_action,
    jsonable,
    npv,
)

__all__ = ["HandoverFlow"]

# ---------------------------------------------------------------------------
# 官方常量与流程参数(数值全部照抄参考实现 / 官方 config,一个都不改)
# ---------------------------------------------------------------------------

#: 官方三路 RGB 相机(高视角 + 左右腕视角)。
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")

LEFT = "left"
RIGHT = "right"
LEFT_ARM = "left_arm"
RIGHT_ARM = "right_arm"

#: 官方 14 维动作布局 ``[left_arm(6), left_eef, right_arm(6), right_eef]``。
#: 只在 ``active_joint_ids`` 推导出的通道下标不可用时作为兜底(见 ``_eef_channel``)。
LEFT_EEF_CHANNEL_DEFAULT = 6
RIGHT_EEF_CHANNEL_DEFAULT = 13

#: 夹爪指令语义(参考实现 / 官方动作库一致):0.0 = 闭合,1.0 = 张开。
GRIP_LEFT_CLOSED = 0.0
GRIP_LEFT_OPEN = 1.0
GRIP_RIGHT_CLOSED = 0.0
GRIP_RIGHT_OPEN = 1.0

#: 官方 ``configs/items_handover/*/gym_config.json`` 的步数上限(公开常量):
#: ``random = 350``,``clear = 300``。两者都是**硬上限** —— 撞上去会让官方 gym
#: 的 TimeLimit 截断并重置环境,该 episode 直接判 0。
#: 本文件只把 350 当作 ``max_env_steps`` 的**默认值**;真正生效的上限会在
#: 绑定环境时用 :meth:`HandoverFlow._probe_env_step_limit` **只降不升**地夹一次
#: (``clear`` 设置下自动变成 300),所以既不打任何补丁,也不会超步。
OFFICIAL_STEP_LIMIT_RANDOM = 350
OFFICIAL_STEP_LIMIT_CLEAR = 300
OFFICIAL_STEP_LIMIT = OFFICIAL_STEP_LIMIT_RANDOM

#: 各状态步数预算(= docstring 里那张表)。
CHUNK1_STEPS = 50
PRIMITIVE_BUDGET = 55
ALIGNMENT_STEPS = 20
CONTINUATION_BUDGET = 100
LEFT_TAKEOVER_BUDGET = 20
RIGHT_RELEASE_STEPS = 9
MANUAL_TARGET_BUDGET = 43
FINAL_PLACE_BUDGET = 48

#: PRIMITIVE 子阶段顺序与各自的硬上限(合计 = PRIMITIVE_BUDGET = 55)。
PRIMITIVE_SUBSTAGES = (
    "LOOK_PEN",
    "VERTICAL_ALIGN",
    "PRESS",
    "CLOSE",
    "SQUEEZE",
    "PAUSE",
    "LIFT",
)
PRIMITIVE_SUBSTAGE_BUDGET: dict[str, int] = {
    "LOOK_PEN": 4,
    "VERTICAL_ALIGN": 20,
    "PRESS": 6,
    "CLOSE": 9,
    "SQUEEZE": 3,
    "PAUSE": 3,
    "LIFT": 10,
}

#: 每个状态**之后**的那些相位还需要多少步(用于反推当前相位的可用步数上限)。
#:
#: 例:``PRIMITIVE`` 的可用步数 = ``max_env_steps - 已用步数 - 240``,其中 240 是
#: PRIMITIVE 之后所有相位的预算和。这样任何一个相位都不可能把后面的相位
#: "饿死",也就永远撞不到官方 350 步的 gym TimeLimit(撞上会导致环境重置,
#: 该任务直接判 0)。
TAIL_RESERVE: dict[str, int] = {
    "SETTLE": (
        CHUNK1_STEPS
        + PRIMITIVE_BUDGET
        + ALIGNMENT_STEPS
        + CONTINUATION_BUDGET
        + LEFT_TAKEOVER_BUDGET
        + RIGHT_RELEASE_STEPS
        + MANUAL_TARGET_BUDGET
        + FINAL_PLACE_BUDGET
    ),
    "CHUNK1": (
        PRIMITIVE_BUDGET
        + ALIGNMENT_STEPS
        + CONTINUATION_BUDGET
        + LEFT_TAKEOVER_BUDGET
        + RIGHT_RELEASE_STEPS
        + MANUAL_TARGET_BUDGET
        + FINAL_PLACE_BUDGET
    ),
    "PRIMITIVE": (
        ALIGNMENT_STEPS
        + CONTINUATION_BUDGET
        + LEFT_TAKEOVER_BUDGET
        + RIGHT_RELEASE_STEPS
        + MANUAL_TARGET_BUDGET
        + FINAL_PLACE_BUDGET
    ),
    "ALIGNMENT": (
        CONTINUATION_BUDGET
        + LEFT_TAKEOVER_BUDGET
        + RIGHT_RELEASE_STEPS
        + MANUAL_TARGET_BUDGET
        + FINAL_PLACE_BUDGET
    ),
    "CONTINUATION": (
        LEFT_TAKEOVER_BUDGET
        + RIGHT_RELEASE_STEPS
        + MANUAL_TARGET_BUDGET
        + FINAL_PLACE_BUDGET
    ),
    "LEFT_TAKEOVER": (
        RIGHT_RELEASE_STEPS + MANUAL_TARGET_BUDGET + FINAL_PLACE_BUDGET
    ),
    "RIGHT_RELEASE": MANUAL_TARGET_BUDGET + FINAL_PLACE_BUDGET,
    "MANUAL_TARGET": FINAL_PLACE_BUDGET,
    "HOLDER_LOCATE": FINAL_PLACE_BUDGET,
    "FINAL_PLACE": 0,
    "DONE": 0,
    "ACT_FILL": 0,
    "FALLBACK_ACT": 0,
}

# ---- 抓取原语(参考 test_grasp_primitive.py 的 argparse 默认值 / 批处理脚本) ----

#: ``VERTICAL_ALIGN_STEPS`` / ``VERTICAL_DESCENT_M`` / ``VERTICAL_ALIGN_TOL_DEG``
VERTICAL_ALIGN_DESCENT_M = 0.12
VERTICAL_ALIGN_TOL_DEG = 8.0
#: ``PRE_GRASP_X_M`` / ``PRE_GRASP_Y_M``
PRE_GRASP_XY_M = (0.02, -0.005)
#: ``z_descent(step_m=0.020, max_descent_m=0.40, stall_eps_m=0.001, stall_steps=1)``
#: 的等价参数(接触 = 连续 ``stall_steps`` 步 |Δz| ≤ ``stall_eps_m``)。
PRESS_STEP_M = 0.020
PRESS_STALL_EPS_M = 0.001
PRESS_STALL_STEPS = 1
#: 接触判据之二("相对衰减"):位置控制器每步**跟不满**指令步长(2026-10-11 实测:
#: 命令 20 mm/步,实测只走 ≈8.5 mm/步,≈42%),所以绝对阈值 ``stall_eps_m`` 在
#: 这种控制器上几乎永远不会触发。这里再叠一条**相对自身基线**的判据:下降速率
#: 相对本次下压的**峰值速率**(running max,不是"指令的某个比例",因此不受
#: 控制器跟随率影响)骤降到该比例以下,即判为"被桌面挡住"。
PRESS_CONTACT_DECAY_RATIO = 0.30
#: ``CLOSE_STEPS`` / ``SQUEEZE_STEPS`` / ``PAUSE_STEPS``
CLOSE_STEPS = 9
SQUEEZE_STEPS = 3
PAUSE_STEPS = 3
#: ``z_lift(distance_m=0.020, step_m=0.002)`` → 10 步
LIFT_DISTANCE_M = 0.020
LIFT_STEP_M = 0.002

# ---- LOOK_PEN(参考 look_pen.py / look_pen_exec.py 的常量与默认值) ----
#: 腕相机图像里"笔应该出现"的固定像素(640x480 帧,17 帧实测)。
PEN_TARGET_UV = (350.0, 365.0)
PEN_TOL_PX = 100.0
#: ``EXTRA_GAIN`` / ``LOOK_PEN_SCALE_M_PER_PX`` / ``LOOK_PEN_MAX_STEP_M``
LOOK_PEN_SCALE_M_PER_PX = 1.2e-3
LOOK_PEN_MAX_STEP_M = 0.05
LOOK_PEN_MOVE_STEPS = 4
#: LOOK_PEN 用右腕相机(参考 ``look_pen_cycle(camera="cam_right_wrist")``)。
LOOK_PEN_CAMERA = "cam_right_wrist"

# ---- 接管 / 释放(参考 run_single_place_seed.sh 的导出值) ----
LEFT_EXTEND_M = 0.04
LEFT_EXTEND_STEPS = 8
LEFT_CLOSE_STEPS = 9
LEFT_HOLD_STEPS = 3
#: 接管方向的最小模长(``direction_norm >= 0.005`` 才做 extend)。
TAKEOVER_MIN_DIRECTION_M = 0.005
#: 接管方向用的左 TCP 历史长度(参考 ``deque(maxlen=10)``)。
LEFT_TCP_HISTORY_LEN = 10
#: ``LOCKED_RIGHT_GRIPPER`` = 0.0,``LOCK_RIGHT_GRIPPER_STEPS`` = 0 → 全程锁定。
LOCKED_RIGHT_GRIPPER = 0.0

# ---- 放置(参考 place_above_release.py / place_above_release2.py) ----
#: ``PLACE_CLEARANCE_M``:先到筒口正上方 **11 cm**,再开爪。
#:
#: **WS-K 改动(2026-10-11,根因是接近高度,不是放宽俯仰)**:200 万关节角批量
#: FK 扫描(左臂,目标点邻域)显示:5 cm 接近高度在 **20 mm 位置容差**内的最优
#: TCP X 轴轴偏差是 **26.06°**(不达标);把接近高度抬高 6 cm(筒口正上方
#: **11 cm**)后,同一 20 mm 容差内是 **9.209°**(达标);抬到 17 cm 是 11.88°
#: (也达标)。之前沿"俯仰 ±8/15/25/35°"放宽的方向是错的 —— 姿态本身不缺,
#: 缺的是**够得着的高度**。
#:
#: **代价(如实写在这里,不要只写好处)**:笔从 11 cm(而不是 5 cm)高处落下,
#: 空中时间更长、落地速度更大,落点姿态更容易翻/跳、更容易弹出筒外;而且 11 cm
#: 处只能做到 **~9° 的轴偏差 + ~10~20 mm 的位置残差**(不是 0°/5 mm),
#: 这是"TCP X 轴垂直桌面"这条约束在当前工作空间里能换到的极限。
#: 用户 2026-10-11 明确接受该代价("只要笔能放进去就行,管他有多高")。
PLACE_CLEARANCE_M = 0.11
#: ``move_steps``(压缩后固定 20 步,与 expert 同款)
PLACE_MOVE_STEPS = 20
#: ``FINAL_PLACE_PRE_HOLD_STEPS`` 释放前握持保持
PLACE_PRE_HOLD_STEPS = 10
PLACE_RELEASE_STEPS = 9
PLACE_SETTLE_STEPS = 9
#: 放笔时 TCP 的 X 轴必须对齐的世界方向(**垂直于桌面**)。默认 ``(0,0,-1)``;
#: 可用环境变量 :data:`PLACE_TCP_X_ENV` 覆盖,便于在真机上试 ``+Z`` / ``-Z``
#: 两个朝向(非法值一律退回默认值)。字段含义见模块 docstring 的
#: "放置姿态约束"一节。
PLACE_TCP_X_WORLD: tuple[float, float, float] = (0.0, 0.0, -1.0)
#: 覆盖 :data:`PLACE_TCP_X_WORLD` 的环境变量名(取值形如 ``0,0,-1``)。
PLACE_TCP_X_ENV = "HYBRID_PLACE_TCP_X"
#: 姿态自检的容差:TCP 的 X 轴与世界 Z 的夹角超过这个度数就记 degradation。
PLACE_TCP_X_TOL_DEG = 15.0
#: 世界 Z 轴(桌面法向的**桌面坐标系**约定:重力反方向)。
PLACE_WORLD_Z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
#: 多起点 IK 的残留阈值与随机起点数(参考 ``place_above_release2.solve_ik``)。
#: 该阈值 = :data:`PLACE_POSITION_TOL_LADDER_M` 的**第 1 级**(10 mm),既作
#: ``_solve_left_pose`` 的早停,也作解析 IK 路径的第一级接受容差。
PLACE_IK_POSITION_TOL_M = 0.010
PLACE_IK_RANDOM_SEEDS = 60
PLACE_IK_RNG_SEED = 0
#: 目标姿态的**分级放宽族**(参考 ``place_above_release.py`` 的 fallback 模式:
#: "绕 TCP→筒心水平方向的水平垂轴做俯仰变体 ±12/±25/±40°")。顺序 =
#: 先精确姿态,再逐级加俯仰。**不删姿态约束**:每一级都是显式构造的目标姿态,
#: 只是把"完全垂直"逐步让成"接近垂直"。注意 0° 只在第一级出现一次,
#: 避免重复求解同一个姿态。
PLACE_PITCH_LADDER_DEG: tuple[float, ...] = (8.0, 15.0, 25.0, 35.0)
#: **接近高度的搜索族(米),按优先级**:11 cm(:data:`PLACE_CLEARANCE_M`)第一优先,
#: 其次 **17 cm**,最后才回落 **5 cm**。顺序依据 2026-10-11 的 200 万样本 FK 扫描
#: (左臂;括号内是"20 mm 位置容差内的最优 TCP X 轴轴偏差"):
#: 11 cm → 9.209° ✅ / 17 cm → 11.88° ✅ / 5 cm → 26.06° ❌。
#: 5 cm 保留作最后兜底(几何上更接近参考实现,虽然姿态做不到)。
PLACE_CLEARANCE_LADDER_M: tuple[float, ...] = (0.17, 0.05)
#: **位置残差容差阶梯(米),先严后松**:先按 <10 mm 找,找不到才放宽到 <20 mm;
#: 实际用了哪一级会**如实**写进 ``final_place.position_tol_mm`` 与
#: ``final_place.accepted_variant["position_tol_mm"]``,绝不默默放宽。
#: 20 mm 这一级是必须的:11 cm 处 10 mm 邻域内最优轴偏差仍有 26.5°(不达标),
#: 达标的那条解落在 10~20 mm 环带里(9.209°)。
PLACE_POSITION_TOL_LADDER_M: tuple[float, ...] = (0.010, 0.020)
#: FK 搜索式位姿求解的参数。解析 IK 判"无解"时改用**纯 FK 采样 + 局部精修**
#: 直接优化"位置残差 + TCP X 轴轴偏差"这两个量(见 :meth:`_place_search_pose`)。
#: ``PLACE_SEARCH_COARSE`` 是均匀采样总数:10 mm/20 mm 环带的体积占比只有
#: ~1e-5,几千个样本根本碰不到;2,000,000 正好复现上面那张扫描表的结论
#: (实测批量 FK 2M 条约 3 s,纯计算、不消耗环境步数)。
PLACE_SEARCH_COARSE = 2_000_000
#: 均匀采样的分批大小(避免一次性占满显存;与诊断脚本同口径的 50k)。
PLACE_SEARCH_CHUNK = 50_000
PLACE_SEARCH_ROUNDS = 6
PLACE_SEARCH_PER_ROUND = 1500
PLACE_SEARCH_TOP_K = 12
PLACE_SEARCH_SIGMA0 = 0.30
PLACE_SEARCH_SIGMA_DECAY = 0.55


def _axis_angle_matrix(axis: Any, angle_rad: float) -> np.ndarray:
    """Rodrigues 公式:绕单位轴 ``axis`` 转 ``angle_rad`` 的 3x3 旋转矩阵。

    只用于 :meth:`HandoverFlow._place_pose_variants` 里构造俯仰变体(纯几何,
    与仿真/物体真值无关)。轴退化(非有限或模长 < 1e-6)时返回单位矩阵,
    **绝不外抛、绝不返回奇异矩阵**。
    """
    vector = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-6:
        return np.eye(3, dtype=np.float64)
    kx, ky, kz = (vector / norm).tolist()
    skew = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(float(angle_rad)) * skew
        + (1.0 - math.cos(float(angle_rad))) * (skew @ skew)
    )


def _place_tcp_x_world() -> np.ndarray:
    """解析 :data:`PLACE_TCP_X_WORLD`(可被环境变量覆盖),返回单位向量。

    环境变量 :data:`PLACE_TCP_X_ENV` 支持 ``"0,0,-1"`` / ``"0 0 1"`` 两种写法;
    未设置、格式非法、含非有限值或模长退化(<1e-6)时**一律退回模块常量**
    (绝不外抛,也绝不返回零向量)。
    """
    raw = os.environ.get(PLACE_TCP_X_ENV)
    vector: np.ndarray | None = None
    if raw is not None and str(raw).strip():
        try:
            parts = [
                float(chunk)
                for chunk in str(raw).replace(",", " ").split()
            ]
            if len(parts) == 3:
                vector = np.asarray(parts, dtype=np.float64)
        except Exception:  # noqa: BLE001 - 环境变量非法一律退回默认值
            vector = None
    if vector is None or not bool(np.all(np.isfinite(vector))):
        vector = np.asarray(PLACE_TCP_X_WORLD, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-6:
        vector = np.asarray(PLACE_TCP_X_WORLD, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
    return vector / norm


# ---- HOLDER_LOCATE(参考 holder_vlm.py / holder_inner_ellipse.py) ----
#: 内圈椭圆只在上部 ``top_frac`` 的 ROI 里找(参考 ``fit_inner_rim(top_frac=0.80)``)。
INNER_RIM_TOP_FRAC = 0.80
#: 椭圆长短轴比上限、最小短半轴、最小面积占比(照抄 ``fit_inner_rim``)。
INNER_RIM_MAX_AXIS_RATIO = 1.90
INNER_RIM_MIN_MINOR_PX = 10.0
INNER_RIM_MIN_AREA_FRAC = 0.20
#: 内圈比外圈小一点(照抄 ``inner_scale = 0.95``)。
INNER_RIM_SCALE = 0.95
#: VLM 找不到 bbox 时的兜底方框半边长(参考 ``cx ± 45``)。
INNER_RIM_BBOX_HALF_PX = 45.0

#: VLM 调用重试与超时(与 flow.py 同口径)。
VLM_ATTEMPTS = 2
VLM_TIMEOUT_S = 180
VLM_RETRY_SLEEP_S = 2.0

#: 一个 ACT chunk 的步数兜底值(参考 ``CHUNK_STEPS``=50 / ``act_step``=50)。
ACT_CHUNK_STEPS_DEFAULT = 50

#: 一次 ``eval()`` 内允许的连续纯计算推进上限(与 ``flow.py`` 同值)。
MAX_PURE_ADVANCES = 32
MAX_DEGRADES = 8
MAX_FAILURES = 64
MAX_ACT_STATE = 400
#: 流程走完后是否把剩余步数交给纯 ACT(与 ``flow.DONE_RUNS_PURE_ACT`` 同口径)。
DONE_RUNS_PURE_ACT = True

#: ``MANUAL_TARGET`` 的固定 14 通道路点(``manual_targets/
#: seed10_manual_target_latest_right_left.json`` 的 ``qpos``)。
#:
#: 来源:参考包里的离线遥操作路点(左臂/gripper 21:43 记录,右臂 21:47 记录),
#: 全场次共用同一个文件(``run_single_place_seed.sh`` 固定 ``MANUAL_TARGET_JSON``),
#: 因此它是**与场景无关的固定路点**,不是任何一局的真值。
#: 布局:``[left_arm(6), left_eef, right_arm(6), right_eef]``。
MANUAL_TARGET_QPOS = np.array(
    [
        -0.5700929164886475,
        1.0966812372207642,
        -1.3750786781311035,
        0.16330480575561523,
        1.1171445846557617,
        -0.2754136621952057,
        0.008062767796218395,
        1.2533549070358276,
        0.9268044233322144,
        -0.8253798484802246,
        -0.7313986420631409,
        0.6249051690101624,
        0.955680251121521,
        0.049999840557575226,
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# VLM 提示词:一律要求 0..1 归一化坐标(见模块 docstring 的"归一化坐标约定")
# ---------------------------------------------------------------------------

#: holder 定位提示词 —— **必须**是归一化 0..1 坐标。
HOLDER_PROMPT = (
    "This is a robot wrist-camera view. On the table stands a short, wide, "
    "open-top cylindrical CUP (the pen holder) with a dark hollow circular "
    "opening. It may be partially occluded by the robot, shadows, or clutter. "
    "The long thin object held by the gripper is the PEN - it is NOT the holder.\n"
    "Answer with strict JSON only, no markdown:\n"
    '{"visible": true|false, "bbox_xyxy_norm": [x1,y1,x2,y2] | null, '
    '"center_uv_norm": [u,v] | null, "confidence": 0..1, "note": "short"}\n'
    "All coordinates are NORMALIZED to 0..1 (pixel divided by image width and "
    "height). x=0 is left, y=0 is top. If the holder is not visible, return "
    "false and nulls; do not guess."
)

#: pen 定位提示词 —— 同样要求归一化坐标(参考的像素版在 VL 编码器空间里错位)。
PEN_PROMPT = (
    "Look at this image from a robot wrist camera. A pen (or a similar slim "
    "elongated object) lies somewhere in the scene; the scene may be cluttered "
    "by the robot's own fingers and shadow.\n"
    "Answer with strict JSON only, no markdown, no extra text:\n"
    '{"pen_visible": true|false, "pen_uv_norm": [u, v] | null, '
    '"confidence": 0..1, "note": "short reason"}\n'
    "u and v are NORMALIZED to 0..1 (u = pixel_x / image_width, "
    "v = pixel_y / image_height), origin at the top-left. Report the CENTRE of "
    "the pen body. If you cannot see a pen, set pen_visible to false and "
    "pen_uv_norm to null - do not guess."
)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _json_from_text(text: Any) -> dict | None:
    """从模型回复里容错地抠出 JSON 对象(容忍 markdown 围栏)。"""
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


def _norm_to_pixels(values: Any, width: int, height: int) -> list[float] | None:
    """把 VLM 返回的坐标换算回像素。

    参考实现的两个分支都保留:

    * 值全部在 ``[0, 1]`` → 视为归一化坐标,乘以宽/高换算回像素
      (**主路径**,VL 编码器空间的问题就靠这一步绕开);
    * 否则视为已经是像素(模型没听归一化的要求),原样透传。
    """
    try:
        seq = [float(v) for v in values]
    except Exception:
        return None
    if not seq:
        return None
    if all(0.0 <= v <= 1.0 for v in seq):
        if len(seq) == 4:
            return [
                seq[0] * float(width),
                seq[1] * float(height),
                seq[2] * float(width),
                seq[3] * float(height),
            ]
        if len(seq) == 2:
            return [seq[0] * float(width), seq[1] * float(height)]
    return seq


def _rotation_matrix(axis: Any, angle: float) -> np.ndarray:
    """Rodrigues 旋转矩阵(与参考 ``Rotation.from_rotvec`` 等价)。"""
    vector = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    unit = vector / norm
    skew = np.array(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def _rotation_error_rad(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    """两个旋转矩阵之间的测地角(弧度)。"""
    cos_angle = (float(np.trace(np.asarray(rotation_a)[:3, :3].T @ np.asarray(rotation_b)[:3, :3])) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _horizontal_lateral_axis(camera_to_world: np.ndarray) -> np.ndarray:
    """与相机前向垂直的水平世界轴(参考 ``look_pen.horizontal_lateral_axis``)。"""
    rotation = np.asarray(camera_to_world, dtype=np.float64)[:3, :3]
    forward = rotation[:, 2]
    forward_xy = np.array([forward[0], forward[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(forward_xy))
    if norm < 1e-6:
        forward_xy = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        forward_xy /= norm
    lateral = np.cross(np.array([0.0, 0.0, 1.0]), forward_xy)
    lateral[2] = 0.0
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm < 1e-6:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return lateral / lateral_norm


def _plan_lateral_move(
    error_px: Any,
    camera_to_world: np.ndarray,
    *,
    scale_m_per_px: float = LOOK_PEN_SCALE_M_PER_PX,
    max_step_m: float = LOOK_PEN_MAX_STEP_M,
) -> np.ndarray:
    """把像素横向误差换成世界系横向位移(参考 ``plan_lateral_move``)。"""
    du = float(np.asarray(error_px, dtype=np.float64).reshape(-1)[0])
    axis = _horizontal_lateral_axis(camera_to_world)
    delta = axis * (du * float(scale_m_per_px))
    delta[2] = 0.0
    norm = float(np.linalg.norm(delta))
    if norm > float(max_step_m):
        delta *= float(max_step_m) / norm
    return delta


def _fit_inner_rim(
    image: np.ndarray,
    bbox: Any,
    *,
    top_frac: float = INNER_RIM_TOP_FRAC,
) -> dict:
    """拟合 holder 的**内圈**椭圆(逐行对应 ``holder_inner_ellipse.fit_inner_rim``)。

    ``cv2`` 不可用或找不到候选时退化为 bbox 中心(参考的
    ``partial_visibility_bbox`` 兜底路径,同样 ``0.65`` 轴长、``0.95`` 内缩),
    这样 HOLDER_LOCATE 在只有 numpy 的环境里也还能跑。
    """
    array = np.asarray(image)
    if array.ndim != 3:
        return {"ok": False, "error": "image is not HxWx3"}
    height, width = array.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0, x1 = sorted((max(0.0, min(width - 1.0, x0)), max(0.0, min(width - 1.0, x1))))
    y0, y1 = sorted((max(0.0, min(height - 1.0, y0)), max(0.0, min(height - 1.0, y1))))
    bbox_center = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]

    def _bbox_fallback(reason: str) -> dict:
        return {
            "ok": True,
            "method": "partial_visibility_bbox",
            "fallback_reason": reason,
            "centre_uv": [float(v) for v in bbox_center],
            "axes_px": [
                INNER_RIM_SCALE * 0.65 * float(x1 - x0),
                INNER_RIM_SCALE * 0.65 * float(y1 - y0),
            ],
            "angle_deg": 0.0,
            "axis_ratio": 1.0,
            "n_points": 0,
            "inner_scale": INNER_RIM_SCALE,
        }

    try:
        import cv2
    except Exception as exc:  # noqa: BLE001 - cv2 缺失时走 bbox 兜底
        return _bbox_fallback(f"cv2 unavailable: {type(exc).__name__}")

    ix0, iy0, ix1, iy1 = int(x0), int(y0), int(x1), int(y1)
    ry1 = int(iy0 + (iy1 - iy0) * float(top_frac))
    roi = array[iy0:ry1, ix0:ix1]
    if roi.size == 0:
        return _bbox_fallback("empty ROI")
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 140)
    contours, _hierarchy = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    bbox_area = max(1.0, float((ix1 - ix0) * (iy1 - iy0)))
    candidates: list[dict] = []
    for contour in contours:
        if len(contour) < 20:
            continue
        try:
            ellipse = cv2.fitEllipse(contour)
        except cv2.error:
            continue
        (cx, cy), (d1, d2), angle = ellipse
        a, b = max(d1, d2) / 2.0, min(d1, d2) / 2.0
        if b < INNER_RIM_MIN_MINOR_PX:
            continue
        gx, gy = cx + ix0, cy + iy0
        ratio = a / max(b, 1e-6)
        if ratio > INNER_RIM_MAX_AXIS_RATIO:
            continue
        if not (ix0 <= gx <= ix1 and iy0 <= gy <= iy1):
            continue
        if gy > iy0 + 0.75 * (iy1 - iy0):
            continue
        area = float(np.pi * a * b)
        if area < INNER_RIM_MIN_AREA_FRAC * bbox_area:
            continue
        candidates.append(
            {
                "area": area,
                "centre_uv": [float(gx), float(gy)],
                "axes_px": [float(d1), float(d2)],
                "angle_deg": float(angle),
                "axis_ratio": float(ratio),
                "n_points": int(len(contour)),
            }
        )

    best: dict | None = None
    if candidates:
        centre = np.array(bbox_center, dtype=np.float64)
        diag = max(1.0, float(np.hypot(ix1 - ix0, iy1 - iy0)))
        nearest = min(
            candidates,
            key=lambda item: float(
                np.linalg.norm(np.asarray(item["centre_uv"], dtype=np.float64) - centre)
            ),
        )
        if (
            float(
                np.linalg.norm(
                    np.asarray(nearest["centre_uv"], dtype=np.float64) - centre
                )
            )
            <= 0.30 * diag
        ):
            best = dict(nearest)
    if best is None:
        return _bbox_fallback("no inner-rim contour candidate near the bbox centre")
    best["axes_px"] = [INNER_RIM_SCALE * float(v) for v in best["axes_px"]]
    best["inner_scale"] = INNER_RIM_SCALE
    best.setdefault("method", "inner_contour")
    best["ok"] = True
    return best


def _projection_matrix(K: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    """``P = K @ [R^T | -R^T t]``(world -> pixel),与参考一致。"""
    rotation = np.asarray(camera_to_world, dtype=np.float64)[:3, :3]
    translation = np.asarray(camera_to_world, dtype=np.float64)[:3, 3]
    return np.asarray(K, dtype=np.float64) @ np.c_[
        rotation.T, -rotation.T @ translation
    ]


def _triangulate_dlt(
    projections: list[np.ndarray], pixels: list[Any]
) -> np.ndarray | None:
    """线性三角化(与 ``cv2.triangulatePoints`` 同一组方程的 DLT 解)。

    参考 ``holder_stereo_inner_eval._triangulate`` 用 ``cv2``;这里用 numpy SVD
    实现同一组方程,免得为了三角化再引入 OpenCV 依赖。
    """
    rows: list[np.ndarray] = []
    for projection, pixel in zip(projections, pixels):
        u, v = float(np.asarray(pixel, dtype=np.float64).reshape(-1)[0]), float(
            np.asarray(pixel, dtype=np.float64).reshape(-1)[1]
        )
        rows.append(u * projection[2] - projection[0])
        rows.append(v * projection[2] - projection[1])
    try:
        _u, _s, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    except Exception:
        return None
    homogeneous = vt[-1]
    if abs(float(homogeneous[3])) < 1e-9:
        return None
    return np.asarray(homogeneous[:3] / homogeneous[3], dtype=np.float64)


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------


class HandoverFlow(HandleBasketFlow):
    """items_handover 的有限状态机(依赖注入版 ``main()``)。

    继承 :class:`HandleBasketFlow` 的 **env 绑定 / info 探测 / 降级 / ACT 微步 /
    调试落盘 / 计时** 机制,只替换状态与阶段实现 —— 这样两个流程的 ``eval()``
    语义逐条同构,SPEC §4.1 要求的"接口逐字一致"由继承直接保证。

    ``act_policy`` 需要提供 ``select_chunk_action(obs) -> (action, ran_inference)``
    与 ``action_dim``;本类**不**加载任何 ACT 权重。
    """

    # ---- 状态名(与 SPEC §4.1 的顺序一一对应)----
    SETTLE = "SETTLE"
    CHUNK1 = "CHUNK1"
    PRIMITIVE = "PRIMITIVE"
    ALIGNMENT = "ALIGNMENT"
    CONTINUATION = "CONTINUATION"
    LEFT_TAKEOVER = "LEFT_TAKEOVER"
    RIGHT_RELEASE = "RIGHT_RELEASE"
    MANUAL_TARGET = "MANUAL_TARGET"
    HOLDER_LOCATE = "HOLDER_LOCATE"
    FINAL_PLACE = "FINAL_PLACE"
    DONE = "DONE"
    ACT_FILL = "ACT_FILL"
    FALLBACK_ACT = "FALLBACK_ACT"

    #: 全部状态(顺序 = 相位顺序;``FALLBACK_ACT`` 是任意点都可跳入的降级态)。
    STATES = (
        SETTLE,
        CHUNK1,
        PRIMITIVE,
        ALIGNMENT,
        CONTINUATION,
        LEFT_TAKEOVER,
        RIGHT_RELEASE,
        MANUAL_TARGET,
        HOLDER_LOCATE,
        FINAL_PLACE,
        DONE,
        ACT_FILL,
        FALLBACK_ACT,
    )

    #: 每个状态的步数预算(docstring 那张表的机器可读版本)。
    STEP_BUDGET = {
        SETTLE: 0,
        CHUNK1: CHUNK1_STEPS,
        PRIMITIVE: PRIMITIVE_BUDGET,
        ALIGNMENT: ALIGNMENT_STEPS,
        CONTINUATION: CONTINUATION_BUDGET,
        LEFT_TAKEOVER: LEFT_TAKEOVER_BUDGET,
        RIGHT_RELEASE: RIGHT_RELEASE_STEPS,
        MANUAL_TARGET: MANUAL_TARGET_BUDGET,
        HOLDER_LOCATE: 0,
        FINAL_PLACE: FINAL_PLACE_BUDGET,
        DONE: 0,
        ACT_FILL: 0,
        FALLBACK_ACT: 0,
    }

    # ------------------------------------------------------------------
    # 构造 / reset
    # ------------------------------------------------------------------

    def __init__(
        self,
        act_policy: Any,
        settings: Any,
        *,
        act_chunks: int = 6,
        max_env_steps: int = OFFICIAL_STEP_LIMIT,
        debug_dir: Path | None = None,
    ) -> None:
        super().__init__(
            act_policy,
            settings,
            act_chunks=act_chunks,
            max_env_steps=max_env_steps,
            debug_dir=debug_dir,
        )

    # ------------------------------------------------------------------
    # episode 状态
    # ------------------------------------------------------------------

    def _init_episode_state(self) -> None:
        """清空一个 episode 的全部运行时状态(父类的字段 + handover 专有字段)。"""
        super()._init_episode_state()
        self._state = self.SETTLE
        # 与父类一致:SETTLE 是起始态而不是"进入"的态;为了让 ``stages`` 审计
        # 从头就完整,这里手动补一条起点记录。
        self._history.append({"state": self.SETTLE, "env_step": 0})

        # 每个状态实际用掉的步数(仅用于 summary / 步数审计)。
        self._budget: dict[str, int] = {name: 0 for name in self.STATES}
        self._prim_budget_used: dict[str, int] = {
            name: 0 for name in PRIMITIVE_SUBSTAGES
        }
        # 当前相位的实际步数上限(由 ``_open_state_cap`` 写入)
        self._state_cap_value: int = 0
        # 是否曾因步数预算不足而压缩/提前降级
        self._budget_truncated: list[dict] = []
        # 环境自带的官方步数上限(只读探测;绑定 env 时写入)
        if not hasattr(self, "_env_step_limit"):
            self._env_step_limit: int | None = None
        if not hasattr(self, "_requested_max_env_steps"):
            self._requested_max_env_steps: int = int(self.max_env_steps)

        # CHUNK1 末态(ALIGNMENT 的回归目标)
        self._align_target: np.ndarray | None = None

        # PRIMITIVE
        self._prim_started: bool = False
        self._prim_stage: int = 0
        self._prim_i: int = 0
        self._prim_steps: int = 0
        self._prim_allowance: int = PRIMITIVE_BUDGET
        self._prim_records: list[dict] = []
        self._prim_ok: bool = True
        # 子阶段初始化标记(用 ``_prim_i`` 判"首次进入"会在纯计算首步上死循环)
        self._prim_init: bool = False
        self._prim_ref: dict = {}
        self._look_pen: dict = {}

        # CONTINUATION
        self._continuation_steps: int = 0
        self._continuation_success: bool = False
        self._continuation_start: np.ndarray | None = None
        self._left_tcp_history: deque = deque(maxlen=LEFT_TCP_HISTORY_LEN)

        # LEFT_TAKEOVER / RIGHT_RELEASE / MANUAL_TARGET
        self._takeover: dict | None = None
        self._takeover_ref: dict = {}
        self._release_ref: dict = {}
        self._manual_ref: dict = {}

        # HOLDER_LOCATE / FINAL_PLACE
        self._holder: dict | None = None
        self._holder_estimate: np.ndarray | None = None
        self._holder_attempts: list[dict] = []
        self._place_ref: dict = {}
        #: FK 搜索式位姿求解的最近一次结果(见 :meth:`_place_search_pose`)。
        self._place_search: dict | None = None
        self._place: dict | None = None

        # ACT 夹爪覆盖(CONTINUATION 期间右夹爪锁死)
        self._grip_override: dict[int, float] | None = None

    def _bind_env(self, env: Any) -> None:
        super()._bind_env(env)
        self._grip_override = None
        self._requested_max_env_steps = int(self.max_env_steps)
        probed = self._probe_env_step_limit(env)
        self._env_step_limit = probed
        if probed is not None and 0 < int(probed) < int(self.max_env_steps):
            # 只降不升:官方 gym config 的 max_episode_steps 才是真正的硬上限
            # (random 350 / clear 300)。这里**只读**该公开属性,不写任何补丁。
            self._record_degradation(
                "BIND",
                f"官方 gym config 的步数上限 {probed} < 传入的 "
                f"{self.max_env_steps};流程自用上限夹到 {probed}",
            )
            self.max_env_steps = int(probed)

    @staticmethod
    def _probe_env_step_limit(env: Any) -> int | None:
        """只读地探测环境自身的 ``max_episode_steps``(公开配置,不是物体真值)。

        官方 ``scripts/eval_policy.py::resolve_episode_max_steps`` 会用
        ``gym_config["max_episode_steps"]`` 建环境,这个值最终挂在 env 或它的
        gym ``TimeLimit`` 包装上。探测失败(没有该属性/包装器名字变了)返回
        ``None``,此时沿用传入的 ``max_env_steps``。
        """
        candidates: list[Any] = [env, getattr(env, "unwrapped", None)]
        cursor = env
        for _ in range(8):
            cursor = getattr(cursor, "env", None)
            if cursor is None:
                break
            candidates.append(cursor)
        for obj in candidates:
            if obj is None:
                continue
            for attr in ("max_episode_steps", "_max_episode_steps"):
                try:
                    value = getattr(obj, attr, None)
                except Exception:  # noqa: BLE001
                    continue
                if value is None:
                    continue
                try:
                    if hasattr(value, "item"):
                        value = value.item()
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if number > 0:
                    return number
        return None

    # ------------------------------------------------------------------
    # 步数审计 / 预算
    # ------------------------------------------------------------------

    def _step_env_step(self, env: Any, action: Any) -> tuple[Any, bool]:
        """执行 1 次 ``env.step()`` 并按状态累计步数(其余语义与父类一致)。"""
        obs, truncated = super()._step_env_step(env, action)
        self._budget[self._state] = self._budget.get(self._state, 0) + 1
        return obs, truncated

    def _budget_left(self, state: str) -> int:
        """当前状态下"还能用多少步"——扣掉后续状态的预留预算。"""
        reserve = int(TAIL_RESERVE.get(str(state), 0))
        return max(0, int(self.max_env_steps) - int(self._steps_taken) - reserve)

    def _state_cap(self, state: str, nominal: int) -> int:
        """把某相位的名义步数夹到"不会挤掉后续相位"的可用量。

        * 默认配置(``max_env_steps=350`` + 名义预算)下 ``cap == nominal``,
          行为与参考流程一致;
        * 预算不够时记一次 ``degradation`` 并把该相位压缩到 ``cap`` 步;
        * ``cap == 0`` 时调用方必须**提前降级为纯 ACT**(绝不冒险超步,
          超步会撞官方 gym 的 350 步 TimeLimit → 环境重置 → 该任务判 0)。
        """
        allowance = self._budget_left(state)
        cap = int(min(int(nominal), int(allowance)))
        if cap < int(nominal):
            self._budget_truncated.append(
                {
                    "state": str(state),
                    "nominal": int(nominal),
                    "allowance": int(allowance),
                    "cap": int(cap),
                    "env_step": int(self._steps_taken),
                }
            )
            self._record_degradation(
                state,
                f"步数预算不足(可用 {allowance} < 名义 {nominal}),该相位压缩到 "
                f"{cap} 步",
            )
        return cap

    def _open_state_cap(self, state: str, nominal: int) -> int:
        """相位入口统一处理:算 cap;``cap == 0`` 直接判失败并降级为纯 ACT。"""
        cap = self._state_cap(state, nominal)
        self._state_cap_value = cap
        if cap <= 0:
            self._record_failure(
                state,
                f"步数预算耗尽(已用 {self._steps_taken}/{self.max_env_steps}),"
                "提前降级为纯 ACT 跑完剩余步数",
            )
            self._enter(self.FALLBACK_ACT)
        return cap

    def _eef_channel(self, arm: str) -> int:
        """某条臂的夹爪在"生效动作向量"里的下标。

        优先用父类 ``_probe_eef_channels`` 从 ``active_joint_ids`` 推出的结果;
        推不出来时退回官方 14 维布局的固定下标(6 / 13)。
        """
        key = "left_eef_channel" if arm == LEFT else "right_eef_channel"
        channel = self._gate.get(key)
        if channel is None:
            return LEFT_EEF_CHANNEL_DEFAULT if arm == LEFT else RIGHT_EEF_CHANNEL_DEFAULT
        return int(channel)

    def _current_action_vector(self) -> np.ndarray:
        """当前关节状态在"生效动作向量"里的 14 维表示(供插值用)。"""
        q_left, grip_left = self._arm_joints(self._base, LEFT)
        q_right, grip_right = self._arm_joints(self._base, RIGHT)
        return np.asarray(
            full_action(self._base, q_left, grip_left, q_right, grip_right),
            dtype=np.float64,
        ).reshape(-1)

    # ------------------------------------------------------------------
    # ACT 夹爪守卫(CONTINUATION 期间右夹爪锁死;其余状态不覆盖)
    # ------------------------------------------------------------------

    def _act_guard(
        self, action: np.ndarray, obs: Any
    ) -> tuple[np.ndarray, bool, float | None]:
        """ACT 动作的夹爪覆盖(逐行对应参考 ``recording_step`` 的 phase 分支)。

        * ``CONTINUATION``:右夹爪通道每一步都被写成
          :data:`LOCKED_RIGHT_GRIPPER`(0.0),对应参考
          ``LOCK_RIGHT_GRIPPER=1, LOCK_RIGHT_GRIPPER_STEPS=0``(全程锁定);
        * ``CHUNK1`` / ``ACT_FILL`` / ``FALLBACK_ACT``:``self._grip_override``
          为 ``None``,不覆盖任何通道。
        """
        row = np.asarray(npv(action), dtype=np.float32).copy().reshape(-1)
        override = self._grip_override
        cmd_value: float | None = None
        if override:
            for raw_channel, value in override.items():
                try:
                    channel = int(raw_channel)
                except Exception:  # noqa: BLE001 - 非法下标直接跳过
                    continue
                if 0 <= channel < row.size:
                    row[channel] = float(value)
                    cmd_value = float(value)
        return row, False, cmd_value

    # ------------------------------------------------------------------
    # 阶段 0:SETTLE(0 步 · 纯计算)
    # ------------------------------------------------------------------

    def _stage_settle(self, env: Any, inference_times: list[float]) -> None:
        """SETTLE:记录左右臂关节基线并立刻进入 CHUNK1(**不消耗 env.step**)。

        参考流程没有单独的 settle 相位;这里只做一次纯计算微步,用来固定
        "另一条臂保持不动"的基线,以及给 ``summary()`` 留一帧调试三路相机图。
        """
        if self._stage_i == 0:
            self._capture_holds()
            self._record_debug_three_cam("handover_settle_three_cam.png")
        if self._left_hold_q is None or self._right_hold_q is None:
            raise RuntimeError("无法读取初始关节角(robot.get_qpos 失败)")
        self._enter(self.CHUNK1)

    # ------------------------------------------------------------------
    # 阶段 1:CHUNK1(50 步 ACT)
    # ------------------------------------------------------------------

    def _stage_chunk1(self, env: Any, inference_times: list[float]) -> None:
        """CHUNK1:参考的 ``chunk1`` —— 1 个 ACT chunk(``CHUNK_STEPS``=50)。"""
        if self._stage_i == 0:
            self._grip_override = None
            self._record_debug_three_cam("handover_chunk1_before_three_cam.png")
            cap = self._open_state_cap(self.CHUNK1, CHUNK1_STEPS)
            if cap <= 0:
                return
            self._chunk1_cap = int(min(cap, max(1, int(self._act_chunk_budget))))
        self._act_micro_step(env, inference_times, guarded=False)
        self._stage_i += 1
        if self._stage_i >= int(getattr(self, "_chunk1_cap", CHUNK1_STEPS)):
            # 参考实现:act_end_action = read_action_vector(base)(chunk 结束时的
            # 当前关节状态),ALIGNMENT 用它当回归目标。
            self._align_target = self._current_action_vector()
            self._record_debug_three_cam("handover_chunk1_after_three_cam.png")
            self._enter(self.PRIMITIVE)

    # ------------------------------------------------------------------
    # 阶段 2:PRIMITIVE(≤55 步 · 右臂抓笔)
    # ------------------------------------------------------------------

    def _stage_primitive(self, env: Any, inference_times: list[float]) -> None:
        """PRIMITIVE:右臂垂直化 → 接触下压 → 闭爪 → 保持 → 抬起(参考 ``primitive``)。

        子阶段顺序见 :data:`PRIMITIVE_SUBSTAGES`;整状态让步数由
        :meth:`_budget_left` 夹紧,合计不超过 :data:`PRIMITIVE_BUDGET`。
        """
        if not self._prim_started:
            self._prim_started = True
            self._prim_stage = 0
            self._prim_i = 0
            self._prim_steps = 0
            self._prim_ok = True
            self._prim_records = []
            self._prim_ref = {}
            self._prim_init = False
            self._prim_allowance = self._open_state_cap(
                self.PRIMITIVE, PRIMITIVE_BUDGET
            )
            if self._prim_allowance <= 0:
                return
            if self._align_target is None:
                self._align_target = self._current_action_vector()

        guard = 0
        while self._prim_stage < len(PRIMITIVE_SUBSTAGES):
            # 防御:任何子阶段都不允许"既不 step 也不前进"(否则会在这里空转)。
            guard += 1
            if guard > 2 * len(PRIMITIVE_SUBSTAGES):
                self._record_failure(
                    self.PRIMITIVE,
                    "PRIMITIVE 子阶段推进停滞(纯计算子阶段未前进),提前降级",
                )
                self._finish_primitive()
                return
            if self._prim_steps >= self._prim_allowance:
                self._record_failure(
                    self.PRIMITIVE,
                    f"PRIMITIVE 步数预算耗尽({self._prim_steps}/"
                    f"{self._prim_allowance}),剩余子阶段未执行",
                )
                # 预算不够就**提前降级为纯 ACT**,绝不带着"半截抓取"继续往下走。
                self._prim_ok = False
                self._finish_primitive()
                return
            name = PRIMITIVE_SUBSTAGES[self._prim_stage]
            handler = getattr(self, f"_prim_{name.lower()}")
            before = int(self._steps_taken)
            handler(env)
            if self._state != self.PRIMITIVE:
                # 子阶段内部已经把状态切走(收尾/降级),这里必须立刻退出,
                # 否则会在新状态上继续跑 PRIMITIVE 的子阶段。
                return
            used = int(self._steps_taken) - before
            self._prim_steps += used
            if used > 0:
                self._prim_budget_used[name] = (
                    self._prim_budget_used.get(name, 0) + used
                )
                return
        self._finish_primitive()

    def _prim_next(self, record: dict, *, fatal: bool = True) -> None:
        """结束当前子阶段并跳到下一个(顺手记录证据)。

        ``fatal=False`` 时,``record["ok"] is False`` **不再**把整条 PRIMITIVE 判死:
        证据照原样记进 ``_prim_records``(保持诚实),但子阶段指针照常前进,
        由调用方负责先写一条 degradation 说明为什么可以继续。

        典型用途:PRESS 的接触判定是启发式(见 :meth:`_prim_press`),判不出来
        不等于抓不到笔,而"探到没探到"不该阻断后面 HOLDER_LOCATE / FINAL_PLACE
        这两个用户真正要验证的相位。
        """
        name = PRIMITIVE_SUBSTAGES[self._prim_stage]
        record = dict(record)
        record["substage"] = name
        record["steps"] = int(self._prim_budget_used.get(name, 0))
        self._prim_records.append(jsonable(record))
        if record.get("ok") is False and fatal:
            # 参考 ``grasp_sequence``:垂直对齐/下压任一 ``ok=False`` 就 **立刻
            # return**,后面的闭爪/抬升不再执行。这里把子阶段指针直接推到末尾,
            # 由 ``_stage_primitive`` 收尾(既不浪费步数,也不"半截抓取"往下走)。
            self._prim_ok = False
            self._prim_stage = len(PRIMITIVE_SUBSTAGES)
            self._prim_i = 0
            self._prim_init = False
            return
        self._prim_stage += 1
        self._prim_i = 0
        self._prim_init = False

    def _finish_primitive(self) -> None:
        self._prim_started = False
        if not self._prim_ok:
            self._record_failure(
                self.PRIMITIVE,
                "抓取原语失败(垂直对齐/接触/闭爪/抬升之一未达成)",
            )
            self._enter(self.FALLBACK_ACT)
            return
        self._record_debug_three_cam("handover_primitive_after_three_cam.png")
        self._enter(self.ALIGNMENT)

    # ---- 2.1 LOOK_PEN(≤4 步)----

    def _prim_look_pen(self, env: Any) -> None:
        """LOOK_PEN:VLM 判断笔是否落在夹爪开口的固定像素点上,不在就横移一次。

        参考 ``look_pen_cycle(max_moves=2, move_steps=4, step_budget=10)``。
        移植版给该子阶段 4 步预算(= 一次 4 步横移),VLM 不可用时只记一次
        ``degradation`` 并继续(参考实现在 VLM 报错时同样继续)。
        """
        cap = int(PRIMITIVE_SUBSTAGE_BUDGET["LOOK_PEN"])
        if not self._prim_init:
            self._prim_init = True
            self._look_pen = self._look_pen_verdict()
            record = dict(self._look_pen)
            record["ok"] = True  # 定位失败不阻断抓取(与参考一致)
            if not self._look_pen.get("ok"):
                self._record_degradation(
                    self.PRIMITIVE,
                    f"LOOK_PEN 跳过:{self._look_pen.get('error')}",
                )
                self._prim_next(record)
                return
            if self._look_pen.get("inside"):
                record["moved"] = False
                self._prim_next(record)
                return
            plan = self._look_pen_move()
            record["move"] = jsonable(plan)
            if not plan.get("ok"):
                self._prim_next(record)
                return
            self._prim_ref = plan
        if self._prim_i >= cap:
            self._prim_next(self._look_pen)
            return
        ref = self._prim_ref
        if not ref:
            self._prim_next(self._look_pen)
            return
        alpha = float(self._prim_i + 1) / float(max(1, cap))
        q = (1.0 - alpha) * ref["q_start"] + alpha * ref["q_target"]
        action = full_action(
            self._base, q, ref["grip_left"], ref["q_right"], ref["grip_right"]
        )
        self._step_env_step(env, action)
        self._prim_i += 1
        if self._prim_i >= cap:
            self._prim_next(self._look_pen)

    def _look_pen_verdict(self) -> dict:
        """问一次 VLM:笔在右腕相机图像的哪个归一化位置,离固定点多远。"""
        vlm = _lazy("vlm")
        try:
            if not vlm.vlm_available():
                return {"ok": False, "error": "VLM 未配置(HYBRID_VLM_URL 为空)"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"vlm.vlm_available 失败: {exc}"}
        rgb = _safe_rgb_of(self._obs, LOOK_PEN_CAMERA)
        if rgb is None:
            return {"ok": False, "error": f"缺少 {LOOK_PEN_CAMERA} 图像"}
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        parsed: dict | None = None
        result: dict = {}
        for attempt in range(int(VLM_ATTEMPTS)):
            result = vlm.vlm_chat(
                PEN_PROMPT,
                [rgb],
                timeout_s=int(VLM_TIMEOUT_S),
            )
            parsed = _json_from_text(result.get("text"))
            if parsed is not None:
                break
            if attempt + 1 < int(VLM_ATTEMPTS):
                time.sleep(float(VLM_RETRY_SLEEP_S))
        if parsed is None:
            return {
                "ok": False,
                "error": result.get("error") or "VLM 未返回可解析 JSON",
                "latency_s": result.get("latency_s"),
            }
        if not bool(parsed.get("pen_visible", False)):
            return {
                "ok": False,
                "visible": False,
                "error": "VLM 报告笔不可见",
                "latency_s": result.get("latency_s"),
            }
        raw = parsed.get("pen_uv_norm")
        if raw is None:
            raw = parsed.get("pen_uv")
        uv = _norm_to_pixels(raw, width, height)
        if uv is None or len(uv) != 2:
            return {"ok": False, "error": "VLM 未返回可用归一化坐标"}
        error_px = [float(uv[0]) - PEN_TARGET_UV[0], float(uv[1]) - PEN_TARGET_UV[1]]
        inside = abs(error_px[0]) <= PEN_TOL_PX and abs(error_px[1]) <= PEN_TOL_PX
        return {
            "ok": True,
            "inside": bool(inside),
            "uv_px": [float(v) for v in uv],
            "uv_norm": [float(uv[0]) / max(1.0, float(width)), float(uv[1]) / max(1.0, float(height))],
            "error_px": error_px,
            "target_uv_px": list(PEN_TARGET_UV),
            "tolerance_px": float(PEN_TOL_PX),
            "latency_s": result.get("latency_s"),
        }

    def _look_pen_move(self) -> dict:
        """按像素误差算一次世界系横移,并用 FK+IK 落到关节目标(4 步插值)。"""
        stereo = _lazy("stereo")
        camera = stereo.wrist_camera_pose(self._base, RIGHT)
        delta = _plan_lateral_move(self._look_pen.get("error_px"), camera)
        target_pose = self._fk(RIGHT_ARM)
        target_pose[:3, 3] = target_pose[:3, 3] + delta
        q_start, grip_left = self._arm_joints(self._base, LEFT)
        q_right, grip_right = self._arm_joints(self._base, RIGHT)
        q_target = self._ik(RIGHT_ARM, target_pose, q_right)
        if q_target is None:
            return {"ok": False, "error": "LOOK_PEN 横移 IK 无解", "delta_m": delta.tolist()}
        return {
            "ok": True,
            "delta_m": delta.tolist(),
            "q_start": q_start,
            "q_target": q_target,
            "grip_left": float(grip_left),
            "q_right": q_right,
            "grip_right": float(grip_right),
        }

    # ---- 2.2 VERTICAL_ALIGN(≤20 步)----

    def _prim_vertical_align(self, env: Any) -> None:
        """VERTICAL_ALIGN:边下降边把右爪 z 轴转到竖直(参考 ``descend_verticalize``)。

        ``steps=20`` / ``descent_m=0.12`` / ``approach_xy_m=(0.02,-0.005)`` /
        ``tolerance_deg=8``;每步在 3 个四元数混合比例 × 21 个 xy 偏置里选一个
        代价最小的可达 IK 解。``ok`` 的判据与参考一致:至少走 2 步、z 下降超过
        5 mm、且末端倾角比开始时更小(``tolerance`` 只作诊断,不作硬停)。
        """
        cap = int(PRIMITIVE_SUBSTAGE_BUDGET["VERTICAL_ALIGN"])
        if not self._prim_init:
            self._prim_init = True
            start_pose = self._fk(RIGHT_ARM)
            if start_pose is None:
                self._prim_next(
                    {
                        "ok": False,
                        "error": f"FK unavailable: {self._fk_error or '未记录'}",
                    }
                )
                return
            target_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            start_angle = float(
                np.degrees(
                    np.arccos(
                        float(np.clip(np.dot(start_pose[:3, 2], target_z), -1.0, 1.0))
                    )
                )
            )
            self._prim_ref = {
                "start_pose": start_pose,
                "start_xy": start_pose[:2, 3].copy(),
                "start_z": float(start_pose[2, 3]),
                "prev_z": float(start_pose[2, 3]),
                "start_angle": start_angle,
                "target_z": target_z,
                "step_m": float(VERTICAL_ALIGN_DESCENT_M) / float(max(1, cap)),
                "last_xy": [0.0, 0.0],
                "last_remaining_deg": None,
            }
            return  # 纯计算微步
        ref = self._prim_ref
        if not ref:
            self._prim_next({"ok": False, "error": "垂直对齐初始化失败"})
            return
        if self._prim_i >= cap:
            self._finish_vertical_align(ref)
            return

        step_index = self._prim_i
        pose = self._fk(RIGHT_ARM)
        if pose is None:
            self._finish_vertical_align(ref)
            return
        best = self._vertical_align_candidate(pose, ref, step_index, cap)
        if best is None:
            self._finish_vertical_align(ref)
            return
        _score, dx, dy, remaining_deg, q_target = best
        # 接近阶段夹爪必须**张开**:爪子要从两侧跨住笔才能闭合夹住它。
        # 闭合只发生在 CLOSE 子阶段(那里做 1.0 → 0.0 的插值)。
        self._step_right_q(env, q_target, grip=GRIP_RIGHT_OPEN)
        self._prim_i += 1
        ref["last_xy"] = [float(dx), float(dy)]
        ref["last_remaining_deg"] = float(remaining_deg)
        after = self._fk(RIGHT_ARM)
        if after is not None:
            z_after = float(after[2, 3])
            moved = abs(float(ref["prev_z"]) - z_after)
            ref["prev_z"] = z_after
            if moved < 0.0005:
                self._finish_vertical_align(ref)
                return
        if self._prim_i >= cap:
            self._finish_vertical_align(ref)

    def _vertical_align_candidate(
        self, pose: np.ndarray, ref: dict, step_index: int, cap: int
    ) -> tuple | None:
        """在 (四元数混合比例 × xy 偏置) 网格里挑代价最小的可达 IK 目标。"""
        offset_grid = [(0.0, 0.0)]
        for radius in (0.005, 0.010, 0.015, 0.020):
            offset_grid.extend(
                [
                    (radius, 0.0),
                    (-radius, 0.0),
                    (0.0, radius),
                    (0.0, -radius),
                    (radius, radius),
                    (radius, -radius),
                    (-radius, radius),
                    (-radius, -radius),
                ]
            )
        target_z = ref["target_z"]
        current_z = pose[:3, 2].copy()
        cross = np.cross(current_z, target_z)
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm < 1e-6:
            full_rotation = pose[:3, :3].copy()
        else:
            axis = cross / cross_norm
            angle = float(np.arccos(np.clip(np.dot(current_z, target_z), -1.0, 1.0)))
            full_rotation = _rotation_matrix(axis, angle) @ pose[:3, :3]

        q_from = self._matrix_to_quat(pose[:3, :3])
        q_to = self._matrix_to_quat(full_rotation)
        if float(np.dot(q_from, q_to)) < 0.0:
            q_to = -q_to
        q_current = self._right_q(ref)
        best: tuple | None = None
        for alpha in (0.60, 0.35, 0.18):
            q_mix = (1.0 - alpha) * q_from + alpha * q_to
            norm = float(np.linalg.norm(q_mix))
            if norm < 1e-9:
                continue
            target_rotation = self._quat_to_matrix(q_mix / norm)
            remaining_deg = float(
                np.degrees(
                    np.arccos(
                        float(np.clip(np.dot(target_rotation[:3, 2], target_z), -1.0, 1.0))
                    )
                )
            )
            progress = float(step_index + 1) / float(max(1, cap))
            desired_xy = np.asarray(ref["start_xy"], dtype=np.float64) + progress * np.asarray(
                PRE_GRASP_XY_M, dtype=np.float64
            )
            for dx, dy in offset_grid:
                target_pose = pose.copy()
                target_pose[:3, :3] = target_rotation
                target_pose[0, 3] = desired_xy[0] + dx
                target_pose[1, 3] = desired_xy[1] + dy
                target_pose[2, 3] -= float(ref["step_m"])
                q_target = self._ik(RIGHT_ARM, target_pose, q_current)
                if q_target is None or q_target.size != 6:
                    continue
                score = (
                    float(np.hypot(dx, dy))
                    + 0.0005 * remaining_deg
                    + 0.02 * float(np.linalg.norm(q_target - q_current))
                )
                if best is None or score < best[0]:
                    best = (score, float(dx), float(dy), float(remaining_deg), q_target)
        return best

    def _finish_vertical_align(self, ref: dict) -> None:
        cap = int(PRIMITIVE_SUBSTAGE_BUDGET["VERTICAL_ALIGN"])
        final_pose = self._fk(RIGHT_ARM)
        target_z = ref.get("target_z", np.array([0.0, 0.0, -1.0]))
        final_angle = None
        if final_pose is not None:
            final_angle = float(
                np.degrees(
                    np.arccos(
                        float(np.clip(np.dot(final_pose[:3, 2], target_z), -1.0, 1.0))
                    )
                )
            )
        steps_used = int(self._prim_i)
        descent = float(ref.get("start_z", 0.0)) - float(ref.get("prev_z", 0.0))
        within_tolerance = final_angle is not None and final_angle <= float(
            VERTICAL_ALIGN_TOL_DEG
        )
        made_progress = (
            steps_used >= 2
            and descent > 0.005
            and final_angle is not None
            and final_angle < float(ref.get("start_angle", 0.0))
        )
        self._prim_next(
            {
                "ok": bool(made_progress),
                "steps": steps_used,
                "start_angle_deg": ref.get("start_angle"),
                "final_angle_deg": final_angle,
                "within_tolerance": bool(within_tolerance),
                "tolerance_deg": float(VERTICAL_ALIGN_TOL_DEG),
                "descent_m": descent,
                "last_xy_offset": ref.get("last_xy"),
                "cap": cap,
            }
        )

    # ---- 2.3 PRESS(≤6 步 · 接触触发下压)----

    def _prim_press(self, env: Any) -> None:
        """PRESS:沿世界 z 下压直到接触(参考 ``z_descent``)。

        ``step_m=0.020`` / ``stall_eps_m=0.001`` / ``stall_steps=1``;参考的
        ``max_descent_m=0.40``(20 步)在本移植里由子阶段预算(6 步)夹紧
        —— **预算表不动**(仍是 ``4/20/6/9/3/3/10``)。

        接触判据有两条,满足任一即算接触(连续 ``PRESS_STALL_STEPS`` 步):

        ==========================  =========================================
        绝对                        ``|Δz| ≤ PRESS_STALL_EPS_M``(1 mm)
        相对衰减                    ``|Δz| < PRESS_CONTACT_DECAY_RATIO × 基线``
        ==========================  =========================================

        ``基线`` = 本次下压**实测 |Δz| 的 running max**(不是指令步长)。加这条的
        原因:2026-10-11 实测位置控制器每步只走指令的 ≈42%(命令 20 mm、实测
        ≈8.5 mm),FK 会一直跟着指令下降,绝对阈值几乎永不触发。用**自身峰值
        速率**做基线就与"控制器跟不跟得满"无关。

        **失败一律非致命**(``fatal=False`` + 一条 degradation):接触判定只是
        启发式,判不出来不等于抓不到笔;而只有继续走完 CLOSE/SQUEEZE/PAUSE/
        LIFT,才可能到达 HOLDER_LOCATE / FINAL_PLACE。仅"下压 IK 无解"/"FK
        不可用"这类**连一步都走不动**的情形同样不阻断后续相位。
        """
        cap = int(PRIMITIVE_SUBSTAGE_BUDGET["PRESS"])

        def _give_up(reason: str, extra: dict | None = None) -> None:
            """非致命收尾:记 degradation + 证据,继续下一个子阶段。"""
            record: dict = {"ok": False, "error": reason, "steps": int(self._prim_i)}
            if extra:
                record.update(extra)
            self._record_degradation(self.PRIMITIVE, f"PRESS 跳过:{reason}")
            self._prim_next(record, fatal=False)

        if not self._prim_init:
            self._prim_init = True
            self._prim_ref = {
                "prev_z": None,
                "start_z": None,
                "stalled": 0,
                "step_peak_m": 0.0,
            }
        if self._prim_i >= cap:
            ref = self._prim_ref if isinstance(self._prim_ref, dict) else {}
            _give_up(
                f"{cap} 步内未检测到接触",
                {
                    "descent_m": (
                        None
                        if ref.get("start_z") is None or ref.get("prev_z") is None
                        else float(ref["start_z"]) - float(ref["prev_z"])
                    ),
                    "step_peak_m": ref.get("step_peak_m"),
                    "contact": False,
                },
            )
            return
        pose = self._fk(RIGHT_ARM)
        if pose is None:
            _give_up(f"FK unavailable: {self._fk_error or '未记录'}")
            return
        z_now = float(pose[2, 3])
        q_right, _grip_right = self._arm_joints(self._base, RIGHT)
        target_pose = pose.copy()
        # 参考 ``z_descent``:``target_pose[2, 3] = z_before - step_m``(向下压)。
        target_pose[2, 3] = z_now - float(PRESS_STEP_M)
        q_target = self._ik(RIGHT_ARM, target_pose, q_right)
        if q_target is None or q_target.size != 6:
            _give_up("下压 IK 无解", {"z_m": z_now})
            return
        self._step_right_q(env, q_target, grip=GRIP_RIGHT_OPEN)
        self._prim_i += 1
        ref = self._prim_ref
        if ref.get("start_z") is None:
            ref["start_z"] = z_now
        after = self._fk(RIGHT_ARM)
        z_after = float(after[2, 3]) if after is not None else z_now
        previous = ref.get("prev_z")
        moved = (
            abs(float(previous) - z_after) if previous is not None else None
        )
        if moved is not None:
            peak = float(ref.get("step_peak_m") or 0.0)
            baseline = max(peak, float(moved))
            ref["step_peak_m"] = baseline
            if moved <= float(PRESS_STALL_EPS_M) or (
                baseline > 0.0
                and moved < float(PRESS_CONTACT_DECAY_RATIO) * baseline
            ):
                ref["stalled"] = int(ref.get("stalled", 0)) + 1
            else:
                ref["stalled"] = 0
        ref["prev_z"] = z_after
        ref["last_q"] = q_target
        contact = int(ref.get("stalled", 0)) >= int(PRESS_STALL_STEPS)
        if contact:
            self._prim_next(
                {
                    "ok": True,
                    "contact": True,
                    "steps": int(self._prim_i),
                    "start_z_m": ref.get("start_z"),
                    "contact_z_m": z_after,
                    "descent_m": float(ref.get("start_z", 0.0)) - z_after,
                    "step_m": float(PRESS_STEP_M),
                    "stall_eps_m": float(PRESS_STALL_EPS_M),
                    "stall_steps": int(PRESS_STALL_STEPS),
                    "step_peak_m": ref.get("step_peak_m"),
                    "last_step_m": moved,
                    "decay_ratio": float(PRESS_CONTACT_DECAY_RATIO),
                    "criterion": (
                        "absolute"
                        if moved is not None and moved <= float(PRESS_STALL_EPS_M)
                        else "relative_decay"
                    ),
                }
            )
            return
        if self._prim_i >= cap:
            _give_up(
                f"{cap} 步内未检测到接触",
                {
                    "descent_m": float(ref.get("start_z", 0.0)) - z_after,
                    "step_peak_m": ref.get("step_peak_m"),
                    "last_step_m": moved,
                    "contact": False,
                },
            )

    # ---- 2.4/2.5/2.6 CLOSE / SQUEEZE / PAUSE ----

    def _prim_close(self, env: Any) -> None:
        """CLOSE:在接触位姿上把右夹爪 1.0 → 0.0 插值闭合(参考 ``close_gripper``)。"""
        cap = int(PRIMITIVE_SUBSTAGE_BUDGET["CLOSE"])
        if not self._prim_init:
            self._prim_init = True
            self._prim_ref = self._hold_q_grip()
            self._prim_ref["grip_start"] = float(self._prim_ref["grip_right"])
            self._prim_ref["grip_target"] = float(GRIP_RIGHT_CLOSED)
        ref = self._prim_ref
        if not ref:
            self._prim_next({"ok": False, "error": "CLOSE 初始化失败"})
            return
        alpha = float(self._prim_i + 1) / float(max(1, cap))
        grip = (1.0 - alpha) * float(ref["grip_start"]) + alpha * float(
            ref["grip_target"]
        )
        action = full_action(
            self._base,
            ref["q_left"],
            ref["grip_left"],
            ref["q_right_press"],
            grip,
        )
        self._step_env_step(env, action)
        self._prim_i += 1
        if self._prim_i >= cap:
            self._prim_next(
                {
                    "ok": True,
                    "steps": int(self._prim_i),
                    "gripper_before": ref.get("grip_start"),
                    "gripper_after": float(grip),
                    "arm_held": True,
                }
            )

    def _prim_squeeze(self, env: Any) -> None:
        """SQUEEZE:保持接触位姿、夹爪保持闭合(参考 ``close_gripper`` 的 squeeze)。"""
        cap = int(PRIMITIVE_SUBSTAGE_BUDGET["SQUEEZE"])
        self._prim_hold(env, cap, "squeeze")

    def _prim_pause(self, env: Any) -> None:
        """PAUSE:抬起前的停顿(参考 ``grasp_sequence`` 的 ``pause_before_lift``)。

        参考的原话:专家演示在夹紧结束到开始抬升之间留了约 6 个控制步,
        "夹爪还在上升时就闭合"正是掉笔的原因 —— 所以这 3 步必须保留。
        """
        cap = int(PRIMITIVE_SUBSTAGE_BUDGET["PAUSE"])
        self._prim_hold(env, cap, "pause_before_lift")

    def _prim_hold(self, env: Any, cap: int, phase: str) -> None:
        """在接触位姿上保持 ``cap`` 步、夹爪保持闭合(闭爪后共 6 步)。"""
        self._prim_hold_common(env, cap, phase)

    def _prim_hold_common(self, env: Any, cap: int, phase: str) -> None:
        ref = self._prim_ref
        if not ref or "q_right_press" not in ref:
            ref = self._hold_q_grip()
            ref["q_right_press"] = ref["q_right"]
            self._prim_ref = ref
        action = full_action(
            self._base,
            ref["q_left"],
            ref["grip_left"],
            ref["q_right_press"],
            GRIP_RIGHT_CLOSED,
        )
        self._step_env_step(env, action)
        self._prim_i += 1
        if self._prim_i >= cap:
            self._prim_next({"ok": True, "phase": phase, "steps": int(self._prim_i)})

    def _hold_q_grip(self) -> dict:
        """快照当前两臂关节与夹爪(运动原语"另一条臂保持不动"用)。

        ``q_right_press`` = 接触位姿(``PRESS`` 阶段记下的右臂关节);没有该
        记录时退回当前右臂关节。
        """
        q_left, grip_left = self._arm_joints(self._base, LEFT)
        q_right, grip_right = self._arm_joints(self._base, RIGHT)
        pressed = None
        if isinstance(self._prim_ref, dict):
            pressed = self._prim_ref.get("last_q")
        return {
            "q_left": q_left,
            "grip_left": float(grip_left),
            "q_right": q_right,
            "grip_right": float(grip_right),
            "q_right_press": np.asarray(
                q_right if pressed is None else pressed, dtype=np.float64
            ),
        }

    # ---- 2.7 LIFT(≤10 步)----

    def _prim_lift(self, env: Any) -> None:
        """LIFT:保持姿态沿世界 z 抬升 0.02 m(参考 ``z_lift``,2 mm/步 = 10 步)。

        **合规说明**:参考在抬升后用 ``pen_pose()``(真值)复核"笔是否跟着爪
        起来了"。该检查无法在不读真值的前提下完成,**已删除**;这里只用
        本体感受信号:FK 测得 TCP 的 z 确实上升了(``rise_m > 0``)且 IK 全程
        可解。``summary()["grasp"]["pen_in_gripper_check"] = "omitted"``。
        """
        cap = int(PRIMITIVE_SUBSTAGE_BUDGET["LIFT"])
        if not self._prim_init:
            self._prim_init = True
            pose = self._fk(RIGHT_ARM)
            if pose is None:
                self._prim_next(
                    {
                        "ok": False,
                        "error": f"FK unavailable: {self._fk_error or '未记录'}",
                    }
                )
                return
            ref = dict(self._prim_ref) if self._prim_ref else self._hold_q_grip()
            ref["lift_start_z"] = float(pose[2, 3])
            ref.setdefault("q_right_press", ref.get("q_right"))
            self._prim_ref = ref
        ref = self._prim_ref
        if self._prim_i >= cap:
            self._finish_lift(ref)
            return
        pose = self._fk(RIGHT_ARM)
        if pose is None:
            self._finish_lift(ref)
            return
        q_right, _grip_right = self._arm_joints(self._base, RIGHT)
        target_pose = pose.copy()
        target_pose[2, 3] = float(pose[2, 3]) + float(LIFT_STEP_M)
        q_target = self._ik(RIGHT_ARM, target_pose, q_right)
        if q_target is None or q_target.size != 6:
            self._finish_lift(ref)
            return
        self._step_right_q(env, q_target, grip=GRIP_RIGHT_CLOSED)
        self._prim_i += 1
        if self._prim_i >= cap:
            self._finish_lift(ref)

    def _finish_lift(self, ref: dict) -> None:
        final_pose = self._fk(RIGHT_ARM)
        start_z = ref.get("lift_start_z")
        end_z = float(final_pose[2, 3]) if final_pose is not None else None
        rise = None
        if start_z is not None and end_z is not None:
            rise = float(end_z) - float(start_z)
        self._prim_next(
            {
                "ok": bool(rise is not None and rise > 0.0),
                "steps": int(self._prim_i),
                "start_z_m": start_z,
                "end_z_m": end_z,
                "rise_m": rise,
                "step_m": float(LIFT_STEP_M),
                "pen_in_gripper_check": "omitted",
            }
        )

    # ------------------------------------------------------------------
    # 阶段 3:ALIGNMENT(20 步)
    # ------------------------------------------------------------------

    def _stage_alignment(self, env: Any, inference_times: list[float]) -> None:
        """ALIGNMENT:插值回 CHUNK1 末态,右夹爪保持闭合(参考 ``alignment``)。

        参考 ``ALIGN_TO_ACT_END`` 分支:``align_target[13] = 0.0``,按
        ``np.linspace(1/steps, 1, steps)`` 线性插值 ``align_steps`` 步。
        目标不可用(动作维度不足 14)时记一次 ``degradation`` 后跳过。
        """
        if self._stage_i == 0:
            cap = self._open_state_cap(self.ALIGNMENT, ALIGNMENT_STEPS)
            if cap <= 0:
                return
            self._align_cap = int(cap)
            target = self._align_target
            if target is None or target.size < 14:
                self._record_degradation(
                    self.ALIGNMENT,
                    "CHUNK1 末态不可用(动作维度 < 14),ALIGNMENT 跳过",
                )
                self._enter(self.CONTINUATION)
                return
            self._release_ref["align_target"] = np.asarray(target, dtype=np.float64)
            self._release_ref["align_start"] = self._hold_q_grip()
            self._release_ref["align_target_pose"] = {
                "q_left": np.asarray(target[0:6], dtype=np.float64),
                "grip_left": float(target[6]),
                "q_right": np.asarray(target[7:13], dtype=np.float64),
                "grip_right": float(GRIP_RIGHT_CLOSED),  # align_target[13] = 0.0
            }
        start = self._release_ref.get("align_start")
        goal = self._release_ref.get("align_target_pose")
        if not start or not goal:
            self._enter(self.CONTINUATION)
            return
        cap = int(getattr(self, "_align_cap", ALIGNMENT_STEPS))
        alpha = float(self._stage_i + 1) / float(max(1, cap))
        q_left = (1.0 - alpha) * start["q_left"] + alpha * goal["q_left"]
        grip_left = (1.0 - alpha) * float(start["grip_left"]) + alpha * float(
            goal["grip_left"]
        )
        q_right = (1.0 - alpha) * start["q_right"] + alpha * goal["q_right"]
        action = full_action(
            self._base, q_left, grip_left, q_right, float(GRIP_RIGHT_CLOSED)
        )
        self._step_env_step(env, action)
        self._stage_i += 1
        if self._stage_i >= cap:
            self._enter(self.CONTINUATION)

    # ------------------------------------------------------------------
    # 阶段 4:CONTINUATION(100 步 ACT · 右夹爪锁死)
    # ------------------------------------------------------------------

    def _stage_continuation(self, env: Any, inference_times: list[float]) -> None:
        """CONTINUATION:最多 2 个 ACT chunk,右夹爪通道锁死 0.0。

        参考 ``CONTINUATION_MAX_STEPS=100`` + ``LOCK_RIGHT_GRIPPER=1`` /
        ``LOCK_RIGHT_GRIPPER_STEPS=0``(全程锁定) + 每步把左 TCP 记进
        ``deque(maxlen=10)``(LEFT_TAKEOVER 的运动方向来源) + 一旦
        ``is_task_success()`` 就提前结束。
        """
        if self._stage_i == 0:
            cap = self._open_state_cap(self.CONTINUATION, CONTINUATION_BUDGET)
            if cap <= 0:
                return
            self._continuation_cap = int(cap)
            self._grip_override = {
                self._eef_channel(RIGHT): float(LOCKED_RIGHT_GRIPPER)
            }
            self._continuation_start = self._fk(LEFT_ARM)
            self._left_tcp_history.clear()
            self._record_debug_three_cam("handover_continuation_start_three_cam.png")
        cap = int(getattr(self, "_continuation_cap", CONTINUATION_BUDGET))
        if self._continuation_steps >= cap:
            self._grip_override = None
            self._enter(self.LEFT_TAKEOVER)
            return
        self._act_micro_step(env, inference_times, guarded=True)
        self._continuation_steps += 1
        self._stage_i += 1
        pose = self._fk(LEFT_ARM)
        if pose is not None:
            self._left_tcp_history.append(pose)
        try:
            if self._query_success():
                self._continuation_success = True
                self._grip_override = None
                self._record_debug_three_cam(
                    "handover_continuation_success_three_cam.png"
                )
                self._enter(self.DONE)
                return
        except Exception:  # noqa: BLE001 - 成功判定不可用时照常继续
            pass
        if self._continuation_steps >= cap:
            self._grip_override = None
            self._record_debug_three_cam("handover_continuation_end_three_cam.png")
            self._enter(self.LEFT_TAKEOVER)

    # ------------------------------------------------------------------
    # 阶段 5:LEFT_TAKEOVER(20 步 = 8 + 9 + 3)
    # ------------------------------------------------------------------

    def _stage_left_takeover(self, env: Any, inference_times: list[float]) -> None:
        """LEFT_TAKEOVER:左臂沿 continuation 的运动方向延伸并闭合接管。

        ``extend``(≤8 步,方向来自 CONTINUATION 期间的左 TCP 历史,退化到
        "continuation 首末位姿之差")→ ``close``(9 步,左夹爪 → 0.0)→
        ``hold``(3 步)。参考实现里 extend 的 IK 失败**不算失败**:闭爪仍然执行。
        """
        cap = int(LEFT_TAKEOVER_BUDGET)
        # 首访用 ``_takeover_ref`` 判空,**不能**用 ``_stage_i == 0``:extend 的规划
        # 是纯计算微步(不 step),用 ``_stage_i`` 判首访会反复进入同一初始化块,
        # 一直空转到父类的 MAX_PURE_ADVANCES 兜底截断。
        if not self._takeover_ref:
            cap = self._open_state_cap(self.LEFT_TAKEOVER, LEFT_TAKEOVER_BUDGET)
            if cap <= 0:
                return
            self._takeover_cap = int(cap)
            start_pose = self._continuation_start
            end_pose = self._fk(LEFT_ARM)
            direction, source = self._takeover_direction(start_pose, end_pose)
            ref = {
                "direction": direction,
                "direction_source": source,
                "direction_norm": float(np.linalg.norm(direction))
                if direction is not None
                else 0.0,
                "end_pose": end_pose,
                "extend_ok": False,
                "extend_error": None,
                "start_action": self._hold_q_grip(),
            }
            self._takeover_ref = ref
            if (
                direction is None
                or ref["direction_norm"] < float(TAKEOVER_MIN_DIRECTION_M)
                or end_pose is None
            ):
                ref["extend_error"] = "左臂运动方向过小或 FK 不可用"
            else:
                target_pose = end_pose.copy()
                target_pose[:3, 3] = target_pose[:3, 3] + direction * float(
                    LEFT_EXTEND_M
                )
                q_target = self._ik(LEFT_ARM, target_pose, ref["start_action"]["q_left"])
                if q_target is not None and q_target.size == 6:
                    ref["q_target"] = q_target
                    ref["extend_ok"] = True
                else:
                    ref["extend_error"] = "左臂 extend IK 无解"
            self._record_debug_three_cam("handover_takeover_start_three_cam.png")
            if not ref["extend_ok"]:
                return  # 纯计算微步;下一次 _advance 直接进 CLOSE 段
        ref = self._takeover_ref
        if not ref:
            self._enter(self.RIGHT_RELEASE)
            return
        cap = int(getattr(self, "_takeover_cap", LEFT_TAKEOVER_BUDGET))
        if ref["extend_ok"] and self._stage_i < int(LEFT_EXTEND_STEPS):
            index = self._stage_i
            alpha = float(index + 1) / float(max(1, int(LEFT_EXTEND_STEPS)))
            start = ref["start_action"]
            q_left = (1.0 - alpha) * start["q_left"] + alpha * ref["q_target"]
            action = full_action(
                self._base,
                q_left,
                start["grip_left"],
                start["q_right"],
                start["grip_right"],
            )
            self._step_env_step(env, action)
            self._stage_i += 1
            if self._stage_i >= int(LEFT_EXTEND_STEPS):
                ref["close_start"] = self._hold_q_grip()
            return

        if "close_start" not in ref:
            ref["close_start"] = self._hold_q_grip()
        if self._stage_i < int(LEFT_EXTEND_STEPS) + int(LEFT_CLOSE_STEPS) + int(
            LEFT_HOLD_STEPS
        ):
            close_start = ref["close_start"]
            action = full_action(
                self._base,
                close_start["q_left"],
                float(GRIP_LEFT_CLOSED),
                close_start["q_right"],
                close_start["grip_right"],
            )
            self._step_env_step(env, action)
            self._stage_i += 1
            if self._stage_i >= cap:
                self._finish_takeover()
            return
        self._finish_takeover()

    def _takeover_direction(
        self, start_pose: np.ndarray | None, end_pose: np.ndarray | None
    ) -> tuple[np.ndarray | None, str]:
        """接管方向:优先用 continuation 的左 TCP 历史,退化到首末位姿之差。"""
        if len(self._left_tcp_history) >= 2:
            first = self._left_tcp_history[0]
            last = self._left_tcp_history[-1]
            return (
                np.asarray(last[:3, 3] - first[:3, 3], dtype=np.float64),
                "left_tcp_history_10",
            )
        if start_pose is not None and end_pose is not None:
            return (
                np.asarray(end_pose[:3, 3] - start_pose[:3, 3], dtype=np.float64),
                "continuation_start_end_fallback",
            )
        return None, "unavailable"

    def _finish_takeover(self) -> None:
        ref = self._takeover_ref
        direction = ref.get("direction")
        norm = float(ref.get("direction_norm", 0.0))
        self._takeover = jsonable(
            {
                "direction": None
                if direction is None or norm < 1e-9
                else (direction / norm).tolist(),
                "direction_source": ref.get("direction_source"),
                "history_len": len(self._left_tcp_history),
                "direction_norm_before": norm,
                "extend_ok": bool(ref.get("extend_ok")),
                "extend_error": ref.get("extend_error"),
                "extend_m": float(LEFT_EXTEND_M) if ref.get("extend_ok") else 0.0,
                "close_steps": int(LEFT_CLOSE_STEPS),
                "hold_steps": int(LEFT_HOLD_STEPS),
                "left_grip_after": float(self._arm_joints(self._base, LEFT)[1]),
            }
        )
        if not ref.get("extend_ok"):
            self._record_degradation(
                self.LEFT_TAKEOVER,
                f"左臂 extend 未执行({ref.get('extend_error')}),仍按参考语义闭爪接管",
            )
        self._record_debug_three_cam("handover_takeover_end_three_cam.png")
        self._enter(self.RIGHT_RELEASE)

    # ------------------------------------------------------------------
    # 阶段 6:RIGHT_RELEASE(9 步)
    # ------------------------------------------------------------------

    def _stage_right_release(self, env: Any, inference_times: list[float]) -> None:
        """RIGHT_RELEASE:左夹爪 0.0(握住)、右夹爪 1.0(张开)保持 9 步。"""
        if self._stage_i == 0:
            cap = self._open_state_cap(self.RIGHT_RELEASE, RIGHT_RELEASE_STEPS)
            if cap <= 0:
                return
            self._release_cap = int(cap)
            self._release_ref["release"] = self._hold_q_grip()
            self._record_debug_three_cam("handover_release_before_three_cam.png")
        ref = self._release_ref.get("release")
        if not ref:
            self._enter(self.MANUAL_TARGET)
            return
        action = full_action(
            self._base,
            ref["q_left"],
            float(GRIP_LEFT_CLOSED),
            ref["q_right"],
            float(GRIP_RIGHT_OPEN),
        )
        self._step_env_step(env, action)
        self._stage_i += 1
        if self._stage_i >= int(getattr(self, "_release_cap", RIGHT_RELEASE_STEPS)):
            self._record_debug_three_cam("handover_release_after_three_cam.png")
            self._enter(self.MANUAL_TARGET)

    # ------------------------------------------------------------------
    # 阶段 7:MANUAL_TARGET(43 步 = 35 + 8)
    # ------------------------------------------------------------------

    def _stage_manual_target(self, env: Any, inference_times: list[float]) -> None:
        """MANUAL_TARGET:双臂 smoothstep 到固定路点并保持(参考 ``manual_target``)。

        ``move_to_joint_target(steps=35, hold_steps=8)`` + 参考 ``recording_step``
        在该相位的覆盖 ``action[6] = LEFT_GRIPPER_LOCK``(左夹爪 0.0,笔不掉)。
        平滑插值 ``s = 0.5*(1-cos(pi*alpha))`` 是参考为减小惯性力矩专门加的。
        """
        move_steps = int(MANUAL_TARGET_BUDGET) - 8
        hold_steps = 8
        if self._stage_i == 0:
            cap = self._open_state_cap(self.MANUAL_TARGET, MANUAL_TARGET_BUDGET)
            if cap <= 0:
                return
            hold_steps = int(min(8, cap))
            move_steps = int(cap - hold_steps)
            start = self._hold_q_grip()
            raw = np.asarray(MANUAL_TARGET_QPOS, dtype=np.float64).reshape(-1)
            goal = {
                "q_left": np.asarray(raw[0:6], dtype=np.float64),
                "grip_left": float(GRIP_LEFT_CLOSED),
                "q_right": np.asarray(raw[7:13], dtype=np.float64),
                "grip_right": float(raw[13]),
            }
            self._manual_ref = {
                "start": start,
                "goal": goal,
                "raw_target": raw,
                "move_steps": move_steps,
                "hold_steps": hold_steps,
            }
            self._record_debug_three_cam("handover_manual_target_before_three_cam.png")
        ref = self._manual_ref
        if not ref:
            self._enter(self.HOLDER_LOCATE)
            return
        move_steps = int(ref.get("move_steps", MANUAL_TARGET_BUDGET - 8))
        hold_steps = int(ref.get("hold_steps", 8))
        start, goal = ref["start"], ref["goal"]
        if self._stage_i < move_steps:
            alpha = float(self._stage_i + 1) / float(max(1, move_steps))
            s = 0.5 * (1.0 - float(np.cos(np.pi * alpha)))
            q_left = (1.0 - s) * start["q_left"] + s * goal["q_left"]
            q_right = (1.0 - s) * start["q_right"] + s * goal["q_right"]
            grip_right = (1.0 - s) * float(start["grip_right"]) + s * float(
                goal["grip_right"]
            )
        else:
            q_left, q_right = goal["q_left"], goal["q_right"]
            grip_right = float(goal["grip_right"])
        action = full_action(
            self._base, q_left, float(GRIP_LEFT_CLOSED), q_right, grip_right
        )
        self._step_env_step(env, action)
        self._stage_i += 1
        if self._stage_i >= move_steps + hold_steps:
            self._manual_ref["final"] = self._hold_q_grip()
            self._manual_ref["final_error_max_abs"] = float(
                np.max(np.abs(self._current_action_vector() - ref["raw_target"]))
            )
            self._record_debug_three_cam("handover_manual_target_after_three_cam.png")
            self._debug_write_json(
                "handover_manual_target.json",
                {
                    "target_qpos": ref["raw_target"].tolist(),
                    "final_error_max_abs": self._manual_ref["final_error_max_abs"],
                },
            )
            self._enter(self.HOLDER_LOCATE)

    # ------------------------------------------------------------------
    # 阶段 8:HOLDER_LOCATE(0 步 · 纯计算)
    # ------------------------------------------------------------------

    def _stage_holder_locate(self, env: Any, inference_times: list[float]) -> None:
        """HOLDER_LOCATE:双腕 VLM(归一化坐标)+ 内圈椭圆拟合 + 双目三角化。

        **不消耗 env.step**(双臂停在 MANUAL_TARGET 路点上不动);输出
        ``estimated_top_center_xyz`` 作为放置目标。参考的
        ``holder_stereo_inner_eval.evaluate_inner_ellipse`` 用严格同一套几何
        (内参 ``priors.camera_K``、腕相机位姿走 FK + 公开安装常量、
        内圈椭圆中心做双目三角化),移植版删掉了它内部的 GT 投影/误差字段
        (``gt_top_center`` / ``error_*_mm``),只保留估计链路。

        任何一步失败 → 记 ``failure`` 并跳 ``FALLBACK_ACT``(纯 ACT 跑完剩余
        步数),不会带着错误的筒口位置去放笔。
        """
        arms = (LEFT, RIGHT)
        images: dict[str, np.ndarray] = {}
        poses: dict[str, np.ndarray] = {}
        intrinsics: dict[str, np.ndarray] = {}
        entries: dict[str, dict] = {}
        stereo = _lazy("stereo")
        priors = _lazy("priors")
        for arm in arms:
            rgb, _obs = stereo.capture(self._base, arm)
            images[arm] = rgb
            poses[arm] = stereo.wrist_camera_pose(self._base, arm)
            intrinsics[arm] = priors.camera_K(f"cam_{arm}_wrist")
            answer = self._ask_holder_vlm(rgb, arm)
            entries[arm] = {"vlm": answer}
            self._holder_attempts.append({"arm": arm, **jsonable(answer)})
            bbox = answer.get("bbox_px")
            if not bbox and answer.get("center_px") is not None:
                cx, cy = (float(v) for v in answer["center_px"])
                half = float(INNER_RIM_BBOX_HALF_PX)
                bbox = [cx - half, cy - half, cx + half, cy + half]
            if not bbox:
                self._record_failure(
                    self.HOLDER_LOCATE,
                    f"{arm} 腕相机未定位到 holder(VLM: {answer.get('error')})",
                )
                self._holder = jsonable(
                    {
                        "ok": False,
                        "arms": entries,
                        "coordinate_convention": "vlm_normalized_0_1",
                    }
                )
                self._enter(self.FALLBACK_ACT)
                return
            ellipse = _fit_inner_rim(images[arm], bbox)
            entries[arm]["bbox_px"] = [float(v) for v in bbox]
            entries[arm]["ellipse"] = jsonable(ellipse)
            if not ellipse.get("ok"):
                self._record_failure(
                    self.HOLDER_LOCATE,
                    f"{arm} 内圈椭圆拟合失败: {ellipse.get('error')}",
                )
                self._holder = jsonable(
                    {
                        "ok": False,
                        "arms": entries,
                        "coordinate_convention": "vlm_normalized_0_1",
                    }
                )
                self._enter(self.FALLBACK_ACT)
                return
            entries[arm]["centre_uv"] = [float(v) for v in ellipse["centre_uv"]]

        projections = [
            _projection_matrix(intrinsics[arm], poses[arm]) for arm in arms
        ]
        pixels = [entries[arm]["centre_uv"] for arm in arms]
        point = _triangulate_dlt(projections, pixels)
        baseline = float(np.linalg.norm(poses[RIGHT][:3, 3] - poses[LEFT][:3, 3]))
        self._holder = jsonable(
            {
                "ok": bool(point is not None),
                "method": "dual_wrist_inner_ellipse_triangulation",
                "arms": entries,
                "estimated_top_center_xyz": None
                if point is None
                else point.tolist(),
                "baseline_m": baseline,
                "coordinate_convention": "vlm_normalized_0_1",
                "left_camera_xyz": poses[LEFT][:3, 3].tolist(),
                "right_camera_xyz": poses[RIGHT][:3, 3].tolist(),
            }
        )
        self._debug_write_json("handover_holder_locate.json", self._holder)
        if point is None:
            self._record_failure(
                self.HOLDER_LOCATE, "双目三角化退化(齐次分量过小)"
            )
            self._enter(self.FALLBACK_ACT)
            return
        self._holder_estimate = np.asarray(point, dtype=np.float64)
        self._enter(self.FINAL_PLACE)

    def _ask_holder_vlm(self, rgb: np.ndarray, arm: str) -> dict:
        """问一次 VLM 并把**归一化**坐标换算回像素(见模块 docstring)。"""
        output_dir = self._debug_subdir(f"holder_vlm/{arm}")
        vlm = _lazy("vlm")
        try:
            if not vlm.vlm_available():
                return {"ok": False, "error": "VLM 未配置(HYBRID_VLM_URL 为空)"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"vlm.vlm_available 失败: {exc}"}
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        parsed: dict | None = None
        result: dict = {}
        for attempt in range(int(VLM_ATTEMPTS)):
            result = vlm.vlm_chat(
                HOLDER_PROMPT,
                [rgb],
                timeout_s=int(VLM_TIMEOUT_S),
            )
            parsed = _json_from_text(result.get("text"))
            if parsed is not None:
                break
            if attempt + 1 < int(VLM_ATTEMPTS):
                time.sleep(float(VLM_RETRY_SLEEP_S))
        self._debug_write_json(
            f"holder_vlm/{arm}_answer.json",
            {"parsed": parsed, "error": result.get("error"), "attempts": attempt + 1},
        )
        if parsed is None:
            return {
                "ok": False,
                "error": result.get("error") or "VLM 未返回可解析 JSON",
                "latency_s": result.get("latency_s"),
            }
        if not bool(parsed.get("visible", False)):
            return {
                "ok": False,
                "visible": False,
                "error": "VLM 报告 holder 不可见",
                "latency_s": result.get("latency_s"),
            }
        raw_bbox = parsed.get("bbox_xyxy_norm")
        if raw_bbox is None:
            raw_bbox = parsed.get("bbox_xyxy")
        bbox_px = _norm_to_pixels(raw_bbox, width, height) if raw_bbox else None
        if bbox_px is not None and len(bbox_px) == 4:
            x0, x1 = sorted(
                (max(0.0, min(width - 1.0, bbox_px[0])), max(0.0, min(width - 1.0, bbox_px[2])))
            )
            y0, y1 = sorted(
                (max(0.0, min(height - 1.0, bbox_px[1])), max(0.0, min(height - 1.0, bbox_px[3])))
            )
            bbox_px = [x0, y0, x1, y1]
        else:
            bbox_px = None
        raw_center = parsed.get("center_uv_norm")
        if raw_center is None:
            raw_center = parsed.get("center_uv")
        center_px = _norm_to_pixels(raw_center, width, height) if raw_center else None
        if center_px is not None and len(center_px) != 2:
            center_px = None
        if bbox_px is None and center_px is None:
            return {"ok": False, "error": "VLM 未返回可用 bbox/center"}
        self._debug_write_png(f"holder_vlm/{arm}_input.png", rgb)
        return {
            "ok": True,
            "visible": True,
            "bbox_px": bbox_px,
            "center_px": center_px,
            "bbox_norm": raw_bbox,
            "center_norm": raw_center,
            "image_size": [int(width), int(height)],
            "confidence": float(parsed.get("confidence") or 0.0),
            "latency_s": result.get("latency_s"),
        }

    # ------------------------------------------------------------------
    # 阶段 9:FINAL_PLACE(48 步 = 20 + 10 + 9 + 9)
    # ------------------------------------------------------------------

    def _stage_final_place(self, env: Any, inference_times: list[float]) -> None:
        """FINAL_PLACE:左臂到筒口正上方(姿态受约束)→ 保持 → 释放 → 稳定。

        参考 ``place_above_release2.place_above_and_release``:
        ``above = fk_pose(left); above[:3,3] = center + [0,0,clearance]``,
        多起点 IK(当前关节 + 60 个随机起点),再 ``move_left_to(steps=25)`` +
        ``release_left(steps=9)``。移植版把移动压到 20 步、并在释放前加 10 步
        握持保持、释放后 9 步稳定(= 参考压缩版的 48 步账:
        "移动固定 20 步(与 expert 同款)、删 10 步保持")。

        **WS-I 的差异(姿态)**:目标姿态不再"保持当前 TCP 姿态",而是由
        :meth:`_place_target_pose` 显式给出(TCP 的 X 轴垂直于桌面、Z 轴落在
        桌面平面内)。步数账与释放时序完全不变(仍是 48 步 = 20+10+9+9)。

        **WS-K 的差异(接近高度与位置容差,2026-10-11 根因修复)**:

        1. **接近高度族 = :data:`PLACE_CLEARANCE_M`(11 cm)→
           :data:`PLACE_CLEARANCE_LADDER_M`(17 cm)→ 5 cm**。之前一直在
           "筒口 +5 cm"上放宽俯仰角,方向错了:200 万样本 FK 扫描显示 5 cm 处
           20 mm 邻域内的最优 TCP X 轴轴偏差是 26.06°(永远不达标),抬高到
           11 cm 后同一邻域是 9.209°(达标)。
        2. **位置容差先严后松**:先按 :data:`PLACE_POSITION_TOL_LADDER_M[0]`
           (<10 mm)找,找不到才放宽到 <20 mm;实际用的那一级**如实**写进
           ``final_place.position_tol_mm`` 与 ``accepted_variant``,不默默放宽。
        3. **姿态自检门槛不删**:``constraint_ok = 轴偏差 < 15°``
           (:data:`PLACE_TCP_X_TOL_DEG`),读数照旧来自 :meth:`_place_tcp_constraint_check`。
           只有在"位置进容差 **且** 姿态 <15°"时才把该变体标成 ``ok``。
        4. 若 11/17 cm + 20 mm 全都拿不到姿态达标的解,就按**位置优先**取全局最近
           位置的候选(用户 2026-10-11:"只要笔能放进去就行"),并如实记
           ``constraint_ok=False`` + 一条 degradation;连位置候选都没有才原地释放。

        **代价(如实记录,不掩饰)**:11 cm 比参考的 5 cm 高 6 cm,笔落下时空中
        时间更长、落地速度更大,落点姿态更容易翻/弹;同时该高度上只能做到
        ~9° 轴偏差与 10~20 mm 位置残差,不是 0°/5 mm。

        移动完成后用**自己的 FK** 做一次姿态自检(见
        :meth:`_place_tcp_constraint_check`),不读任何物体真值。
        """
        if self._holder_estimate is None:
            self._record_failure(
                self.FINAL_PLACE, "没有可用的筒口估计(HOLDER_LOCATE 未产出)"
            )
            self._enter(self.FALLBACK_ACT)
            return
        if not self._place_ref:
            cap = self._open_state_cap(self.FINAL_PLACE, FINAL_PLACE_BUDGET)
            if cap <= 0:
                return
            # 四个子段的名义账:移动 20 + 释放前保持 10 + 释放 9 + 稳定 9。
            # 预算被压缩时按同一顺序夹紧(释放永远排在稳定之前)。
            move = int(min(PLACE_MOVE_STEPS, cap))
            pre_hold = int(min(PLACE_PRE_HOLD_STEPS, max(0, cap - move)))
            release_n = int(min(PLACE_RELEASE_STEPS, max(0, cap - move - pre_hold)))
            settle_n = int(max(0, cap - move - pre_hold - release_n))
            pose = self._fk(LEFT_ARM)
            if pose is None:
                self._record_failure(self.FINAL_PLACE, "左臂 FK 不可用")
                self._enter(self.FALLBACK_ACT)
                return
            # 目标姿态 = **显式约束**(TCP X 轴 ⊥ 桌面)而不是"保持当前姿态";
            # 位置 = 筒口中心正上方 PLACE_CLEARANCE_M(11 cm)。只用到 center
            # (VLM + 双目估计)与自己的 TCP 位姿(本体感受),不读物体真值。
            exact = self._place_target_pose(self._holder_estimate, pose)
            snapshot = self._hold_q_grip()
            q_start = np.asarray(snapshot["q_left"], dtype=np.float64)
            # **位置容差先严后松**:外层遍历 PLACE_POSITION_TOL_LADDER_M
            # (<10 mm → <20 mm),内层遍历"接近高度族 × 俯仰族"构造的候选
            # (见 _place_pose_variants)。接受条件 = 位置残差进当前容差
            # **且** 解出来的 FK 姿态轴偏差 < PLACE_TCP_X_TOL_DEG(15°);
            # 姿态约束不删,退让量与实际用的容差都记录在案。
            chosen: tuple | None = None
            chosen_pose = exact
            chosen_variant: dict | None = None
            attempts: list[dict] = []
            for tol_m in PLACE_POSITION_TOL_LADDER_M:
                for candidate, variant in self._place_pose_variants(
                    self._holder_estimate, pose
                ):
                    solved = self._solve_left_pose(candidate, q_start)
                    residual = None if solved is None else float(solved[1])
                    achieved_dev = (
                        None if solved is None
                        else self._place_x_dev_deg_of_q(solved[0])
                    )
                    attempts.append(
                        {
                            **variant,
                            "position_tol_mm": float(tol_m) * 1000.0,
                            "solved": solved is not None,
                            "ik_error_mm": (
                                None if residual is None else residual * 1000.0
                            ),
                            "achieved_x_dev_deg": achieved_dev,
                            "ik_detail": getattr(self, "_ik_error", None),
                        }
                    )
                    if (
                        solved is not None
                        and residual is not None
                        and residual < float(tol_m)
                        and achieved_dev is not None
                        and achieved_dev < float(PLACE_TCP_X_TOL_DEG)
                    ):
                        chosen = solved
                        chosen_pose = candidate
                        chosen_variant = {
                            **variant,
                            "position_tol_mm": float(tol_m) * 1000.0,
                            "solver": "multi_seed_ik",
                            "achieved_x_dev_deg": achieved_dev,
                        }
                        break
                if chosen is not None:
                    break
            if chosen is None:
                # 解析 IK 全族判"无解"。**判无解 ≠ 工作空间里没有**:2026-10-11 实测
                # 解析 IK(OPW)在这个目标上(甚至"只换位置"或"只换姿态"的单侧目标)
                # 恒 success=0,而 200 万关节角 FK 扫描显示"TCP X 轴垂直桌面"在
                # **抬高到 11/17 cm** 后可达。所以改用 **纯 FK 采样 + 局部精修**
                # 直接优化"位置残差 + X 轴轴偏差",并按**接近高度族**逐个试:
                # 11 cm(第一优先)→ 17 cm → 5 cm,每个高度内部再走 10 mm→20 mm
                # 的容差阶梯(见 _place_search_pose 的 pos_tol_ladder)。
                clearance_family = [
                    float(PLACE_CLEARANCE_M),
                    *[float(c) for c in PLACE_CLEARANCE_LADDER_M],
                ]
                best_failed_search: dict | None = None
                best_failed_clearance: float | None = None
                for clearance in clearance_family:
                    goal_pose = exact.copy()
                    goal_pose[:3, 3] = np.asarray(
                        self._holder_estimate, dtype=np.float64
                    ).reshape(-1)[:3] + np.array(
                        [0.0, 0.0, float(clearance)], dtype=np.float64
                    )
                    search = self._place_search_pose(LEFT_ARM, goal_pose, q_start)
                    self._place_search = search
                    if search is not None and bool(search.get("ok")):
                        clearance_mm = float(clearance) * 1000.0
                        tol_mm = float(search.get("pos_tol_m") or 0.0) * 1000.0
                        self._record_degradation(
                            self.FINAL_PLACE,
                            "解析 IK 全族无解,改用 FK 搜索式位姿求解:接近高度 "
                            f"{clearance_mm:.0f} mm、位置容差 {tol_mm:.0f} mm"
                            "(先严后松的阶梯)、位置残差 "
                            f"{float(search['pos_err_m']) * 1000.0:.1f} mm、"
                            "TCP X 轴轴偏差 "
                            f"{float(search['x_dev_deg']):.1f}°"
                            f"(评估 {int(search['n_eval'])} 个关节角)",
                        )
                        chosen = (
                            search["q"],
                            float(search["pos_err_m"]),
                            -1,
                        )
                        chosen_pose = goal_pose
                        chosen_variant = {
                            "pitch_deg": 0.0,
                            "clearance_m": float(clearance),
                            "clearance_mm": clearance_mm,
                            "position_tol_mm": tol_mm,
                            "label": (
                                f"fk_search_clr{clearance_mm:.0f}mm_"
                                f"tol{tol_mm:.0f}mm"
                            ),
                            "solver": "fk_search",
                            "search_pos_err_m": float(search["pos_err_m"]),
                            "search_pos_tol_m": float(
                                search.get("pos_tol_m") or 0.0
                            ),
                            "search_x_dev_deg": float(search["x_dev_deg"]),
                            "search_n_eval": int(search["n_eval"]),
                            "search_best_x_dev_deg_per_tol": search.get(
                                "best_x_dev_deg_per_tol"
                            ),
                            "search_best_pos_err_m_per_tol": search.get(
                                "best_pos_err_m_per_tol"
                            ),
                        }
                        break
                    if search is not None:
                        # 记住"全局最近位置"的那次搜索,供位置优先兜底用。
                        if best_failed_search is None or float(
                            search.get("pos_err_m", 1e9)
                        ) < float(best_failed_search.get("pos_err_m", 1e9)):
                            best_failed_search = search
                            best_failed_clearance = float(clearance)
                if chosen is None and best_failed_search is not None:
                    # 没有任何"位置进容差且姿态 <15°"的解。按用户 2026-10-11 的
                    # 优先级(**位置优先:只要笔能放进去**),取全局最近位置的候选
                    # 而不是原地释放;姿态不达标如实记 degradation。
                    self._record_degradation(
                        self.FINAL_PLACE,
                        "放置姿态约束在 11/17/5 cm × 10/20 mm 全族内均未达标,"
                        "按**位置优先**取全局最近位置候选(接近高度 "
                        f"{float(best_failed_clearance or PLACE_CLEARANCE_M) * 1000.0:.0f} mm、"
                        "位置残差 "
                        f"{float(best_failed_search['pos_err_m']) * 1000.0:.1f} mm、"
                        "TCP X 轴轴偏差 "
                        f"{float(best_failed_search['x_dev_deg']):.1f}° >= "
                        f"{float(PLACE_TCP_X_TOL_DEG):.0f}°)",
                    )
                    chosen = (
                        best_failed_search["q"],
                        float(best_failed_search["pos_err_m"]),
                        -1,
                    )
                    chosen_pose = exact.copy()
                    chosen_pose[:3, 3] = np.asarray(
                        self._holder_estimate, dtype=np.float64
                    ).reshape(-1)[:3] + np.array(
                        [0.0, 0.0, float(best_failed_clearance or PLACE_CLEARANCE_M)],
                        dtype=np.float64,
                    )
                    chosen_variant = {
                        "pitch_deg": 0.0,
                        "clearance_m": float(
                            best_failed_clearance or PLACE_CLEARANCE_M
                        ),
                        "position_tol_mm": None,
                        "label": "fk_search_position_priority_pose_not_met",
                        "solver": "fk_search",
                        "search_pos_err_m": float(
                            best_failed_search["pos_err_m"]
                        ),
                        "search_x_dev_deg": float(best_failed_search["x_dev_deg"]),
                        "search_n_eval": int(best_failed_search["n_eval"]),
                    }
            if chosen is None:
                # 两种机制都失败(FK 连一个样本都算不出来):仍**非致命** ——
                # 记 degradation,保持当前左臂位姿走完释放/稳定步
                # (不谎报成功;姿态自检照样如实读一次)。
                self._record_degradation(
                    self.FINAL_PLACE,
                    "放置目标位姿:解析 IK 全族 + FK 搜索均无解(接近高度 "
                    "11/17/5 cm × 位置容差 10/20 mm × 俯仰 ±8/±15/±25/±35°);"
                    "改为原地释放并如实记录姿态偏差",
                )
                chosen_pose = exact
                chosen_variant = {
                    "pitch_deg": None,
                    "clearance_m": float(PLACE_CLEARANCE_M),
                    "position_tol_mm": None,
                    "label": "ik_failed_hold_current",
                }
                chosen = (
                    np.asarray(q_start, dtype=np.float64),
                    0.0,
                    0,
                )
            q_target, err, seed_index = chosen
            position_tol_m = chosen_variant.get("position_tol_mm")
            self._place_ref = {
                "snapshot": snapshot,
                "q_start": q_start,
                "q_target": np.asarray(q_target, dtype=np.float64),
                "ik_error_mm": float(err) * 1000.0,
                "ik_seed_index": int(seed_index),
                "target_xyz": np.asarray(chosen_pose[:3, 3], dtype=np.float64),
                "target_rot": np.asarray(chosen_pose[:3, :3], dtype=np.float64),
                "accepted_variant": dict(chosen_variant or {}),
                # 实际用的位置容差(米);None = 姿态优先失败、改走位置优先兜底。
                "position_tol_m": (
                    None if position_tol_m is None else float(position_tol_m) / 1000.0
                ),
                "ik_attempts": attempts,
                "ik_failed": chosen_variant is not None
                and chosen_variant.get("pitch_deg") is None,
                "move": move,
                "pre_hold": pre_hold,
                "release": release_n,
                "settle": settle_n,
                # 移动完成后填一次姿态自检(本体感受,见 _place_tcp_constraint_check)。
                "constraint": None,
            }
            self._place = jsonable(
                {
                    "ok": False,
                    "mode": "above_rim_tcp_x_normal_to_table",
                    "estimated_top_center_xyz": self._holder_estimate.tolist(),
                    "clearance_m": float(chosen_variant.get("clearance_m")),
                    "position_tol_mm": (
                        None
                        if position_tol_m is None
                        else float(position_tol_m)
                    ),
                    "target_tcp_xyz": self._place_ref["target_xyz"].tolist(),
                    "target_tcp_rot": self._place_ref["target_rot"].tolist(),
                    "place_tcp_x_world": _place_tcp_x_world().tolist(),
                    "ik_error_mm": self._place_ref["ik_error_mm"],
                    "ik_seed_index": self._place_ref["ik_seed_index"],
                    "accepted_variant": self._place_ref["accepted_variant"],
                    "ik_attempts": self._place_ref["ik_attempts"],
                    "ik_failed": self._place_ref["ik_failed"],
                    "constraint_ok": None,
                    "tcp_x_angle_to_world_z_deg": None,
                    "tcp_x_axis_deviation_deg": None,
                    "tcp_x_alignment_to_place_axis_deg": None,
                    "tcp_x_world": None,
                    "steps": {
                        "move": int(move),
                        "pre_hold": int(pre_hold),
                        "release": int(release_n),
                        "settle": int(settle_n),
                    },
                }
            )
        ref = self._place_ref
        snapshot = ref["snapshot"]
        i = int(self._stage_i)
        move = int(ref.get("move", PLACE_MOVE_STEPS))
        pre_hold = move + int(ref.get("pre_hold", PLACE_PRE_HOLD_STEPS))
        release = pre_hold + int(ref.get("release", PLACE_RELEASE_STEPS))
        settle = release + int(ref.get("settle", PLACE_SETTLE_STEPS))

        if i < move:
            alpha = float(i + 1) / float(max(1, move))
            q_left = (1.0 - alpha) * ref["q_start"] + alpha * ref["q_target"]
            grip_left = float(GRIP_LEFT_CLOSED)
        else:
            q_left = ref["q_target"]
            grip_left = (
                float(GRIP_LEFT_CLOSED)
                if i < release
                else float(GRIP_LEFT_OPEN)
            )
            # **移动已完成**(上一步已经命令到 q_target):立刻用自己的 FK 读一次
            # TCP 姿态,做 §2.4 的姿态自检。只读本体感受,不读任何物体真值。
            if ref.get("constraint") is None:
                ref["constraint"] = self._place_tcp_constraint_check()
        action = full_action(
            self._base,
            q_left,
            grip_left,
            snapshot["q_right"],
            snapshot["grip_right"],
        )
        self._step_env_step(env, action)
        self._stage_i += 1
        if self._stage_i >= settle:
            final_pose = self._fk(LEFT_ARM)
            reached = None
            if final_pose is not None:
                reached = float(
                    np.linalg.norm(
                        np.asarray(final_pose[:3, 3], dtype=np.float64)
                        - np.asarray(ref["target_xyz"], dtype=np.float64)
                    )
                )
            # 自检读数:正常情况下是"移动刚完成"时读的那一次;若移动子段被预算
            # 压到 0 步(没走到上面的分支),就退化为此刻补读一次。
            constraint = ref.get("constraint")
            if constraint is None:
                constraint = self._place_tcp_constraint_check()
                ref["constraint"] = constraint
            # 位置是否达标:阈值 = **实际采用的那一级容差**(10 mm 或 20 mm),
            # 不是写死的 10 mm。用了 20 mm 就在这里如实体现。
            tol_used_m = ref.get("position_tol_m")
            if tol_used_m is None:
                tol_used_m = float(PLACE_IK_POSITION_TOL_M)
            tol_used_m = float(tol_used_m)
            self._place = jsonable(
                {
                    "ok": bool(reached is not None and reached <= tol_used_m),
                    "mode": "above_rim_tcp_x_normal_to_table",
                    "estimated_top_center_xyz": self._holder_estimate.tolist(),
                    "target_tcp_xyz": ref["target_xyz"].tolist(),
                    "target_tcp_rot": np.asarray(
                        ref["target_rot"], dtype=np.float64
                    ).tolist(),
                    "place_tcp_x_world": _place_tcp_x_world().tolist(),
                    "reached_tcp_xyz": None
                    if final_pose is None
                    else final_pose[:3, 3].tolist(),
                    "residual_mm": None if reached is None else reached * 1000.0,
                    "ik_error_mm": ref["ik_error_mm"],
                    "ik_seed_index": ref["ik_seed_index"],
                    # ---- 分级放宽用了哪一级(用户要核对的字段) ----
                    "accepted_variant": ref.get("accepted_variant"),
                    "position_tol_mm": (
                        None if tol_used_m is None else tol_used_m * 1000.0
                    ),
                    "ik_attempts": ref.get("ik_attempts"),
                    "ik_failed": bool(ref.get("ik_failed", False)),
                    "fk_search": getattr(self, "_place_search", None),
                    "clearance_m": (
                        (ref.get("accepted_variant") or {}).get("clearance_m")
                    ),
                    "release_open_value": float(GRIP_LEFT_OPEN),
                    # ---- §2.4 姿态自检(本体感受) ----
                    "constraint_ok": constraint.get("constraint_ok"),
                    "tcp_x_angle_to_world_z_deg": constraint.get(
                        "tcp_x_angle_to_world_z_deg"
                    ),
                    "tcp_x_axis_deviation_deg": constraint.get(
                        "tcp_x_axis_deviation_deg"
                    ),
                    "tcp_x_alignment_to_place_axis_deg": constraint.get(
                        "tcp_x_alignment_to_place_axis_deg"
                    ),
                    "tcp_x_world": constraint.get("tcp_x_world"),
                    "tcp_z_world": constraint.get("tcp_z_world"),
                    "tcp_x_tol_deg": float(PLACE_TCP_X_TOL_DEG),
                    "constraint_fk_available": constraint.get("fk_available"),
                    "steps": {
                        "move": move,
                        "pre_hold": int(ref.get("pre_hold", PLACE_PRE_HOLD_STEPS)),
                        "release": int(ref.get("release", PLACE_RELEASE_STEPS)),
                        "settle": int(ref.get("settle", PLACE_SETTLE_STEPS)),
                    },
                }
            )
            if not self._place.get("ok"):
                self._record_failure(
                    self.FINAL_PLACE,
                    f"最终 TCP 残留 {self._place.get('residual_mm')} mm > "
                    f"{tol_used_m * 1000.0:.0f} mm(实际采用的位置容差)",
                    dedup=False,
                )
            if self._place.get("constraint_ok") is False:
                # 如实记录:姿态约束没达成时只记 degradation(不谎报成功);
                # 任务成败由官方 eval_policy.py 判定,本文件不做自评估。
                self._record_degradation(
                    self.FINAL_PLACE,
                    "放置姿态约束未达成:TCP X 轴与桌面法向的轴偏差 "
                    f"{self._place.get('tcp_x_axis_deviation_deg')}° "
                    f">= {float(PLACE_TCP_X_TOL_DEG)}°"
                    "(有符号夹角 "
                    f"{self._place.get('tcp_x_angle_to_world_z_deg')}°,"
                    "偏差过大通常意味着多起点 IK 只满足了位置、没满足姿态)",
                )
            self._record_debug_write_place()
            self._enter(self.DONE)

    def _record_debug_write_place(self) -> None:
        self._record_debug_three_cam("handover_final_place_after_three_cam.png")
        self._debug_write_json("handover_final_place.json", self._place)

    # ------------------------------------------------------------------
    # 几何原语(FK / IK / 四元数),device-aware
    # ------------------------------------------------------------------

    def _device(self) -> Any:
        """张量该放的设备:优先 ``robot.device``,否则 ``act_policy`` 的设备。"""
        device = getattr(self._base, "device", None)
        if device is None:
            device = getattr(getattr(self._base, "robot", None), "device", None)
        if device is None:
            device = _timing_device(self.act_policy)
        return device

    def _fk(self, arm: str) -> np.ndarray | None:
        """``<arm>_arm`` 的 FK 4x4(与参考 ``left_tcp_pose`` / ``fk_pose`` 同口径)。

        口径(2026-10-11 在真实 RTX 4090 + ``EmbodiChain_main 0.2.4`` 上实测):

        * ``robot.compute_fk(q6, name="<arm>_arm", to_matrix=True)`` 返回的是该
          control part 的 **tool/TCP 帧**,``link_names=[...]`` / ``end_link_name``
          **被忽略**(传与不传逐位相同);
        * 这与 ``compute_ik(pose, name="<arm>_arm")`` 期望的输入帧**是同一个帧**
          ——本方法与 :meth:`_ik` 必须同口径,否则目标位姿会整体偏
          ``TOOL_FROM_LINK6``(实测 143 mm / 180°);
        * ``robot.get_link_pose(link_name="<arm>_link6", to_matrix=True)`` 可用,
          返回的才是 ``<arm>_link6`` 连杆帧,比 FK 返回值恰好少一个
          ``priors.TOOL_FROM_LINK6``(实测残差 1e-06)。

        输入必须是 **torch 张量**:``compute_fk`` 收到 ``np.ndarray`` 会抛
        ``AttributeError: 'numpy.ndarray' object has no attribute 'dim'``,
        形状不是 ``(N, 6)`` 会抛 ``RuntimeError: Joint positions shape mismatch``。

        三级回退,**全部保持 tool/TCP 口径**(绝不静默返回别的帧):

        1. ``compute_fk(q6, name=..., to_matrix=True)``;
        2. ``compute_fk(q6, name=..., to_matrix=False)``(pos+quat 自己拼矩阵);
        3. ``get_link_pose("<arm>_link6") @ TOOL_FROM_LINK6``(复用 :mod:`priors`
           里实测解出的 ``link6 -> tool`` 常量)。

        每次失败都把真实异常写进 ``self._fk_error``,不再静默返回 ``None``。
        """
        text = str(arm)
        name = text if text.endswith("_arm") else f"{text}_arm"
        link_name = f"{text[: -len('_arm')] if text.endswith('_arm') else text}_link6"
        self._fk_error = None
        robot = getattr(self._base, "robot", None)
        if robot is None:
            self._fk_error = "绑定的 base env 上没有 robot 句柄"
            return None

        def _coerce(value: Any) -> np.ndarray | None:
            """把 FK 返回值(4x4 或 7 维 pos+quat)统一成 4x4;失败返回 None。"""
            try:
                matrix = np.asarray(npv(value), dtype=np.float64)
            except Exception as exc:  # noqa: BLE001
                self._fk_error = f"FK 返回值无法转成 array: {type(exc).__name__}: {exc}"
                return None
            while matrix.ndim > 2:
                matrix = matrix[0]
            if matrix.shape == (7,):
                # ``compute_fk(to_matrix=False)`` 返回 ``[pos(3), quat(4)]``,四元数
                # 由 ``embodichain.utils.math.quat_from_matrix`` 给出,顺序 **(w,x,y,z)**;
                # ``_quat_to_matrix`` 收的是 (x,y,z,w),这里显式换序。
                w, x, y, z = (float(v) for v in matrix[3:])
                out = np.eye(4, dtype=np.float64)
                out[:3, 3] = matrix[:3]
                out[:3, :3] = HandoverFlow._quat_to_matrix(np.array([x, y, z, w]))
                return out
            if matrix.shape == (3, 4):
                out = np.eye(4, dtype=np.float64)
                out[:3, :4] = matrix
                return out
            if matrix.shape == (4, 4):
                return matrix.copy()
            self._fk_error = f"FK 返回值形状不可解读: {matrix.shape}"
            return None

        try:
            import torch

            raw = robot.get_qpos(name=name)
            cpu = torch.as_tensor(npv(raw), dtype=torch.float32).reshape(1, -1)
        except Exception as exc:  # noqa: BLE001
            self._fk_error = f"读取 {name} 关节角失败: {type(exc).__name__}: {exc}"
            return None
        if cpu.shape[1] <= 0 or not bool(torch.isfinite(cpu).all().item()):
            self._fk_error = f"{name} 关节角不可用(shape={tuple(cpu.shape)})"
            return None

        # ``compute_fk`` 只接受 torch 张量;cpu 与设备上的版本都试一遍。
        candidates = [cpu]
        device = self._device()
        if device is not None:
            try:
                moved = cpu.to(device)
                if moved.device != cpu.device:
                    candidates.append(moved)
            except Exception:  # noqa: BLE001 - 放不过去就只用 cpu 版本
                pass

        for tensor in candidates:
            for to_matrix in (True, False):
                try:
                    value = robot.compute_fk(tensor, name=name, to_matrix=to_matrix)
                except Exception as exc:  # noqa: BLE001
                    self._fk_error = (
                        f"compute_fk(name={name!r}, to_matrix={to_matrix}) 抛错: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                if value is None:
                    self._fk_error = (
                        f"compute_fk(name={name!r}, to_matrix={to_matrix}) 返回 None"
                        "(该 control part 没有 solver)"
                    )
                    continue
                out = _coerce(value)
                if out is not None:
                    return out

        # 末级回退:link6 连杆帧 @ link6->tool 实测常量 == tool/TCP 帧(同口径)。
        try:
            pose6 = _coerce(robot.get_link_pose(link_name=link_name, to_matrix=True))
            if pose6 is not None:
                tool_from_link6 = np.asarray(
                    _lazy("priors").TOOL_FROM_LINK6, dtype=np.float64
                )
                return pose6 @ tool_from_link6
            self._fk_error = f"{self._fk_error} | get_link_pose({link_name}) 形状不可解读"
        except Exception as exc:  # noqa: BLE001
            self._fk_error = (
                f"{self._fk_error} | get_link_pose(link_name={link_name!r}) 抛错: "
                f"{type(exc).__name__}: {exc}"
            )
        return None

    def _ik(self, arm: str, target_pose: np.ndarray, seed: Any) -> np.ndarray | None:
        """``<arm>_arm`` 的 IK,返回 6 维关节角(失败返回 ``None``,绝不外抛)。

        逐行对应参考 ``test_grasp_primitive.left_ik``:``compute_ik(pose,
        joint_seed=seed, name=...)``,检查 success 位,取第一个 batch 元素。

        与 :meth:`_fk` 同口径:``name`` 必须是 ``"<arm>_arm"``(solver 的键),
        ``pose`` 必须是 **torch 张量** 且形状 ``(N, 4, 4)``。失败原因写进
        ``self._ik_error``(不再静默 ``None``,否则无解与"参数传错"分不开)。
        """
        self._ik_error: str | None = None
        try:
            import torch

            device = self._device()
            pose = torch.as_tensor(
                np.asarray(target_pose, dtype=np.float32)
            )
            seed_t = torch.as_tensor(np.asarray(seed, dtype=np.float32).reshape(-1))
            if device is not None:
                try:
                    pose = pose.to(device)
                    seed_t = seed_t.to(device)
                except Exception:  # noqa: BLE001
                    pass
            if pose.ndim == 2:
                pose = pose.unsqueeze(0)
            if seed_t.ndim == 1:
                seed_t = seed_t.unsqueeze(0)
            computed = self._base.robot.compute_ik(
                pose, joint_seed=seed_t, name=arm
            )
        except Exception as exc:  # noqa: BLE001
            self._ik_error = f"compute_ik 抛错: {type(exc).__name__}: {exc}"
            return None
        if computed is None:
            self._ik_error = f"compute_ik(name={arm!r}) 返回 None(该 control part 没有 solver)"
            return None
        try:
            import torch

            success, qpos = computed
            if not bool(torch.as_tensor(success).all().item()):
                self._ik_error = (
                    f"compute_ik(name={arm!r}) success={npv(success).reshape(-1).tolist()}"
                    "(求解器判为无解)"
                )
                return None
            array = np.asarray(npv(qpos), dtype=np.float64)
            if array.ndim > 1:
                array = array[0]
            return array.reshape(-1)[:6]
        except Exception as exc:  # noqa: BLE001
            self._ik_error = f"IK 返回值无法解读: {type(exc).__name__}: {exc}"
            return None

    def _place_target_pose(
        self, center: Any, current_tcp: Any
    ) -> np.ndarray:
        """构造 FINAL_PLACE 的目标 TCP 位姿(4x4),**姿态受显式约束**。

        约束(SPEC §2.2):**TCP 的 X 轴垂直于桌面**(对齐
        :data:`PLACE_TCP_X_WORLD`,可由 ``HYBRID_PLACE_TCP_X`` 覆盖),
        **TCP 的 Z 轴落在桌面平面内**(取当前 TCP 接近方向在该平面上的投影,
        尽量少改手腕位形)。

        ==========  ==========================================================
        轴          构造
        ==========  ==========================================================
        ``x``       ``normalize(PLACE_TCP_X_WORLD)``
        ``z``       ``z_cur - (z_cur·x) x`` 再归一化;退化(‖·‖<1e-6)时换基向量
        ``y``       ``cross(z, x)``
        ``z``       再用 ``cross(x, y)`` 正交化(保证 R 正交且 ``det=+1``)
        位置        ``center + (0, 0, PLACE_CLEARANCE_M)``(当前默认 **11 cm**,
                    详见模块 docstring 的 "WS-K" 一节)
        ==========  ==========================================================

        本函数**不读任何物体真值**:``center`` 来自 VLM + 双目估计,
        ``current_tcp`` 来自 :meth:`_fk`(``robot.compute_fk``,本体感受)。
        ``current_tcp`` 不可用或退化(接近方向与 ``x`` 共线)时一律走**确定性的
        兜底构造**,单步不抛异常;``center`` 形状非法这类调用方错误会抛出,由
        ``eval()`` 的阶段级 ``try/except`` 记 failure 并降级为纯 ACT
        (**不会外抛到调用方**,这也是本文件唯一安全的兜底:绝不用一个编造的
        位置去驱动机器人)。
        """
        # 1) X 轴 = 世界方向(默认 (0,0,-1),⊥ 桌面)
        x_axis = _place_tcp_x_world()
        center_xyz = np.asarray(center, dtype=np.float64).reshape(-1)[:3]

        # 2) 当前 TCP 的接近方向(Z 轴),投影到垂直于 x 的平面
        z_cur = np.zeros(3, dtype=np.float64)
        try:
            matrix = np.asarray(current_tcp, dtype=np.float64)
            z_cur = matrix[:3, 2].reshape(3).copy()
        except Exception:  # noqa: BLE001 - 姿态不可用时用 0 触发下面的退化分支
            z_cur = np.zeros(3, dtype=np.float64)
        z_axis = z_cur - float(np.dot(z_cur, x_axis)) * x_axis
        if float(np.linalg.norm(z_axis)) < 1e-6:
            # 退化(当前接近方向与 x 共线,或 current_tcp 不可用):
            # 先按 SPEC 用 (1,0,0) 再投影;若它也与 x 共线,依次换另外两个基向量。
            for candidate in (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ):
                fallback = np.asarray(candidate, dtype=np.float64)
                z_axis = fallback - float(np.dot(fallback, x_axis)) * x_axis
                if float(np.linalg.norm(z_axis)) >= 1e-6:
                    break
        norm_z = float(np.linalg.norm(z_axis))
        if not np.isfinite(norm_z) or norm_z < 1e-6:
            # 理论上到不了这里(x 是单位向量,三个基向量里必有一个不与之共线);
            # 兜底给一个与 x 正交的确定性方向,绝不返回奇异矩阵。
            z_axis = np.cross(x_axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            if float(np.linalg.norm(z_axis)) < 1e-6:
                z_axis = np.cross(x_axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            norm_z = float(np.linalg.norm(z_axis))
        z_axis = z_axis / norm_z

        # 3) y = cross(z, x),再用 z = cross(x, y) 正交化(列 = 三个轴)
        y_axis = np.cross(z_axis, x_axis)
        norm_y = float(np.linalg.norm(y_axis))
        if not np.isfinite(norm_y) or norm_y < 1e-6:
            y_axis = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), x_axis)
            norm_y = float(np.linalg.norm(y_axis))
            if not np.isfinite(norm_y) or norm_y < 1e-6:
                y_axis = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), x_axis)
                norm_y = float(np.linalg.norm(y_axis))
        y_axis = y_axis / norm_y
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / float(np.linalg.norm(z_axis))

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
        pose[:3, 3] = center_xyz + np.array(
            [0.0, 0.0, float(PLACE_CLEARANCE_M)], dtype=np.float64
        )
        return pose

    def _place_pose_variants(
        self, center: Any, current_tcp: Any
    ) -> list[tuple[np.ndarray, dict]]:
        """FINAL_PLACE 目标位姿的**分级放宽候选族**(按优先级排序)。

        第 1 个候选 = :meth:`_place_target_pose` 的**精确姿态**(TCP X 轴 ⊥ 桌面,
        :data:`PLACE_CLEARANCE_M` = 11 cm 接近高度)。精确姿态被多起点 IK 判为
        不可达时,按参考 ``place_above_release.py`` 的 fallback 模式**分级放宽**
        —— 只放宽姿态与接近高度,**不删约束**:

        1. 绕"TCP → 筒心水平方向"的**水平垂轴**做俯仰 ``±8° / ±15° / ±25° / ±35°``
           (每个角度两个方向,接近高度仍是该级的高度);
        2. 姿态族仍无解时换下一级接近高度:11 cm → **17 cm** → **5 cm**
           (:data:`PLACE_CLEARANCE_M` + :data:`PLACE_CLEARANCE_LADDER_M`),
           每个高度重跑同一俯仰族。

        容差(10 mm 先、20 mm 后)不在这里,而是在调用方
        :meth:`_stage_final_place` 用 :data:`PLACE_POSITION_TOL_LADDER_M` 外层套。

        每个候选都是**显式构造**的 4x4 目标姿态(不是"保持当前姿态"),且只用到
        ``center``(VLM + 双目估计)与自己的 TCP 位姿(本体感受),**不读任何物体真值**;
        本方法纯计算,**不消耗环境步数**。

        返回 ``[(pose(4x4), variant), ...]``。``variant`` =
        ``{"pitch_deg": float, "clearance_m": float, "label": str}``,会被原样写进
        ``final_place.accepted_variant``,便于事后核对"最后用的是哪一级"。
        """
        base = self._place_target_pose(center, current_tcp)
        center_xyz = np.asarray(center, dtype=np.float64).reshape(-1)[:3]
        x_axis = np.asarray(_place_tcp_x_world(), dtype=np.float64).reshape(3)

        # 俯仰轴 = 水平面内、与"TCP → 筒心"的水平方向垂直的单位向量。
        try:
            tcp_xyz = np.asarray(current_tcp, dtype=np.float64).reshape(4, 4)[:3, 3]
            horizontal = np.asarray(center_xyz - tcp_xyz, dtype=np.float64)
        except Exception:  # noqa: BLE001 - 位姿不可用时用确定性方向
            horizontal = np.zeros(3, dtype=np.float64)
        horizontal[2] = 0.0
        norm_h = float(np.linalg.norm(horizontal))
        if not np.isfinite(norm_h) or norm_h < 1e-6:
            horizontal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            norm_h = 1.0
        horizontal = horizontal / norm_h
        pitch_axis = np.cross(x_axis, horizontal)
        norm_axis = float(np.linalg.norm(pitch_axis))
        if not np.isfinite(norm_axis) or norm_axis < 1e-6:
            pitch_axis = np.cross(x_axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            norm_axis = float(np.linalg.norm(pitch_axis))
            if not np.isfinite(norm_axis) or norm_axis < 1e-6:
                pitch_axis = np.cross(
                    x_axis, np.array([0.0, 1.0, 0.0], dtype=np.float64)
                )
                norm_axis = float(np.linalg.norm(pitch_axis))
        pitch_axis = pitch_axis / norm_axis

        # 俯仰阶梯:0° 只出现一次,之后 ±每一步。
        pitches: list[float] = [0.0]
        for step in PLACE_PITCH_LADDER_DEG:
            pitches.extend([+float(step), -float(step)])

        clearances = [float(PLACE_CLEARANCE_M), *PLACE_CLEARANCE_LADDER_M]
        variants: list[tuple[np.ndarray, dict]] = []
        for clearance in clearances:
            for pitch_deg in pitches:
                pose = base.copy()
                pose[:3, 3] = center_xyz + np.array(
                    [0.0, 0.0, float(clearance)], dtype=np.float64
                )
                if pitch_deg != 0.0:
                    pose[:3, :3] = _axis_angle_matrix(
                        pitch_axis, math.radians(float(pitch_deg))
                    ) @ np.asarray(base[:3, :3], dtype=np.float64)
                variants.append(
                    (
                        pose,
                        {
                            "pitch_deg": float(pitch_deg),
                            "clearance_m": float(clearance),
                            "label": (
                                f"pitch{pitch_deg:+.0f}deg_clr"
                                f"{float(clearance) * 1000.0:.0f}mm"
                            ),
                        },
                    )
                )
        return variants

    def _place_x_dev_deg_of_q(self, q: Any) -> float | None:
        """给定关节角,用 FK 算 TCP X 轴与世界 Z 的**轴偏差**(度)。

        只读本体感受(``compute_fk``),不消耗环境步数、不读物体真值;
        用于在接受一个多起点 IK 解之前先验一遍"这个解的姿态到底达不达标"
        (``_solve_left_pose`` 只按**位置**残差选解,姿态要靠这里复核)。
        FK 不可用返回 ``None``(调用方据此拒绝该解,不冒充达标)。
        """
        pose = self._fk_with_q(LEFT_ARM, q)
        if pose is None:
            return None
        try:
            tcp_x = np.asarray(pose[:3, 0], dtype=np.float64).reshape(3)
            cosine = float(np.clip(float(np.dot(tcp_x, PLACE_WORLD_Z)), -1.0, 1.0))
            angle_deg = float(math.degrees(math.acos(cosine)))
            return float(min(angle_deg, 180.0 - angle_deg))
        except Exception:  # noqa: BLE001
            return None

    def _place_tcp_constraint_check(self) -> dict:
        """放置姿态自检(SPEC §2.4):读**自己的** TCP 姿态,算 X 轴与世界 Z 的夹角。

        纯本体感受(``robot.compute_fk``),**不读任何物体真值**,也不消耗
        环境步数。返回字段:
        ``tcp_x_angle_to_world_z_deg``(SPEC §2.4 的有符号夹角,对世界 +Z)、
        ``tcp_x_axis_deviation_deg``(X 轴与桌面法向**轴**的偏差,0° = 完全垂直
        于桌面,判定用这个)、``tcp_x_alignment_to_place_axis_deg``(X 轴与配置的
        :data:`PLACE_TCP_X_WORLD` 的夹角)、``tcp_x_world`` / ``tcp_z_world`` /
        ``constraint_ok`` / ``fk_available`` / ``tol_deg``。
        FK 不可用时 ``constraint_ok`` 为 ``None``(未知,**不冒充达标**)。
        """
        tol_deg = float(PLACE_TCP_X_TOL_DEG)
        result: dict[str, Any] = {
            "tcp_x_angle_to_world_z_deg": None,
            "tcp_x_axis_deviation_deg": None,
            "tcp_x_alignment_to_place_axis_deg": None,
            "tcp_x_world": None,
            "tcp_z_world": None,
            "constraint_ok": None,
            "fk_available": False,
            "tol_deg": tol_deg,
            "place_tcp_x_world": _place_tcp_x_world().tolist(),
        }
        fk = self._fk(LEFT_ARM)
        if fk is None:
            return result
        try:
            tcp_x = np.asarray(fk[:3, 0], dtype=np.float64).reshape(3)
            tcp_z = np.asarray(fk[:3, 2], dtype=np.float64).reshape(3)
            cosine = float(
                np.clip(float(np.dot(tcp_x, PLACE_WORLD_Z)), -1.0, 1.0)
            )
            angle_deg = float(math.degrees(math.acos(cosine)))
            # **符号说明(重要,别把这一条读成 bug)**:SPEC §2.4 给的公式是对世界
            # **+Z** 的有符号夹角。而默认 X 轴方向是 ``(0,0,-1)``(朝下),所以
            # 完美达标的姿态在这里会读到 ``180°``。物理上的要求是"X 轴**垂直于
            # 桌面**",即 X 轴与世界 Z **轴**共线(朝上或朝下都算),所以真正用于
            # 判定的量是**轴偏差** ``min(angle, 180-angle)``(0° = 完全垂直)。
            # 官方 ``_is_fall_x`` 用的是同一个有符号公式且阈值是 1.309 rad(≈75°),
            # 即它要求物体的 X 轴**朝上**;两者是否一致取决于"笔长轴 ≈ TCP 哪个轴"
            # 这个尚未验证的假设(见模块 docstring 的"放置姿态约束")。
            axis_deviation_deg = float(min(angle_deg, 180.0 - angle_deg))
            place_axis = _place_tcp_x_world()
            alignment_deg = float(
                math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                float(np.dot(tcp_x, place_axis)),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
            )
            result["tcp_x_angle_to_world_z_deg"] = angle_deg
            result["tcp_x_axis_deviation_deg"] = axis_deviation_deg
            result["tcp_x_alignment_to_place_axis_deg"] = alignment_deg
            result["tcp_x_world"] = tcp_x.tolist()
            result["tcp_z_world"] = tcp_z.tolist()
            result["constraint_ok"] = bool(axis_deviation_deg < tol_deg)
            result["fk_available"] = True
        except Exception:  # noqa: BLE001 - 自检失败只记 None,绝不外抛
            return result
        return result

    def _solve_left_pose(
        self, target_pose: np.ndarray, q_seed: np.ndarray
    ) -> tuple[np.ndarray, float, int] | None:
        """左臂多起点 IK(参考 ``place_above_release2.solve_ik``)。

        起点 = 当前关节 + ``PLACE_IK_RANDOM_SEEDS`` 个在关节限位内均匀采样的
        随机起点(``np.random.default_rng(0)`` 复现);按 FK 位置残差选最优,
        残差 < :data:`PLACE_IK_POSITION_TOL_M`(10 mm,容差阶梯第 1 级)立即早停。
        返回的只是**位置最优**解,姿态是否达标由调用方用
        :meth:`_place_x_dev_deg_of_q` 复核后再决定接受。
        """
        try:
            limits = np.asarray(
                npv(self._base.robot.get_qpos_limits(name=LEFT_ARM)), dtype=np.float64
            )
            while limits.ndim > 2:
                limits = limits[0]
            lower, upper = limits[:, 0], limits[:, 1]
        except Exception:
            return None
        rng = np.random.default_rng(int(PLACE_IK_RNG_SEED))
        starts: list[np.ndarray] = [np.asarray(q_seed, dtype=np.float64).reshape(-1)]
        starts.extend(
            np.clip(rng.uniform(lower, upper), lower, upper)
            for _ in range(int(PLACE_IK_RANDOM_SEEDS))
        )
        best: tuple[np.ndarray, float, int] | None = None
        for index, seed in enumerate(starts):
            q = self._ik(LEFT_ARM, target_pose, seed)
            if q is None:
                continue
            reached = self._fk_with_q(LEFT_ARM, q)
            if reached is None:
                continue
            error = float(
                np.linalg.norm(
                    np.asarray(reached[:3, 3], dtype=np.float64)
                    - np.asarray(target_pose[:3, 3], dtype=np.float64)
                )
            )
            if best is None or error < best[1]:
                best = (q, error, index)
            if error < float(PLACE_IK_POSITION_TOL_M):
                break
        return best

    def _fk_many(self, arm: str, qs: Any) -> np.ndarray | None:
        """批量 FK:``(N,6)`` 关节角 -> ``(N,4,4)`` 世界位姿(工具/TCP 帧)。

        只用于 :meth:`_place_search_pose` 的纯计算搜索,**不消耗环境步数**,
        也不读任何物体真值(纯本体感受运动学)。

        两条路,全部同口径:

        1. 求解器自身的 ``get_fk``(``robot.compute_fk`` 内部走的就是这一条),
           再左乘 root link 的世界位姿 —— 支持 batch,快;
        2. 逐样本 ``robot.compute_fk``(公开 API,单 env 只能一条一条来),
           作为求解器句柄不可用时的兜底。

        两条都失败返回 ``None``(调用方退回解析 IK / 原地释放)。
        """
        try:
            import torch

            name = arm if str(arm).endswith("_arm") else f"{arm}_arm"
            rows = np.atleast_2d(np.asarray(qs, dtype=np.float32))
            robot = getattr(self._base, "robot", None)
            if robot is None or rows.size == 0:
                return None
            device = self._device()
            tensor = torch.as_tensor(rows, dtype=torch.float32)
            if device is not None:
                try:
                    tensor = tensor.to(device)
                except Exception:  # noqa: BLE001
                    pass
            solver = None
            try:
                solver = getattr(robot, "_solvers", {}).get(name)
            except Exception:  # noqa: BLE001
                solver = None
            if solver is not None:
                try:
                    root = getattr(solver, "root_link_name", None)
                    base_pose = robot.get_link_pose(link_name=root, to_matrix=True)
                    base_pose = base_pose.to(tensor.device).to(torch.float32)
                    raw = solver.get_fk(qpos=tensor)
                    if isinstance(raw, dict):
                        raw = torch.stack(
                            [raw[key].get_matrix() for key in raw], dim=0
                        )
                    raw = raw.to(torch.float32)
                    if raw.dim() == 2:
                        raw = raw.reshape(-1, 4, 4)
                    if raw.dim() == 3 and tuple(raw.shape[-2:]) == (4, 4):
                        world = torch.matmul(base_pose, raw)
                        out = np.asarray(
                            world.detach().cpu().numpy(), dtype=np.float64
                        ).reshape(-1, 4, 4)
                        if out.shape[0] == rows.shape[0]:
                            return out
                except Exception:  # noqa: BLE001 - 退回逐样本
                    pass
            mats = []
            for row in np.atleast_2d(np.asarray(qs, dtype=np.float32)):
                value = robot.compute_fk(
                    torch.as_tensor(row, dtype=torch.float32).reshape(1, -1),
                    name=name,
                    to_matrix=True,
                )
                mats.append(
                    np.asarray(npv(value), dtype=np.float64).reshape(4, 4)
                )
            return np.stack(mats, axis=0)
        except Exception:  # noqa: BLE001
            return None

    def _place_search_pose(
        self,
        arm: str,
        target_pose: np.ndarray,
        q_seed: Any,
        pos_tol_ladder: Any = None,
    ) -> dict | None:
        """**大面积 FK 采样 + 局部精修**的位姿求解,直接优化"位置残差 + X 轴轴偏差"。

        为什么需要它:2026-10-11 实测解析 IK(OPW)在"筒口正上方 + TCP X 轴 ⊥ 桌面"
        这个目标上(含俯仰 ±8/±15/±25/±35°、接近高度 5/8/12/16/17/3/2 cm)
        **恒返回 success=0**;隔离实验里连"只换位置"或"只换姿态"的单侧目标也解不出。
        解析求解器判"无解"**不等于**工作空间里没有,所以这里不依赖它,直接对 FK
        做采样搜索(纯本体感受运动学)。

        目标("先位置后姿态",与 2026-10-11 的判据优先级一致):

        1. 位置残差进入容差阶梯(``pos_tol_ladder``,默认
           :data:`PLACE_POSITION_TOL_LADDER_M` = 10 mm → 20 mm),**先严后松**;
        2. 同级容差内取 TCP X 轴与目标 X 轴的**轴偏差最小**的解;
        3. 只有轴偏差 < :data:`PLACE_TCP_X_TOL_DEG`(15°)才算"姿态达标"。

        搜索 = 关节限位内 **2,000,000** 个均匀样本(分批喂批量 FK;10~20 mm 环带
        的体积占比只有 ~1e-5,样本少了根本碰不到)+ 围绕"最松容差内轴偏差最小"的
        前 ``K`` 名候选做多轮高斯精修(每轮标准差 ×
        :data:`PLACE_SEARCH_SIGMA_DECAY`)。全程只调 FK,**纯计算、不消耗环境步数、
        不读物体真值**。

        返回字段:``ok``(位置进容差且 15° 姿态达标)、``q`` / ``pos_err_m`` /
        ``x_dev_deg`` / ``pos_tol_m``(实际用的容差级,单位米)/ ``n_eval`` /
        ``pos_tol_ladder_m`` / ``best_x_dev_deg_per_tol``(每级容差内的最优轴偏差,
        便于事后核对)/ ``best_effort``(全局最近位置候选)。
        ``ok=False`` 时 ``q`` 就是 ``best_effort`` 的关节角(仍比"原地释放"更靠近
        筒口),调用方按"位置优先"的优先级决定用不用它;完全搜不到返回 ``None``。
        """
        try:
            robot = getattr(self._base, "robot", None)
            if robot is None:
                return None
            name = arm if str(arm).endswith("_arm") else f"{arm}_arm"
            limits = np.asarray(npv(robot.get_qpos_limits(name=name)), dtype=np.float64)
            while limits.ndim > 2:
                limits = limits[0]
            lower, upper = limits[:, 0], limits[:, 1]
        except Exception:  # noqa: BLE001
            return None

        goal = np.asarray(target_pose, dtype=np.float64).reshape(4, 4)
        goal_pos = goal[:3, 3]
        goal_x = goal[:3, 0]
        goal_x = goal_x / max(1e-12, float(np.linalg.norm(goal_x)))
        tol_levels = [
            float(t) for t in (pos_tol_ladder or PLACE_POSITION_TOL_LADDER_M)
        ]
        if not tol_levels:
            tol_levels = list(PLACE_POSITION_TOL_LADDER_M)
        tol_soft = float(tol_levels[-1])
        rng = np.random.default_rng(int(PLACE_IK_RNG_SEED))
        n_eval = 0
        # 每一级容差内的最优(轴偏差最小)候选;best_any = 不限容差的最近位置。
        best_by_tol: list[dict | None] = [None] * len(tol_levels)
        best_any: dict | None = None
        # 精修种子池:最松容差内轴偏差最小的前若干个(总量截到 200)。
        pool: list[tuple[np.ndarray, float, float]] = []

        def evaluate(qs: Any) -> tuple[Any, Any, Any]:
            """批量 FK -> ``(q64, pos_err_m, x_dev_deg)``;FK 不可用返回全 ``None``。"""
            nonlocal n_eval
            fk = self._fk_many(arm, qs)
            if fk is None:
                return None, None, None
            n_eval += int(np.asarray(qs).shape[0])
            pos_err = np.linalg.norm(fk[:, :3, 3] - goal_pos[None, :], axis=1)
            xax = fk[:, :3, 0]
            xax = xax / np.maximum(1e-12, np.linalg.norm(xax, axis=1, keepdims=True))
            ang = np.degrees(np.arccos(np.clip(xax @ goal_x, -1.0, 1.0)))
            # 轴偏差:X 轴与目标 X 轴**轴**共线即可(朝上/朝下都算垂直桌面)。
            return (
                np.asarray(qs, dtype=np.float64),
                pos_err,
                np.minimum(ang, 180.0 - ang),
            )

        def absorb(qs64: Any, pos_err: Any, x_dev: Any) -> None:
            """把一批样本并进 ``best_by_tol`` / ``best_any`` / 精修池。"""
            nonlocal best_any
            i_near = int(np.argmin(pos_err))
            if best_any is None or float(pos_err[i_near]) < float(best_any["pos_err_m"]):
                best_any = {
                    "q": np.asarray(qs64[i_near], dtype=np.float64),
                    "pos_err_m": float(pos_err[i_near]),
                    "x_dev_deg": float(x_dev[i_near]),
                }
            for level, tol in enumerate(tol_levels):
                mask = pos_err <= tol
                if not bool(mask.any()):
                    continue
                idx = np.flatnonzero(mask)
                k = int(idx[int(np.argmin(x_dev[idx]))])
                current = best_by_tol[level]
                if current is None or float(x_dev[k]) < float(current["x_dev_deg"]):
                    best_by_tol[level] = {
                        "q": np.asarray(qs64[k], dtype=np.float64),
                        "pos_err_m": float(pos_err[k]),
                        "x_dev_deg": float(x_dev[k]),
                    }
            # 精修池只收"最松容差内"的样本(按轴偏差升序),再保底放最近位置。
            mask = pos_err <= tol_soft
            if bool(mask.any()):
                idx = np.flatnonzero(mask)
                order = np.lexsort((pos_err[idx], x_dev[idx]))
                for k in idx[order[:50]]:
                    pool.append((qs64[k], float(pos_err[k]), float(x_dev[k])))
            pool.append((qs64[i_near], float(pos_err[i_near]), float(x_dev[i_near])))
            pool.sort(key=lambda item: (0 if item[1] <= tol_soft else 1, item[2], item[1]))
            del pool[200:]

        # 1) 均匀粗采样(分批,避免一次性占满显存)
        remaining = int(PLACE_SEARCH_COARSE)
        chunk = max(1, int(PLACE_SEARCH_CHUNK))
        first = True
        while remaining > 0:
            n = min(chunk, remaining)
            q = np.clip(
                rng.uniform(lower, upper, size=(n, 6)), lower, upper
            ).astype(np.float32)
            if first:
                q[0] = np.asarray(q_seed, dtype=np.float32).reshape(-1)[:6]
                first = False
            qs64, pos_err, x_dev = evaluate(q)
            if qs64 is None:
                return None
            absorb(qs64, pos_err, x_dev)
            remaining -= n

        # 2) 局部精修:围绕池里的前 K 名做高斯扰动,逐轮收紧标准差
        sigma = float(PLACE_SEARCH_SIGMA0)
        for _ in range(int(PLACE_SEARCH_ROUNDS)):
            seeds = pool[: int(PLACE_SEARCH_TOP_K)]
            if not seeds:
                break
            per = max(1, int(PLACE_SEARCH_PER_ROUND) // max(1, len(seeds)))
            batch = []
            for q0, _err, _dev in seeds:
                noise = rng.normal(0.0, sigma, size=(per, 6))
                batch.append(np.clip(q0[None, :] + noise, lower, upper))
            qs64, pos_err, x_dev = evaluate(
                np.concatenate(batch, axis=0).astype(np.float32)
            )
            if qs64 is None:
                break
            absorb(qs64, pos_err, x_dev)
            # 每一级容差都已经有"姿态达标"的解就收工
            if all(
                b is not None and float(b["x_dev_deg"]) < float(PLACE_TCP_X_TOL_DEG)
                for b in best_by_tol
            ):
                break
            sigma *= float(PLACE_SEARCH_SIGMA_DECAY)

        diagnostics = {
            "pos_tol_ladder_m": [float(t) for t in tol_levels],
            "best_x_dev_deg_per_tol": [
                None if b is None else float(b["x_dev_deg"]) for b in best_by_tol
            ],
            "best_pos_err_m_per_tol": [
                None if b is None else float(b["pos_err_m"]) for b in best_by_tol
            ],
            "coarse_samples": int(PLACE_SEARCH_COARSE),
            "n_eval": int(n_eval),
        }
        # 3) 容差阶梯"先严后松":第 1 个"位置进容差且 15° 姿态达标"的级胜出。
        for level, tol in enumerate(tol_levels):
            found = best_by_tol[level]
            if found is not None and float(found["x_dev_deg"]) < float(
                PLACE_TCP_X_TOL_DEG
            ):
                return {
                    "ok": True,
                    "q": found["q"],
                    "pos_err_m": float(found["pos_err_m"]),
                    "x_dev_deg": float(found["x_dev_deg"]),
                    "pos_tol_m": float(tol),
                    "best_effort": best_any,
                    **diagnostics,
                }
        if best_any is None:
            return None
        # 没有任何"姿态达标"的解:把全局最近位置的候选交回调用方(位置优先兜底)。
        return {
            "ok": False,
            "q": best_any["q"],
            "pos_err_m": float(best_any["pos_err_m"]),
            "x_dev_deg": float(best_any["x_dev_deg"]),
            "pos_tol_m": None,
            "best_effort": best_any,
            **diagnostics,
        }

    def _fk_with_q(self, arm: str, q: Any) -> np.ndarray | None:
        """给定关节角的 FK(参考 ``_fk_left``)。"""
        try:
            import torch

            device = self._device()
            tensor = torch.as_tensor(np.asarray(q, dtype=np.float32).reshape(1, -1))
            if device is not None:
                try:
                    tensor = tensor.to(device)
                except Exception:  # noqa: BLE001
                    pass
            value = self._base.robot.compute_fk(tensor, name=arm, to_matrix=True)
        except Exception:
            return None
        try:
            matrix = np.asarray(npv(value), dtype=np.float64)
            while matrix.ndim > 2:
                matrix = matrix[0]
            return matrix.copy()
        except Exception:
            return None

    @staticmethod
    def _matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
        """旋转矩阵 → 单位四元数 (x, y, z, w)(与 ``Rotation.as_quat`` 同序)。"""
        matrix = np.asarray(rotation, dtype=np.float64)[:3, :3]
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = float(np.sqrt(trace + 1.0)) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
        else:
            index = int(np.argmax(np.diag(matrix)))
            if index == 0:
                scale = float(np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
                quat = np.array(
                    [
                        0.25 * scale,
                        (matrix[0, 1] + matrix[1, 0]) / scale,
                        (matrix[0, 2] + matrix[2, 0]) / scale,
                        (matrix[2, 1] - matrix[1, 2]) / scale,
                    ]
                )
            elif index == 1:
                scale = float(np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
                quat = np.array(
                    [
                        (matrix[0, 1] + matrix[1, 0]) / scale,
                        0.25 * scale,
                        (matrix[1, 2] + matrix[2, 1]) / scale,
                        (matrix[0, 2] - matrix[2, 0]) / scale,
                    ]
                )
            else:
                scale = float(np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
                quat = np.array(
                    [
                        (matrix[0, 2] + matrix[2, 0]) / scale,
                        (matrix[1, 2] + matrix[2, 1]) / scale,
                        0.25 * scale,
                        (matrix[1, 0] - matrix[0, 1]) / scale,
                    ]
                )
        norm = float(np.linalg.norm(quat))
        return quat / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])

    @staticmethod
    def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
        """单位四元数 (x, y, z, w) → 旋转矩阵。"""
        x, y, z, w = (float(v) for v in np.asarray(quat, dtype=np.float64).reshape(4))
        norm = float(np.linalg.norm([x, y, z, w]))
        if norm < 1e-12:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _right_q(self, ref: dict) -> np.ndarray:
        """抓取原语里的右臂 IK 种子(优先本次原语的快照,其次当前 qpos)。"""
        snapshot = ref.get("snapshot") if isinstance(ref, dict) else None
        if isinstance(snapshot, dict) and snapshot.get("q_right") is not None:
            return np.asarray(snapshot["q_right"], dtype=np.float64)
        q_right, _grip = self._arm_joints(self._base, RIGHT)
        return np.asarray(q_right, dtype=np.float64)

    def _step_right_q(
        self, env: Any, q_right: Any, *, grip: float = GRIP_RIGHT_OPEN
    ) -> None:
        """只动右臂、左臂保持当前位姿的一步动作(抓取原语共用)。

        默认夹爪值是 :data:`GRIP_RIGHT_OPEN`(张开),**不是闭合** ——
        接近(``VERTICAL_ALIGN``)与下压(``PRESS``)阶段爪子必须从两侧跨住笔,
        只有 ``CLOSE`` 子阶段才把它 1.0 → 0.0 插值闭合,``LIFT`` 之后才保持闭合。

        历史教训(2026-10-11):该默认值原本是 ``GRIP_RIGHT_CLOSED``,
        而 ``VERTICAL_ALIGN`` / ``PRESS`` 两个调用点都没显式传 ``grip=``,
        于是右夹爪从第一步起就被命令闭合、整条 PRIMITIVE 都没张开过,
        笔永远夹不住(``LIFT`` 抬升量仅 ≈4 mm)。改为 OPEN 后,
        "忘记传 grip" 的后果从"夹不住笔"退化为"多张开一会儿",不再致命。
        """
        q_left, grip_left = self._arm_joints(self._base, LEFT)
        action = full_action(
            self._base, q_left, grip_left, np.asarray(q_right, dtype=np.float64), grip
        )
        self._step_env_step(env, action)

    # ------------------------------------------------------------------
    # 阶段 10/11:DONE → ACT_FILL / FALLBACK_ACT(继承父类实现)
    # ------------------------------------------------------------------

    def _stage_done(self, env: Any, inference_times: list[float]) -> None:
        """DONE:正常跑完相位,剩余步数交给纯 ACT(与父类同口径)。"""
        self._grip_override = None
        self._enter(self.ACT_FILL)

    def _stage_act_fill(self, env: Any, inference_times: list[float]) -> None:
        """ACT_FILL:用剩余步数继续跑纯 ACT(不覆盖任何夹爪通道)。"""
        self._grip_override = None
        if DONE_RUNS_PURE_ACT:
            self._act_micro_step(env, inference_times, guarded=True)
        else:
            self._step_env_step(env, self._hold_action())

    def _stage_fallback_act(self, env: Any, inference_times: list[float]) -> None:
        """FALLBACK_ACT:任一阶段失败后的纯 ACT 执行(不做任何夹爪覆盖)。"""
        self._grip_override = None
        self._act_micro_step(env, inference_times, guarded=True)

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """返回本 episode 的可序列化摘要(含逐状态步数审计)。"""
        timings = list(self._inference_times_all)
        budget_total = sum(
            int(self.STEP_BUDGET.get(name, 0))
            for name in self.STATES
            if name != self.ACT_FILL
        )
        return jsonable(
            {
                "task": "items_handover",
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
                "step_budget": {
                    "official_limit": int(OFFICIAL_STEP_LIMIT),
                    "official_limits": {
                        "random": int(OFFICIAL_STEP_LIMIT_RANDOM),
                        "clear": int(OFFICIAL_STEP_LIMIT_CLEAR),
                    },
                    "requested_max_env_steps": int(self._requested_max_env_steps),
                    "env_step_limit_probed": self._env_step_limit,
                    "effective_max_env_steps": int(self.max_env_steps),
                    "by_state": {
                        name: int(self.STEP_BUDGET.get(name, 0))
                        for name in self.STATES
                    },
                    "planned_total": int(budget_total),
                    "actual_total": int(sum(self._budget.values())),
                    "actual_by_state": dict(self._budget),
                    "primitive_by_substage": dict(self._prim_budget_used),
                    "within_official_limit": bool(
                        int(self._steps_taken) <= int(self.max_env_steps)
                    ),
                    # 因步数预算不足而压缩/提前降级的相位(空 = 预算充裕)。
                    "compressed_states": list(self._budget_truncated),
                    "exhausted": bool(
                        int(self._steps_taken) >= int(self.max_env_steps)
                    ),
                    "exhausted_reason": (
                        f"因步数预算耗尽而截断:已用 {int(self._steps_taken)} 步 "
                        f"达到 max_env_steps={int(self.max_env_steps)}"
                        if int(self._steps_taken) >= int(self.max_env_steps)
                        else None
                    ),
                },
                "chunk1": {
                    "steps": int(CHUNK1_STEPS),
                    "act_chunk_steps": int(self._act_chunk_budget),
                    "align_target_captured": self._align_target is not None,
                },
                "primitive": {
                    "steps": int(self._prim_steps),
                    "allowance": int(self._prim_allowance),
                    "ok": bool(self._prim_ok),
                    "substages": list(self._prim_records),
                    "look_pen": jsonable(self._look_pen),
                },
                "continuation": {
                    "steps": int(self._continuation_steps),
                    "budget": int(CONTINUATION_BUDGET),
                    "success": bool(self._continuation_success),
                    "right_gripper_lock": float(LOCKED_RIGHT_GRIPPER),
                    "left_tcp_history_len": len(self._left_tcp_history),
                },
                "left_takeover": self._takeover,
                "manual_target": {
                    "waypoint": "参考包内的离线段点(manual_targets/*.json)",
                    "steps": int(MANUAL_TARGET_BUDGET),
                    "final_error_max_abs": None
                    if not self._manual_ref
                    else self._manual_ref.get("final_error_max_abs"),
                },
                "holder_locate": self._holder,
                "holder_vlm_attempts": list(self._holder_attempts),
                "final_place": self._place,
                "grasp": {
                    # 参考用 pen 真值复核"笔是否随爪起来";本移植**已省略**该检查
                    # (无法在不读真值的前提下完成),只保留本体感受信号。
                    "pen_in_gripper_check": "omitted",
                    "truth_reads_removed": [
                        "SPEC §3 第 1 行:pen 真值位姿辅助函数(pen_pose)",
                        "SPEC §3 第 2 行:任意刚体真值位姿辅助函数(_pose)",
                    ],
                },
                "act": {
                    "chunks_requested": self.act_chunks,
                    "chunk_steps": int(self._act_chunk_budget),
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
                    "debug_dir": str(self.debug_dir) if self.debug_dir else None,
                },
            }
        )
