# 复现指南

本文件描述如何在一台**有 GPU 的机器**上把这个提交包跑起来。
打包所在的机器没有可用的 GPU 设备(见 `docs/COMPLIANCE.md` §5),
因此下列步骤**尚未被端到端执行过**;请把它当作待验证的操作手册,而不是已验证的配方。

---

## 0. 前提

| 需求 | 说明 |
| --- | --- |
| 官方仿真环境 | 已按官方文档装好 EmbodiChain,`python scripts/eval_policy.py --help` 可用 |
| Python | 3.11(PyTorch 2.7.0+cu128) |
| `lerobot` | `0.4.4` |
| GPU | ≥24 GB 显存推荐(VLM 7B Q4 + 仿真器 + ACT) |
| 磁盘 | ≥25 GB(`FoundationStereo` 权重 0.8 GB、VLM 5.5 GB、ACT 0.2 GB、缓存) |
| 网络 | 能访问 GitHub;HF 默认走 `HF_ENDPOINT=https://hf-mirror.com` |

---

## 1. 把提交包放进官方仓库

```bash
cd /path/to/RoboSynChallenge
git clone https://github.com/tworedsun/robosynchallenge-hybrid-submission /tmp/hybrid

cp -r /tmp/hybrid/policy/hybrid    policy/
cp -r /tmp/hybrid/robosyn_hybrid   .
```

> ⚠️ **只拷贝 `policy/hybrid/` 和 `robosyn_hybrid/`**。不要把提交包顶层的
> `scripts/` 拷到官方仓库根目录——那里已经有 `scripts/eval_policy.py`。
> 本提交包的所有脚本都在 `robosyn_hybrid/scripts/` 下。

自检:

```bash
python -c "
import sys; sys.path.insert(0,'.')
from policy.hybrid import get_model, eval, reset_model
print('adapter ok:', get_model, eval, reset_model)
"
```

---

## 2. 安装额外依赖

```bash
pip install -r robosyn_hybrid/requirements/robosyn-extra.txt
```

主要新增:`pillow`、`opencv-python`、`ultralytics`(SAM2 加载器)、`requests`、
`huggingface_hub`。

---

## 3. 下载第三方权重

```bash
export ROBOSYN_HYBRID_ASSETS=$PWD/robosyn_hybrid_assets
bash robosyn_hybrid/scripts/download_assets.sh
```

脚本会下载并**逐个校验 SHA-256**:

| 资产 | 来源 | 大小 | SHA-256 |
| --- | --- | ---: | --- |
| `sam2.1_t.pt` | `github.com/ultralytics/assets` release `v8.3.0` | 78,105,722 B | `3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3` |
| `FoundationStereo/` 源码 | `github.com/NVlabs/FoundationStereo` | — | — |
| `.../11-33-40/model_best_bp2.pth` | `hf-mirror.com/shriarul5273/FoundationStereo_models` | 787,711,942 B | `8d7850b9dc68d1366722a02a39745704b1db41471211be6abd93ef463727e6be` |
| `vlm/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | `hf-mirror.com/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` | ~4.7 GB | 由 HTTP 层保证完整性 |
| `vlm/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf` | 同上 | ~1.35 GB | 同上 |

不需要 VLM 时用 `bash robosyn_hybrid/scripts/download_assets.sh --no-vlm`。
需要研究阶段实际使用的 27B 模型时用 `VLM_SIZE=large`(总 17 GB,需 ~24 GB 显存独占)。

> `FoundationStereo` 的 11-33-40 权重官方只给出 Google Drive 链接,
> 这里改用 HF 镜像 `shriarul5273/FoundationStereo_models`,其文件大小与
> 我们实测的 787,711,942 B 完全一致。作者本人的
> `Felix-Zhenghao/FoundationStereo` 仓库里那个权重是 **23-51-11(ViT-large,
> 3.3 GB)**,不是本策略使用的 11-33-40,不要下错。

---

## 4. 构建独立的 FoundationStereo 环境

FoundationStereo 的 CUDA 扩展与仿真器环境冲突,因此它跑在自己的环境里:

```bash
bash robosyn_hybrid/scripts/setup_fs_env.sh
export FS_PYTHON=$PWD/robosyn_hybrid_assets/fs_venv/bin/python
```

`stereo.py` 通过 `subprocess` 用 `FS_PYTHON` 调 `fs_infer.py`;
优先走常驻 worker(`FS_STEREO_SOCKET`,默认 `/tmp/fs_stereo.sock`),
worker 不存在时自动回退到每次调用一个子进程(慢但不失败)。

> 常驻 worker 的意义:实测加载 FoundationStereo 约 86 s,单次前向约 0.34 s,
> 不用 worker 的话单次 `fetch_depth` 有 96/103 秒花在重复加载上。

---

## 5. 启动 VLM 端点

**推荐用 ollama + `qwen3vl:8b`。** 参考实现的 holder / 按钮定位精度(筒口 3D 误差
**3.17 mm**)就是在 ollama + `qwen3vl:8b` 上实测出来的,本仓库的
`settings.vlm_model` 默认值也跟随它。ollama 原生提供 OpenAI 兼容端点,不需要 mmproj。

```bash
ollama pull qwen3vl:8b
VLM_BACKEND=ollama bash robosyn_hybrid/scripts/run_vlm_server.sh &
export HYBRID_VLM_URL=http://127.0.0.1:11434/v1/chat/completions
export HYBRID_VLM_MODEL=qwen3vl:8b
```

**没有 ollama** 时可用 llama.cpp 的 `llama-server`(需要 GGUF 主权重 + 配套 mmproj):

```bash
MODEL_PATH=/path/Qwen3-VL-8B-Instruct-Q4_K_M.gguf \
MMPROJ_PATH=/path/mmproj-F16.gguf \
VLM_PORT=8080 bash robosyn_hybrid/scripts/run_vlm_server.sh &
export HYBRID_VLM_URL=http://127.0.0.1:8080/v1/chat/completions
export HYBRID_VLM_MODEL=qwen3vl:8b        # llama-server 会忽略该字段,填什么都可以
```

验证端点:

```bash
curl -s "$HYBRID_VLM_URL" -H 'Content-Type: application/json' -d '{
  "model":"qwen3vl:8b","max_tokens":32,
  "messages":[{"role":"user","content":[{"type":"text","text":"Reply with the single word ok"}]}]
}' | head -c 300
```

> 冷启动第一次调用可能 30 s 左右(模型加载),热态约 2.5 s。如果日志里出现
> `timeout after 120s`,先把 `HYBRID_VLM_TIMEOUT_S` 调大再跑第二次。

**端点没起来也没关系**:`HYBRID_VLM_URL` 未设置时,三条混合流程(`handle_basket` /
`click_bell` / `items_handover`)都会自动降级为纯 ACT,episode 仍然跑完。

---

## 6. 冒烟测试(强烈建议先跑这一步)

```bash
export ROBOSYN_HYBRID_ASSETS=$PWD/robosyn_hybrid_assets
export HYBRID_VLM_URL=http://127.0.0.1:11434/v1/chat/completions
export HYBRID_DEBUG_DIR=$PWD/hybrid_debug/seed0

HYBRID_CKPT=./ACT_sim_handle_basket \
bash policy/hybrid/eval.sh handle_basket random smoke_seed0 0 \
  --max_episodes 1 --eval_expert_check false --headless true
```

成功的标志:

1. 日志里出现 `Loading policy: hybrid`;
2. 出现 `ACT adapter: external pre/post processors (lerobot 0.4.4 format)`
   (如果你的 checkpoint 是 legacy 格式,则会打印 `legacy built-in normalizers`);
3. `hybrid_debug/` 下出现 `depth` / `vlm` / `sam2` / `ik` 等阶段的产物;
4. episode 结束时 `eval_result/handle_basket/hybrid/random/.../evaluation_metrics.json` 生成。

然后**关掉 VLM 再跑一次**,确认降级路径成立:

```bash
unset HYBRID_VLM_URL
HYBRID_CKPT=./ACT_sim_handle_basket \
bash policy/hybrid/eval.sh handle_basket random smoke_noact 0 \
  --max_episodes 1 --eval_expert_check false
```

这一次应当在日志里看到 VLM 不可用的原因,episode 仍正常结束。

### 6b. `click_bell` 混合流程冒烟(本次新增,尚未在真机上跑过)

```bash
export ROBOSYN_HYBRID_ASSETS=$PWD/robosyn_hybrid_assets
export HYBRID_VLM_URL=http://127.0.0.1:11434/v1/chat/completions
export HYBRID_DEBUG_DIR=$PWD/hybrid_debug/clickbell_seed0

# 不需要 HYBRID_CKPT:click_bell 复用官方 released checkpoint(pinned revision)
bash policy/hybrid/eval.sh click_bell random smoke_clickbell 0 \
  --max_episodes 1 --eval_expert_check false --headless true
```

要看的产物与信号:

1. 日志出现 `hybrid flow active for click_bell: ClickBellFlow ... max_env_steps=361`;
2. `hybrid_debug/clickbell_seed0/` 下的 `depth_reference_rgb.png`、
   `vlm_button.json`、`button_estimation.json`、`press_ik.json`;
3. episode 结束后的 `summary`(在评测日志/metrics 里)应当有
   `approach.converged`、`approach.trace`、`press.trace`、`press.pressed_m`;
4. `press.trace[*].success` 为 `true` 表示按压命中(成功是粘滞锁存的);
5. 若看到 `failed=true` 且有 `failures`,看 `stage` 判断是
   `VLM_LOCATE`(端点/解析)、`ESTIMATE_POSE`(反投影采样不足)、`PRESS_IK`(IK
   无解)还是 `APPROACH`。

**已知几何风险(必须在真机上确认)**:参考实现 `servo_to_bell` 只把工具带到
`按钮 + APPROACH_CLEARANCE_M(0.04)`;`clickbell.py` 因此照参考实现的编排在粗接近
收敛后补了一次"落到按压面"(`move_to_stage` 偏移 `STAGE_OFFSET_M = 0.0`,常量
`APPROACH_SURFACE_STAGE` 可关)。这一步是否到位决定 8 mm 的按压能不能压到
4.8 mm,只能靠真机验证。

### 6c. `items_handover` 混合流程冒烟(本次新增,尚未在真机上跑过)

```bash
export ROBOSYN_HYBRID_ASSETS=$PWD/robosyn_hybrid_assets
export HYBRID_VLM_URL=http://127.0.0.1:11434/v1/chat/completions
export HYBRID_DEBUG_DIR=$PWD/hybrid_debug/handover_seed0

# 不需要 HYBRID_CKPT:items_handover 复用官方 released checkpoint(pinned revision)
bash policy/hybrid/eval.sh items_handover random smoke_handover 0 \
  --max_episodes 1 --eval_expert_check false --headless true
```

要看的产物与信号:

1. 日志出现 `hybrid flow active for items_handover: HandoverFlow ... max_env_steps=350`;
2. `hybrid_debug/handover_seed0/` 下的 `handover_settle_three_cam.png`、
   `handover_primitive_after_three_cam.png`、`holder_vlm/{left,right}_answer.json`、
   `handover_holder_locate.json`、`handover_final_place.json`;
3. episode 结束后的 `summary` 里应有:
   * `step_budget.effective_max_env_steps == 350`、
     `step_budget.within_official_limit == true`、
     `step_budget.planned_total == 345`,
   * `step_budget.actual_by_state`(逐状态真实步数)与
     `step_budget.primitive_by_substage`(抓取原语 7 个子阶段),
   * `holder_locate.coordinate_convention == "vlm_normalized_0_1"`、
     `holder_locate.estimated_top_center_xyz`,
   * `primitive.substages`、`left_takeover`、`final_place`;
4. `step_budget.compressed_states` 为空 = 预算充裕;非空 = 该相位被压缩
   (此时要检查是不是官方 setting 的 `max_episode_steps` 小于 350);
   `step_budget.exhausted == true` 表示跑到上限被截断;
5. 若看到 `failed=true`,看 `failures[*].stage` 判断是 `PRIMITIVE`(垂直对齐/接触
   /IK)、`HOLDER_LOCATE`(VLM 或椭圆拟合)还是 `FINAL_PLACE`(多起点 IK 无解)。

**已知限制(如实保留,不要顺手修)**:基线的放置模块在传了 `expert_actions` 时会
**直接 return**,永远走不到后面的候选姿态扫描;本移植同样**不做候选族兜底** ——
筒口正上方的多起点 IK 无解就直接降级为纯 ACT。另外参考实现已记录的物理缺陷
(左手接管后笔在爪内滑转 ~63°,导致放置时倾斜入筒口磕壁弹出)**不在本次封装范围内**。

---

## 7. 十个任务的完整评测

```bash
export ROBOSYN_HYBRID_ASSETS=$PWD/robosyn_hybrid_assets
export HYBRID_VLM_URL=http://127.0.0.1:11434/v1/chat/completions

for T in handle_basket click_bell water_pouring table_rearrangement \
         drawer_open_place mixer_operating items_handover item_assembly \
         manipulate_pipette sample_loading; do
  bash policy/hybrid/eval.sh "$T" random "hybrid_${T}" 0 \
    --max_episodes 20 --headless true
done
```

结果落在 `eval_result/<task>/hybrid/random/<train_config>/<model>/<timestamp>/evaluation_metrics.json`。

如果要复现官方 100 episode 协议,把 `--max_episodes 20` 改成 `--max_episodes 100`。
注意评分里动作效率占 20%,而失败 episode 按该任务最大步数计,
所以**成功率之外的任何指标都不要单独优化**。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `FileNotFoundError: Required asset 'sam2.1_t.pt'` | 没跑 `download_assets.sh`,或 `ROBOSYN_HYBRID_ASSETS` 指错 |
| `RuntimeError: sam2 包不可用` | `pip install -r robosyn_hybrid/requirements/robosyn-extra.txt` |
| 日志里 `vlm_disabled` | `HYBRID_VLM_URL` 未设置。这是预期的降级,不是崩溃 |
| 日志里 `timeout after 120s` | VLM 太慢。调大 `HYBRID_VLM_TIMEOUT_S`,或换更小的模型 |
| `no image-horizontal nudge found for this pose` | 起始关节位姿下找不到满足基线的横向平移。属流程失败,会自动降级到 ACT |
| `no feasible pose in scale/yaw/center candidate set` | IK 搜索失败。同上,自动降级 |
| FoundationStereo 报 CUDA 扩展编译错误 | 检查 `setup_fs_env.sh` 用的 CUDA 版本与 `nvcc` 是否匹配仿真器驱动 |
| `Policy action has dim X, but env expects 14` | checkpoint 与任务不匹配(见 `robosyn_hybrid/registry.py`) |

---

## 9. 第三方权重许可

| 组件 | 许可 |
| --- | --- |
| SAM2.1 (Ultralytics 构建) | Apache-2.0(Ultralytics)/ Apache-2.0(Meta SAM 2) |
| FoundationStereo | 见 `NVlabs/FoundationStereo` 仓库 LICENSE( NVIDIA Source Code License ) |
| Qwen2.5-VL | Apache-2.0 |
| 官方 ACT checkpoints | Apache-2.0(随 RoboSynChallenge 发布) |

`robosyn_hybrid/scripts/download_assets.sh` 只是下载,不重新分发这些权重;请在使用前自行确认
各上游许可。本仓库自身以 Apache-2.0 发布。
