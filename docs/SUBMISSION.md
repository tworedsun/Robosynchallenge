# 提交操作手册(预赛,RoBoSynChallenge @ NeurIPS 2026)

> 本文件是**内部操作文档**,不随 GitHub 仓库发布。
> 所有事实均来自官网线上站点与官方仓库,出处见每节末尾。

---

## 0. 截止时间(最容易搞错的一点)

| 表述 | 时刻 |
| --- | --- |
| 官方截止 | **2026-10-11 23:59:59 AoE (UTC−12)** |
| 等价 UTC | **2026-10-12 11:59:59 UTC** |
| 等价北京时间 | **2026-10-12 19:59:59 (UTC+8)** |

FAQ 原文:

> "Teams may submit multiple times before **October 11, 2026, Anywhere on Earth (AoE, UTC−12)**. … We will evaluate each team only once, using its latest valid submission received before the deadline."

⚠️ **每队只评一次**,取截止前**最后一次有效提交**。不要卡点提交(限流:提交 10 次/邮箱/小时;注册 5 次/邮箱/小时)。

出处:`https://robosyn-bench.net/static/js/pages-app.js`(线上站点前端 bundle,FAQ 段落)。

---

## 1. 提交物到底需要什么

FAQ 原文("What should a policy submission include?"):

> **Code URL.** Use a GitHub repository for the evaluation code. … Private repositories are allowed if **robosynchallenge** is invited as a collaborator.
>
> **Run and dependency notes.** Include the commands, environment details, and dependencies needed by the organizers. **The repository should expose the RoboSynChallenge policy adapter entrypoints and include `eval.sh`, `deploy_policy.yml`, and `deploy_policy.py`.**
>
> **Hugging Face checkpoint URL.** Provide checkpoints covering all 10 tasks, with a clear task-to-checkpoint mapping in the run notes or README.

后端 `handleSubmitPolicy_` 的硬校验:

```js
const artifactName = requiredString_(submission.artifact_name, "Artifact name");
const title        = requiredString_(submission.title, "Experiment name");
const shortDescription = requiredString_(submission.short_description, "Short description");
const codeLink     = requiredUrl_(submission.code_link, "Code URL");
const checkpointLink = requiredHuggingFaceUrl_(submission.checkpoint_link);
const dataSourceText = requiredString_(submission.data_source_text, "Data source");
const technicalNotes = requiredString_(submission.technical_notes, "Run and dependency notes");
```

`requiredHuggingFaceUrl_` 的正则:

```js
if (!/^https:\/\/huggingface\.co\//i.test(text)) throw new Error("Checkpoint link must be a https://huggingface.co/... URL.");
```

⚠️ **一次申请必须覆盖全部 10 个任务**。线上表单原文:

> "**Each evaluation request must cover all 10 tasks.** List every task's checkpoint and run command in the run notes or repository README. Task-specific checkpoints are allowed within one request; **partial-task requests are not valid.**"

---

## 2. 操作步骤

### 步骤 1 — 申请 access token(如果还没拿到)

打开 `https://robosyn-bench.net/#/register`,填:

| 字段 | 填什么 |
| --- | --- |
| `email` | 你的常用邮箱 |
| `full_name` | **英文逗号分隔、最多 5 人、不允许空项、注册后不可改** |
| `team_name` | 队名 |
| `affiliation` | 单位 |
| `intended_use` | 例如 `RoboSynChallenge preliminary round submission` |
| `misc` | 可选 |

提交后组织者会往 `robosynchallenge@gmail.com` 发审批邮件,通过后把 token 邮件发给你。

### 步骤 2 — 登录

`https://robosyn-bench.net/#/login`,email + token。**session 只有 24 小时**,超时要重登。

### 步骤 3 — 注册 base model

打开 `https://robosyn-bench.net/#/models`(或 My Models),填:

| 字段 | 填什么 |
| --- | --- |
| Base model name | `RoboSynChallenge hybrid ACT adapter (handle_basket + click_bell + items_handover hybrid flows + official ACT baselines)` |
| Checkpoint link | `https://huggingface.co/dimofu/ACT_sim_handle_basket` |
| Code link | `https://github.com/tworedsun/Robosynchallenge` |

**不注册 base model 就无法提交评测。**

### 步骤 4 — 提交评测申请

打开 `https://robosyn-bench.net/#/evaluation`,按下表填写。

---

## 3. 提交表单逐字段填写内容(可直接粘贴)

| 字段 | 值 |
| --- | --- |
| `artifact_name` | `robosynchallenge-hybrid-submission` |
| `title` | `Hybrid analytic flows (handle_basket, click_bell, items_handover) + official ACT baselines for the other seven tasks` |
| `short_description` | 见下方 §3.1 |
| `code_link` | `https://github.com/tworedsun/Robosynchallenge` |
| `checkpoint_link` | `https://huggingface.co/dimofu/ACT_sim_handle_basket` |
| `data_source_text` | 见下方 §3.2 |
| `technical_notes` | 见下方 §3.3 |
| `all_tasks_confirmed` | ✅ 勾选 |
| `is_ranked` | ✅ 勾选(不勾选只是 Test run,不进排行榜) |

### 3.1 `short_description`

```
The submission is a single RoboSynChallenge policy adapter named `hybrid`.

For the handle_basket task it runs an RGB-only analytic grasp-and-place stage
(motion-stereo depth estimated with FoundationStereo, milk localisation with a
self-hosted open-weights VLM, SAM2.1 mask refinement, simulator-free FK/IK grasp
search using the publicly released robot and camera description) and then hands
over to an ACT action-chunk policy for the place-and-carry phase.

For `click_bell` and `items_handover` it runs two more RGB-only analytic flows
(press / hand-over) that hand the remaining steps to the released official ACT
checkpoints of those tasks. For the other seven tasks the same adapter runs the
released official ACT checkpoints (`RoboSynChallenge/ACT_sim_<task>`) through the
unmodified official ACT inference path, so one request covers all ten tasks.

No object ground-truth pose, no simulator depth buffer and no privileged scene
state is read at evaluation time; observations are the three RGB cameras and the
robot's own joint feedback.
```

### 3.2 `data_source_text`

```
Training data:
- handle_basket ACT component: the released official checkpoint
  RoboSynChallenge/ACT_sim_handle_basket (official simulated data), re-packaged
  in the LeRobot 0.4.4 layout. No additional training was performed.
- click_bell / items_handover ACT components and the other seven tasks: the
  released official ACT checkpoints RoboSynChallenge/ACT_sim_<task> (official
  simulated data), used unmodified.
- the analytic perception stage uses no learned component that was trained on
  RoboSynChallenge data: FoundationStereo (open weights) and SAM2.1 (open
  weights) are used as released, and the VLM is a self-hosted open-weights
  model.

No official real data was used. No non-official data was used for training.
```

### 3.3 `technical_notes`(最重要的一段,必须含 10 任务映射与运行命令)

```
ENTRYPOINTS:
All ten tasks use one adapter, installed as policy/hybrid/ in the official
RoboSynChallenge repository:

    <repo>/policy/hybrid/__init__.py        exports get_model / eval / reset_model
    <repo>/policy/hybrid/deploy_policy.py
    <repo>/policy/hybrid/deploy_policy.yml
    <repo>/policy/hybrid/eval.sh

INSTALL:
cd RoboSynChallenge
git clone https://github.com/tworedsun/Robosynchallenge /tmp/hybrid
cp -r /tmp/hybrid/policy/hybrid policy/
cp -r /tmp/hybrid/robosyn_hybrid .
bash robosyn_hybrid/scripts/download_assets.sh      # SAM2.1 + FoundationStereo weights
bash robosyn_hybrid/scripts/run_vlm_server.sh &     # self-hosted VLM, prints HYBRID_VLM_URL
export HYBRID_VLM_URL=http://127.0.0.1:11434/v1/chat/completions

TASK -> CHECKPOINT MAP:
handle_basket         https://huggingface.co/dimofu/ACT_sim_handle_basket
click_bell            https://huggingface.co/RoboSynChallenge/ACT_sim_click_bell
water_pouring         https://huggingface.co/RoboSynChallenge/ACT_sim_water_pouring
table_rearrangement   https://huggingface.co/RoboSynChallenge/ACT_sim_table_rearrangement
drawer_open_place     https://huggingface.co/RoboSynChallenge/ACT_sim_drawer_open_place
mixer_operating       https://huggingface.co/RoboSynChallenge/ACT_sim_mixer_operating
items_handover        https://huggingface.co/RoboSynChallenge/ACT_sim_items_handover
item_assembly         https://huggingface.co/RoboSynChallenge/ACT_sim_item_assembly
manipulate_pipette    https://huggingface.co/RoboSynChallenge/ACT_sim_manipulate_pipette
sample_loading        https://huggingface.co/RoboSynChallenge/ACT_sim_sample_loading

handle_basket / click_bell / items_handover 走混合流程(解析阶段 + ACT 接管/填充),
其余七个任务为纯 ACT。click_bell 与 items_handover 的混合流程都复用上表各自那个
官方 released checkpoint 做 ACT 阶段(失败时也是纯 ACT 兜底),所以对它们
HYBRID_CKPT 是可选的;handle_basket 的迁移权重托管在
https://huggingface.co/dimofu/ACT_sim_handle_basket,同样可以省略
HYBRID_CKPT(显式本地路径仍然优先)。
items_handover 的解析阶段受官方 gym config 的 350 步(clear 为 300 步)硬约束,
逐相位预算合计 345 步,预算不足会提前降级为纯 ACT,绝不超步。

### RUN COMMANDS (all ten tasks) ===
for T in handle_basket click_bell water_pouring table_rearrangement \
         drawer_open_place mixer_operating items_handover item_assembly \
         manipulate_pipette sample_loading; do
  HYBRID_CKPT=<local path or HF repo id for that task> \
  bash policy/hybrid/eval.sh "$T" random "hybrid_$T" 0
done
# click_bell / items_handover 与另外七个 released 任务可以省掉 HYBRID_CKPT
# (自动用 registry 里 pinned 的 Hub revision);handle_basket 省略时自动用
# dimofu/ACT_sim_handle_basket。

PINNED REVISIONS:
All ten checkpoints are pinned in robosyn_hybrid/registry.py: the nine released
official checkpoints, five of which are the exact revisions published in
evaluation_results/released_checkpoint_results.json, plus the migrated
handle_basket upload revision. This way the run reproduces the released
baseline rather than a moving branch:

handle_basket         231ec7a8c5d80876636c2445070489b5b9a25e5d
click_bell            677e65fbb15974024ff840893496197ef7db26d4
water_pouring         0bf0fcfc931a69c52871385f28068fcf873cf07a
table_rearrangement   3a46b36fade1772176b791dd87349f479d0a98c8
drawer_open_place     592e30434aad83cf7bd8f1ee105d7c401488743d
mixer_operating       0f12c53a2a6e093ae5e1e28f20480296b45fdf2b
items_handover        86f6ca7f5decd17a92c3218f295de53497383d14
item_assembly         58955958519157176712a1eff319018d78177ddb
manipulate_pipette    ca013b2e202d594bf347530c61c3ba28c06d1bc2
sample_loading        917bf2796f18abd0a16f723573382161f1a5c5c8

PROTOCOL:
Task:                 all ten official task names, setting random
Observation state:    14-dim joint positions + gripper
Observation images:   cam_high, cam_left_wrist, cam_right_wrist (RGB only)
Action output:        14-dim absolute joint positions + gripper
output_chunk_size:    50
execution_chunk_size: 50
gripper_threshold:    0.05
Inference timing:     official policy/inference_timing.py boundary (env.step excluded)

ENVIRONMENT / DEPENDENCIES:
  * Official RoboSynChallenge conda environment (PyTorch 2.7.0+cu128, Python 3.11).
  * lerobot 0.4.4 (the released ACT checkpoints are migrated to this layout).
  * A self-hosted OpenAI-compatible VLM endpoint, started by
    robosyn_hybrid/scripts/run_vlm_server.sh. HYBRID_VLM_URL may also point at
    any compatible server. If it is unset, the handle_basket pipeline degrades
    to the plain ACT path automatically.
  * FoundationStereo runs in its own environment to keep its CUDA extensions
    away from the simulator environment; robosyn_hybrid/scripts/setup_fs_env.sh
    builds it and FS_PYTHON points at it.
  * SAM2.1 (sam2.1_t.pt) for mask refinement, downloaded by
    robosyn_hybrid/scripts/download_assets.sh.

FAILURE BEHAVIOUR:
Every stage of the handle_basket pipeline is wrapped: if the VLM endpoint, the
stereo depth model or the IK search is unavailable, the adapter logs the reason
and continues with the plain ACT path instead of aborting the episode, so the
remaining nine tasks and the ACT fallback are unaffected by a partial
environment.
```

> ⚠️ 提交前请**逐条核对 10 个 HF 仓库名**(可用 `https://hf-mirror.com/api/models/RoboSynChallenge/ACT_sim_<task>` 验证返回 200;handle_basket 对应 `https://huggingface.co/dimofu/ACT_sim_handle_basket`)。

---

## 4. 提交前 checklist

- [ ] GitHub 仓库已公开(或已把 `robosynchallenge` 加为 collaborator)
- [ ] 仓库根目录含 `policy/hybrid/{__init__.py,deploy_policy.py,deploy_policy.yml,eval.sh}`
- [ ] `policy/hybrid/__init__.py` 顶层能导出 `get_model` / `eval` / `reset_model`
- [ ] README 里有 10 任务 → checkpoint 的映射表
- [ ] HF checkpoint 仓库已上传,链接是 `https://huggingface.co/...`(后端正则强制)
- [ ] 10 个任务的 checkpoint 全部可访问
- [ ] `data_source_text` 与 `technical_notes` 已填(后端必填)
- [ ] `all_tasks_confirmed` 已勾选
- [ ] `is_ranked` 已勾选(否则不进排行榜)
- [ ] 提交时间早于 **2026-10-12 19:59:59 北京时间**
- [ ] 记住:每队只评一次,取截止前最后一次有效提交

---

## 5. 评分公式(决定我们该优化什么)

FAQ 原文:

> **Overall Score = 75% Success Rate + 20% Action Efficiency + 5% Inference Efficiency**
> Episode Action Efficiency = (1 − Used Action Steps / H) × 100
> Episode Inference Efficiency = max(0, 1 − Measured Inference Time / T) × 100

- 失败 episode 按该任务最大步数计 → **动作效率 0**。
- `handle_basket` 的 `H = 500`。
- 因此:**成功率是绝对大头**;在成功的前提下,**越早触发 `is_task_success()` 越好**(我们的状态机在成功时立即返回,就是为了这一点)。

---

## 6. 已知风险(提交前请知悉)

1. **未在本机验证过。** 这台容器当前没有挂载 GPU(`/dev/nvidia*` 不存在,`torch.cuda.is_available() == False`),封装后的代码**没有跑过任何一个 episode**。
2. **VLM 端点换了实现。** 原始 12.9% 的成绩是在 Codex CLI(`deepseek-flash`)下取得的;新包装改用自托管 OpenAI 兼容端点,即使权重相同,**prompt 响应分布也会变**,成功率不可外推。
3. **合规灰区。** 官网没有任何关于 FK/IK、固定先验、解析式流程的条款(全站检索零命中),只有两条明确的输入限制:评测只提供 RGB;任务只能基于视觉 + 本体感受。建议在提交的同时发邮件到 `robosynchallenge@gmail.com` 书面确认。
4. **"10 个任务"是人工审查项。** 后端并不校验 `all_tasks_confirmed`,但规则明确写了 partial-task 无效。
