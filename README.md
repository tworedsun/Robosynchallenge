# RoboSynChallenge hybrid submission

Evaluation code for our **RoboSynChallenge (NeurIPS 2026) preliminary-round**
entry. The repository exposes a single RoboSynChallenge policy adapter,
`policy/hybrid/`, that covers **all ten official tasks** — as the submission
rules require:

* **`handle_basket`** — an RGB-only analytic grasp-and-place stage (motion-stereo
  depth, self-hosted VLM localisation, SAM2.1 mask refinement, FK/IK grasp
  search) followed by an **ACT** action-chunk hand-over for the place-and-carry
  phase.
* **`click_bell`** — an RGB-only analytic press stage (motion-stereo depth on the
  right wrist camera, self-hosted VLM button localisation, FK/IK approach and a
  relative press against the official 4.8 mm success threshold) followed by an
  **ACT** fill that reuses the released `ACT_sim_click_bell` checkpoint.
* **`items_handover`** — an RGB-only analytic hand-over stage (right-arm grasp
  primitives: verticalise → contact press → close → lift; ACT continuation with
  the right gripper locked; left-arm take-over; VLM + inner-rim-ellipse holder
  localisation from both wrist cameras and a place above the rim) followed by an
  **ACT** fill that reuses the released `ACT_sim_items_handover` checkpoint and
  stays inside the official **350-step** budget.
* **the remaining seven tasks** — the released official ACT checkpoints driven
  through the unmodified official ACT inference path.

A hybrid stage never aborts an episode: any failure degrades to pure ACT with
the remaining step budget.

No object ground-truth pose, no simulator depth buffer and no privileged scene
state is read at evaluation time. See [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md)
for the full signal-by-signal declaration.

---

## Task → checkpoint map

| Task | Checkpoint | How it runs |
| --- | --- | --- |
| `handle_basket` | `https://huggingface.co/tworedsun/ACT_sim_handle_basket` | hybrid flow + ACT fill |
| `click_bell` | `https://huggingface.co/RoboSynChallenge/ACT_sim_click_bell` | hybrid flow + ACT fill |
| `water_pouring` | `https://huggingface.co/RoboSynChallenge/ACT_sim_water_pouring` | pure ACT |
| `table_rearrangement` | `https://huggingface.co/RoboSynChallenge/ACT_sim_table_rearrangement` | pure ACT |
| `drawer_open_place` | `https://huggingface.co/RoboSynChallenge/ACT_sim_drawer_open_place` | pure ACT |
| `mixer_operating` | `https://huggingface.co/RoboSynChallenge/ACT_sim_mixer_operating` | pure ACT |
| `items_handover` | `https://huggingface.co/RoboSynChallenge/ACT_sim_items_handover` | hybrid flow + ACT fill |
| `item_assembly` | `https://huggingface.co/RoboSynChallenge/ACT_sim_item_assembly` | pure ACT |
| `manipulate_pipette` | `https://huggingface.co/RoboSynChallenge/ACT_sim_manipulate_pipette` | pure ACT |
| `sample_loading` | `https://huggingface.co/RoboSynChallenge/ACT_sim_sample_loading` | pure ACT |

> Replace `tworedsun` with the account that hosts the migrated checkpoint. The
> handle_basket checkpoint is the released official ACT weights re-packaged in
> the **LeRobot 0.4.4** layout; no additional training was performed. The
> migration and the normalisation-key pitfall it works around are documented in
> the model card of that repository.
>
> `click_bell` and `items_handover` keep their **official** released
> checkpoints: their hybrid flows
> (see [`robosyn_hybrid/clickbell.py`](robosyn_hybrid/clickbell.py) and
> [`robosyn_hybrid/handover.py`](robosyn_hybrid/handover.py)) do the analytic
> stage and hand the remaining steps to that same ACT policy, so the two paths
> of each task share one set of weights. `HYBRID_CKPT` / `--checkpoint_path` is
> therefore optional for them — the pinned Hub revision is used when no local
> path is given.
>
> `items_handover` is capped by the official
> `configs/items_handover/random/gym_config.json` at **350 steps** (the `clear`
> setting allows 300). The flow's per-phase budget adds up to **345** and is
> hard-clamped per phase, so it never reaches the gym `TimeLimit`: when a budget
> is short the affected phase is compressed and, if nothing is left, the flow
> degrades to pure ACT instead of overrunning.

The mapping also lives in code at
[`robosyn_hybrid/registry.py`](robosyn_hybrid/registry.py) so the two can never
drift apart silently. Each of the nine released checkpoints is **pinned to a
commit sha** there, five of which are the exact revisions the organizers
published in `evaluation_results/released_checkpoint_results.json`, so a
re-evaluation reproduces the published baseline instead of following a moving
branch.

---

## Install

Requires the official RoboSynChallenge environment (EmbodiChain + PyTorch 2.7 +
`lerobot==0.4.4`). Start from a working official checkout:

```bash
# 1. official simulator environment must already run:
#    python scripts/eval_policy.py --help

# 2. drop this repository's payload into the official checkout
cd /path/to/RoboSynChallenge
git clone https://github.com/tworedsun/robosynchallenge-hybrid-submission /tmp/hybrid
cp -r /tmp/hybrid/policy/hybrid   policy/
cp -r /tmp/hybrid/robosyn_hybrid  .
cp -r /tmp/hybrid/scripts         robosyn_hybrid/scripts

# 3. extra Python packages (SAM2 loader, VLM client, HF hub)
pip install -r robosyn_hybrid/requirements/robosyn-extra.txt

# 4. third-party weights: SAM2.1 + FoundationStereo + a vision model
export ROBOSYN_HYBRID_ASSETS=$PWD/robosyn_hybrid_assets
bash robosyn_hybrid/scripts/download_assets.sh

# 5. FoundationStereo gets its own environment (its CUDA extensions conflict
#    with the simulator environment)
bash robosyn_hybrid/scripts/setup_fs_env.sh

# 6. start the OpenAI-compatible VLM endpoint; it prints the URL to export
bash robosyn_hybrid/scripts/run_vlm_server.sh &
export HYBRID_VLM_URL=http://127.0.0.1:11434/v1/chat/completions
```

Nothing above needs an internet connection to `huggingface.co` directly;
`robosyn_hybrid/scripts/download_assets.sh` uses `HF_ENDPOINT` (default `https://hf-mirror.com`)
and all large files are verified against recorded SHA-256 digests.

---

## Run

```bash
HYBRID_CKPT=/path/to/ACT_sim_handle_basket \
bash policy/hybrid/eval.sh handle_basket random hybrid_handle_basket 0
```

All ten tasks:

```bash
for T in handle_basket click_bell water_pouring table_rearrangement \
         drawer_open_place mixer_operating items_handover item_assembly \
         manipulate_pipette sample_loading; do
  bash policy/hybrid/eval.sh "$T" random "hybrid_${T}" 0
done
```

`HYBRID_CKPT` may be a local directory or a Hugging Face repo id; when it is
omitted the adapter falls back to the checkpoint registered in
`robosyn_hybrid/registry.py`. That fallback works for `click_bell`,
`items_handover` and the other seven released checkpoints; `handle_basket` ships
its own weights (point `HYBRID_CKPT` at them, or at the repo id that hosts them).

### Environment switches

| Variable | Default | Meaning |
| --- | --- | --- |
| `HYBRID_ENABLED` | `1` | `0` disables the analytic stage and runs plain ACT for `handle_basket` / `click_bell` / `items_handover` |
| `HYBRID_VLM_URL` | *(unset)* | OpenAI-compatible endpoint; unset disables the VLM |
| `HYBRID_VLM_MODEL` | `qwen3vl:8b` | model name sent in the request body (matches `ollama pull qwen3vl:8b`) |
| `HYBRID_VLM_TIMEOUT_S` | `120` | per-request timeout |
| `ROBOSYN_HYBRID_ASSETS` | `<repo>/robosyn_hybrid_assets` | downloaded weights |
| `SAM2_CKPT` | `<assets>/sam2.1_t.pt` | SAM2.1 checkpoint |
| `FS_ROOT` | `<assets>/FoundationStereo` | FoundationStereo checkout |
| `FS_CKPT` | `<FS_ROOT>/pretrained_models/11-33-40/model_best_bp2.pth` | stereo weights |
| `FS_PYTHON` | current interpreter | interpreter for the FoundationStereo worker |
| `FS_STEREO_SOCKET` | `/tmp/fs_stereo.sock` | persistent stereo worker socket |
| `HYBRID_DEBUG_DIR` | *(unset)* | when set, dumps per-stage artefacts here |

---

## Protocol

| | |
| --- | --- |
| Task names | the ten official names above |
| Setting | `random` (domain randomisation on) |
| Observation — state | 14-dim joint positions + gripper (`obs["robot"]["qpos"]`) |
| Observation — images | `cam_high`, `cam_left_wrist`, `cam_right_wrist`, RGB, 480×640 |
| Action output | 14-dim absolute joint positions + gripper: `[left_arm(0-5), left_gripper(6), right_arm(7-12), right_gripper(13)]` |
| `output_chunk_size` | 50 |
| `execution_chunk_size` | 50 |
| `gripper_threshold` | 0.05 |
| Inference timing | the official `policy/inference_timing.py` boundary (`env.step()` excluded) |

---

## Failure behaviour

The analytic stage is best-effort by construction. Every stage is wrapped: if
the VLM endpoint, the stereo model, the SAM2 weights, the IK search or the ACT
forward pass fails, the flow records the reason and **switches to the plain ACT
path** for the rest of the episode instead of aborting it. A partially
provisioned machine therefore degrades to baseline ACT behaviour rather than
scoring zero, and the other tasks are unaffected.

Three consequences worth knowing before you debug a low `handle_basket` score:

1. **The VLM endpoint is effectively required for the hybrid path.** It supplies
   the milk bounding box and yaw, which seed depth back-projection and the IK
   grasp search. Without it the flow degrades to plain ACT on the first stage,
   and `handle_basket` scores whatever the released ACT baseline scores. Check
   `HYBRID_VLM_URL` first.
2. **The hybrid path needs a CUDA device.** `robosyn_hybrid/grasp.py` performs
   FK/IK on the GPU, matching the reference implementation. On a CPU-only box
   the grasp stages raise, degrade to ACT, and `handle_basket` silently becomes
   the baseline.
3. **A failed release judgement locks the left gripper.** This is faithful to
   the evaluated design: until `judge_place_release` approves, the left gripper
   command is overwritten with the closed value on every control step. If the
   VLM answers but never approves, the episode cannot complete the release.

Every degradation is recorded. `HandleBasketFlow.summary()` returns the full
list, and with `HYBRID_DEBUG_DIR` set each stage also writes its artefacts to
disk, so a silent fallback is always traceable to a concrete stage and reason.

---

## License

Apache-2.0, matching the upstream RoboSynChallenge release. Third-party weights
downloaded by `robosyn_hybrid/scripts/download_assets.sh` keep their own licenses — see
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

---

## Repository layout

```
policy/hybrid/          the adapter the official harness imports
  __init__.py           exports get_model / eval / reset_model
  deploy_policy.py      dispatch: three hybrid flows, official ACT elsewhere
  deploy_policy.yml     protocol defaults
  eval.sh               official-style launcher

robosyn_hybrid/         the library the adapter uses
  settings.py           every path/switch, environment-driven
  priors.py             public constants: grasp frame, sizes, camera calibration
  calibration.json      public intrinsics and wrist-camera mount
  registry.py           task -> checkpoint map
  act_wrapper.py        LeRobot ACT, both 0.4.4 and legacy b883328 layouts
  stereo.py             motion-stereo depth (RGB only, no ground truth)
  fs_infer.py           FoundationStereo single-shot inference
  vlm.py                OpenAI-compatible VLM client
  sam2_mask.py          SAM2.1 mask refinement
  grasp.py              grasp-pose search and IK
  flow.py               the handle_basket state machine
  clickbell.py          the click_bell state machine
  handover.py           the items_handover state machine
  scripts/              asset download, FoundationStereo env, VLM server, pre-flight
  requirements/         extra packages and the isolated stereo environment

docs/                   reproduction, submission notes, compliance declaration
```

> All of this package's own scripts live under `robosyn_hybrid/scripts/`, never
> in the official checkout's top-level `scripts/` directory, which already
> contains `eval_policy.py`. See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) §1.
