# ----------------------------------------------------------------------------
# RoboSynChallenge hybrid policy adapter.
#
# Contract required by scripts/eval_policy.py (see load_policy_adapter):
#
#     get_model(usr_args: dict) -> model
#     eval(env, model, obs) -> (obs, info, truncated, inference_times_s)
#     reset_model(model) -> None
#
# Two evaluation paths are provided behind that single contract:
#
#   * tasks served by the released baseline ACT checkpoints -> pure ACT,
#     behaviourally identical to the official policy/act adapter;
#   * handle_basket -> the in-repo hybrid flow (depth + VLM + SAM2 + grasp +
#     ACT takeover);
#   * click_bell -> the in-repo hybrid flow (motion-stereo depth + VLM button
#     localisation + FK/IK approach + relative press + ACT fill), which reuses
#     the official released ACT checkpoint for its fill phase;
#   * items_handover -> the in-repo hybrid flow (right-arm grasp primitives +
#     ACT continuation + left-arm take-over + VLM/ellipse holder localisation +
#     place above the rim + ACT fill), which likewise reuses the official
#     released ACT checkpoint and never exceeds the official 350-step budget.
#   If a flow cannot be imported or constructed for any reason, the adapter
#   degrades to pure ACT instead of failing the episode.
#
# No absolute path is hard-coded here.
# ----------------------------------------------------------------------------

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

# Make `robosyn_hybrid` importable when the adapter is imported first.  All
# participant runtime code lives inside policy/: the package is
# policy/hybrid/robosyn_hybrid, so policy/hybrid must be on sys.path.  Both
# locations are derived from this file (policy/hybrid/deploy_policy.py), never
# hard-coded; the official eval.sh also puts them on PYTHONPATH.
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[1]
for _candidate in (str(_PKG_DIR), str(_REPO_ROOT)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import numpy as np  # noqa: E402  (import order is intentional)
import torch  # noqa: E402

from robosyn_hybrid import registry  # noqa: E402
from robosyn_hybrid.act_wrapper import ACTPolicyWrapper  # noqa: E402

__all__ = ["HybridPolicy", "get_model", "eval", "reset_model"]

_LOG_PREFIX = "[hybrid]"

#: Fallback ACT action-chunk length when `hybrid_act_step` is not configured.
DEFAULT_ACT_STEP = 50
#: Fallback episode budget handed to a hybrid flow when `max_steps` is unset.
#: Official gym configs: handle_basket 500, click_bell 361, items_handover 350.
DEFAULT_MAX_ENV_STEPS = 500
#: Per-task fallback budgets (SPEC: click_bell must default to 361 and
#: items_handover to the official 350 -- no step-budget patch is ever applied).
DEFAULT_MAX_ENV_STEPS_BY_TASK = {
    "handle_basket": 500,
    "click_bell": 361,
    "items_handover": 350,
}
#: Default number of ACT chunks taken over by the flow after the grasp phases.
DEFAULT_ACT_CHUNKS = 6
#: ``task_name -> (module name, class name)`` for the in-repo hybrid flows.
FLOW_FACTORIES = {
    "handle_basket": ("robosyn_hybrid.flow", "HandleBasketFlow"),
    "click_bell": ("robosyn_hybrid.clickbell", "ClickBellFlow"),
    "items_handover": ("robosyn_hybrid.handover", "HandoverFlow"),
}


def _default_max_env_steps(task_name: str) -> int:
    """Per-task fallback episode budget (click_bell 361, items_handover 350)."""
    return int(DEFAULT_MAX_ENV_STEPS_BY_TASK.get(str(task_name), DEFAULT_MAX_ENV_STEPS))


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _log(message: str) -> None:
    print(f"{_LOG_PREFIX} {message}", flush=True)


def _warn(message: str) -> None:
    print(f"{_LOG_PREFIX} WARNING: {message}", flush=True)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _as_int(value, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        _warn(f"could not parse {value!r} as an integer; using {default}.")
        return default


def _as_done(value) -> bool:
    """Mirror RecordingEnvProxy._as_done for terminated/truncated flags."""
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, np.ndarray):
        return bool(value.any())
    return bool(value)


def _read_elapsed_steps(env):
    """Best-effort read of the environment's episode step counter.

    Returns ``None`` when the running environment does not expose one; the
    caller then simply relies on ``reset_model`` for episode boundaries.
    """
    candidates = (
        lambda: env.get_wrapper_attr("elapsed_steps"),
        lambda: getattr(env.unwrapped, "elapsed_steps"),
        lambda: getattr(env.unwrapped, "_elapsed_steps"),
        lambda: getattr(env, "elapsed_steps"),
    )
    for getter in candidates:
        try:
            value = getter()
        except Exception:
            continue
        if value is None:
            continue
        try:
            if hasattr(value, "item"):
                value = value.item()
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


# ----------------------------------------------------------------------------
# checkpoint resolution
# ----------------------------------------------------------------------------
def _hf_unavailable_error(repo_id: str, revision: str) -> RuntimeError:
    """Actionable error for a missing huggingface_hub installation."""
    lines = [
        "huggingface_hub is not installed, so the released checkpoint "
        f"'{repo_id}' (revision '{revision}') cannot be downloaded.",
    ]
    if _as_bool(os.environ.get("HF_HUB_OFFLINE"), False):
        lines.append(
            "HF_HUB_OFFLINE=1 is set, so this run must use an already-downloaded "
            "checkpoint directory."
        )
    if _as_bool(os.environ.get("HYBRID_ALLOW_NO_HF"), False):
        lines.append(
            "HYBRID_ALLOW_NO_HF=1 allows running without the Hub, but only when a "
            "local checkpoint is provided; none was given."
        )
    lines.extend(
        [
            "Do one of the following:",
            "  1) pip install huggingface_hub   (then re-run the evaluation), or",
            f"  2) hf download {repo_id} --repo-type model "
            f"--revision {revision} --local-dir checkpoints/{repo_id.split('/')[-1]}",
            "     and pass that directory via --checkpoint_path "
            "(policy/hybrid/eval.sh reads HYBRID_CKPT).",
        ]
    )
    return RuntimeError("\n".join(lines))


def _resolve_checkpoint(task_name: str, checkpoint_path, checkpoint_revision) -> str:
    """Return a local checkpoint directory for ``task_name``.

    An explicit ``checkpoint_path`` always wins.  Otherwise the registry maps
    the task to its released Hub repository and ``snapshot_download`` fetches it
    at a pinned revision, in this order of precedence:

    1. ``usr_args["checkpoint_revision"]`` (explicit override),
    2. ``registry.revision_for(task)`` -- the sha recorded when this submission
       was prepared,
    3. ``main``, with an explicit "not pinned" notice in the log.

    A failing pinned fetch falls back to ``main`` with a loud warning, so a
    stale pin degrades reproducibility instead of aborting the evaluation.
    """
    if checkpoint_path is not None and str(checkpoint_path).strip():
        resolved = str(Path(str(checkpoint_path)).expanduser())
        if not Path(resolved).exists():
            _warn(
                f"checkpoint_path '{resolved}' does not exist; passing it through "
                "and letting LeRobot report the details."
            )
        _log(f"checkpoint: using local path {resolved}")
        return resolved

    # Raises ValueError with an actionable hint for the hybrid task and for
    # unknown task names.
    repo_id = registry.checkpoint_for(task_name)

    requested_revision = None
    if checkpoint_revision is not None and str(checkpoint_revision).strip():
        requested_revision = str(checkpoint_revision).strip()
    registry_revision = None
    try:
        registry_revision = registry.revision_for(task_name)
    except Exception:  # noqa: BLE001 - registry revision is best-effort
        registry_revision = None
    revision = requested_revision or registry_revision or "main"

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # ImportError and any broken install
        raise _hf_unavailable_error(repo_id, revision) from exc

    if requested_revision:
        _log(f"checkpoint: snapshot_download({repo_id!r}, revision={revision!r}) [pinned by override]")
    elif registry_revision:
        _log(
            f"checkpoint: snapshot_download({repo_id!r}, revision={revision!r}) "
            "[pinned by robosyn_hybrid.registry]"
        )
    else:
        _log(
            f"checkpoint: snapshot_download({repo_id!r}, revision='main') "
            "[NOT pinned: no revision in the registry and none passed via "
            "--checkpoint_revision; re-evaluation will not be reproducible]"
        )
    try:
        return snapshot_download(repo_id=repo_id, revision=revision)
    except Exception as exc:  # noqa: BLE001
        if requested_revision or not registry_revision:
            raise
        _warn(
            f"pinned revision {revision!r} of {repo_id!r} could not be fetched "
            f"({type(exc).__name__}: {exc}); falling back to the moving 'main' "
            "branch, which breaks reproducibility."
        )
        return snapshot_download(repo_id=repo_id, revision="main")


# ----------------------------------------------------------------------------
# model container
# ----------------------------------------------------------------------------
class HybridPolicy:
    """Container holding the ACT wrapper and, for a hybrid task, the flow.

    Attributes
    ----------
    act:
        :class:`robosyn_hybrid.act_wrapper.ACTPolicyWrapper` instance.
    flow:
        ``HandleBasketFlow`` / ``ClickBellFlow`` instance, or ``None`` for the
        pure-ACT path (including a degraded hybrid run).
    degrade_reason:
        Non-empty string explaining why the hybrid flow is not active.
    """

    def __init__(
        self,
        task_name: str,
        act: ACTPolicyWrapper,
        *,
        flow=None,
        act_step: int = DEFAULT_ACT_STEP,
        checkpoint_path: str | None = None,
        hybrid_enabled: bool = False,
        degrade_reason: str | None = None,
    ):
        self.task_name = str(task_name)
        self.act = act
        self.flow = flow
        self.act_step = int(act_step)
        self.checkpoint_path = checkpoint_path
        self.hybrid_enabled = bool(hybrid_enabled)
        self.degrade_reason = degrade_reason
        # Episode bookkeeping for lazy flow reset (reset_model has no env).
        self._needs_episode_reset = True
        self._flow_started = False
        self._last_elapsed: int | None = None

    @property
    def action_dim(self) -> int:
        return int(self.act.action_dim)

    @property
    def using_flow(self) -> bool:
        return self.flow is not None

    def summary(self) -> dict:
        info = {
            "task_name": self.task_name,
            "checkpoint_path": self.checkpoint_path,
            "act_step": self.act_step,
            "action_dim": self.action_dim,
            "hybrid_enabled": self.hybrid_enabled,
            "using_flow": self.using_flow,
            "degrade_reason": self.degrade_reason,
            "normalization_path": getattr(self.act, "normalization_path", None),
        }
        if self.flow is not None and hasattr(self.flow, "summary"):
            try:
                info["flow"] = self.flow.summary()
            except Exception as exc:  # pragma: no cover - flow specific
                info["flow_error"] = f"{type(exc).__name__}: {exc}"
        return info

    def reset(self) -> None:
        """Model-level reset; mirrors ``reset_model`` for convenience."""
        # Dump the finished episode's hybrid summary BEFORE wiping it.  The
        # official metrics file only records success/steps/inference, so
        # without this the flow's own diagnostics (which state failed, the
        # step budget accounting, the placement constraint check) would be
        # invisible after a run -- which makes a failed episode impossible to
        # diagnose.  One JSON per episode, written next to the flow's other
        # debug artifacts.
        self._dump_episode_summary()
        try:
            self.act.reset()
        except Exception as exc:  # pragma: no cover - defensive
            _warn(f"ACT reset failed ({type(exc).__name__}: {exc}).")

    def _dump_episode_summary(self) -> None:
        """Persist ``summary()`` as ``episode_summary_<n>.json`` under the debug dir."""
        if self.flow is None:
            return
        debug_dir = getattr(self.flow, "debug_dir", None) or getattr(
            self.flow, "_debug_dir", None
        )
        if debug_dir is None:
            return
        try:
            from pathlib import Path as _Path

            out_dir = _Path(str(debug_dir))
            out_dir.mkdir(parents=True, exist_ok=True)
            self._summary_seq = getattr(self, "_summary_seq", 0) + 1
            path = out_dir / f"episode_summary_{self._summary_seq:03d}.json"
            path.write_text(
                json.dumps(self.summary(), indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            _log(f"episode summary written: {path}")
        except Exception as exc:  # pragma: no cover - never break the episode
            _warn(f"could not write episode summary ({type(exc).__name__}: {exc})")
        self._needs_episode_reset = True
        self._flow_started = False
        self._last_elapsed = None


# ----------------------------------------------------------------------------
# flow construction (lazy, fully guarded)
# ----------------------------------------------------------------------------
def _build_flow(act, usr_args: dict, act_chunks: int, task_name: str):
    """Build the hybrid flow for ``task_name``; return ``(flow, degrade_reason)``.

    Dispatches on ``task_name`` (``handle_basket`` -> ``HandleBasketFlow``,
    ``click_bell`` -> ``ClickBellFlow``, ``items_handover`` -> ``HandoverFlow``).
    All three flows share the same four-method contract and are constructed with
    the same keyword arguments, so the only task-specific value here is the
    default step budget.

    Never raises: any import or construction failure degrades to pure ACT with
    an explicit reason, because a broken optional dependency must not abort
    ``load_policy_adapter`` / the whole evaluation.
    """
    factory = FLOW_FACTORIES.get(str(task_name))
    if factory is None:
        reason = f"no hybrid flow is registered for task '{task_name}'."
        _warn(reason)
        return None, reason
    module_name, class_name = factory

    try:
        module = importlib.import_module(module_name)
        flow_class = getattr(module, class_name)
    except Exception as exc:
        reason = (
            f"{module_name}.{class_name} is unavailable "
            f"({type(exc).__name__}: {exc}); running pure ACT for {task_name}."
        )
        _warn(reason)
        return None, reason

    try:
        from robosyn_hybrid.settings import get_settings

        settings = get_settings()
    except Exception as exc:
        reason = (
            "robosyn_hybrid.settings could not be loaded "
            f"({type(exc).__name__}: {exc}); running pure ACT for {task_name}."
        )
        _warn(reason)
        return None, reason

    max_env_steps = _as_int(
        usr_args.get("max_steps"), _default_max_env_steps(task_name)
    )
    debug_dir = getattr(settings, "debug_dir", None)

    try:
        flow = flow_class(
            act,
            settings,
            act_chunks=int(act_chunks),
            max_env_steps=int(max_env_steps),
            debug_dir=debug_dir,
        )
    except Exception as exc:
        reason = (
            f"{class_name} could not be constructed "
            f"({type(exc).__name__}: {exc}); running pure ACT for {task_name}."
        )
        _warn(reason)
        return None, reason

    _log(
        f"hybrid flow active for {task_name}: {class_name}, act_chunks="
        f"{int(act_chunks)}, max_env_steps={int(max_env_steps)}, "
        f"debug_dir={debug_dir}"
    )
    return flow, None


# ----------------------------------------------------------------------------
# public contract
# ----------------------------------------------------------------------------
def get_model(usr_args: dict) -> HybridPolicy:
    """Build the hybrid (or pure ACT) policy described by ``usr_args``."""
    usr_args = dict(usr_args or {})

    task_name = usr_args.get("task_name")
    if not task_name:
        raise ValueError(
            "usr_args['task_name'] is required (set it in deploy_policy.yml or "
            "pass --task_name)."
        )
    task_name = str(task_name)

    device = usr_args.get("device") or usr_args.get("pytorch_device") or "cuda"
    act_step = _as_int(usr_args.get("hybrid_act_step"), DEFAULT_ACT_STEP)
    hybrid_act_chunks = _as_int(
        usr_args.get("hybrid_act_chunks"), DEFAULT_ACT_CHUNKS
    )
    hybrid_enabled = _as_bool(usr_args.get("hybrid_enabled"), True)

    checkpoint_path = _resolve_checkpoint(
        task_name,
        usr_args.get("checkpoint_path"),
        usr_args.get("checkpoint_revision"),
    )

    act = ACTPolicyWrapper(
        checkpoint_path,
        device=device,
        act_step=act_step,
        state_obs_path=usr_args.get("state_obs_path", "robot/qpos"),
        strict_action_dim=_as_bool(usr_args.get("strict_action_dim"), True),
        image_key_map=usr_args.get("image_key_map"),
    )

    flow = None
    degrade_reason = None
    if registry.is_hybrid_task(task_name):
        if not hybrid_enabled:
            degrade_reason = (
                f"hybrid_enabled is false; running pure ACT for {task_name}."
            )
            _warn(degrade_reason)
        else:
            flow, degrade_reason = _build_flow(
                act, usr_args, hybrid_act_chunks, task_name
            )
    else:
        _log(f"task '{task_name}': pure ACT path (released baseline checkpoint)")

    model = HybridPolicy(
        task_name,
        act,
        flow=flow,
        act_step=act_step,
        checkpoint_path=checkpoint_path,
        hybrid_enabled=hybrid_enabled,
        degrade_reason=degrade_reason,
    )
    _log(
        f"ready: task={task_name}, checkpoint={checkpoint_path}, "
        f"action_dim={model.action_dim}, act_step={act_step}, "
        f"flow={'on' if flow is not None else 'off'}"
    )
    return model


def _maybe_reset_flow(env, model) -> None:
    """Lazily reset the flow at episode boundaries.

    ``reset_model(model)`` is invoked by ``eval_policy.py`` right after
    ``env.reset(...)`` but without the env, so the actual ``flow.reset(env)``
    happens here, on the first ``eval`` call of the episode.  Two independent
    triggers are honoured:

    * the flag set by ``reset_model`` / the first call ever, and
    * an ``elapsed_steps`` counter that went backwards or restarted at the
      beginning of an episode (in case ``reset_model`` is not called).
    """
    elapsed = _read_elapsed_steps(env)
    last_elapsed = getattr(model, "_last_elapsed", None)

    needs_reset = bool(getattr(model, "_needs_episode_reset", False)) or not bool(
        getattr(model, "_flow_started", False)
    )
    if elapsed is not None and last_elapsed is not None and elapsed < last_elapsed:
        needs_reset = True
    if (
        elapsed is not None
        and last_elapsed is not None
        and last_elapsed > 1
        and elapsed <= 1
    ):
        needs_reset = True

    if needs_reset:
        try:
            model.flow.reset(env)
            _log(
                "flow.reset(env) at episode start "
                f"(elapsed_steps={elapsed}, previous={last_elapsed})."
            )
        except Exception as exc:
            _warn(
                "flow.reset(env) failed "
                f"({type(exc).__name__}: {exc}); continuing with the flow state."
            )
        model._needs_episode_reset = False
        model._flow_started = True

    model._last_elapsed = elapsed


def _to_env_action(env, action: np.ndarray, act: ACTPolicyWrapper):
    """Convert a wrapper action to the tensor shape the environment expects."""
    action_tensor = torch.as_tensor(action, dtype=torch.float32)

    try:
        env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    except Exception:
        env_action_dim = None

    if env_action_dim is not None:
        policy_action_dim = int(action_tensor.shape[-1])
        if policy_action_dim != env_action_dim:
            message = (
                f"Policy action has dim {policy_action_dim}, but env expects "
                f"{env_action_dim}."
            )
            if act.strict_action_dim or policy_action_dim < env_action_dim:
                raise ValueError(message)
            _warn(message + " Truncating the action (strict_action_dim is false).")
            action_tensor = action_tensor[:, :env_action_dim]

    try:
        device = env.unwrapped.device
    except Exception:
        device = None
    if device is not None:
        action_tensor = action_tensor.to(device=device, dtype=torch.float32)
    return action_tensor


def _eval_act(env, model, obs):
    """Pure ACT rollout step: identical semantics to official policy/act."""
    act = model.act
    act_step = int(model.act_step)

    final_obs = obs
    info = None
    truncated = False
    inference_times_s: list[float] = []

    for _ in range(act_step):
        action, runs_model_inference = act.select_chunk_action(final_obs)
        if runs_model_inference:
            # Timing is recorded inside the wrapper around the model forward;
            # cached queue pops contribute nothing, exactly like the official
            # adapter's `runs_model_inference` branch.
            inference_times_s.extend(act.drain_inference_times())

        final_obs, reward, terminated, truncated, info = env.step(
            _to_env_action(env, action, act)
        )

        is_truncated = _as_done(truncated)
        if not is_truncated and env.get_wrapper_attr("is_task_success")():
            break
        if is_truncated:
            break

    return final_obs, info, truncated, inference_times_s


def eval(env, model, obs):
    """Advance the episode; returns ``(obs, info, truncated, inference_times_s)``."""
    flow = getattr(model, "flow", None)
    if flow is not None:
        _maybe_reset_flow(env, model)
        result = flow.eval(env, obs)
        # Persist the flow summary as soon as the episode ends.  ``reset_model``
        # also dumps it, but that only fires at the START of the next episode,
        # so a single-episode run would otherwise leave no diagnostic artifact.
        if _as_done(result[2]):
            model._dump_episode_summary()
        return result
    return _eval_act(env, model, obs)


def reset_model(model) -> None:
    """Per-episode reset; called by eval_policy.py right after ``env.reset``.

    Only the model is available here -- not the env -- so the flow's env-aware
    reset is deferred to the next ``eval`` call via ``_needs_episode_reset``.
    """
    act = getattr(model, "act", None)
    if act is None:
        act = model
    reset = getattr(act, "reset", None)
    if callable(reset):
        try:
            reset()
        except Exception as exc:
            _warn(f"ACT reset failed ({type(exc).__name__}: {exc}).")

    if hasattr(model, "_needs_episode_reset"):
        model._needs_episode_reset = True
        model._flow_started = False
        model._last_elapsed = None
    flow = getattr(model, "flow", None)
    if flow is not None:
        _log(
            "episode reset requested; flow.reset(env) will run on the next "
            "eval() call (reset_model has no env)."
        )
