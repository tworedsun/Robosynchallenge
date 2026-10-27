# ----------------------------------------------------------------------------
# LeRobot ACT wrapper used by the RoboSynChallenge hybrid policy.
#
# Ported from the server-side `policy/act/deploy_policy.py`, which supports two
# LeRobot checkpoint generations:
#
#   1. lerobot b883328 ("legacy") checkpoints whose ACTPolicy carries built-in
#      normalizers, so `select_action()` already returns denormalised actions.
#   2. lerobot 0.4.4 checkpoints that ship `policy_preprocessor.json` /
#      `policy_postprocessor.json` next to `pretrained_model/`; normalization
#      happens outside the policy and must be applied explicitly.
#
# The wrapper keeps the original observation construction, the non-finite
# action guard and the internal action-queue semantics, and additionally
# reports whether a given call actually executed a model forward so that the
# caller can time inference with `policy.inference_timing`.
# ----------------------------------------------------------------------------

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy

__all__ = ["ACTPolicyWrapper", "DEFAULT_IMAGE_KEY_MAP", "resolve_device"]

#: LeRobot image feature key -> RoboSynChallenge observation camera name.
DEFAULT_IMAGE_KEY_MAP: dict[str, str] = {
    "observation.images.cam_high": "cam_high",
    "observation.images.cam_right_wrist": "cam_right_wrist",
    "observation.images.cam_left_wrist": "cam_left_wrist",
}

_TIMING_HOOKS: tuple[Any, Any] | None = None


def _local_start_inference(device: Any = None):
    """Fallback timing start used when `policy.inference_timing` is missing."""
    if device is not None and str(device).startswith("cuda"):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
        except Exception:  # pragma: no cover - synchronize is best effort
            pass
    return time.perf_counter()


def _local_finish_inference(started_at: float, samples: list, device: Any = None):
    """Fallback timing stop; mirrors `policy.inference_timing.finish_inference`."""
    if device is not None and str(device).startswith("cuda"):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
        except Exception:  # pragma: no cover - synchronize is best effort
            pass
    samples.append(time.perf_counter() - started_at)


def _timing_hooks():
    """Return the official timing hooks, falling back to local ones.

    The import is deferred (and only attempted once) so that importing this
    module never depends on the simulator repository being importable.
    """
    global _TIMING_HOOKS
    if _TIMING_HOOKS is None:
        try:
            from policy.inference_timing import finish_inference, start_inference

            _TIMING_HOOKS = (start_inference, finish_inference)
        except Exception as exc:  # pragma: no cover - depends on the checkout
            print(
                "[act_wrapper] policy.inference_timing unavailable "
                f"({type(exc).__name__}: {exc}); using the local timing fallback.",
                flush=True,
            )
            _TIMING_HOOKS = (_local_start_inference, _local_finish_inference)
    return _TIMING_HOOKS


def resolve_device(device: Any) -> str:
    """Return a usable device string, degrading CUDA to CPU when needed."""
    resolved = str(device or "cuda")
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"[act_wrapper] device '{resolved}' was requested but "
            "torch.cuda.is_available() is False; falling back to 'cpu'.",
            flush=True,
        )
        return "cpu"
    return resolved


def _image_features(config: Any) -> dict:
    """Return the config's image-feature mapping across lerobot versions."""
    features = getattr(config, "image_features", None)
    if features:
        return dict(features)
    features = getattr(config, "input_features", None)
    if isinstance(features, dict):
        return {
            key: value
            for key, value in features.items()
            if str(key).startswith("observation.images.")
        }
    return {}


def _shape_last_dim(shape: Any) -> int | None:
    if shape is None:
        return None
    if isinstance(shape, int):
        return int(shape)
    try:
        dims = tuple(int(dim) for dim in shape)
    except TypeError:
        return None
    return dims[-1] if dims else None


def _action_dim_from_config(config: Any) -> int:
    """Read the action dimension declared by the checkpoint config."""
    output_features = getattr(config, "output_features", None)
    feature = None
    if isinstance(output_features, dict):
        feature = output_features.get("action")
    elif output_features is not None:
        feature = getattr(output_features, "action", None)

    for candidate in (feature, getattr(config, "action_feature", None)):
        dim = _shape_last_dim(getattr(candidate, "shape", None))
        if dim:
            return int(dim)

    raise ValueError(
        "Could not infer the ACT action dimension from the checkpoint config "
        f"({type(config).__name__}): neither 'output_features[\"action\"].shape' "
        "nor 'action_feature.shape' is available."
    )


class ACTPolicyWrapper:
    """Thin, explicitly-timed wrapper around a LeRobot ``ACTPolicy``.

    The class owns the action queue semantics of the original adapter: one
    action is returned per call, a new chunk is produced only when the queue is
    empty (or when temporal ensembling is enabled, in which case every call runs
    the model).  ``ran_model_inference`` tells the caller whether this
    particular call paid for a forward pass.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        act_step: int = 50,
        state_obs_path: str = "robot/qpos",
        strict_action_dim: bool = True,
        image_key_map: dict | None = None,
    ):
        if checkpoint_path is None or not str(checkpoint_path).strip():
            raise ValueError("checkpoint_path must be a non-empty path.")
        checkpoint_path = str(checkpoint_path)

        self.checkpoint_path = checkpoint_path
        self.device = resolve_device(device)
        self.act_step = int(act_step)
        if self.act_step <= 0:
            raise ValueError(f"act_step must be positive, got {self.act_step}.")
        self.state_obs_path = str(state_obs_path or "robot/qpos")
        self.strict_action_dim = bool(strict_action_dim)

        # Timing samples recorded by this wrapper, consumed via
        # `drain_inference_times()`.
        self.inference_times_s: list[float] = []

        policy = self._load_policy(checkpoint_path, self.device)
        self.policy = policy

        resolved_image_key_map = dict(DEFAULT_IMAGE_KEY_MAP)
        resolved_image_key_map.update(image_key_map or {})
        image_keys = list(_image_features(policy.config).keys())
        for image_key in image_keys:
            resolved_image_key_map.setdefault(
                image_key,
                str(image_key).removeprefix("observation.images."),
            )
        self.image_key_map = resolved_image_key_map
        self.act_image_keys = image_keys

        self.act_device = next(policy.parameters()).device
        self._action_dim = _action_dim_from_config(policy.config)

        # Keep the pre-0.4.4 attribute names available on the policy object as
        # well: downstream helpers (and older copies of this adapter) read them
        # from the model instance.
        policy.act_device = self.act_device
        policy.act_step = self.act_step
        policy.state_obs_path = self.state_obs_path
        policy.strict_action_dim = self.strict_action_dim
        policy.act_image_keys = self.act_image_keys
        policy.image_key_map = self.image_key_map

        self._normalization_path = self._resolve_normalization_path()

        self._temporal_ensemble = (
            getattr(policy.config, "temporal_ensemble_coeff", None) is not None
        )
        self._bind_action_queue()

        print(
            "[act_wrapper] loaded ACT checkpoint "
            f"(device={self.act_device}, act_step={self.act_step}, "
            f"action_dim={self.action_dim}, cameras={self.act_image_keys}, "
            f"state_path={self.state_obs_path}, "
            f"temporal_ensemble={self._temporal_ensemble})",
            flush=True,
        )

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_policy(checkpoint_path: str, device: str):
        """`ACTPolicy.from_pretrained` with the official cli_overrides fallback."""
        cli_overrides = [f"--device={device}"]
        try:
            policy = ACTPolicy.from_pretrained(
                checkpoint_path,
                cli_overrides=cli_overrides,
            )
        except TypeError as exc:
            if "cli_overrides" not in str(exc):
                raise
            config = PreTrainedConfig.from_pretrained(
                checkpoint_path,
                cli_overrides=cli_overrides,
            )
            policy = ACTPolicy.from_pretrained(checkpoint_path, config=config)
        policy.eval()
        return policy

    @staticmethod
    def _legacy_normalizers_are_usable(policy) -> bool:
        """True for the b883328 ACTPolicy with statistics already loaded."""
        if not (
            hasattr(policy, "normalize_inputs")
            and hasattr(policy, "unnormalize_outputs")
        ):
            return False

        stat_names = ("normalize_inputs.", "unnormalize_outputs.")
        for name, buf in policy.named_buffers():
            if (
                name.startswith(stat_names)
                and buf.is_floating_point()
                and name.endswith(("mean", "std", "min", "max"))
            ):
                if not bool(torch.isinf(buf.float()).any()):
                    return True
        return False

    @staticmethod
    def _load_external_processors(policy, checkpoint_path: str | Path):
        """Load the 0.4.4 pre/post processors stored next to the model."""
        checkpoint_path = Path(checkpoint_path)
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            if not (checkpoint_path / name).exists():
                raise FileNotFoundError(
                    f"Missing '{name}' next to the checkpoint. "
                    "This checkpoint was saved by legacy LeRobot (b883328) and must be "
                    "evaluated with policy/act/.venv or another environment pinned to that commit."
                )

        try:
            from lerobot.policies.factory import make_pre_post_processors
        except Exception as exc:
            raise RuntimeError(
                "The checkpoint has external 0.4.4 processor files, but the current "
                "LeRobot cannot create them. Use the same conda env used for training "
                "(lerobot 0.4.4) for evaluation."
            ) from exc

        try:
            policy.act_preprocessor, policy.act_postprocessor = make_pre_post_processors(
                policy_cfg=policy.config,
                pretrained_path=str(checkpoint_path),
                preprocessor_overrides={
                    "device_processor": {"device": policy.config.device}
                },
                postprocessor_overrides={"device_processor": {"device": "cpu"}},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load 0.4.4 processors from {checkpoint_path}: {exc}"
            ) from exc

    def _resolve_normalization_path(self) -> str:
        """Choose built-in vs external normalization and announce the choice."""
        if self._legacy_normalizers_are_usable(self.policy):
            # b883328 ACTPolicy unnormalizes inside select_action already.
            self.policy.use_external_processors = False
            self.use_external_processors = False
            print(
                "[act_wrapper] normalization path: legacy built-in normalizers "
                "(lerobot b883328 format)",
                flush=True,
            )
        else:
            # lerobot 0.4.4 moved normalization outside the policy; the raw
            # model output must be unnormalized with the saved postprocessor.
            self._load_external_processors(self.policy, self.checkpoint_path)
            self.policy.use_external_processors = True
            self.use_external_processors = True
            print(
                "[act_wrapper] normalization path: external pre/post processors "
                "(lerobot 0.4.4 format)",
                flush=True,
            )
        return (
            "external_0.4.4" if self.use_external_processors else "legacy_builtin"
        )

    # ------------------------------------------------------------------
    # action queue
    # ------------------------------------------------------------------
    def _queue_maxlen(self) -> int | None:
        n_action_steps = getattr(self.policy.config, "n_action_steps", None)
        try:
            n_action_steps = int(n_action_steps)
        except (TypeError, ValueError):
            n_action_steps = 0
        if n_action_steps <= 0:
            n_action_steps = self.act_step
        if self._temporal_ensemble:
            n_action_steps = 1
        return n_action_steps

    def _bind_action_queue(self) -> None:
        """Alias this wrapper's action queue to the policy's own queue.

        LeRobot's ``ACTPolicy.select_action`` refills ``self._action_queue``
        whenever it is empty and pops one action per call, which is exactly the
        semantics of the original adapter.  Aliasing the deque (instead of
        keeping a second buffer) guarantees the two never drift apart, and lets
        ``reset()`` clear the real queue.  If the installed LeRobot no longer
        exposes ``_action_queue``, an empty deque is installed on the policy so
        the wrapper still reports "no cached action" correctly.
        """
        existing = getattr(self.policy, "_action_queue", None)
        if existing is None:
            existing = collections.deque(maxlen=self._queue_maxlen())
            try:
                self.policy._action_queue = existing
            except Exception:  # pragma: no cover - read-only policy objects
                pass
        self._action_queue = existing

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        """Action dimension declared by the checkpoint configuration."""
        return int(self._action_dim)

    @property
    def normalization_path(self) -> str:
        """'legacy_builtin' or 'external_0.4.4' -- which path was loaded."""
        return self._normalization_path

    # ------------------------------------------------------------------
    # observation batch
    # ------------------------------------------------------------------
    def _build_policy_batch(self, obs) -> dict:
        """Build the LeRobot batch dict from a simulator observation."""
        state = obs
        for key in str(self.state_obs_path).split("/"):
            if key:
                state = state[key]
        state_tensor = (
            state.detach().to(device=self.act_device, dtype=torch.float32)
            if isinstance(state, torch.Tensor)
            else torch.as_tensor(state, dtype=torch.float32, device=self.act_device)
        )
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        batch = {"observation.state": state_tensor}
        for image_key in self.act_image_keys:
            camera_name = self.image_key_map.get(
                image_key,
                str(image_key).removeprefix("observation.images."),
            )
            image = obs["sensor"][camera_name]["color"]
            image_tensor = (
                image.detach().to(device=self.act_device, dtype=torch.float32)
                if isinstance(image, torch.Tensor)
                else torch.as_tensor(
                    image, dtype=torch.float32, device=self.act_device
                )
            )
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor[..., :3].permute(0, 3, 1, 2).contiguous()
            if torch.max(image_tensor) > 1.5:
                image_tensor = image_tensor / 255.0
            batch[image_key] = image_tensor
        return batch

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _infer_raw_action(self, obs):
        """Run one forward pass (and refill the queue) for a fresh chunk."""
        batch = self._build_policy_batch(obs)
        if self.use_external_processors:
            batch = self.policy.act_preprocessor(batch)
        return self.policy.select_action(batch)

    def _postprocess(self, action):
        if self.use_external_processors:
            return self.policy.act_postprocessor(action)
        return action

    def _validate_action(self, action) -> np.ndarray:
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action)
        if not torch.isfinite(action).all():
            raise ValueError(
                "Non-finite ACT action after postprocessor; refusing to step the env: "
                f"{action.detach().cpu().numpy()}"
            )
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim != 2:
            raise ValueError(
                f"Expected policy action shape [B, D], but got {tuple(action.shape)}."
            )
        array = action.detach().to(device="cpu", dtype=torch.float32).numpy()
        return np.ascontiguousarray(array)

    def select_chunk_action(self, obs) -> tuple[np.ndarray, bool]:
        """Return ``(action, ran_model_inference)`` for a single env step.

        The returned action is a ``(1, action_dim)`` float32 array, i.e. one
        action ready to be passed to ``env.step``.  ``ran_model_inference`` is
        ``True`` exactly when a model forward pass happened during this call, so
        the caller can attribute the recorded inference time correctly (cached
        queue pops are free).
        """
        runs_model_inference = (
            self._temporal_ensemble or len(self._action_queue) == 0
        )

        if runs_model_inference:
            start_inference, finish_inference = _timing_hooks()
            started_at = start_inference(self.act_device)
            raw_action = self._infer_raw_action(obs)
            action = self._postprocess(raw_action)
            finish_inference(started_at, self.inference_times_s, self.act_device)
        else:
            raw_action = self._action_queue.popleft()
            action = self._postprocess(raw_action)

        return self._validate_action(action), runs_model_inference

    def drain_inference_times(self) -> list[float]:
        """Return and clear the inference samples recorded so far."""
        samples = list(self.inference_times_s)
        self.inference_times_s.clear()
        return samples

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the action queue and any policy-level per-episode state."""
        try:
            self.policy.reset()
        except Exception as exc:  # pragma: no cover - policy specific
            print(
                "[act_wrapper] policy.reset() failed "
                f"({type(exc).__name__}: {exc}); clearing the action queue directly.",
                flush=True,
            )
        # `policy.reset()` may replace the deque, so re-alias before clearing.
        self._bind_action_queue()
        self._action_queue.clear()
        self.inference_times_s.clear()

    def close(self) -> None:
        """Best-effort release of cached tensors."""
        self._action_queue.clear()
        self.inference_times_s.clear()

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"ACTPolicyWrapper(checkpoint={self.checkpoint_path!r}, "
            f"device={self.act_device}, act_step={self.act_step}, "
            f"action_dim={self.action_dim}, "
            f"normalization={self._normalization_path})"
        )
