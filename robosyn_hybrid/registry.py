"""Task registry: which of the ten tasks needs which ACT checkpoint.

The RoboSynChallenge preliminary is evaluated on all ten tasks with a single
submission.  Seven of them are served by the released baseline ACT checkpoints
published under the ``RoboSynChallenge`` Hugging Face organisation; the other
three are served by this repository's hybrid flows:

* ``handle_basket`` (depth + VLM + SAM2 + grasp + ACT takeover) uses the
  migrated ACT weights hosted at ``dimofu/ACT_sim_handle_basket``; an explicit
  local ``checkpoint_path`` (``HYBRID_CKPT``) still overrides it;
* ``click_bell`` (motion-stereo depth + VLM button localisation + FK/IK press +
  ACT fill) keeps the **official** released checkpoint
  ``RoboSynChallenge/ACT_sim_click_bell``: the hybrid flow takes over the press
  and hands the remaining steps to that same ACT policy, so the two paths share
  one set of weights;
* ``items_handover`` (right-arm grasp primitives + ACT continuation + left-arm
  take-over + VLM/ellipse holder localisation + place) likewise keeps the
  official released checkpoint ``RoboSynChallenge/ACT_sim_items_handover`` for
  its ACT phases and for the pure-ACT fallback.

``HYBRID_TASK_NAMES`` is therefore the authoritative list of tasks that go
through a hybrid flow, and it is deliberately independent of "has no Hub
checkpoint".

Nothing in this module downloads anything: it only answers "which repository
id / which local path belongs to this task".

Hard constraint: no absolute path may appear in this file.  Every location is
either an upstream Hub repository id or a path supplied by the caller.
"""

from __future__ import annotations

__all__ = [
    "TASK_CHECKPOINTS",
    "TASK_REVISIONS",
    "TASK_NAMES",
    "HYBRID_TASK_NAMES",
    "HYBRID_TASKS",
    "checkpoint_for",
    "checkpoint_url_for",
    "revision_for",
    "is_hybrid_task",
    "describe_registry",
]

#: ``task_name -> Hub repository id``.  Every task resolves to a Hub
#: repository; an explicit ``--checkpoint_path`` / ``HYBRID_CKPT`` still wins
#: over the registered entry.  The defensive ``None`` branch in
#: :func:`checkpoint_for` is kept in case a task is ever re-pointed at
#: in-repo weights.
TASK_CHECKPOINTS: dict[str, str | None] = {
    "handle_basket": "dimofu/ACT_sim_handle_basket",
    "click_bell": "RoboSynChallenge/ACT_sim_click_bell",
    "water_pouring": "RoboSynChallenge/ACT_sim_water_pouring",
    "table_rearrangement": "RoboSynChallenge/ACT_sim_table_rearrangement",
    "drawer_open_place": "RoboSynChallenge/ACT_sim_drawer_open_place",
    "mixer_operating": "RoboSynChallenge/ACT_sim_mixer_operating",
    "items_handover": "RoboSynChallenge/ACT_sim_items_handover",
    "item_assembly": "RoboSynChallenge/ACT_sim_item_assembly",
    "manipulate_pipette": "RoboSynChallenge/ACT_sim_manipulate_pipette",
    "sample_loading": "RoboSynChallenge/ACT_sim_sample_loading",
}

#: Canonical ordering of the ten evaluated tasks (same order as the spec table).
TASK_NAMES: tuple[str, ...] = tuple(TASK_CHECKPOINTS)

#: ``task_name -> pinned Hugging Face revision`` for the released checkpoints.
#:
#: These are the commit shas that were current when this submission was
#: prepared.  Five of them are the exact revisions the organizers published in
#: ``evaluation_results/released_checkpoint_results.json`` for click_bell,
#: water_pouring, table_rearrangement, drawer_open_place and mixer_operating;
#: the remaining four are the head of each repository at the same time.
#:
#: Pinning matters: an unpinned ``snapshot_download`` silently follows a moving
#: branch, which would make a re-evaluation non-reproducible.
TASK_REVISIONS: dict[str, str] = {
    "click_bell": "677e65fbb15974024ff840893496197ef7db26d4",
    "water_pouring": "0bf0fcfc931a69c52871385f28068fcf873cf07a",
    "table_rearrangement": "3a46b36fade1772176b791dd87349f479d0a98c8",
    "drawer_open_place": "592e30434aad83cf7bd8f1ee105d7c401488743d",
    "mixer_operating": "0f12c53a2a6e093ae5e1e28f20480296b45fdf2b",
    "items_handover": "86f6ca7f5decd17a92c3218f295de53497383d14",
    "item_assembly": "58955958519157176712a1eff319018d78177ddb",
    "manipulate_pipette": "ca013b2e202d594bf347530c61c3ba28c06d1bc2",
    "sample_loading": "917bf2796f18abd0a16f723573382161f1a5c5c8",
}

#: Tasks served by an in-repo hybrid flow (analytic stage + ACT hand-over).
#: ``handle_basket`` has no Hub checkpoint of its own; ``click_bell`` reuses the
#: official released checkpoint for its ACT fill phase, which is why this set is
#: declared explicitly instead of being derived from ``TASK_CHECKPOINTS``.
#: ``items_handover`` **is** a hybrid task again.  It was briefly switched to
#: plain ACT after 11 real runs scored 0/11, but that run of failures was
#: traced to a bug in the grasp primitive: ``_step_right_q`` defaulted its
#: gripper argument to ``GRIP_RIGHT_CLOSED``, and the ``VERTICAL_ALIGN`` /
#: ``PRESS`` call sites never overrode it -- so the right gripper was commanded
#: shut from the first primitive step and never opened to straddle the pen
#: (``LIFT rise_m`` was only ~4 mm).  With that fixed (gripper open during the
#: approach, closed only at ``CLOSE``) a real GPU episode reports
#: ``success rate: 1/1`` on seed 39 -- see ``docs/COMPLIANCE.md`` §3.5.
HYBRID_TASK_NAMES: set[str] = {"handle_basket", "click_bell", "items_handover"}

#: Canonical, ordered view of :data:`HYBRID_TASK_NAMES` (same order as the ten
#: task table), kept for logging and documentation.
HYBRID_TASKS: tuple[str, ...] = tuple(
    name for name in TASK_NAMES if name in HYBRID_TASK_NAMES
)

_MISSING_CHECKPOINT_HINT = (
    "Pass --checkpoint_path <local checkpoint directory> "
    "(policy/hybrid/eval.sh reads the HYBRID_CKPT environment variable), or "
    "pre-download the repository from the Hugging Face Hub."
)


def checkpoint_for(task_name: str, override: str | None = None) -> str:
    """Return the checkpoint to load for ``task_name``.

    ``override`` (typically ``usr_args["checkpoint_path"]``) always wins when it
    is a non-empty string.  Otherwise the registry entry is used; every task
    resolves to a Hub repository id -- ``handle_basket`` to its migrated
    ``dimofu/ACT_sim_handle_basket`` weights, and ``click_bell`` /
    ``items_handover`` to their official released repository ids, which their
    hybrid flows' ACT phases (and pure-ACT fallbacks) reuse.  A ``ValueError``
    is only raised for unknown tasks or a defensive ``None`` entry.
    """
    if override is not None and str(override).strip():
        return str(override)

    if task_name not in TASK_CHECKPOINTS:
        known = ", ".join(TASK_NAMES)
        raise ValueError(
            f"Unknown task '{task_name}'. Known tasks: {known}."
        )

    repo_id = TASK_CHECKPOINTS[task_name]
    if repo_id is None:
        raise ValueError(
            f"Task '{task_name}' has no released Hub checkpoint in this registry: "
            "it is served by the in-repo hybrid flow together with its own ACT "
            "weights, so an explicit checkpoint path is required. "
            f"{_MISSING_CHECKPOINT_HINT}"
        )
    return repo_id


def is_hybrid_task(task_name: str) -> bool:
    """True when ``task_name`` is handled by an in-repo hybrid flow.

    Uses :data:`HYBRID_TASK_NAMES`, not "the task has no Hub checkpoint":
    ``click_bell`` and ``items_handover`` are hybrid tasks *and* have official
    released checkpoints, because the hybrid flows' ACT phases run those very
    checkpoints.
    """
    return str(task_name) in HYBRID_TASK_NAMES


def revision_for(task_name: str) -> str | None:
    """Pinned Hub revision for ``task_name``, or ``None`` when unpinned.

    Returns ``None`` for ``handle_basket`` (its migrated repository is not
    revision-pinned; it is served from ``main``) and for any task without a
    recorded revision, so callers can fall back to ``main`` with an explicit
    warning instead of silently tracking a moving branch.
    ``click_bell`` and ``items_handover`` are hybrid tasks *and* have pinned
    revisions, because their hybrid flows load the released checkpoints.
    """
    return TASK_REVISIONS.get(task_name)


def describe_registry() -> dict[str, str]:
    """Human-readable ``task_name -> source`` map for logs and documentation."""
    out: dict[str, str] = {}
    for name, repo_id in TASK_CHECKPOINTS.items():
        if repo_id is None:
            out[name] = "<in-repo hybrid ACT weights>"
        elif name in HYBRID_TASK_NAMES:
            out[name] = f"{repo_id} (released checkpoint; hybrid flow + ACT fill)"
        else:
            out[name] = repo_id
    return out


def checkpoint_url_for(task_name: str) -> str | None:
    """Browser / submission-form URL for ``task_name``'s released checkpoint.

    The registry stores a ``repo_id`` because that is what
    ``huggingface_hub.snapshot_download`` expects, whereas the
    RoboSynChallenge submission backend requires an
    ``https://huggingface.co/...`` link.  The two must not be confused, so the
    conversion lives here instead of being re-derived by callers.
    Every registered task resolves to a URL.
    """
    repo_id = TASK_CHECKPOINTS.get(task_name)
    if repo_id is None:
        return None
    return f"https://huggingface.co/{repo_id}"
