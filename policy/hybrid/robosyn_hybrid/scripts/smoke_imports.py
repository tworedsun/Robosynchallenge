#!/usr/bin/env python
"""Pre-flight self-check for the hybrid submission.

Runs without a GPU and without the simulator, so it can be executed immediately
after installing the package and again on the evaluation machine before a long
run.  It answers four questions:

1. can the adapter contract be imported at all
   (``policy.hybrid`` exports ``get_model`` / ``eval`` / ``reset_model``)?
2. do the public constants resolve into sane matrices?
3. is every one of the ten official tasks mapped to a well-formed checkpoint?
4. which third-party assets are present, and where?

Exit code 0 means "no blocker found"; 1 means at least one check failed.

USAGE
    python policy/hybrid/robosyn_hybrid/scripts/smoke_imports.py
    python policy/hybrid/robosyn_hybrid/scripts/smoke_imports.py --strict   # missing assets fail
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# <repo>/policy/hybrid/robosyn_hybrid/scripts/smoke_imports.py
REPO_ROOT = Path(__file__).resolve().parents[4]
HYBRID_PARENT = Path(__file__).resolve().parents[2]   # <repo>/policy/hybrid
for _candidate in (str(REPO_ROOT), str(HYBRID_PARENT)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

FAILURES: list[str] = []
WARNINGS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def warn(label: str, detail: str = "") -> None:
    print(f"  [warn] {label}" + (f" — {detail}" if detail else ""))
    WARNINGS.append(label)


# --------------------------------------------------------------------------- #
def check_adapter() -> None:
    print("\n[1] adapter contract (official load_policy_adapter requirement)")
    try:
        from policy.hybrid import eval as adapter_eval  # noqa: A004
        from policy.hybrid import get_model, reset_model
    except Exception as exc:  # noqa: BLE001
        check("import policy.hybrid", False, f"{type(exc).__name__}: {exc}")
        return
    for name, fn in (("get_model", get_model), ("eval", adapter_eval), ("reset_model", reset_model)):
        check(f"exports {name}", callable(fn))


def check_settings() -> None:
    print("\n[2] settings resolve from the environment")
    try:
        from robosyn_hybrid import get_settings
    except Exception as exc:  # noqa: BLE001
        check("import robosyn_hybrid", False, f"{type(exc).__name__}: {exc}")
        return
    settings = get_settings()
    check("repo_root exists", settings.repo_root.is_dir(), str(settings.repo_root))
    check("assets_dir path resolved", bool(settings.assets_dir), str(settings.assets_dir))
    if settings.vlm_available:
        check("VLM endpoint configured", True, settings.vlm_url)
    else:
        warn("VLM endpoint not configured (HYBRID_VLM_URL unset) -> plain ACT fallback")


def check_priors() -> None:
    print("\n[3] public constants")
    try:
        import numpy as np

        from robosyn_hybrid import priors
    except Exception as exc:  # noqa: BLE001
        check("import robosyn_hybrid.priors", False, f"{type(exc).__name__}: {exc}")
        return

    for arm in ("left", "right"):
        mount = priors.camera_mount(arm)
        check(f"camera_mount({arm!r}) is 4x4", mount.shape == (4, 4), str(mount.shape))
        rot = mount[:3, :3]
        orthonormal = np.allclose(rot @ rot.T, np.eye(3), atol=1e-6)
        check(f"camera_mount({arm!r}) rotation orthonormal", orthonormal)
        check(f"camera_mount({arm!r}) det == +1", abs(np.linalg.det(rot) - 1.0) < 1e-6)

    for uid in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        k = priors.camera_K(uid)
        check(
            f"camera_K({uid!r}) principal point inside image",
            0 < k[0, 2] < 640 and 0 < k[1, 2] < 480,
            f"cx={k[0, 2]:.1f} cy={k[1, 2]:.1f}",
        )

    check("MILK_GRASP_OBJECT is 4x4", priors.MILK_GRASP_OBJECT.shape == (4, 4))
    check("MILK_PRIOR_SIZE has 3 entries", priors.MILK_PRIOR_SIZE.shape == (3,))


def check_registry() -> None:
    print("\n[4] ten-task checkpoint registry")
    try:
        from robosyn_hybrid import registry
    except Exception as exc:  # noqa: BLE001
        check("import robosyn_hybrid.registry", False, f"{type(exc).__name__}: {exc}")
        return

    expected = {
        "click_bell",
        "handle_basket",
        "water_pouring",
        "table_rearrangement",
        "items_handover",
        "drawer_open_place",
        "mixer_operating",
        "item_assembly",
        "manipulate_pipette",
        "sample_loading",
    }
    check("registry covers all ten official tasks", set(registry.TASK_NAMES) == expected,
          f"{len(registry.TASK_NAMES)} tasks")
    missing = expected - set(registry.TASK_NAMES)
    if missing:
        check("no missing task", False, f"missing {sorted(missing)}")

    for task in registry.TASK_NAMES:
        entry = registry.TASK_CHECKPOINTS.get(task)
        if entry is None:
            # Only the hybrid task is allowed to have no Hub checkpoint.
            check(
                f"{task}: in-repo hybrid weights",
                registry.is_hybrid_task(task),
                "no Hub repo by design — pass HYBRID_CKPT / --checkpoint_path",
            )
            continue

        # The registry stores a repo_id (what snapshot_download needs); the
        # submission form needs the URL form.  Check both, plus the pin.
        url = registry.checkpoint_url_for(task)
        url_ok = bool(url) and url.startswith("https://huggingface.co/")
        revision = registry.revision_for(task)
        pinned_ok = bool(revision) and len(str(revision)) == 40
        check(
            f"{task}: repo_id + HF link + pinned revision",
            url_ok and pinned_ok,
            f"{entry} | {url} | rev={revision or 'UNPINNED'}",
        )


def check_assets(strict: bool) -> None:
    print("\n[5] third-party assets")
    try:
        from robosyn_hybrid import get_settings
    except Exception as exc:  # noqa: BLE001
        check("import robosyn_hybrid", False, f"{type(exc).__name__}: {exc}")
        return
    settings = get_settings()
    report = [
        ("SAM2.1 checkpoint", settings.sam2_ckpt, settings.sam2_available),
        ("FoundationStereo checkout", settings.foundationstereo_root,
         settings.foundationstereo_root.is_dir()),
        ("FoundationStereo weights", settings.fs_ckpt, settings.fs_ckpt.is_file()),
    ]
    for label, path, present in report:
        if present:
            size = path.stat().st_size if path.is_file() else "-"
            check(label, True, f"{path} ({size})")
        elif strict:
            check(label, False, f"missing: {path}")
        else:
            warn(f"{label} missing", str(path))
    vlm_dir = settings.assets_dir / "vlm"
    if vlm_dir.is_dir() and any(vlm_dir.glob("*.gguf")):
        check("VLM weights present", True, str(vlm_dir))
    else:
        warn("VLM weights not found", str(vlm_dir))


def check_no_absolute_paths() -> None:
    print("\n[6] no hard-coded build-server paths")
    # Assembled from fragments so that this checker does not itself trip the
    # pre-publish grep in ../_publish/publish_github.sh.
    needles = (
        "/root/" + "gpufree" + "-data",
        "gpufree" + "-data",
        "/usr/local/bin/" + "co" + "dex",
    )
    hits: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".sh", ".yml", ".yaml", ".json"}:
            continue
        if "__pycache__" in path.parts or path.name in {"SPEC.md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for needle in needles:
            if needle in text:
                hits.append(f"{path.relative_to(REPO_ROOT)}: {needle}")
    check("no absolute build-server paths", not hits, "; ".join(hits[:5]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="treat missing third-party assets as failures")
    args = parser.parse_args()

    print("== RoboSynChallenge hybrid pre-flight ==")
    print(f"   repository root: {REPO_ROOT}")

    check_adapter()
    check_settings()
    check_priors()
    check_registry()
    check_assets(strict=args.strict)
    check_no_absolute_paths()

    print("\n== result ==")
    print(f"   failures: {len(FAILURES)}   warnings: {len(WARNINGS)}")
    for item in FAILURES:
        print(f"   FAIL {item}")
    if FAILURES:
        print("\nFix the failures above before starting a long evaluation run.")
        return 1
    print("\nNo blocker found. Next: docs/REPRODUCE.md step 6 (smoke episode).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
