"""RoboSynChallenge hybrid submission package.

Only the environment-driven settings live at the package root so that simply
importing :mod:`robosyn_hybrid` never pulls in heavyweight optional
dependencies (torch, lerobot, FoundationStereo, SAM2, ...).  Every heavier
module (``act_wrapper``, ``stereo``, ``vlm``, ``flow``, ...) is imported
explicitly by whoever needs it, which keeps the policy adapter free to degrade
to a pure ACT path when an optional component is unavailable.

Public surface required by the submission spec::

    from robosyn_hybrid import get_settings, asset
    robosyn_hybrid.__version__  # "1.0.0"
"""

from __future__ import annotations

from .settings import Settings, asset, get_settings

__version__ = "1.0.0"

__all__ = ["Settings", "asset", "get_settings", "__version__"]
