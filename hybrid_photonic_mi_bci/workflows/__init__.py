"""Workflow helpers for the FBCSP-based design lines.

Heavy optional dependencies such as torch are imported by the concrete workflow
modules, not by this package initializer.
"""

from .common import FBCSPDesignConfig, FBCSPPreparedData, prepare_fbcsp_data


def __getattr__(name: str):
    if name == "run_fbcsp_reference":
        from .fbcsp_reference import run_fbcsp_reference

        return run_fbcsp_reference
    if name == "run_small_network_line":
        from .small_network_line import run_small_network_line

        return run_small_network_line
    if name == "run_experience_photonic_line":
        from .experience_photonic_line import run_experience_photonic_line

        return run_experience_photonic_line
    if name == "run_full_design_comparison":
        from .full_design import run_full_design_comparison

        return run_full_design_comparison
    raise AttributeError(name)


__all__ = [
    "FBCSPDesignConfig",
    "FBCSPPreparedData",
    "prepare_fbcsp_data",
    "run_experience_photonic_line",
    "run_fbcsp_reference",
    "run_full_design_comparison",
    "run_small_network_line",
]
