"""phaselp: low-pass filtering with ramped phase randomization.

Implements the methods of Dang & Jiang, "Low-pass filtering revisited: ramped
phase randomization for cryo-EM map comparison and data augmentation":

* the filter of equation (2) with the coherence envelopes of equations (5)-(7),
  for signals, images and density maps (``ramped_phase_filter``), and the
  comparison filters used in the paper;
* the cryo-EM map comparison workflow: registration, filtering, radial
  amplitude matching, FSC, difference maps and residual statistics;
* ramped phase augmentation of training data (``phase_augment``,
  ``PhaseAugment``);
* simple plots of each step.
"""

from .augment import SIGMA_MAX, PhaseAugment, phase_augment, ramp_sigma
from .compare import (
    Comparison,
    compare_maps,
    difference_map,
    local_rms,
    residual_variance,
    union_mask,
    variance_ratio,
)
from .envelope import (
    COHERENCE_FLOOR,
    KAPPA,
    coherence,
    coherence_from_sigma,
    envelope,
    expected_fsc_filtered_vs_other,
    expected_fsc_independent,
    expected_fsc_mean_of_realizations,
    full_map_envelope,
    halfmap_envelope,
    sigma_from_coherence,
    ssnr,
)
from .filters import (
    FilterProfile,
    amplitude_only_filter,
    binary_phase_filter,
    filter_profile,
    hard_cutoff_filter,
    ramped_phase_filter,
)
from .fsc import fsc, match_radial_amplitude, radial_amplitude
from .maps import Map, read_map, resample_to_grid, standardize, write_map
from .registration import Registration, apply_similarity, register

__version__ = "1.1.1"

__all__ = [
    "COHERENCE_FLOOR",
    "KAPPA",
    "SIGMA_MAX",
    "Comparison",
    "FilterProfile",
    "Map",
    "PhaseAugment",
    "Registration",
    "amplitude_only_filter",
    "apply_similarity",
    "binary_phase_filter",
    "coherence",
    "coherence_from_sigma",
    "compare_maps",
    "difference_map",
    "envelope",
    "expected_fsc_filtered_vs_other",
    "expected_fsc_independent",
    "expected_fsc_mean_of_realizations",
    "filter_profile",
    "fsc",
    "full_map_envelope",
    "halfmap_envelope",
    "hard_cutoff_filter",
    "local_rms",
    "match_radial_amplitude",
    "phase_augment",
    "radial_amplitude",
    "ramp_sigma",
    "ramped_phase_filter",
    "read_map",
    "register",
    "resample_to_grid",
    "residual_variance",
    "sigma_from_coherence",
    "ssnr",
    "standardize",
    "union_mask",
    "variance_ratio",
    "write_map",
]
