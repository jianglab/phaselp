"""Coherence envelopes and the expected-FSC relations (manuscript Section 2).

Notation follows the manuscript: ``s`` is the frequency magnitude (1/angstrom
for maps, cycles per sample for signals), ``R`` a resolution in the
reciprocal unit, ``kappa`` the dimensionless SSNR scale, ``c`` the coherence
E[cos delta] of equation (3) and ``a`` an envelope, the correlation of a map
with the ideal, noiseless map.
"""

from __future__ import annotations

import numpy as np

#: SSNR scale kappa used throughout the manuscript (Section 5.2).
KAPPA = 100.0

#: Coefficients whose coherence falls below this value are set to zero
#: (Section 5.2), which gives the filter a finite stopband.
COHERENCE_FLOOR = 1e-6


def _check_resolution(resolution: float, name: str) -> float:
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return resolution


def _check_kappa(kappa: float) -> float:
    kappa = float(kappa)
    if not np.isfinite(kappa) or kappa <= 1.0 / 3.0:
        raise ValueError("kappa must be finite and greater than 1/3")
    return kappa


def ssnr(s, resolution: float, kappa: float = KAPPA):
    """Gaussian SSNR model of equation (5): ``kappa * exp(-B_R s^2 / 4)``.

    ``B_R = 4 R^2 ln(3 kappa)`` places SSNR = 1/3, and so ``c = 0.5``, at
    ``s = 1/R``.
    """
    resolution = _check_resolution(resolution, "resolution")
    kappa = _check_kappa(kappa)
    b_factor = 4.0 * resolution**2 * np.log(3.0 * kappa)
    return kappa * np.exp(-b_factor * np.asarray(s, dtype=float) ** 2 / 4.0)


def envelope(s, resolution: float, kappa: float = KAPPA):
    """Envelope ``a_R(s) = sqrt(SSNR / (1 + SSNR))`` of equation (5).

    The correlation that a map at resolution ``R`` shares with the ideal
    map; it equals 0.5 at ``s = 1/R``.
    """
    u = ssnr(s, resolution, kappa)
    return np.sqrt(u / (1.0 + u))


def coherence(s, target_res: float, parent_res: float | None = None, kappa: float = KAPPA):
    """Coherence profile ``c(s)`` of the filter.

    Without ``parent_res`` the parent is treated as ideal and ``c`` is the
    target envelope of equation (5). With ``parent_res`` the filter supplies
    only the remaining loss of correlation, equation (7):
    ``c = min(1, a_T / a_H)``. Frequencies where ``a_H`` is zero get ``c = 0``.
    """
    target = envelope(s, target_res, kappa)
    if parent_res is None or parent_res == 0:
        return target
    parent_res = _check_resolution(parent_res, "parent_res")
    if parent_res >= target_res:
        raise ValueError(
            f"parent_res ({parent_res}) must be finer (smaller) than target_res ({target_res})"
        )
    parent = envelope(s, parent_res, kappa)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(parent > 0.0, target / np.where(parent > 0.0, parent, 1.0), 0.0)
    return np.minimum(1.0, ratio)


def sigma_from_coherence(c):
    """Width of the normal offsets, ``sigma = sqrt(-2 ln c)`` (equation 6).

    ``c`` is clipped to ``[COHERENCE_FLOOR, 1]``; coefficients below the
    floor are removed by the filter, so their width is irrelevant.
    """
    c = np.clip(np.asarray(c, dtype=float), COHERENCE_FLOOR, 1.0)
    return np.sqrt(np.abs(-2.0 * np.log(c)))  # abs turns -0.0 at c = 1 into 0.0


def coherence_from_sigma(sigma):
    """Coherence of wrapped-normal offsets, ``c = exp(-sigma^2 / 2)`` (equation S1)."""
    return np.exp(-0.5 * np.asarray(sigma, dtype=float) ** 2)


# --------------------------------------------------------------------------
# Expected FSCs (equations 4 and 8, Supplementary Sections S1 and S2)
# --------------------------------------------------------------------------


def expected_fsc_independent(a_x, a_y):
    """Expected FSC between two independent maps, ``a_X * a_Y`` (equation 8)."""
    return np.asarray(a_x) * np.asarray(a_y)


def full_map_envelope(rho_half):
    """Envelope of the full map from the half-map FSC: ``sqrt(2 rho / (1 + rho))``.

    A half-map FSC of 0.143 (1/7) gives 0.5.
    """
    rho = np.asarray(rho_half, dtype=float)
    return np.sqrt(2.0 * rho / (1.0 + rho))


def halfmap_envelope(s, resolution: float, kappa: float = KAPPA):
    """Envelope of one half map whose average follows equation (5).

    Each half map has half the SSNR ``u`` of the full map, giving
    ``sqrt(u / (2 + u))`` (Supplementary Section S1); its square, the
    expected half-map FSC, is 1/7 at ``s = 1/R``.
    """
    u = ssnr(s, resolution, kappa)
    return np.sqrt(u / (2.0 + u))


def expected_fsc_mean_of_realizations(c, n: int):
    """Expected FSC with the parent of the mean of ``n`` filtered maps.

    ``c / sqrt(c^2 + (1 - c^2) / n)`` rises from ``c`` (one map) toward 1:
    averaging realizations removes the phase randomization (Section S1).
    """
    if int(n) < 1:
        raise ValueError("n must be at least 1")
    c = np.asarray(c, dtype=float)
    return c / np.sqrt(c**2 + (1.0 - c**2) / int(n))


def expected_fsc_filtered_vs_other(c, fsc_parent_other):
    """Expected FSC between a filtered parent and another map L: ``c * FSC(H, L)``.

    Supplementary Section S2; it also holds after radial amplitude matching.
    """
    return np.asarray(c) * np.asarray(fsc_parent_other)
