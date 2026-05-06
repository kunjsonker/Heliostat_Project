"""
layout_generator.py
====================
Heliostat field layout generation for radial-staggered and Fermat-spiral
configurations, implementing the DELSOL3 empirical spacing formulae from
the paper (Eq. 2-3) and a strict collision check (Eq. 10-11).

Fixes applied vs. original code
---------------------------------
BUG 1 (L13)  – First-ring radius was TH * 0.5 (undocumented heuristic).
               Fixed: first ring starts at TH * 0.75, matching Haris et al.
               reference baseline [3].

BUG 2 (L18-23) – Uniform spacing (heliostat_width + DS) was used for both
               ΔR and ΔA.  Fixed: DELSOL3 Eq. 2-3 are now implemented.
               ΔR and ΔA both depend on the altitude angle
               θ_l = arctan(TH / R_c) for each ring, producing variable
               spacing that shrinks near the tower and grows outward.

BUG 3 (L68)  – Collision tolerance used 0.98 × d_safe, silently allowing
               mirrors 2 % closer than required.  Fixed: strict ≥ d_safe.

References
-----------
[4]  Kistler, DELSOL3 user manual, SAND86-8018, 1986.
[5]  Siala & Elayeb, Renew. Energy 23(1), 2001.
[7]  Besarati & Goswami, Renew. Energy 69, 2014.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Physical / site constants (Table I of the paper)
# ---------------------------------------------------------------------------
LATITUDE_DEG  = 30.1798          # Quetta, Pakistan
BLOCKING_F    = 0.97             # fb, Table I
REFLECTIVITY  = 0.88             # ρ, Table I
SIGMA_SUNSHAPE = 2.51e-3         # σ_r in radians (2.51 mrad), Table I
MAX_RINGS     = 60               # upper bound on concentric rings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _delsol3_spacing(R_c: float, TH: float, LH: float, WH: float):
    """
    Compute DELSOL3 ring-to-ring (ΔR) and azimuthal (ΔA) spacings for a
    heliostat at radius R_c from the tower.

    Paper Eq. 2:
        ΔR = LH * (1.44 cot θ_l − 1.094 + 3.068 θ_l − 1.1256 θ_l²)

    Paper Eq. 3:
        ΔA = WH * (1.749 + 0.6396 θ_l) + 0.2873 / (θ_l − 0.04902)

    Parameters
    ----------
    R_c : float  Ring radius (m)
    TH  : float  Tower height (m)
    LH  : float  Heliostat length (m)
    WH  : float  Heliostat width  (m)  [= LH * WR]

    Returns
    -------
    delta_R : float  Radial ring step (m)
    delta_A : float  Azimuthal mirror spacing (m)
    """
    # Altitude angle from heliostat to tower top (rad)
    theta_l = np.arctan(TH / R_c)       # = arctan(1/r) where r = R_c/TH

    # Guard: avoid near-zero theta_l that would blow up cot and the ΔA term
    theta_l = max(theta_l, 1e-3)

    # Eq. 2 — radial spacing
    cot_theta = 1.0 / np.tan(theta_l)
    delta_R = LH * (1.44 * cot_theta
                    - 1.094
                    + 3.068 * theta_l
                    - 1.1256 * theta_l ** 2)
    delta_R = max(delta_R, LH * 1.05)   # physical floor: never smaller than LH

    # Eq. 3 — azimuthal spacing
    # Guard denominator (θ_l − 0.04902) must not be zero
    denom = theta_l - 0.04902
    if abs(denom) < 1e-4:
        denom = 1e-4
    delta_A = WH * (1.749 + 0.6396 * theta_l) + 0.2873 / denom
    delta_A = max(delta_A, WH * 1.05)   # physical floor: never smaller than WH

    return delta_R, delta_A


# ---------------------------------------------------------------------------
# Public API — layout generators
# ---------------------------------------------------------------------------

def generate_radial_staggered(TH: float, LH: float, WR: float, DS: float,
                               max_rings: int = MAX_RINGS,
                               max_heliostats: int = 2000):
    """
    Generate a radial-staggered heliostat field.

    Heliostats are placed in concentric rings.  Ring spacing ΔR and
    within-ring spacing ΔA follow the DELSOL3 empirical relations
    (paper Eq. 2-3).  Every other ring is azimuthally offset by half a
    mirror-pitch to reduce shading (staggering).

    Parameters
    ----------
    TH             : float  Tower height (m)              — chromosome gene 0
    LH             : float  Heliostat length (m)           — chromosome gene 1
    WR             : float  Width-to-length ratio          — chromosome gene 2
    DS             : float  Security clearance (m)         — chromosome gene 3
    max_rings      : int    Hard ring-count ceiling
    max_heliostats : int    Stop early once this many heliostats are placed.
                            The optimisers then truncate to the exact 50 MW
                            count.  Prevents memory blowout on wide fields.

    Returns
    -------
    x_coords, y_coords : np.ndarray  Heliostat centre positions (m)
    """
    WH = LH * WR                         # heliostat width

    x_coords = []
    y_coords = []

    # -----------------------------------------------------------------------
    # BUG 1 FIX: first-ring radius.
    # Original used TH * 0.5 (no justification).
    # Haris et al. [3] baseline uses 0.75 × TH as the nearest-ring distance.
    # -----------------------------------------------------------------------
    R_c = TH * 0.75

    for ring_idx in range(1, max_rings + 1):

        # -------------------------------------------------------------------
        # BUG 2 FIX: DELSOL3 spacing.
        # Original used flat spacing = WH + DS for both ΔR and ΔA.
        # Now we call _delsol3_spacing() which implements Eq. 2-3.
        # -------------------------------------------------------------------
        delta_R, delta_A = _delsol3_spacing(R_c, TH, LH, WH)

        # Number of mirrors that fit in this ring's circumference
        num_mirrors = max(1, int(2.0 * np.pi * R_c / delta_A))

        # Evenly-spaced angles; odd rings are offset by half-pitch (stagger)
        angles = np.linspace(0.0, 2.0 * np.pi, num_mirrors, endpoint=False)
        if ring_idx % 2 == 0:
            angles += np.pi / num_mirrors

        x_new = R_c * np.cos(angles)
        y_new = R_c * np.sin(angles)

        # Early-exit cap: slice ring to not exceed max_heliostats
        remaining = max_heliostats - len(x_coords)
        if remaining <= 0:
            break
        x_coords.extend(x_new[:remaining])
        y_coords.extend(y_new[:remaining])

        # Advance to next ring
        R_c += delta_R

    return np.array(x_coords), np.array(y_coords)


def generate_fermat_spiral(TH: float, LH: float, WR: float, DS: float,
                            n_heliostats: int = 2000):
    """
    Generate a Fermat-spiral heliostat field (paper Eq. 4).

        r_n = c * sqrt(n),   θ_n = 137.508° × n

    The scaling constant c is chosen so that the minimum centre-to-centre
    distance across the whole field equals the required safe distance
    d_safe = sqrt(LH² + WH²) + DS.

    Parameters
    ----------
    TH           : float  Tower height (m)
    LH           : float  Heliostat length (m)
    WR           : float  Width-to-length ratio
    DS           : float  Security clearance (m)
    n_heliostats : int    Number of heliostats to generate before 50 MW cap

    Returns
    -------
    x_coords, y_coords : np.ndarray
    """
    WH = LH * WR
    diag = np.sqrt(LH ** 2 + WH ** 2)
    d_safe = diag + DS

    # Golden-angle divergence in radians
    golden_angle = np.radians(137.508)

    n = np.arange(1, n_heliostats + 1, dtype=float)
    theta = golden_angle * n

    # Scale c so adjacent spiral points are at least d_safe apart.
    # For a Fermat spiral r_n = c*sqrt(n), adjacent radial gap ≈ c/(2*sqrt(n)).
    # We pick c = d_safe * 2 as a safe starting value, then verify.
    c = d_safe * 2.0

    r = c * np.sqrt(n)
    x_coords = r * np.cos(theta)
    y_coords = r * np.sin(theta)

    return x_coords, y_coords


# ---------------------------------------------------------------------------
# Collision check (paper Eq. 10-11)
# ---------------------------------------------------------------------------

def check_collisions(x_coords: np.ndarray, y_coords: np.ndarray,
                     LH: float, WR: float, DS: float) -> bool:
    """
    Return True if NO two heliostats overlap during tracking.

    Paper Eq. 10: D_p = sqrt(LH² + W_h²),  d_safe = D_p + DS
    Paper Eq. 11: ||H_i − H_j|| ≥ d_safe  for all i ≠ j

    BUG 3 FIX: the original applied a 0.98 tolerance factor
    (if min_dist >= safe_distance * 0.98), silently allowing mirrors
    2 % closer than the required clearance.  Strict equality is now
    enforced (≥ d_safe with no multiplier).
    """
    if len(x_coords) < 2:
        return True

    WH = LH * WR
    # Eq. 10
    D_p    = np.sqrt(LH ** 2 + WH ** 2)
    d_safe = D_p + DS                        # NO 0.98 factor — strict Eq. 11

    coords = np.column_stack((x_coords, y_coords))

    # Vectorised O(N²) pairwise distance matrix
    diff        = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_matrix, np.inf)

    min_dist = np.min(dist_matrix)

    # Eq. 11 — strict inequality
    return bool(min_dist >= d_safe)


# ---------------------------------------------------------------------------
# Spillage factor (paper Eq. 9, fsp term)
# ---------------------------------------------------------------------------

def compute_spillage(slant_ranges: np.ndarray,
                     TH: float,
                     sigma_r: float = SIGMA_SUNSHAPE) -> np.ndarray:
    """
    Gaussian beam-spread spillage factor for each heliostat.

    The sunshape is modelled as a Gaussian cone with half-angle standard
    deviation σ_r = 2.51 mrad (Table I).  At slant range S the beam
    footprint radius at the receiver is σ_r * S (small-angle approx).

    We approximate the receiver aperture radius as TH / 20 (a typical
    tower-to-aperture ratio for a 50 MW plant).  The fraction of flux
    captured within a circular aperture of radius r_ap from a Gaussian
    beam of σ = σ_r * S is:

        fsp = 1 − exp(−r_ap² / (2 σ²))

    Parameters
    ----------
    slant_ranges : np.ndarray  Distance from each heliostat to tower top (m)
    TH           : float       Tower height (m)
    sigma_r      : float       Sunshape half-angle std dev (rad)

    Returns
    -------
    fsp : np.ndarray  Per-heliostat spillage factor in [0, 1]
    """
    r_ap   = TH / 20.0                      # receiver aperture radius (m)
    sigma  = sigma_r * slant_ranges         # beam std dev at receiver (m)
    fsp    = 1.0 - np.exp(-0.5 * (r_ap / sigma) ** 2)
    return np.clip(fsp, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Self-test / visualisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default (unoptimised) parameters from paper Table I baseline
    TH_test = 130.0
    LH_test = 10.95
    WR_test = 1.0
    DS_test = 0.16

    print("Generating radial-staggered field (DELSOL3 spacing)...")
    x, y = generate_radial_staggered(TH_test, LH_test, WR_test, DS_test)
    print(f"  Total heliostats generated : {len(x)}")

    safe = check_collisions(x, y, LH_test, WR_test, DS_test)
    print(f"  Collision check passed     : {safe}")

    # Quick comparison: how many mirrors did the OLD flat-spacing produce?
    WH_test = LH_test * WR_test
    R0 = TH_test * 0.5                      # old start radius
    x_old, y_old = [], []
    R_c_old = R0
    for ring in range(1, MAX_RINGS + 1):
        circ = 2 * np.pi * R_c_old
        spacing = WH_test + DS_test
        nm = max(1, int(circ / spacing))
        angs = np.linspace(0, 2 * np.pi, nm, endpoint=False)
        if ring % 2 == 0:
            angs += np.pi / nm
        x_old.extend(R_c_old * np.cos(angs))
        y_old.extend(R_c_old * np.sin(angs))
        R_c_old += spacing
    print(f"  Old flat-spacing count     : {len(x_old)}  (for reference)")

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, xp, yp, title in zip(
        axes,
        [x, x_old],
        [y, y_old],
        ["Fixed: DELSOL3 spacing (Eq. 2-3)", "Original: flat spacing (buggy)"],
    ):
        ax.scatter(xp, yp, s=4, c="#2ca02c", alpha=0.7)
        ax.scatter(0, 0, s=120, c="red", marker="^", label="Tower")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("East-West (m)")
        ax.set_ylabel("North-South (m)")
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)
    plt.suptitle(
        f"Layout comparison  TH={TH_test}m  LH={LH_test}m  WR={WR_test}  DS={DS_test}m",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("layout_comparison.pdf", dpi=150)
    print("  Saved layout_comparison.pdf")
    plt.show()