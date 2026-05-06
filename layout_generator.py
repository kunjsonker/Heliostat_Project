import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants (Paper Table I)
# ---------------------------------------------------------------------------
LATITUDE_DEG   = 30.1798    # Quetta, Pakistan
BLOCKING_F     = 0.97       # fb, Table I
REFLECTIVITY   = 0.88       # ρ, Table I
SIGMA_SUNSHAPE = 2.51e-3    # σ_r = 2.51 mrad, Table I
MAX_RINGS      = 60         # upper bound on concentric rings


# ---------------------------------------------------------------------------
# DELSOL3 spacing helper (Paper Eq. 2 and 3)
# ---------------------------------------------------------------------------

def _delsol3_spacing(R_c, TH, LH, WH):
    """
    Compute DELSOL3 ring-to-ring (ΔR) and azimuthal (ΔA) spacings.

    Paper Eq. 2:
        ΔR = LH * (1.44 cot θ_l − 1.094 + 3.068 θ_l − 1.1256 θ_l²)

    Paper Eq. 3:
        ΔA = WH * (1.749 + 0.6396 θ_l) + 0.2873 / (θ_l − 0.04902)

    Parameters
    ----------
    R_c : float  Ring radius (m)
    TH  : float  Tower height (m)
    LH  : float  Heliostat length (m)
    WH  : float  Heliostat width (m) = LH * WR
    """
    # Altitude angle θ_l = arctan(TH / R_c)
    theta_l = np.arctan(TH / R_c)
    theta_l = max(theta_l, 1e-3)   # guard against near-zero

    # Eq. 2 — radial spacing
    cot_theta = 1.0 / np.tan(theta_l)
    delta_R = LH * (1.44 * cot_theta
                    - 1.094
                    + 3.068 * theta_l
                    - 1.1256 * theta_l ** 2)
    delta_R = max(delta_R, LH * 1.05)   # floor: never smaller than LH

    # Eq. 3 — azimuthal spacing
    denom = theta_l - 0.04902
    if abs(denom) < 1e-4:
        denom = 1e-4
    delta_A = WH * (1.749 + 0.6396 * theta_l) + 0.2873 / denom
    delta_A = max(delta_A, WH * 1.05)   # floor: never smaller than WH

    return delta_R, delta_A


# ---------------------------------------------------------------------------
# Radial staggered layout (Paper Eq. 1, 2, 3)
# ---------------------------------------------------------------------------

def generate_radial_staggered(tower_height, LH, WR, DS, max_rings=60):
    """
    Generate radial staggered heliostat field using DELSOL3 spacing (Eq. 2-3).

    Parameters
    ----------
    tower_height : float  Tower height TH (m)
    LH           : float  Heliostat length (m)
    WR           : float  Width-to-length ratio (width = LH * WR)
    DS           : float  Security clearance (m)
    max_rings    : int    Maximum number of concentric rings

    Returns
    -------
    x_coords, y_coords : np.ndarray  Heliostat Cartesian coordinates (m)
    """
    Hm = LH
    Wm = LH * WR

    x_coords = []
    y_coords = []

    # First ring radius — physically motivated minimum clear zone
    # 0.75 * TH matches Haris et al. [3] reference baseline
    current_radius = max(0.75 * tower_height, Hm + Wm)

    for ring in range(1, max_rings + 1):

        # DELSOL3 spacing for this ring (Eq. 2-3)
        delta_R, delta_A = _delsol3_spacing(current_radius, tower_height, Hm, Wm)

        # Number of heliostats that fit azimuthally
        circumference = 2.0 * np.pi * current_radius
        num_mirrors   = max(1, int(circumference / delta_A))

        # Angular positions — stagger every other ring (Eq. 1)
        angles = np.linspace(0, 2.0 * np.pi, num_mirrors, endpoint=False)
        if ring % 2 == 0:
            angles += np.pi / num_mirrors

        x_coords.extend(current_radius * np.cos(angles))
        y_coords.extend(current_radius * np.sin(angles))

        # Advance radius by ΔR
        current_radius += delta_R

    return np.array(x_coords), np.array(y_coords)


# ---------------------------------------------------------------------------
# Fermat spiral layout (Paper Eq. 4)
# ---------------------------------------------------------------------------

def generate_fermat_spiral(TH, LH, WR, DS, n_heliostats=2000):
    """
    Generate Fermat-spiral heliostat field (Paper Eq. 4).

        r_n = c * sqrt(n),   θ_n = 137.508° × n

    Parameters
    ----------
    TH           : float  Tower height (m)
    LH           : float  Heliostat length (m)
    WR           : float  Width-to-length ratio
    DS           : float  Security clearance (m)
    n_heliostats : int    Number of heliostats to place

    Returns
    -------
    x_coords, y_coords : np.ndarray
    """
    WH     = LH * WR
    diag   = np.sqrt(LH ** 2 + WH ** 2)
    d_safe = diag + DS

    golden_angle = np.radians(137.508)

    n     = np.arange(1, n_heliostats + 1, dtype=float)
    theta = golden_angle * n

    # Scale c so adjacent spiral points are separated by at least d_safe
    c = d_safe * 2.0

    r         = c * np.sqrt(n)
    x_coords  = r * np.cos(theta)
    y_coords  = r * np.sin(theta)

    return x_coords, y_coords


# ---------------------------------------------------------------------------
# Collision check (Paper Eq. 10-11) — strict, no tolerance
# ---------------------------------------------------------------------------

def check_collisions(x_coords, y_coords, LH, WR, DS):
    """
    Return True if NO two heliostats overlap during sun-tracking.

    Paper Eq. 10:  D_p = sqrt(LH² + W_h²),  d_safe = D_p + DS
    Paper Eq. 11:  ||H_i − H_j|| ≥ d_safe   for all i ≠ j

    FIX: original used 0.98 * d_safe tolerance — removed, strict now.
    """
    if len(x_coords) < 2:
        return True

    WH     = LH * WR
    D_p    = np.sqrt(LH ** 2 + WH ** 2)   # Eq. 10
    d_safe = D_p + DS                      # Eq. 10 — no tolerance factor

    coords      = np.column_stack((x_coords, y_coords))
    diff        = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_matrix, np.inf)

    return bool(np.min(dist_matrix) >= d_safe)   # Eq. 11 — strict


# ---------------------------------------------------------------------------
# Spillage factor helper (Paper Eq. 9, f_sp term)
# ---------------------------------------------------------------------------

def compute_spillage(slant_ranges, TH, sigma_r=SIGMA_SUNSHAPE):
    """
    Per-heliostat Gaussian sunshape spillage factor.

    Gaussian beam with σ_r = 2.51 mrad spreads to σ = σ_r * S at the
    receiver (slant range S).  Fraction captured within aperture r_ap:

        f_sp = 1 − exp(−r_ap² / (2σ²))

    Parameters
    ----------
    slant_ranges : np.ndarray  Heliostat-to-tower distances (m)
    TH           : float       Tower height (m)
    sigma_r      : float       Sunshape std dev (rad)

    Returns
    -------
    fsp : np.ndarray  Per-heliostat spillage factor in [0, 1]
    """
    r_ap  = TH / 20.0                            # receiver aperture radius (m)
    sigma = sigma_r * slant_ranges               # beam footprint std dev (m)
    fsp   = 1.0 - np.exp(-0.5 * (r_ap / sigma) ** 2)
    return np.clip(fsp, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TH = 130.0
    LH = 10.95
    WR = 1.0
    DS = 0.16

    print("Generating radial-staggered field (DELSOL3 spacing)...")
    x, y = generate_radial_staggered(TH, LH, WR, DS, max_rings=15)
    print(f"  Heliostats generated : {len(x)}")

    safe = check_collisions(x, y, LH, WR, DS)
    print(f"  Collision check      : {'PASS' if safe else 'FAIL'}")

    # Spillage factor sample
    slant = np.sqrt(x ** 2 + y ** 2 + TH ** 2)
    fsp   = compute_spillage(slant, TH)
    print(f"  Mean spillage f_sp   : {np.mean(fsp):.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, s=6, c="#2ca02c", alpha=0.7, label="Heliostats")
    ax.scatter(0, 0, s=150, c="red", marker="^", label="Tower")
    ax.set_title(f"Radial Staggered — DELSOL3 spacing\n"
                 f"TH={TH}m  LH={LH}m  WR={WR}  DS={DS}m  N={len(x)}")
    ax.set_xlabel("East-West (m)")
    ax.set_ylabel("North-South (m)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig("layout_test.pdf", dpi=150)
    print("  Saved layout_test.pdf")
    plt.show()