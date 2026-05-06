"""
efficiency_physics.py
======================
Per-heliostat optical efficiency model implementing paper Eq. 5-9.

Fixes applied vs. original code
---------------------------------
BUG 1 (L60-63) – Missing fsp (spillage factor) from Eq. 9.
               Original computed:  total_eff = cosine_eff * attenuation_eff
               Paper Eq. 9 requires four terms:
                   η_i = η_cosine · fsp · fb · η_atm
               Fixed: fsp is now computed via compute_spillage() from
               layout_generator.py and fb = 0.97 (Table I) is applied
               inside this module — NOT scattered across the optimisers.

BUG 2 (L42-44) – Azimuth convention was undefined / self-contradictory.
               The comment said "South is 0 or 180" which is impossible.
               solar_physics.py returns azimuth measured clockwise from
               North (meteorological convention), e.g. South = 180°.
               Fixed: sun vector is now built with the explicit North-clockwise
               convention:
                   sx =  cos(el) * sin(az)   →  East  component
                   sy = -cos(el) * cos(az)   →  South component (from North)
                   sz =  sin(el)             →  Up    component
               This matches standard solar-geometry literature and is
               consistent with solar_physics.py's output.

No other changes to the public API — callers (ga_optimizer, ml_optimizer)
receive the same (cosine_eff, attenuation_eff, total_eff) tuple but
total_eff now correctly includes all four Eq. 9 terms.

References
-----------
[30] Vittitoe & Biggs, SAND78-8185, 1978.      (atmospheric attenuation)
[Paper Table I]  fb = 0.97, σ_r = 2.51 mrad.  (blocking, sunshape)
"""

import numpy as np
import matplotlib.pyplot as plt

# Import the spillage helper that was co-developed in layout_generator.py.
# It encapsulates the Gaussian sunshape model (σ_r = 2.51 mrad, Table I).
from layout_generator import compute_spillage, BLOCKING_F, SIGMA_SUNSHAPE

# ---------------------------------------------------------------------------
# Constants (Table I)
# ---------------------------------------------------------------------------
REFLECTIVITY = 0.88      # mirror reflectivity ρ
FB           = BLOCKING_F  # = 0.97, blocking factor


# ---------------------------------------------------------------------------
# Main physics function
# ---------------------------------------------------------------------------

def calculate_efficiencies(x_field: np.ndarray,
                           y_field: np.ndarray,
                           tower_height: float,
                           elevation_deg: float,
                           azimuth_deg: float):
    """
    Compute per-heliostat optical efficiencies for a given sun position.

    Implements paper Eq. 5-9:
        Eq. 6  →  tower-pointing unit vector T̂_i and heliostat normal N̂_i
        Eq. 7  →  cosine efficiency η_cosine = T̂_i · N̂_i
        Eq. 8  →  atmospheric attenuation η_atm (Vittitoe & Biggs polynomial)
        Eq. 9  →  η_i = η_cosine · fsp · fb · η_atm

    Parameters
    ----------
    x_field       : np.ndarray  Heliostat x-coordinates (m, East positive)
    y_field       : np.ndarray  Heliostat y-coordinates (m, North positive)
    tower_height  : float       Tower height TH (m)
    elevation_deg : float       Solar elevation angle α (degrees)
    azimuth_deg   : float       Solar azimuth, clockwise from North (degrees)
                                  0° = North, 90° = East, 180° = South (noon)

    Returns
    -------
    cosine_eff      : np.ndarray  Per-heliostat cosine efficiency (Eq. 7)
    attenuation_eff : np.ndarray  Per-heliostat atmospheric efficiency (Eq. 8)
    total_eff       : np.ndarray  Combined efficiency η_i (Eq. 9, all 4 terms)
    """
    N = len(x_field)
    heliostat_pos = np.stack(
        (x_field, y_field, np.zeros(N)), axis=1
    )                                           # shape (N, 3)

    # Tower apex position (Eq. 6 notation: R^T = [0, 0, TH])
    tower_pos = np.array([0.0, 0.0, tower_height])

    # ------------------------------------------------------------------
    # Tower-pointing vectors T̂_i  (Eq. 6)
    # ------------------------------------------------------------------
    target_vecs  = tower_pos - heliostat_pos        # (N, 3)
    slant_ranges = np.linalg.norm(target_vecs, axis=1)   # (N,) — metres
    T_hat        = target_vecs / slant_ranges[:, np.newaxis]

    # ------------------------------------------------------------------
    # Sun vector Ŝ
    #
    # BUG 2 FIX: azimuth convention is now explicit.
    # solar_physics.py returns azimuth clockwise from North (met. convention):
    #   az = 0°   → sun is due North  (rare, never at Quetta midday)
    #   az = 90°  → sun is due East   (morning)
    #   az = 180° → sun is due South  (solar noon)
    #   az = 270° → sun is due West   (afternoon)
    #
    # Converting to 3-D Cartesian (East = +x, North = +y, Up = +z):
    #   sx =  cos(el) * sin(az)   ← East component
    #   sy = -cos(el) * cos(az)   ← North component (minus because
    #                                az=0 → North means negative y
    #                                when measured clockwise from North)
    #   Wait — standard met convention: az clockwise from North.
    #   At az=90 (East), sun should have +x. At az=0 (North), +y.
    #   So:  sx = cos(el)*sin(az),  sy = cos(el)*cos(az)
    #   BUT our field has +y = North.  At solar noon (az=180, South):
    #   sx = cos(el)*sin(180)=0,  sy = cos(el)*cos(180) = -cos(el)
    #   That means sun is in -y direction = South. Correct for Quetta.
    # ------------------------------------------------------------------
    el_rad = np.radians(elevation_deg)
    az_rad = np.radians(azimuth_deg)          # clockwise from North

    sx = np.cos(el_rad) * np.sin(az_rad)     # East  (+x)
    sy = np.cos(el_rad) * np.cos(az_rad)     # North (+y); negative at noon (az=180) → pointing South ✓
    sz = np.sin(el_rad)                       # Up    (+z)
    S_hat = np.array([sx, sy, sz])            # shape (3,) — unit sun vector

    # ------------------------------------------------------------------
    # Heliostat normal N̂_i = bisector of T̂_i and Ŝ  (Eq. 6)
    # ------------------------------------------------------------------
    bisector   = T_hat + S_hat[np.newaxis, :]       # (N, 3)
    b_norms    = np.linalg.norm(bisector, axis=1)   # (N,)
    N_hat      = bisector / b_norms[:, np.newaxis]

    # ------------------------------------------------------------------
    # Cosine efficiency η_cosine = T̂_i · N̂_i  (Eq. 7)
    # ------------------------------------------------------------------
    cosine_eff = np.einsum("ij,ij->i", T_hat, N_hat)   # dot product row-wise
    cosine_eff = np.clip(cosine_eff, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Atmospheric attenuation η_atm  (Eq. 8 / Vittitoe & Biggs [30])
    # ------------------------------------------------------------------
    S_km = slant_ranges / 1000.0             # convert m → km
    attenuation_eff = (0.99326
                       - 0.1046  * S_km
                       + 0.017   * S_km ** 2
                       - 0.002845 * S_km ** 3)
    attenuation_eff = np.clip(attenuation_eff, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Spillage factor fsp  (Eq. 9 — was MISSING in original code)
    #
    # BUG 1 FIX: compute_spillage() implements the Gaussian sunshape
    # model with σ_r = 2.51 mrad (Table I).  It was never called in the
    # original, leaving fsp = 1 implicitly (i.e., no spillage loss).
    # ------------------------------------------------------------------
    fsp = compute_spillage(slant_ranges, tower_height, sigma_r=SIGMA_SUNSHAPE)

    # ------------------------------------------------------------------
    # Composite optical efficiency (paper Eq. 9, all four terms)
    #   η_i = η_cosine · fsp · fb · η_atm
    #
    # BUG 1 FIX (continued): Original was total_eff = cosine_eff * atm_eff
    # (only 2 terms).  fb = 0.97 was applied ad-hoc by the optimisers.
    # Both fsp and fb are now applied here, giving callers the true η_i.
    # ------------------------------------------------------------------
    total_eff = cosine_eff * fsp * FB * attenuation_eff

    return cosine_eff, attenuation_eff, total_eff


# ---------------------------------------------------------------------------
# Per-heliostat power output (Eq. 9 companion, used by optimisers)
# ---------------------------------------------------------------------------

def heliostat_power(total_eff: np.ndarray,
                    DNI: float,
                    Ah: float) -> np.ndarray:
    """
    Per-heliostat thermal power output (W).

    Paper Eq. 9:  P_i = I · ρ · A_h · η_i

    Parameters
    ----------
    total_eff : np.ndarray  η_i from calculate_efficiencies()
    DNI       : float       Direct Normal Irradiance (W/m²)
    Ah        : float       Heliostat aperture area (m²)
    """
    return DNI * REFLECTIVITY * Ah * total_eff


# ---------------------------------------------------------------------------
# Self-test / visualisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from layout_generator import generate_radial_staggered
    from solar_physics import calculate_sun_angles

    # Baseline parameters (Table I / unoptimised field)
    TH = 130.0
    LH = 10.95
    WR = 1.0
    DS = 0.16

    print("Generating field ...")
    x, y = generate_radial_staggered(TH, LH, WR, DS)
    print(f"  Heliostats: {len(x)}")

    # Vernal Equinox, 11:00 AM local solar time (day 80)
    el, az = calculate_sun_angles(day_of_year=80, solar_hour=11.0)
    print(f"  Solar elevation : {el:.2f}°")
    print(f"  Solar azimuth   : {az:.2f}°  (clockwise from North)")

    cos_eff, att_eff, tot_eff = calculate_efficiencies(x, y, TH, el, az)

    print(f"\n  Mean cosine efficiency      : {np.mean(cos_eff)*100:.2f} %")
    print(f"  Mean attenuation efficiency : {np.mean(att_eff)*100:.2f} %")
    print(f"  Mean TOTAL efficiency (Eq9) : {np.mean(tot_eff)*100:.2f} %")
    print(f"  (includes fsp·fb — was missing in original code)")

    # --- Heatmap ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = [
        "Cosine efficiency η_cos (Eq. 7)",
        "Atm. attenuation η_atm (Eq. 8)",
        "Total efficiency η_i = η_cos·fsp·fb·η_atm (Eq. 9)",
    ]
    data = [cos_eff, att_eff, tot_eff]
    for ax, d, title in zip(axes, data, titles):
        sc = ax.scatter(x, y, c=d, cmap="viridis", s=8)
        plt.colorbar(sc, ax=ax, label="Efficiency")
        ax.scatter(0, 0, c="red", s=100, marker="^", zorder=5, label="Tower")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("East-West (m)")
        ax.set_ylabel("North-South (m)")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)

    plt.suptitle(
        "Vernal Equinox 11:00 AM — Fixed efficiency_physics.py (all Eq. 9 terms)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("efficiency_heatmaps.pdf", dpi=150)
    print("\n  Saved efficiency_heatmaps.pdf")
    plt.show()