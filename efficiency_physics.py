import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants (Paper Table I)
# ---------------------------------------------------------------------------
F_SP         = 0.95   # Spillage factor — Gaussian sunshape approx. (σ_r = 2.51 mrad)
F_B          = 0.97   # Blocking factor
REFLECTIVITY = 0.88   # Mirror reflectivity ρ


# ---------------------------------------------------------------------------
# Main physics function
# ---------------------------------------------------------------------------

def calculate_efficiencies(x_field, y_field, tower_height, elevation_deg, azimuth_deg=0):
    """
    Calculates per-heliostat optical efficiencies.
    Implements Paper Equations 6, 7, 8, 9 exactly.

    Parameters
    ----------
    x_field       : np.ndarray  East-West coordinates of heliostats (m)
    y_field       : np.ndarray  North-South coordinates of heliostats (m)
    tower_height  : float       Tower height TH (m)
    elevation_deg : float       Solar elevation angle α (degrees)
    azimuth_deg   : float       Solar azimuth angle (degrees, clockwise from North)

    Returns
    -------
    cosine_eff      : np.ndarray  Per-heliostat cosine efficiency η^CA_i
    attenuation_eff : np.ndarray  Per-heliostat atmospheric attenuation η_atm,i
    total_eff       : np.ndarray  Per-heliostat total optical efficiency η_i
                                  includes f_sp, f_b, η_cos, η_atm (all Eq. 9 terms)
    """

    num_mirrors = len(x_field)

    # Heliostat positions as (N, 3) matrix — z = 0 (ground level)
    heliostat_positions = np.stack(
        (x_field, y_field, np.zeros(num_mirrors)), axis=1
    )

    # Tower apex position
    tower_position = np.array([0.0, 0.0, tower_height])

    # ------------------------------------------------------------------
    # Step 1 — Tower-pointing unit vector T̂_i (Eq. 6)
    # ------------------------------------------------------------------
    target_vectors      = tower_position - heliostat_positions          # (N, 3)
    slant_ranges        = np.linalg.norm(target_vectors, axis=1)        # (N,)  metres
    target_vectors_norm = target_vectors / slant_ranges[:, np.newaxis]  # (N, 3) unit

    # ------------------------------------------------------------------
    # Step 2 — Sun unit vector Ŝ
    # Azimuth convention: clockwise from North (meteorological standard)
    # This matches the output of solar_physics.calculate_sun_angles()
    #   sx =  cos(el) * sin(az)  →  East  component
    #   sy =  cos(el) * cos(az)  →  North component
    #   sz =  sin(el)            →  Up    component
    # ------------------------------------------------------------------
    el_rad = np.radians(elevation_deg)
    az_rad = np.radians(azimuth_deg)

    sun_vector = np.array([
        np.cos(el_rad) * np.sin(az_rad),   # East
        np.cos(el_rad) * np.cos(az_rad),   # North
        np.sin(el_rad)                      # Up
    ])

    # ------------------------------------------------------------------
    # Step 3 — Heliostat normal N̂_i = normalised bisector of T̂_i and Ŝ
    # (Eq. 6): N̂_i = (T̂_i + Ŝ) / ‖T̂_i + Ŝ‖
    # sun_vector shape (3,) broadcasts correctly over (N, 3)
    # ------------------------------------------------------------------
    bisector       = target_vectors_norm + sun_vector   # (N, 3)
    bisector_norms = np.linalg.norm(bisector, axis=1)   # (N,)
    heliostat_normals = bisector / bisector_norms[:, np.newaxis]  # (N, 3) unit

    # ------------------------------------------------------------------
    # Step 4 — Cosine efficiency η^CA_i = T̂_i · N̂_i  (Eq. 7)
    # einsum gives the row-wise dot product efficiently
    # ------------------------------------------------------------------
    cosine_eff = np.einsum('ij,ij->i', target_vectors_norm, heliostat_normals)
    cosine_eff = np.clip(cosine_eff, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Step 5 — Atmospheric attenuation η_atm,i  (Eq. 8)
    # Vittitoe & Biggs [30] polynomial — slant range in km
    # ------------------------------------------------------------------
    S_km = slant_ranges / 1000.0
    attenuation_eff = (
          0.99326
        - 0.1046   * S_km
        + 0.017    * S_km ** 2
        - 0.002845 * S_km ** 3
    )
    attenuation_eff = np.clip(attenuation_eff, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Step 6 — Total optical efficiency  (Eq. 9)
    # η_i = η^CA_i · f_sp · f_b · η_atm,i
    # Reflectivity ρ is applied in the optimisers when computing power/energy
    # ------------------------------------------------------------------
    total_eff = cosine_eff * F_SP * F_B * attenuation_eff

    return cosine_eff, attenuation_eff, total_eff


# ---------------------------------------------------------------------------
# Self-test / visualisation  (run this file directly to verify)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from layout_generator import generate_radial_staggered
    from solar_physics import calculate_sun_angles

    # Unoptimised baseline field (Table I defaults)
    TH = 130.0
    LH = 10.95
    WR = 1.0
    DS = 0.16

    print("Generating baseline field ...")
    x, y = generate_radial_staggered(TH, LH, WR, DS)
    print(f"  Heliostats generated : {len(x)}")

    # Vernal Equinox 11:00 AM  (day 80)
    el, az = calculate_sun_angles(day_of_year=80, solar_hour=11.0)
    print(f"  Solar elevation      : {el:.2f} deg")
    print(f"  Solar azimuth        : {az:.2f} deg  (clockwise from North)")

    cos_eff, att_eff, tot_eff = calculate_efficiencies(x, y, TH, el, az)

    print(f"\n  Mean cosine efficiency      : {np.mean(cos_eff) * 100:.2f} %")
    print(f"  Mean attenuation efficiency : {np.mean(att_eff) * 100:.2f} %")
    print(f"  Mean TOTAL efficiency       : {np.mean(tot_eff) * 100:.2f} %")
    print(f"  (Eq. 9 — includes f_sp={F_SP}, f_b={F_B})")

    # Heatmap of all three efficiency components
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    datasets = [
        (cos_eff,  "Cosine efficiency η_cos  (Eq. 7)"),
        (att_eff,  "Atm. attenuation η_atm  (Eq. 8)"),
        (tot_eff,  "Total efficiency η_i = η_cos·f_sp·f_b·η_atm  (Eq. 9)"),
    ]

    for ax, (data, title) in zip(axes, datasets):
        sc = ax.scatter(x, y, c=data, cmap="viridis", s=8)
        plt.colorbar(sc, ax=ax, label="Efficiency")
        ax.scatter(0, 0, c="red", s=120, marker="^", zorder=5, label="Tower")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("East-West (m)")
        ax.set_ylabel("North-South (m)")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)

    plt.suptitle(
        f"Vernal Equinox 11:00 AM  |  TH={TH}m  |  N={len(x)} heliostats",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("efficiency_heatmaps.pdf", dpi=150)
    print("\n  Saved efficiency_heatmaps.pdf")
    plt.show()