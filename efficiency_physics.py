import numpy as np
import matplotlib.pyplot as plt
from layout_generator import generate_radial_staggered

# Sunshape standard deviation (mrad)
SIGMA_R_MRAD = 2.51


def calculate_spillage_factor(
    slant_ranges,
    receiver_half_width=6.0,
    receiver_half_height=6.0,
    sigma_r=SIGMA_R_MRAD
):
    """
    Calculate spillage factor using Gaussian sunshape model.
    """

    sigma_r_rad = sigma_r * 1e-3
    sigma_beam = np.maximum(sigma_r_rad * slant_ranges, 1e-9)

    try:
        from scipy.special import erf
    except ImportError:
        from math import erf
        erf = np.vectorize(erf)

    fsp_x = erf(receiver_half_width / (np.sqrt(2) * sigma_beam))
    fsp_y = erf(receiver_half_height / (np.sqrt(2) * sigma_beam))

    return np.clip(fsp_x * fsp_y, 0.0, 1.0)


def calculate_efficiencies(
    x_field,
    y_field,
    tower_height,
    elevation_deg,
    azimuth_deg=0,
    receiver_half_width=6.0,
    receiver_half_height=6.0
):
    """
    Calculate:
    - Cosine efficiency
    - Atmospheric attenuation
    - Spillage factor
    - Total optical efficiency
    """

    num_mirrors = len(x_field)

    heliostat_positions = np.stack(
        (x_field, y_field, np.zeros(num_mirrors)),
        axis=1
    )

    tower_position = np.array([0, 0, tower_height])

    # Tower-pointing vectors
    target_vectors = tower_position - heliostat_positions
    slant_ranges = np.linalg.norm(target_vectors, axis=1)
    target_vectors_norm = target_vectors / slant_ranges[:, np.newaxis]

    # Sun vector
    el_rad = np.radians(elevation_deg)
    az_rad = np.radians(azimuth_deg)

    sx = np.cos(el_rad) * np.sin(az_rad)
    sy = np.cos(el_rad) * np.cos(az_rad)
    sz = np.sin(el_rad)

    sun_vector = np.array([sx, sy, sz])

    # Heliostat normals
    bisector = target_vectors_norm + sun_vector
    bisector_norms = np.linalg.norm(bisector, axis=1)
    heliostat_normals = bisector / bisector_norms[:, np.newaxis]

    # Cosine efficiency
    cosine_eff = np.sum(
        target_vectors_norm * heliostat_normals,
        axis=1
    )

    cosine_eff = np.clip(cosine_eff, 0.0, 1.0)

    # Atmospheric attenuation
    S_km = slant_ranges / 1000.0

    attenuation_eff = (
        0.99326
        - 0.1046 * S_km
        + 0.017 * S_km**2
        - 0.002845 * S_km**3
    )

    attenuation_eff = np.clip(attenuation_eff, 0.0, 1.0)

    # Spillage factor
    spillage_eff = calculate_spillage_factor(
        slant_ranges,
        receiver_half_width,
        receiver_half_height
    )

    # Composite efficiency
    total_eff = cosine_eff * attenuation_eff * spillage_eff

    return cosine_eff, attenuation_eff, spillage_eff, total_eff


if __name__ == "__main__":

    # Baseline parameters
    TH = 130
    LH = 10.95
    WR = 1.0
    DS = 0.16

    print("Generating field...")
    x, y = generate_radial_staggered(
        TH,
        LH * WR,
        DS
    )

    sun_elevation = 56.25
    sun_azimuth = 165

    print("Calculating efficiencies...")

    cos_eff, att_eff, spill_eff, tot_eff = calculate_efficiencies(
        x,
        y,
        TH,
        sun_elevation,
        sun_azimuth
    )

    print(f"Average Cosine Efficiency:      {np.mean(cos_eff):.4f}")
    print(f"Average Attenuation Efficiency: {np.mean(att_eff):.4f}")
    print(f"Average Spillage Factor:        {np.mean(spill_eff):.4f}")
    print(f"Average Total Efficiency:       {np.mean(tot_eff):.4f}")
    print(f"With blocking factor (0.97):    {np.mean(tot_eff)*0.97:.4f}")

    # Plot
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        x,
        y,
        c=tot_eff,
        cmap="viridis",
        s=15
    )

    plt.colorbar(
        scatter,
        label="Optical Efficiency"
    )

    plt.scatter(
        0,
        0,
        c="red",
        s=120,
        marker="^",
        label="Tower"
    )

    plt.title(
        f"Heliostat Field Efficiency Map\nTower Height = {TH} m"
    )

    plt.xlabel("East-West (m)")
    plt.ylabel("North-South (m)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()