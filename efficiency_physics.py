import numpy as np
import matplotlib.pyplot as plt
from layout_generator import generate_radial_staggered
from solar_physics import calculate_sun_angles

def calculate_efficiencies(x_field, y_field, tower_height, elevation_deg, azimuth_deg=0):
    """
    Calculates Cosine and Attenuation efficiencies for the entire field.
    """
    # 1. Setup Vectors
    # Heliostat positions (z is 0 for simplicity, though actual mirrors are slightly raised)
    # We stack them into a (N, 3) matrix: [[x1, y1, 0], [x2, y2, 0], ...]
    num_mirrors = len(x_field)
    heliostat_positions = np.stack((x_field, y_field, np.zeros(num_mirrors)), axis=1)
    
    # Tower position: [0, 0, TH]
    tower_position = np.array([0, 0, tower_height])
    
    # 2. Calculate Target Vector (T)
    # Vector from each mirror to the tower tip
    target_vectors = tower_position - heliostat_positions
    # Calculate slant range (distance) for attenuation later
    slant_ranges = np.linalg.norm(target_vectors, axis=1)
    # Normalize target vectors (make length 1)
    target_vectors_norm = target_vectors / slant_ranges[:, np.newaxis]
    
    # 3. Calculate Sun Vector (S)
    # Convert Elevation/Azimuth to a standard [x, y, z] vector
    el_rad = np.radians(elevation_deg)
    az_rad = np.radians(azimuth_deg) # Assuming South is 0 or 180, simplified here to standard physics vector
    
    # Sun vector S (pointing TO the sun)
    # Simple geometry: z is sin(elevation), x/y depend on azimuth
    sz = np.sin(el_rad)
    hyp = np.cos(el_rad)
    sx = hyp * np.sin(az_rad)
    sy = hyp * np.cos(az_rad)
    sun_vector = np.array([sx, sy, sz])
    
    # 4. Calculate Heliostat Normal Vector (N) - The Bisector
    # The mirror must point exactly halfway between the Sun and the Tower to reflect light correctly
    bisector = (target_vectors_norm + sun_vector)
    # Normalize the bisector
    bisector_norms = np.linalg.norm(bisector, axis=1)
    heliostat_normals = bisector / bisector_norms[:, np.newaxis]
    
    # 5. Calculate Cosine Efficiency (Equation 5)
    # Cosine efficiency = Dot product of Sun Vector and Heliostat Normal
    # This tells us how "on-axis" the reflection is.
    cosine_eff = np.sum(target_vectors_norm * heliostat_normals, axis=1)
    
    # 6. Calculate Atmospheric Attenuation (Equation 14)
    # Formula: 0.99326 - 0.1046*S + ... (S is distance in km)
    S_km = slant_ranges / 1000.0 # Convert meters to km
    attenuation_eff = (0.99326 
                       - 0.1046 * S_km 
                       + 0.017 * S_km**2 
                       - 0.002845 * S_km**3)
    
    # Clip values just in case (efficiency can't be > 1 or < 0)
    attenuation_eff = np.clip(attenuation_eff, 0, 1)
    
    # 7. Total Optical Efficiency (ignoring shading/blocking for now)
    total_eff = cosine_eff * attenuation_eff
    
    return cosine_eff, attenuation_eff, total_eff

# --- Run the Simulation ---
# Constants from Paper
TH = 130  # Tower Height
LH = 10.95
WR = 1.0
DS = 0.16

# 1. Generate the Field
print("Generating field...")
x, y = generate_radial_staggered(TH, LH*WR, DS)

# 2. Get Sun Position (Vernal Equinox, 11 AM)
# We calculated this in the previous step: Elevation ~56.25 deg
sun_elevation = 56.25 
# Note: Azimuth is also needed for the vector. For 11 AM, sun is roughly South-East.
# Let's approximate Azimuth as 165 degrees (15 degrees off South) for this test.
sun_azimuth = 165 

# 3. Calculate Efficiencies
print("Calculating physics...")
cos_eff, att_eff, tot_eff = calculate_efficiencies(x, y, TH, sun_elevation, sun_azimuth)

print(f"Average Cosine Efficiency: {np.mean(cos_eff):.4f}")
print(f"Average Attenuation Efficiency: {np.mean(att_eff):.4f}")
print(f"Average Total Efficiency: {np.mean(tot_eff):.4f}")

# --- Visualize (Heatmap) ---
plt.figure(figsize=(10, 8))
plt.scatter(x, y, c=tot_eff, cmap='viridis', s=15)
plt.colorbar(label='Total Optical Efficiency')
plt.scatter(0, 0, c='red', s=100, marker='^', label='Tower')
plt.title(f"Field Efficiency Map (Vernal Equinox 11:00 AM)\nTower Height: {TH}m")
plt.xlabel("East-West (m)")
plt.ylabel("North-South (m)")
plt.axis('equal')
plt.legend()
plt.show()