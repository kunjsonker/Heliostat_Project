import numpy as np
import matplotlib.pyplot as plt

def generate_radial_staggered(tower_height, heliostat_width, security_distance, max_rings=20):
    """
    Generates x and y coordinates for a radial staggered heliostat field.
    """
    x_coords = []
    y_coords = []
    
    # The first ring (essential ring) starts at a distance related to the tower height
    current_radius = tower_height * 0.5 
    
    for ring in range(1, max_rings + 1):
        # Calculate the circumference of the current ring
        circumference = 2 * np.pi * current_radius
        
        # The physical space one mirror takes up, plus the security gap
        mirror_spacing = heliostat_width + security_distance
        
        # Calculate how many mirrors can fit in this ring
        num_mirrors = int(circumference / mirror_spacing)
        
        # Calculate the angle for each mirror in radians
        angles = np.linspace(0, 2 * np.pi, num_mirrors, endpoint=False)
        
        # Stagger every other ring by shifting the angle slightly so they don't block each other
        if ring % 2 == 0:
            angles += (np.pi / num_mirrors)
            
        # Convert polar coordinates (radius, angle) to Cartesian (x, y)
        x_ring = current_radius * np.cos(angles)
        y_ring = current_radius * np.sin(angles)
        
        x_coords.extend(x_ring)
        y_coords.extend(y_ring)
        
        # Increase the radius for the next ring
        current_radius += (heliostat_width + security_distance)
        
    return np.array(x_coords), np.array(y_coords)



def check_collisions(x_coords, y_coords, LH, WR, DS):
    
    # Checks if any heliostats in the field will collide during rotation.
    # Returns True if the layout is safe, False if there are collisions.
    
    # 1. Calculate the mirror's width based on the ratio
    width = LH * WR
    
    # 2. Calculate the diagonal of the mirror using the Pythagorean theorem
    diagonal = np.sqrt(LH**2 + width**2)
    
    # 3. The absolute minimum safe distance between two center points
    safe_distance = diagonal + DS
    
    # 4. Group x and y into a single matrix of coordinates
    coords = np.column_stack((x_coords, y_coords))
    
    # 5. Calculate the distance between EVERY pair of mirrors (Vectorized math)
    # This creates a matrix of all distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    
    # 6. Ignore the distance of a mirror to itself (which is always 0)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # 7. Find the absolute closest pair of mirrors in the entire field
    min_dist = np.min(dist_matrix)
    
    if min_dist >= safe_distance:
       # print(f"Layout is SAFE! Minimum distance is {min_dist:.2f}m (Required: {safe_distance:.2f}m)")
        return True
    else:
       # print(f"COLLISION DETECTED! Minimum distance is {min_dist:.2f}m (Required: {safe_distance:.2f}m)")
        return False




# --- Test the Generator ---
# Using the constant parameters defined in Table 2 of the paper
TH = 130          # Tower Height in meters
LH = 10.95        # Heliostat Length in meters
WR = 1            # Width to Length ratio is 1, so width is also 10.95m
DS = 0.16         # Security distance in meters

x, y = generate_radial_staggered(tower_height=TH, heliostat_width=(LH * WR), security_distance=DS, max_rings=15)

print(f"Successfully generated {len(x)} heliostat coordinates.")

# --- Visualize the Layout ---
plt.figure(figsize=(8, 8))
plt.scatter(x, y, s=15, c='#2ca02c', alpha=0.7, label='Heliostats')
plt.scatter(0, 0, s=150, c='red', label='Central Tower') 

plt.title("Radial Staggered Heliostat Field Layout")
plt.xlabel("Distance from Tower (meters)")
plt.ylabel("Distance from Tower (meters)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.axis('equal') # Keeps the aspect ratio perfectly circular
# --- Run the Collision Check ---
is_safe = check_collisions(x, y, LH, WR, DS)
plt.show()