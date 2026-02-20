import numpy as np
import pandas as pd

def calculate_sun_angles(day_of_year, solar_hour, latitude_deg=30.1798):
    """
    Calculates the solar declination, hour angle, and elevation angle (alpha)
    for Quetta, Pakistan.
    """
    # Convert latitude to radians
    phi = np.radians(latitude_deg)
    
    # 1. Calculate Solar Declination Angle (delta)
    # This represents the Earth's tilt relative to the sun on a given day (n)
    declination = 23.45 * np.sin(np.radians((360 / 365) * (day_of_year - 81)))
    delta = np.radians(declination)
    
    # 2. Calculate Solar Hour Angle (omega)
    # The sun moves 15 degrees per hour. Solar noon is 0 degrees.
    hour_angle = 15 * (solar_hour - 12)
    omega = np.radians(hour_angle)
    
    # 3. Calculate Solar Elevation/Altitude Angle (alpha) using Equation 6
    # alpha = arcsin(sin(delta) * sin(phi) + cos(delta) * cos(omega) * cos(phi))
    term1 = np.sin(delta) * np.sin(phi)
    term2 = np.cos(delta) * np.cos(omega) * np.cos(phi)
    
    alpha = np.arcsin(term1 + term2)
    
    return np.degrees(alpha), np.degrees(omega)

# --- Test the Physics Engine ---
# Let's test this using the Vernal Equinox (March 21) at 11:00 AM, 
# which is the exact design point used in the paper's Table 3.

# March 21 is the 80th day of the year
test_day = 80
test_hour = 11.0  # 11:00 AM

elevation, hour_angle = calculate_sun_angles(day_of_year=test_day, solar_hour=test_hour)

print(f"--- Sun Position for Vernal Equinox (11:00 AM) ---")
print(f"Solar Elevation Angle (alpha): {elevation:.2f} degrees")
print(f"Hour Angle (omega): {hour_angle:.2f} degrees")