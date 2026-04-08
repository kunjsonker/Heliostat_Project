import numpy as np
import pandas as pd

def calculate_sun_angles(day_of_year, solar_hour, latitude_deg=30.1798):
    """
    Calculates the solar declination, elevation angle (alpha), and 
    true azimuth angle for Quetta, Pakistan.
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
    term1 = np.sin(delta) * np.sin(phi)
    term2 = np.cos(delta) * np.cos(omega) * np.cos(phi)
    alpha = np.arcsin(term1 + term2)
    
    # 4. Calculate True Solar Azimuth Angle (A)
    # Formula mapping the sun's compass direction along the horizon
    term3 = np.sin(delta) * np.cos(phi) - np.cos(delta) * np.sin(phi) * np.cos(omega)
    
    # Avoid division by zero at perfectly straight up (zenith)
    if np.cos(alpha) == 0:
        cos_azimuth = 0
    else:
        cos_azimuth = term3 / np.cos(alpha)
        
    # Clip to [-1, 1] to avoid floating point math errors with arccos
    cos_azimuth = np.clip(cos_azimuth, -1.0, 1.0)
    azimuth = np.arccos(cos_azimuth)
    
    # Adjust azimuth based on morning vs afternoon
    if hour_angle > 0:
        azimuth = 2 * np.pi - azimuth
    
    return np.degrees(alpha), np.degrees(azimuth)


if __name__ == "__main__":
    # --- Test the Physics Engine ---
    # Let's test this using the Vernal Equinox (March 21) at 11:00 AM, 
    # which is the exact design point used in the paper's Table 3.

    # March 21 is the 80th day of the year
    test_day = 80
    test_hour = 11.0  # 11:00 AM

    elevation, azimuth = calculate_sun_angles(day_of_year=test_day, solar_hour=test_hour)

    print(f"--- Sun Position for Vernal Equinox (11:00 AM) ---")
    print(f"Solar Elevation Angle (alpha): {elevation:.2f} degrees")
    print(f"Solar Azimuth Angle: {azimuth:.2f} degrees")