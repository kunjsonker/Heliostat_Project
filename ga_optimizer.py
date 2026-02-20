import numpy as np
import pandas as pd
import pygad
from layout_generator import generate_radial_staggered, check_collisions
from efficiency_physics import calculate_efficiencies
from solar_physics import calculate_sun_angles

# --- 1. Load and Sample the Dataset ---
print("Loading annual weather data...")
df = pd.read_csv('solar-measurementspakistanquettawb-esmapqc.csv', low_memory=False)
df['time'] = pd.to_datetime(df['time'])
daylight_data = df[df['dni'] > 0].copy()

# Extract a representative sample: The 15th of every month at 12:00 PM
annual_sample = daylight_data[(daylight_data['time'].dt.day == 15) & 
                              (daylight_data['time'].dt.hour == 12)].copy()

# Pre-calculate the day of the year for our sample
annual_sample['day_of_year'] = annual_sample['time'].dt.dayofyear

# --- 2. Define the Annual Fitness Function ---
def fitness_func(ga_instance, solution, solution_idx):
    TH, LH, WR, DS = solution[0], solution[1], solution[2], solution[3]
    
    # Use the diagonal to prevent collisions
    width = LH * WR
    diagonal = np.sqrt(LH**2 + width**2)
    x, y = generate_radial_staggered(tower_height=TH, heliostat_width=diagonal, security_distance=DS)
    
    if not check_collisions(x, y, LH, WR, DS):
        return 0.0001 # Fail score for crashing mirrors
    
    total_annual_efficiency = 0
    
    # Loop through our 12 representative days of the year
    for index, row in annual_sample.iterrows():
        day = row['day_of_year']
        hour = 12.0 # Solar noon
        
        # Calculate where the sun is on this specific day
        sun_elevation, sun_azimuth = calculate_sun_angles(day, hour)
        
        # If the sun is below the horizon (elevation < 0), skip it
        if sun_elevation <= 0:
            continue
            
        # We approximate azimuth for solar noon as 180 degrees (South) for Quetta's latitude
        cos_eff, att_eff, tot_eff = calculate_efficiencies(x, y, TH, sun_elevation, 180)
        
        # Multiply by the paper's 0.97 blocking factor
        daily_efficiency = np.mean(tot_eff) * 0.97
        total_annual_efficiency += daily_efficiency
        
    # Return the average efficiency across the whole year
    return total_annual_efficiency / len(annual_sample)

# --- 3. Configure and Run the Genetic Algorithm ---
print("Configuring Genetic Algorithm for Annual Optimization...")
gene_space = [{'low': 50, 'high': 300}, {'low': 5, 'high': 20}, {'low': 1, 'high': 2}, {'low': 0.1, 'high': 0.5}]

ga_instance = pygad.GA(
    num_generations=20,           
    num_parents_mating=10,        
    fitness_func=fitness_func,    
    sol_per_pop=30,               
    num_genes=4,                  
    gene_space=gene_space,        
    mutation_probability=0.2,
    suppress_warnings=True
)

print("Starting annual optimization... This will take a moment.")
ga_instance.run()

# --- 4. Output the Results ---
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("\n--- ANNUAL OPTIMIZATION COMPLETE ---")
print(f"Highest Annual Average Efficiency: {solution_fitness * 100:.2f}%")
print(f"Optimal Tower Height (TH): {solution[0]:.2f} m")
print(f"Optimal Heliostat Length (LH): {solution[1]:.2f} m")
print(f"Optimal Width Ratio (WR): {solution[2]:.2f}")
print(f"Optimal Security Distance (DS): {solution[3]:.2f} m")

ga_instance.plot_fitness(title="Annual Optimization Progress", xlabel="Generation", ylabel="Annual Fitness")