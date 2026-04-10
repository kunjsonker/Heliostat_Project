import numpy as np
import pandas as pd
import pygad
import time
from layout_generator import generate_radial_staggered, check_collisions
from efficiency_physics import calculate_efficiencies
from solar_physics import calculate_sun_angles

# --- 1. Load and Sample the Dataset ---
print("Loading annual weather data...")
df = pd.read_csv('solar-measurementspakistanquettawb-esmapqc.csv', low_memory=False)
df['time'] = pd.to_datetime(df['time'])
daylight_data = df[df['dni'] > 0].copy()

# 12 representative days for the year
annual_sample = daylight_data[(daylight_data['time'].dt.day == 15) & 
                              (daylight_data['time'].dt.hour == 12)].copy()
annual_sample['day_of_year'] = annual_sample['time'].dt.dayofyear

# --- 2. Shared Physics & Cost Engine ---
def calculate_plant_cost(TH, LH, WR, num_mirrors, field_radius):
    tower_cost = 3000000 * np.exp(0.0113 * TH) 
    mirror_area = LH * (LH * WR)
    total_glass_area = mirror_area * num_mirrors
    heliostat_cost = total_glass_area * 150  
    land_area = np.pi * (field_radius ** 2)
    land_cost = land_area * 5 
    
    total_cost = tower_cost + heliostat_cost + land_cost
    capital_recovery_factor = (0.07 * (1 + 0.07)**25) / (((1 + 0.07)**25) - 1)
    annual_cost = total_cost * capital_recovery_factor
    return annual_cost, total_glass_area

def calculate_annual_metrics(TH, LH, WR, DS):
    width = LH * WR
    diagonal = np.sqrt(LH**2 + width**2)
    x, y = generate_radial_staggered(TH, diagonal, DS)
    
    mirror_area = LH * width
    power_per_mirror = 858 * 0.88 * mirror_area * 0.82 
    target_mirrors = int(50000000 / power_per_mirror) 
    
    if target_mirrors <= len(x):
        x = x[:target_mirrors]
        y = y[:target_mirrors]
    else:
        # FIX 1: Return 3 zeros instead of 2 zeros or nothing
        return 0, 0, 0
        
    if not check_collisions(x, y, LH, WR, DS) or len(x) == 0:
        # FIX 2: Return 3 zeros for collision failures
        return 0, 0, 0
        
    field_radius = np.max(np.sqrt(x**2 + y**2)) + diagonal
    annual_cost, total_glass_area = calculate_plant_cost(TH, LH, WR, len(x), field_radius)
    
    total_annual_efficiency = 0
    total_annual_energy_kwh = 0
    valid_days = 0
    
    for _, row in annual_sample.iterrows():
        sun_elevation, sun_azimuth = calculate_sun_angles(row['day_of_year'], 12.0)
        if sun_elevation <= 0: continue
            
        _, _, tot_eff = calculate_efficiencies(x, y, TH, sun_elevation, sun_azimuth)
        
        mean_eff = np.mean(tot_eff) * 0.97 
        total_annual_efficiency += mean_eff
        valid_days += 1
        
        total_annual_energy_kwh += ((row['dni'] * total_glass_area * mean_eff * 0.88) / 1000 * 8) * 30
        
    if total_annual_energy_kwh == 0 or valid_days == 0:
        # FIX 3: Return 3 zeros for zero energy failures
        return 0, 0, 0
        
    # FIX 4: Make sure the final success return gives exactly 3 values: (eff, lcoe, num_mirrors)
    return total_annual_efficiency / valid_days, annual_cost / total_annual_energy_kwh, len(x)


# --- 3. Define the Two Fitness Functions ---
def fitness_func_efficiency(ga_instance, solution, solution_idx):
    """Goal: Maximize the optical percentage."""
    eff, lcoe, num = calculate_annual_metrics(*solution)
    return eff if eff > 0 else 0.0001

def fitness_func_lcoe(ga_instance, solution, solution_idx):
    """Goal: Minimize the dollar cost."""
    eff, lcoe, num = calculate_annual_metrics(*solution)
    return 1.0 / lcoe if lcoe > 0 else 0.0001

# --- 4. Run the Dual Optimizations ---
gene_space = [{'low': 50, 'high': 300}, {'low': 5, 'high': 20}, {'low': 1, 'high': 2}, {'low': 0.1, 'high': 0.5}]
# Lowered population/generations slightly so both runs complete faster
ga_args = {
    'num_generations': 15, 'num_parents_mating': 10, 'sol_per_pop': 20, 
    'num_genes': 4, 'gene_space': gene_space, 'mutation_probability': 0.2, 'suppress_warnings': True
}

overall_start_time = time.time()

print("\n==================================================")
print("RUN 1: OPTIMIZING FOR MAXIMUM ANNUAL EFFICIENCY")
print("==================================================")
eff_start = time.time()
ga_eff = pygad.GA(fitness_func=fitness_func_efficiency, **ga_args)
ga_eff.run()


import matplotlib.pyplot as plt

# --- Generate Figure 6(a): GA Convergence ---
plt.rcParams.update({"font.family": "serif", "figure.dpi": 300})
fig6a, ax = plt.subplots(figsize=(5, 5))

# Extract fitness history and convert to percentage
ga_fitness_history = [v * 100 for v in ga_eff.best_solutions_fitness]

# Plot the convergence curve
ax.plot(ga_fitness_history, color='black', linewidth=2, label=r"Run 1: $\eta$")
ax.set_xlabel("Generation")
ax.set_ylabel(r"Mean $\eta$ (%)")
ax.set_title("(a) GA convergence")
ax.grid(True, linestyle='--', alpha=0.5)

# Optional: Add a shaded region like in the paper
ax.fill_between(range(len(ga_fitness_history)), ga_fitness_history, color='#e6e6fa', alpha=0.5)

plt.tight_layout()
plt.savefig("fig6a_ga_convergence.pdf")
print("Saved Figure 6(a): fig6a_ga_convergence.pdf")





sol_eff, _, _ = ga_eff.best_solution()
eff1, lcoe1, num1 = calculate_annual_metrics(*sol_eff)
eff_time = (time.time() - eff_start) / 60

print(f"\n---> BEST EFFICIENCY RESULTS <---")
print(f"Optical Efficiency: {eff1 * 100:.2f}%")
print(f"LCOE Cost: ${lcoe1:.4f} per kWh")
print(f"Dimensions -> Tower: {sol_eff[0]:.2f}m | Mirrors: {sol_eff[1]:.2f}m x {sol_eff[1]*sol_eff[2]:.2f}m")
print(f"Total Mirrors: {num1} | Safety Distance: {sol_eff[3]:.2f}m")
print(f"Time Taken: {eff_time:.2f} minutes")


print("\n==================================================")
print("RUN 2: OPTIMIZING FOR LOWEST FINANCIAL COST (LCOE)")
print("==================================================")
lcoe_start = time.time()
ga_lcoe = pygad.GA(fitness_func=fitness_func_lcoe, **ga_args)
ga_lcoe.run()
sol_lcoe, _, _ = ga_lcoe.best_solution()
eff2, lcoe2, num2 = calculate_annual_metrics(*sol_lcoe)
lcoe_time = (time.time() - lcoe_start) / 60

print(f"\n---> BEST FINANCIAL RESULTS (LCOE) <---")
print(f"Optical Efficiency: {eff2 * 100:.2f}%")
print(f"LCOE Cost: ${lcoe2:.4f} per kWh")
print(f"Dimensions -> Tower: {sol_lcoe[0]:.2f}m | Mirrors: {sol_lcoe[1]:.2f}m x {sol_lcoe[1]*sol_lcoe[2]:.2f}m")
print(f"Total Mirrors: {num2} | Safety Distance: {sol_lcoe[3]:.2f}m")
print(f"Time Taken: {lcoe_time:.2f} minutes")

total_time = (time.time() - overall_start_time) / 60
print(f"\nAll simulations complete. Total standard GA runtime: {total_time:.2f} minutes.")