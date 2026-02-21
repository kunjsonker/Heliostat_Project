import numpy as np
import pandas as pd
import pygad
import time
from sklearn.ensemble import RandomForestRegressor
from layout_generator import generate_radial_staggered, check_collisions
from efficiency_physics import calculate_efficiencies
from solar_physics import calculate_sun_angles

# --- 1. Load Data & Shared Physics Engine ---
print("Loading weather data...")
df = pd.read_csv('solar-measurementspakistanquettawb-esmapqc.csv', low_memory=False)
df['time'] = pd.to_datetime(df['time'])
daylight_data = df[df['dni'] > 0].copy()
annual_sample = daylight_data[(daylight_data['time'].dt.day == 15) & (daylight_data['time'].dt.hour == 12)].copy()
annual_sample['day_of_year'] = annual_sample['time'].dt.dayofyear

def calculate_plant_cost(TH, LH, WR, num_mirrors, field_radius):
    tower_cost = 3000000 * np.exp(0.0113 * TH) 
    heliostat_cost = (LH * (LH * WR)) * num_mirrors * 150  
    land_cost = (np.pi * (field_radius ** 2)) * 5 
    annual_cost = (tower_cost + heliostat_cost + land_cost) * ((0.07 * (1 + 0.07)**25) / (((1 + 0.07)**25) - 1))
    return annual_cost, (LH * (LH * WR)) * num_mirrors

def calculate_annual_metrics(TH, LH, WR, DS):
    width = LH * WR
    diagonal = np.sqrt(LH**2 + width**2)
    x, y = generate_radial_staggered(TH, diagonal, DS)
    
    if not check_collisions(x, y, LH, WR, DS) or len(x) == 0:
        return 0, 0
        
    field_radius = np.max(np.sqrt(x**2 + y**2)) + diagonal
    annual_cost, total_glass_area = calculate_plant_cost(TH, LH, WR, len(x), field_radius)
    
    total_annual_efficiency = 0
    total_annual_energy_kwh = 0
    valid_days = 0
    
    for _, row in annual_sample.iterrows():
        sun_elevation, _ = calculate_sun_angles(row['day_of_year'], 12.0)
        if sun_elevation <= 0: continue
            
        _, _, tot_eff = calculate_efficiencies(x, y, TH, sun_elevation, 180)
        mean_eff = np.mean(tot_eff) * 0.97
        total_annual_efficiency += mean_eff
        valid_days += 1
        total_annual_energy_kwh += ((row['dni'] * total_glass_area * mean_eff * 0.9) / 1000 * 8) * 30
        
    if total_annual_energy_kwh == 0 or valid_days == 0:
        return 0, 0
        
    return total_annual_efficiency / valid_days, annual_cost / total_annual_energy_kwh

# --- 2. GENERATE TRAINING DATA FOR MACHINE LEARNING ---
print("\n--- PHASE 1: Training the Machine Learning Surrogate Model ---")
print("Mining for 150 perfectly safe layouts to teach the AI... This will take a few minutes.")

X_train = [] # Features: [TH, LH, WR, DS]
y_eff = []   # Target 1: Efficiency
y_lcoe = []  # Target 2: LCOE

target_valid_samples = 150
total_attempts = 0

# Keep guessing until we have exactly 150 safe, collision-free layouts
while len(X_train) < target_valid_samples:
    total_attempts += 1
    
    TH = np.random.uniform(50, 300)
    LH = np.random.uniform(5, 20)
    WR = np.random.uniform(1, 2)
    DS = np.random.uniform(0.1, 0.5)
    
    eff, lcoe = calculate_annual_metrics(TH, LH, WR, DS)
    
    # Only save valid, non-colliding layouts to teach the ML
    if eff > 0:
        X_train.append([TH, LH, WR, DS])
        y_eff.append(eff)
        y_lcoe.append(lcoe)
        
        # Print progress so you know it hasn't frozen
        if len(X_train) % 10 == 0:
            print(f"Data Mining Progress: {len(X_train)}/{target_valid_samples} safe layouts found...")

print(f"\nData generation complete! It took {total_attempts} random guesses to find 150 safe layouts.")

# --- 3. TRAIN THE RANDOM FOREST MODELS ---
print("Training Random Forest AI models on the verified data...")
rf_efficiency = RandomForestRegressor(n_estimators=100, random_state=42)
rf_lcoe = RandomForestRegressor(n_estimators=100, random_state=42)

rf_efficiency.fit(X_train, y_eff)
rf_lcoe.fit(X_train, y_lcoe)
print("Models trained successfully! Optimization will now run at lightspeed.")

# --- 4. THE ML-ACCELERATED FITNESS FUNCTIONS ---
def ml_fitness_func_efficiency(ga_instance, solution, solution_idx):
    prediction = rf_efficiency.predict([solution])[0]
    return prediction if prediction > 0 else 0.0001

def ml_fitness_func_lcoe(ga_instance, solution, solution_idx):
    prediction = rf_lcoe.predict([solution])[0]
    return 1.0 / prediction if prediction > 0 else 0.0001

# --- 5. RUN THE SUPERCHARGED OPTIMIZATION ---
print("\n--- PHASE 2: ML-Accelerated Genetic Algorithm ---")
gene_space = [{'low': 50, 'high': 300}, {'low': 5, 'high': 20}, {'low': 1, 'high': 2}, {'low': 0.1, 'high': 0.5}]

# We can safely use a large population and high generation count because the ML makes it instant
ga_args = {
    'num_generations': 50,         
    'num_parents_mating': 15, 
    'sol_per_pop': 50,             
    'num_genes': 4, 
    'gene_space': gene_space, 
    'mutation_probability': 0.2, 
    'suppress_warnings': True
}

# Run Efficiency Optimization
start_time = time.time()
ga_eff = pygad.GA(fitness_func=ml_fitness_func_efficiency, **ga_args)
ga_eff.run()
sol_eff, _, _ = ga_eff.best_solution()
eff_time = time.time() - start_time

# Run LCOE Optimization
start_time = time.time()
ga_lcoe = pygad.GA(fitness_func=ml_fitness_func_lcoe, **ga_args)
ga_lcoe.run()
sol_lcoe, _, _ = ga_lcoe.best_solution()
lcoe_time = time.time() - start_time

print("\n==================================================")
print("FINAL ML PREDICTION RESULTS")
print("==================================================")
print(f"Efficiency Optimization completed in just {eff_time:.2f} seconds!")
print(f"Optimal Dimensions -> Tower: {sol_eff[0]:.2f}m | LH: {sol_eff[1]:.2f}m | WR: {sol_eff[2]:.2f} | DS: {sol_eff[3]:.2f}m")

print(f"\nLCOE Optimization completed in just {lcoe_time:.2f} seconds!")
print(f"Optimal Dimensions -> Tower: {sol_lcoe[0]:.2f}m | LH: {sol_lcoe[1]:.2f}m | WR: {sol_lcoe[2]:.2f} | DS: {sol_lcoe[3]:.2f}m")