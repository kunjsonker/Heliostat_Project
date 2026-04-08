import numpy as np
import pandas as pd
import pygad
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from layout_generator import generate_radial_staggered, check_collisions
from efficiency_physics import calculate_efficiencies
from solar_physics import calculate_sun_angles

# --- 1. Load Data & Synchronized Physics Engine ---
print("Loading weather data...")
df = pd.read_csv('solar-measurementspakistanquettawb-esmapqc.csv', low_memory=False)
df['time'] = pd.to_datetime(df['time'])
daylight_data = df[df['dni'] > 0].copy()

# Standardizing on 12 representative monthly days
annual_sample = daylight_data[(daylight_data['time'].dt.day == 15) & (daylight_data['time'].dt.hour == 12)].copy()
annual_sample['day_of_year'] = annual_sample['time'].dt.dayofyear

def calculate_plant_cost(TH, LH, WR, num_mirrors, field_radius):
    # Constants from Equations 12, 13, 14
    tower_cost = 3000000 * np.exp(0.0113 * TH) 
    heliostat_cost = (LH * (LH * WR)) * num_mirrors * 150  
    land_cost = (np.pi * (field_radius ** 2)) * 5 
    
    # Capital Recovery Factor (7% interest, 25 years)
    annual_cost = (tower_cost + heliostat_cost + land_cost) * ((0.07 * (1 + 0.07)**25) / (((1 + 0.07)**25) - 1))
    return annual_cost, (LH * (LH * WR)) * num_mirrors

def calculate_annual_metrics(TH, LH, WR, DS):
    width = LH * WR
    diagonal = np.sqrt(LH**2 + width**2)
    x, y = generate_radial_staggered(TH, diagonal, DS)
    
    # Static collision check
    if not check_collisions(x, y, LH, WR, DS) or len(x) == 0:
        return 0, 0

    # --- FIX 1: Enforce 50 MW Capacity Limit ---
    mirror_area = LH * width
    # Estimate power per mirror using Vernal Equinox nominal DNI (858) and baseline efficiencies
    power_per_mirror = 858 * 0.88 * mirror_area * 0.82 
    target_mirrors = int(50000000 / power_per_mirror) # Target 50,000,000 Watts
    
    if target_mirrors <= len(x):
        # Slice the array to cap the plant at exactly 50 MW
        x = x[:target_mirrors]
        y = y[:target_mirrors]
    else:
        # Reject layout if it cannot reach 50 MW
        return 0, 0
    # -------------------------------------------
        
    field_radius = np.max(np.sqrt(x**2 + y**2)) + diagonal
    annual_cost, total_glass_area = calculate_plant_cost(TH, LH, WR, len(x), field_radius)
    
    total_annual_efficiency = 0
    total_annual_energy_kwh = 0
    valid_days = 0
    
    for _, row in annual_sample.iterrows():
        sun_elevation, sun_azimuth = calculate_sun_angles(row['day_of_year'], 12.0)
        if sun_elevation <= 0: continue
            
        # --- FIX 2: Use actual sun azimuth instead of hardcoded 180 ---
        _, _, tot_eff = calculate_efficiencies(x, y, TH, sun_elevation, sun_azimuth)
        
        # Synchronized Constants: 0.97 Blocking, 0.88 Reflectivity
        mean_eff = np.mean(tot_eff) * 0.97 
        total_annual_efficiency += mean_eff
        valid_days += 1
        
        # Annual energy yield estimate (8 hours/day, 30 days/month)
        total_annual_energy_kwh += ((row['dni'] * total_glass_area * mean_eff * 0.88) / 1000 * 8) * 30
        
    if total_annual_energy_kwh == 0 or valid_days == 0:
        return 0, 0
        
    return total_annual_efficiency / valid_days, annual_cost / total_annual_energy_kwh

# --- 2. GENERATE TRAINING DATA FOR MACHINE LEARNING ---
print("\n--- PHASE 1: Training the Machine Learning Surrogate Model ---")
print("Mining for 150 perfectly safe layouts to teach the AI...")

# --- FIX 3: Start the master timer for the entire script ---
total_ml_start_time = time.time() 

X_train, y_eff, y_lcoe = [], [], []
target_valid_samples = 150
total_attempts = 0

while len(X_train) < target_valid_samples:
    total_attempts += 1
    TH = np.random.uniform(50, 300)
    LH = np.random.uniform(5, 20)
    WR = np.random.uniform(1, 2)
    DS = np.random.uniform(0.1, 0.5)
    
    eff, lcoe = calculate_annual_metrics(TH, LH, WR, DS)
    if eff > 0:
        X_train.append([TH, LH, WR, DS])
        y_eff.append(eff)
        y_lcoe.append(lcoe)
        if len(X_train) % 10 == 0:
            print(f"Data Mining Progress: {len(X_train)}/{target_valid_samples} safe layouts found...")

print(f"\nData generation complete! It took {total_attempts} random guesses to find 150 safe layouts.")

# --- 3. TRAIN THE RANDOM FOREST MODELS ---
print("Training Random Forest AI models...")
rf_efficiency = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_eff)
rf_lcoe = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_lcoe)
print("Models trained successfully!")

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
ga_args = {
    'num_generations': 50, 'num_parents_mating': 15, 'sol_per_pop': 50,
    'num_genes': 4, 'gene_space': gene_space, 'mutation_probability': 0.2, 'suppress_warnings': True
}

# Efficiency Optimization
ga_eff = pygad.GA(fitness_func=ml_fitness_func_efficiency, **ga_args)
ga_eff.run()
sol_eff, _, _ = ga_eff.best_solution()

# LCOE Optimization
ga_lcoe = pygad.GA(fitness_func=ml_fitness_func_lcoe, **ga_args)
ga_lcoe.run()
sol_lcoe, _, _ = ga_lcoe.best_solution()

# --- 6. OUTPUT RESULTS & FIGURES ---
final_eff = rf_efficiency.predict([sol_eff])[0] * 100
final_lcoe = rf_lcoe.predict([sol_lcoe])[0]

print("\n" + "="*50)
print("FINAL ML PREDICTION RESULTS")
print("="*50)
print(f"Efficiency Opt: TH={sol_eff[0]:.2f}m | LH={sol_eff[1]:.2f}m | Eff={final_eff:.2f}%")
print(f"LCOE Opt:       TH={sol_lcoe[0]:.2f}m | LH={sol_lcoe[1]:.2f}m | LCOE=${final_lcoe:.3f}/kWh")

plt.rcParams.update({"font.family": "serif", "figure.dpi": 300})

# Figure: Convergence (Paper Fig 6c)
fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.plot([v * 100 for v in ga_eff.best_solutions_fitness], color="#1f77b4", label="Efficiency")
ax1.set_xlabel("Generation"); ax1.set_ylabel("Mean Efficiency (%)", color="#1f77b4")
ax2 = ax1.twinx()
ax2.plot([1.0/v for v in ga_lcoe.best_solutions_fitness], color="#2ca02c", linestyle="--", label="LCOE")
ax2.set_ylabel("LCOE ($/kWh)", color="#2ca02c")
plt.savefig("ml_convergence.pdf")

# Figure: Pareto (Paper Fig 8b)
fig2, ax = plt.subplots(figsize=(6, 5))
ax.scatter([e * 100 for e in y_eff], y_lcoe, alpha=0.3, color="gray", label="Samples")
ax.scatter(final_eff, rf_lcoe.predict([sol_eff])[0], color="blue", marker="*", s=200, label="Max Eff")
ax.scatter(rf_efficiency.predict([sol_lcoe])[0]*100, final_lcoe, color="green", marker="*", s=200, label="Min LCOE")
ax.set_xlabel("Efficiency (%)"); ax.set_ylabel("LCOE ($/kWh)")
ax.legend(); plt.savefig("pareto_plot.pdf")

# Figure: Final Layout (Paper Fig 4 Right)
opt_diag = np.sqrt(sol_lcoe[1]**2 + (sol_lcoe[1]*sol_lcoe[2])**2)
x_ml, y_ml = generate_radial_staggered(sol_lcoe[0], opt_diag, sol_lcoe[3])

# Safely slice layout to 50 MW capacity for plotting
opt_mirror_area = sol_lcoe[1] * (sol_lcoe[1] * sol_lcoe[2])
opt_power_per_mirror = 858 * 0.88 * opt_mirror_area * 0.82 
opt_target_mirrors = int(50000000 / opt_power_per_mirror) 
if opt_target_mirrors <= len(x_ml):
    x_ml, y_ml = x_ml[:opt_target_mirrors], y_ml[:opt_target_mirrors]

fig3, ax3 = plt.subplots(figsize=(6, 6))
dist = np.sqrt(x_ml**2 + y_ml**2)
ax3.scatter(x_ml, y_ml, c=dist, cmap="RdYlGn", s=6)
ax3.plot(0, 0, "k^", ms=10)
ax3.set_title(f"ML-GA LCOE-Optimized Layout (N={len(x_ml)})")
plt.savefig("ml_layout.pdf", bbox_inches="tight")

print("\nAll figures saved: ml_convergence.pdf, pareto_plot.pdf, ml_layout.pdf")

# --- FIX 3 (continued): Calculate and print the true total runtime ---
total_time_mins = (time.time() - total_ml_start_time) / 60
print(f"\nTotal System Runtime (Data Mining + AI Training + Optimization): {total_time_mins:.2f} minutes")