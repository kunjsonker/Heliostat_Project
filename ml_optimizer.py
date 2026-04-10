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
valid_layouts = 0
target_valid_samples = 150
total_attempts = 0

# --- FIX 4: The AI Blindspot Fix ---
while valid_layouts < target_valid_samples:
    total_attempts += 1
    TH = np.random.uniform(50, 300)
    LH = np.random.uniform(5, 20)
    WR = np.random.uniform(1, 2)
    DS = np.random.uniform(0.1, 0.5)
    
    eff, lcoe = calculate_annual_metrics(TH, LH, WR, DS)
    
    # Record EVERY guess so the AI learns the physical boundaries of the field
    X_train.append([TH, LH, WR, DS])
    
    if eff > 0:
        # Successful layout
        y_eff.append(eff)
        y_lcoe.append(lcoe)
        valid_layouts += 1
        if valid_layouts % 10 == 0:
            print(f"Data Mining Progress: {valid_layouts}/{target_valid_samples} safe layouts found...")
    else:
        # Failed layout (collided or couldn't reach 50 MW) -> Give the AI a penalty
        y_eff.append(0.0001)
        y_lcoe.append(9999.0)

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
# --- 6. OUTPUT RESULTS & FIGURES ---
final_eff = rf_efficiency.predict([sol_eff])[0] * 100
final_lcoe = rf_lcoe.predict([sol_lcoe])[0]

# Calculate the exact number of mirrors used for the 50 MW target for both optimal solutions
def get_mirror_count(LH, WR):
    mirror_area = LH * (LH * WR)
    power_per_mirror = 858 * 0.88 * mirror_area * 0.82 
    return int(50000000 / power_per_mirror)

mirrors_eff = get_mirror_count(sol_eff[1], sol_eff[2])
mirrors_lcoe = get_mirror_count(sol_lcoe[1], sol_lcoe[2])

print("\n" + "="*60)
print("FINAL ML PREDICTION RESULTS")
print("="*60)
print(f"Efficiency Opt: TH={sol_eff[0]:.2f}m | LH={sol_eff[1]:.2f}m | Mirrors={mirrors_eff} | Eff={final_eff:.2f}%")
print(f"LCOE Opt:       TH={sol_lcoe[0]:.2f}m | LH={sol_lcoe[1]:.2f}m | Mirrors={mirrors_lcoe} | LCOE=${final_lcoe:.3f}/kWh")
print("="*60)

plt.rcParams.update({"font.family": "serif", "figure.dpi": 300})

# --- Generate Figure 6(c): ML-GA Convergence ---
fig6c, ax1 = plt.subplots(figsize=(5, 5))

# Plot Efficiency (Run 1) on primary Y axis

eff_history = [v * 100 for v in ga_eff.best_solutions_fitness]
line1 = ax1.plot(eff_history, color="#1f4e79", linewidth=2, label=r"Run 1: $\eta$")
ax1.set_xlabel("Generation")
ax1.set_ylabel(r"Mean $\eta$ (%)", color="#1f4e79")

# Plot LCOE (Run 2) on secondary Y axis
ax2 = ax1.twinx()
# Note: PyGAD maximizes fitness, so we invert the fitness score back to raw LCOE
lcoe_history = [1.0/v for v in ga_lcoe.best_solutions_fitness]
line2 = ax2.plot(lcoe_history, color="#2c3e50", linestyle="--", linewidth=2, label="Run 2: LCOE")
ax2.set_ylabel("LCOE (USD/kWh)", color="#2c3e50")
ax2.tick_params(axis='y', labelcolor="#2c3e50")

# Add combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="lower right")

ax1.set_title("(c) ML-GA convergence")
ax1.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("fig6c_ml_convergence.pdf")

# Figure: Pareto (Paper Fig 8b)
fig2, ax = plt.subplots(figsize=(6, 5))
# Filter out the penalty scores (9999 LCOE) so they don't ruin the Pareto plot scale
valid_effs = [e * 100 for e, l in zip(y_eff, y_lcoe) if l < 9999.0]
valid_lcoes = [l for l in y_lcoe if l < 9999.0]
ax.scatter(valid_effs, valid_lcoes, alpha=0.3, color="gray", label="Samples")
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

# --- Calculate and print the true total runtime ---
total_time_mins = (time.time() - total_ml_start_time) / 60
print(f"\nTotal System Runtime (Data Mining + AI Training + Optimization): {total_time_mins:.2f} minutes")

# --- Figure 8a: Tower Height vs Efficiency & Tower Cost ---
# Filter valid samples for plotting
valid_ths = [x[0] for x, l in zip(X_train, y_lcoe) if l < 9999.0]
valid_effs_8a = [e * 100 for e, l in zip(y_eff, y_lcoe) if l < 9999.0]
tower_costs_8a = [3000000 * np.exp(0.0113 * th) / 1e6 for th in valid_ths] # In Millions USD

fig8a, ax1 = plt.subplots(figsize=(6, 5))
# Sort data by TH for a cleaner line/scatter plot
sorted_indices = np.argsort(valid_ths)
ths_sorted = np.array(valid_ths)[sorted_indices]
effs_sorted = np.array(valid_effs_8a)[sorted_indices]
costs_sorted = np.array(tower_costs_8a)[sorted_indices]

ax1.scatter(ths_sorted, effs_sorted, color="#1f77b4", alpha=0.4, s=10)
ax1.set_xlabel("Tower Height (m)")
ax1.set_ylabel("Annual Efficiency (%)", color="#1f77b4")
ax1.tick_params(axis='y', labelcolor="#1f77b4")

ax2 = ax1.twinx()
ax2.plot(ths_sorted, costs_sorted, color="red", linewidth=2, label="Tower Cost")
ax2.set_ylabel("Tower Capital Cost ($M)", color="red")
ax2.tick_params(axis='y', labelcolor="red")

plt.title("Fig 8a: Tower Height impact on Efficiency and Cost")
plt.savefig("fig8a_tower_impact.pdf", bbox_inches="tight")

# --- Figure 8c: Capital Cost Breakdown Comparison ---
def get_cost_breakdown(solution):
    TH, LH, WR, DS = solution
    width = LH * WR
    diagonal = np.sqrt(LH**2 + width**2)
    # Ensure we use max_rings=60 as in the GA optimizer
    x, y = generate_radial_staggered(TH, diagonal, DS, max_rings=60)
    
    # Apply 50MW Slicing logic
    mirror_area = LH * width
    # Physics constants must match calculate_annual_metrics
    power_per_mirror = 858 * 0.88 * mirror_area * 0.82 
    target_mirrors = int(50000000 / power_per_mirror)
    
    # --- FIXED: Slicing both x AND y ---
    if target_mirrors < len(x): 
        x = x[:target_mirrors]
        y = y[:target_mirrors]
    
    # If the layout is too small to even reach 50 MW, don't crash
    if len(x) == 0: return 0, 0, 0
    
    field_radius = np.max(np.sqrt(x**2 + y**2)) + diagonal
    num_mirrors = len(x)
    
    t_cost = 3000000 * np.exp(0.0113 * TH) 
    h_cost = (LH * width) * num_mirrors * 150  
    l_cost = (np.pi * (field_radius ** 2)) * 5 
    return t_cost/1e6, h_cost/1e6, l_cost/1e6 # Return in Millions USD

# Get breakdowns for both solutions
eff_costs = get_cost_breakdown(sol_eff)
lcoe_costs = get_cost_breakdown(sol_lcoe)

labels = ['Tower', 'Heliostat', 'Land']
x_pos = np.arange(len(labels))

fig8c, ax = plt.subplots(figsize=(7, 5))
ax.bar(x_pos - 0.2, eff_costs, 0.4, label='Eff-Opt Layout', color='#1f77b4')
ax.bar(x_pos + 0.2, lcoe_costs, 0.4, label='LCOE-Opt Layout', color='#2ca02c')

ax.set_ylabel('Capital Cost (Millions USD)')
ax.set_title('Fig 8c: Capital Cost Breakdown Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.legend()

plt.savefig("fig8c_cost_breakdown.pdf", bbox_inches="tight")
print("Saved additional figures: fig8a_tower_impact.pdf and fig8c_cost_breakdown.pdf")