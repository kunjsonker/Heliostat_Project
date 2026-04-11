import numpy as np
import pandas as pd
import pygad
import time
import matplotlib.pyplot as plt
from layout_generator import generate_radial_staggered, check_collisions
from efficiency_physics import calculate_efficiencies
from solar_physics import calculate_sun_angles

# --- 1. Load and Sample the Dataset ---
print("Loading annual weather data...")
# Ensure the CSV path is correct for your local environment
try:
    df = pd.read_csv('solar-measurementspakistanquettawb-esmapqc.csv', low_memory=False)
    df['time'] = pd.to_datetime(df['time'])
    daylight_data = df[df['dni'] > 0].copy()

    # 12 representative days for the year (15th of every month at noon)
    annual_sample = daylight_data[(daylight_data['time'].dt.day == 15) & 
                                  (daylight_data['time'].dt.hour == 12)].copy()
    annual_sample['day_of_year'] = annual_sample['time'].dt.dayofyear
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    exit()

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
    
    # Generate layout
    x, y = generate_radial_staggered(TH, diagonal, DS)
    
    mirror_area = LH * width
    # Estimated power to scale the field size to a 50MW target
    power_per_mirror = 858 * 0.88 * mirror_area * 0.82 
    target_mirrors = int(50000000 / power_per_mirror) 
    
    if target_mirrors <= len(x):
        x = x[:target_mirrors]
        y = y[:target_mirrors]
    else:
        return 0, 0, 0
        
    if not check_collisions(x, y, LH, WR, DS) or len(x) == 0:
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
        
        # Simple energy estimation: 8 hours of sun per day, 30 days per month
        total_annual_energy_kwh += ((row['dni'] * total_glass_area * mean_eff * 0.88) / 1000 * 8) * 30
        
    if total_annual_energy_kwh == 0 or valid_days == 0:
        return 0, 0, 0
        
    return total_annual_efficiency / valid_days, annual_cost / total_annual_energy_kwh, len(x)

# --- 3. Define Fitness Functions ---
def fitness_func_efficiency(ga_instance, solution, solution_idx):
    eff, lcoe, num = calculate_annual_metrics(*solution)
    return float(eff) if eff > 0 else 0.0001

def fitness_func_lcoe(ga_instance, solution, solution_idx):
    eff, lcoe, num = calculate_annual_metrics(*solution)
    return 1.0 / lcoe if lcoe > 0 else 0.0001

# --- 4. Optimization Settings ---
gene_space = [
    {'low': 50, 'high': 300},  # Tower Height (TH)
    {'low': 5, 'high': 20},    # Mirror Length (LH)
    {'low': 1, 'high': 2},     # Width Ratio (WR)
    {'low': 0.1, 'high': 0.5}  # Safety Distance (DS)
]

ga_args = {
    'num_generations': 15, 
    'num_parents_mating': 10, 
    'sol_per_pop': 20, 
    'num_genes': 4, 
    'gene_space': gene_space, 
    'mutation_probability': 0.2, 
    'suppress_warnings': True
}

overall_start_time = time.time()

# --- RUN 1: EFFICIENCY ---
print("\n" + "="*50)
print("RUN 1: OPTIMIZING FOR MAXIMUM ANNUAL EFFICIENCY")
print("="*50)
eff_start = time.time()
ga_eff = pygad.GA(fitness_func=fitness_func_efficiency, **ga_args)
ga_eff.run()

# --- PLOTTING CONVERGENCE ---
plt.rcParams.update({"font.family": "serif", "figure.dpi": 200})
# FIX: Unpack the tuple into fig and ax
fig, ax = plt.subplots(figsize=(6, 5))

ga_fitness_history = [v * 100 for v in ga_eff.best_solutions_fitness]

ax.plot(ga_fitness_history, color='black', linewidth=2, label=r"Run 1: $\eta$")
ax.set_xlabel("Generation")
ax.set_ylabel(r"Mean $\eta$ (%)")
ax.set_title("(a) GA Convergence - Optical Efficiency")
ax.grid(True, linestyle='--', alpha=0.5)
ax.fill_between(range(len(ga_fitness_history)), ga_fitness_history, color='#e6e6fa', alpha=0.5)

plt.tight_layout()
plt.savefig("ga_convergence.pdf")
print("Saved ga_convergence.pdf")

# Results for Run 1
sol_eff, _, _ = ga_eff.best_solution()
eff1, lcoe1, num1 = calculate_annual_metrics(*sol_eff)
eff_time = (time.time() - eff_start) / 60

print(f"\n---> BEST EFFICIENCY RESULTS <---")
print(f"Optical Efficiency: {eff1 * 100:.2f}%")
print(f"LCOE Cost: ${lcoe1:.4f} per kWh")
print(f"Dimensions -> Tower: {sol_eff[0]:.2f}m | Mirrors: {sol_eff[1]:.2f}m x {sol_eff[1]*sol_eff[2]:.2f}m")
print(f"Total Mirrors: {num1} | Safety Distance: {sol_eff[3]:.2f}m")
print(f"Time Taken: {eff_time:.2f} minutes")

# --- RUN 2: LCOE ---
print("\n" + "="*50)
print("RUN 2: OPTIMIZING FOR LOWEST FINANCIAL COST (LCOE)")
print("="*50)
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


# ============================================================
# PLOT: GA Layout Figure (a) and (b) — matching plotting.py style
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

def _scatter_field(ax, x, y, title, cmap="RdYlGn", cbar_label="Efficiency (%)"):
    positions = np.column_stack([x, y])
    dist = np.sqrt(x**2 + y**2)
    # Normalize distance: inner=high value (green), outer=low value (red)
    norm_dist = 1 - (dist / dist.max())
    sc = ax.scatter(x, y, c=norm_dist, cmap=cmap, s=6,
                    vmin=0, vmax=1, linewidths=0)
    plt.colorbar(sc, ax=ax, label=cbar_label, fraction=0.04, pad=0.04)
    ax.set_title(title, pad=6)
    ax.set_xlabel("Distance from Tower (m)")
    ax.set_ylabel("Distance from Tower (m)")
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.axvline(0, color="k", lw=0.4, ls="--")
    # Cardinal labels
    r = dist.max() * 0.92
    for txt, (ex, ey) in [("N",(0,1)), ("S",(0,-1)), ("E",(1,0)), ("W",(-1,0))]:
        ax.text(ex*r, ey*r, txt, ha="center", va="center",
                fontsize=7, color="gray")
    # Tower marker
    ax.plot(0, 0, marker="*", color="darkred", markersize=12,
            markeredgecolor="black", markeredgewidth=0.5,
            label="Tower / Receiver", zorder=5)
    ax.legend(fontsize=7, loc="upper right")

# --- Generate GA Eff layout ---
diag_eff = np.sqrt(sol_eff[1]**2 + (sol_eff[1]*sol_eff[2])**2)
x_eff, y_eff_coords = generate_radial_staggered(sol_eff[0], diag_eff, sol_eff[3])
mirror_area_eff = sol_eff[1] * (sol_eff[1]*sol_eff[2])
ppm_eff = 858 * 0.88 * mirror_area_eff * 0.82
tm_eff = int(50000000 / ppm_eff)
if tm_eff <= len(x_eff):
    x_eff, y_eff_coords = x_eff[:tm_eff], y_eff_coords[:tm_eff]

# --- Generate GA LCOE layout ---
diag_lcoe = np.sqrt(sol_lcoe[1]**2 + (sol_lcoe[1]*sol_lcoe[2])**2)
x_lc, y_lc = generate_radial_staggered(sol_lcoe[0], diag_lcoe, sol_lcoe[3])
mirror_area_lc = sol_lcoe[1] * (sol_lcoe[1]*sol_lcoe[2])
ppm_lc = 858 * 0.88 * mirror_area_lc * 0.82
tm_lc = int(50000000 / ppm_lc)
if tm_lc <= len(x_lc):
    x_lc, y_lc = x_lc[:tm_lc], y_lc[:tm_lc]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

_scatter_field(
    axes[0], x_eff, y_eff_coords,
    f"GA-opt Efficiency Radial Staggered\n"
    f"TH={sol_eff[0]:.1f}m, LH={sol_eff[1]:.2f}m×{sol_eff[1]*sol_eff[2]:.2f}m, N={len(x_eff)}"
)
_scatter_field(
    axes[1], x_lc, y_lc,
    f"GA-opt LCOE Radial Staggered\n"
    f"TH={sol_lcoe[0]:.1f}m, LH={sol_lcoe[1]:.2f}m×{sol_lcoe[1]*sol_lcoe[2]:.2f}m, N={len(x_lc)}"
)

fig.suptitle("GA-Optimized Radial Staggered Layouts", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("ga_layout.pdf", bbox_inches="tight")
plt.show()
print("Saved ga_layout.pdf")