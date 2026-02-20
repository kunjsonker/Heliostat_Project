import pandas as pd

# 1. Load the raw dataset
print("Loading dataset...")
file_path = 'solar-measurementspakistanquettawb-esmapqc.csv'
df = pd.read_csv(file_path, low_memory=False)

# 2. Convert the 'time' column to actual datetime objects
# This makes it easy to filter by specific months, days, or hours later
print("Formatting timestamps...")
df['time'] = pd.to_datetime(df['time'])

# 3. Filter out the nighttime data (where DNI is 0)
# We only want to analyze the layout when the sun is providing energy
print("Filtering nighttime data...")
daylight_data = df[df['dni'] > 0].copy()

# 4. Display the results to confirm it worked
print("\n--- Data Preprocessing Complete ---")
print(f"Original data rows: {len(df)}")
print(f"Daylight data rows: {len(daylight_data)}")
print(f"We successfully removed {len(df) - len(daylight_data)} rows of useless nighttime data.")

# Show the first 5 rows of our clean data (Time, DNI, and Temperature)
print("\nFirst 5 rows of clean daylight data:")
print(daylight_data[['time', 'dni', 'air_temperature']].head())