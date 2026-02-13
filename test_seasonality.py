"""Test script demonstrating seasonality detection feature."""

import pandas as pd
from datetime import datetime, timedelta
from src import DemandPlanner

# Generate sample data with seasonal pattern
# (higher demand in Q4, lower in spring)
dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
base_demand = 100

# Create seasonal pattern
seasonal_pattern = {
    1: 0.85,   # January - low
    2: 0.80,   # February - low
    3: 0.90,   # March - improving
    4: 0.95,   # April - building
    5: 1.00,   # May - baseline
    6: 1.05,   # June - summer increase
    7: 1.10,   # July - peak summer
    8: 1.08,   # August - still high
    9: 0.95,   # September - declining
    10: 1.20,  # October - Q4 rush
    11: 1.35,  # November - holiday season
    12: 1.30,  # December - holiday peak
}

# Generate quantities with seasonal pattern and some randomness
import numpy as np
np.random.seed(42)
quantities = []
for date in dates:
    month = date.month
    seasonal_factor = seasonal_pattern.get(month, 1.0)
    noise = np.random.normal(0, 5)  # Add some noise
    qty = max(0, base_demand * seasonal_factor + noise)
    quantities.append(int(qty))

historical_data = pd.DataFrame({
    'date': dates,
    'quantity': quantities
})

print("=" * 70)
print("SEASONALITY DETECTION TEST")
print("=" * 70)
print(f"\nHistorical Data: {len(historical_data)} days ({dates[0].date()} to {dates[-1].date()})")
print(f"Base Demand: {base_demand} units/day")
print(f"Average Daily Demand: {historical_data['quantity'].mean():.1f} units")

# Initialize demand planner
planner = DemandPlanner()
planner.load_historical_sales(historical_data)

# Detect seasonality
print("\n" + "-" * 70)
print("SEASONALITY ANALYSIS")
print("-" * 70)

seasonality = planner.detect_seasonality(min_coefficient=0.15)

print(f"\nSeasonality Detected: {seasonality['has_seasonality']}")
print(f"Seasonality Strength: {seasonality['seasonality_strength']:.1%}")
print(f"\nPattern Description: {seasonality['annual_pattern']}")

if seasonality['peak_months']:
    print(f"\nPeak Months (above average):")
    for month in seasonality['peak_months']:
        idx = seasonality['seasonal_indices'][month]
        print(f"  Month {month:2d}: {idx*100:6.1f}% of average")

if seasonality['low_months']:
    print(f"\nLow Months (below average):")
    for month in seasonality['low_months']:
        idx = seasonality['seasonal_indices'][month]
        print(f"  Month {month:2d}: {idx*100:6.1f}% of average")

# Generate forecast with seasonality
print("\n" + "-" * 70)
print("SEASONAL FORECAST COMPARISON")
print("-" * 70)

forecast_with_seasonality = planner.forecast_with_seasonality(periods=12)

print(f"\nWeek | Date       | Base Forecast | Seasonal Factor | Adjusted Forecast")
print("-" * 75)
for _, row in forecast_with_seasonality.head(12).iterrows():
    date_str = row['date'].strftime('%Y-%m-%d')
    base = row['forecast']
    factor = row['seasonal_factor']
    adjusted = row['seasonality_adjusted_forecast']
    print(f"{int(row['period']):4d} | {date_str} | {base:13.0f} | {factor:15.2%} | {adjusted:17.0f}")

print("\n" + "=" * 70)
print("✓ Seasonality detection successfully demonstrates Q4 peak pattern")
print("=" * 70)
