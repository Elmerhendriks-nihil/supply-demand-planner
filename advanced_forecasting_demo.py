"""Test script demonstrating advanced forecasting features."""

import pandas as pd
import numpy as np
from datetime import datetime
from src import DemandPlanner

print("=" * 80)
print("ADVANCED FORECASTING FEATURES DEMO")
print("=" * 80)

# Generate sample data with trend and noise
dates = pd.date_range(start='2024-01-01', end='2026-01-31', freq='D')
np.random.seed(42)

# Create data with upward trend
base_demand = 100
trend_rate = 0.0001  # Slight upward trend
quantities = [
    max(0, base_demand + (i * trend_rate * base_demand) + np.random.normal(0, 10))
    for i in range(len(dates))
]

historical_data = pd.DataFrame({
    'date': dates,
    'quantity': quantities
})

print(f"\nLoaded {len(historical_data)} days of historical data")
print(f"Date Range: {dates[0].date()} to {dates[-1].date()}")
print(f"Average Demand: {historical_data['quantity'].mean():.1f} units/day")
print(f"Volatility (Std Dev): {historical_data['quantity'].std():.1f} units")

# Initialize planner
planner = DemandPlanner()
planner.load_historical_sales(historical_data)

# ============================================================================
# FEATURE 1: TREND DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("1. TREND DETECTION")
print("=" * 80)

trend = planner.detect_trend(periods=90)

print(f"\nTrend Direction: {trend['trend'].upper()}")
print(f"Trend Strength: {trend['trend_strength']}")
print(f"Growth Rate: {trend['growth_rate']:+.2f}%")
print(f"Recent 45-day Average: {trend['recent_avg']:.1f} units/day")
print(f"Earlier 45-day Average: {trend['earlier_avg']:.1f} units/day")
print(f"Forecast Impact: {trend['forecast_impact']}")

# ============================================================================
# FEATURE 2: EXPONENTIAL SMOOTHING FORECAST
# ============================================================================
print("\n" + "=" * 80)
print("2. EXPONENTIAL SMOOTHING FORECAST")
print("=" * 80)

print("\n(a) Conservative Smoothing (alpha=0.2):")
forecast_conservative = planner.forecast_exponential_smoothing(
    periods=12, alpha=0.2, trend_beta=0.1
)
print("\nWeek | Date       | Forecast | Trend Component")
print("-" * 50)
for _, row in forecast_conservative.head(6).iterrows():
    date_str = row['date'].strftime('%Y-%m-%d')
    forecast = row['forecast']
    trend_comp = row['trend_component']
    print(f"{int(row['period']):4d} | {date_str} | {forecast:8.0f} | {trend_comp:+7.1f}")

print("\n(b) Responsive Smoothing (alpha=0.5):")
forecast_responsive = planner.forecast_exponential_smoothing(
    periods=12, alpha=0.5, trend_beta=0.2
)
print("\nWeek | Date       | Forecast | Trend Component")
print("-" * 50)
for _, row in forecast_responsive.head(6).iterrows():
    date_str = row['date'].strftime('%Y-%m-%d')
    forecast = row['forecast']
    trend_comp = row['trend_component']
    print(f"{int(row['period']):4d} | {date_str} | {forecast:8.0f} | {trend_comp:+7.1f}")

# ============================================================================
# FEATURE 3: FORECAST ACCURACY METRICS
# ============================================================================
print("\n" + "=" * 80)
print("3. FORECAST ACCURACY METRICS")
print("=" * 80)

# Get last 30 days of actual data
last_30_actual = historical_data.tail(30)['quantity'].values

# Create different forecasts to evaluate
actual_values = list(last_30_actual)

# Forecast 1: Simple average
simple_forecast = [historical_data['quantity'].mean()] * 30

# Forecast 2: Exponential smoothing
exp_forecast_values = []
alpha = 0.3
level = last_30_actual[0]
for actual in last_30_actual:
    level = alpha * actual + (1 - alpha) * level
    exp_forecast_values.append(level)

# Compare accuracy
print("\n(a) SIMPLE AVERAGE FORECAST:")
metrics_simple = planner.calculate_forecast_error(actual_values, simple_forecast)
print(f"  Mean Absolute Error (MAE): {metrics_simple['mae']:.2f} units")
print(f"  Root Mean Squared Error (RMSE): {metrics_simple['rmse']:.2f} units")
print(f"  Mean Absolute Percentage Error (MAPE): {metrics_simple['mape']:.2f}%")
print(f"  Forecast Bias: {metrics_simple['bias']:+.2f} (Avg forecast error)")
print(f"  Quality Rating: {metrics_simple['forecast_quality']}")

print("\n(b) EXPONENTIAL SMOOTHING FORECAST:")
metrics_exp = planner.calculate_forecast_error(actual_values, exp_forecast_values)
print(f"  Mean Absolute Error (MAE): {metrics_exp['mae']:.2f} units")
print(f"  Root Mean Squared Error (RMSE): {metrics_exp['rmse']:.2f} units")
print(f"  Mean Absolute Percentage Error (MAPE): {metrics_exp['mape']:.2f}%")
print(f"  Forecast Bias: {metrics_exp['bias']:+.2f} (Avg forecast error)")
print(f"  Quality Rating: {metrics_exp['forecast_quality']}")

# Compare
print("\n(c) COMPARISON:")
improvement = (
    (metrics_simple['mape'] - metrics_exp['mape']) / metrics_simple['mape'] * 100
)
print(f"  Exponential Smoothing is {abs(improvement):.1f}% ", end="")
print("better" if improvement > 0 else "worse")
print(f"  (MAPE: {metrics_simple['mape']:.2f}% → {metrics_exp['mape']:.2f}%)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print(f"""
✓ Trend Analysis:
  - Market is {trend['trend']} with {trend['growth_rate']:+.2f}% growth
  - {trend['forecast_impact']}

✓ Forecast Methods:
  - Exponential Smoothing captures recent patterns better
  - Use alpha=0.2 for stable markets, alpha=0.5 for volatile markets
  - Add trend_beta for markets with sustained growth/decline

✓ Accuracy Metrics:
  - Current best method: {metrics_exp['forecast_quality']} accuracy (MAPE: {metrics_exp['mape']:.2f}%)
  - Bias: {metrics_exp['bias']:+.2f} units (positive = overforecasting)
  - Use these metrics to choose between forecasting methods

✓ Recommendations:
  1. For trending markets, use exponential_smoothing with trend_beta
  2. For seasonal markets, use forecast_with_seasonality
  3. Monitor MAPE monthly and adjust alpha parameters if needed
  4. Compare actual vs forecast to continuously improve accuracy
""")

print("=" * 80)
