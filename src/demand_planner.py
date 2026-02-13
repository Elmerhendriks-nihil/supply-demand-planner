"""Demand forecasting module for supply and demand planning."""

from datetime import timedelta

import pandas as pd


class DemandPlanner:
    """Manages demand forecasting based on historical sales and planned sales."""

    def __init__(self):
        """Initialize the demand planner."""
        self.historical_data = None
        self.forecast_data = None

    def load_historical_sales(self, data: pd.DataFrame):
        """
        Load historical sales data.

        Expected columns: date, quantity, price (optional)
        """
        if "date" not in data.columns or "quantity" not in data.columns:
            raise ValueError("Data must contain 'date' and 'quantity' columns")

        prepared = data.copy()
        prepared["date"] = pd.to_datetime(prepared["date"])
        self.historical_data = prepared.sort_values("date")
        return self.historical_data

    def calculate_average_daily_demand(self, days: int = 30) -> float:
        """Calculate average daily demand from recent sales."""
        if self.historical_data is None:
            raise ValueError("No historical data loaded")

        recent = self.historical_data.tail(days)
        return float(recent["quantity"].mean()) if len(recent) > 0 else 0.0

    def calculate_daily_demand_std(self, days: int = 90) -> float:
        """Calculate daily demand standard deviation from recent sales."""
        if self.historical_data is None:
            raise ValueError("No historical data loaded")

        recent = self.historical_data.tail(days)
        if len(recent) <= 1:
            return 0.0
        return float(recent["quantity"].std(ddof=0))

    def forecast_weekly(
        self, periods: int = 12, ma_weeks: int = 4, start_date: str | pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """
        Generate a weekly forecast using weekly moving average.

        Args:
            periods: Number of weekly periods to forecast
            ma_weeks: Number of historical weeks for moving average

        Returns:
            DataFrame with columns: date, forecast, period
        """
        if self.historical_data is None:
            raise ValueError("No historical data loaded")

        weekly = (
            self.historical_data.set_index("date")["quantity"].resample("W-MON").sum().reset_index()
        )
        if weekly.empty:
            baseline = 0.0
            last_week_date = pd.Timestamp.now().normalize()
        else:
            baseline = float(weekly["quantity"].tail(ma_weeks).mean())
            last_week_date = pd.to_datetime(weekly["date"].max())

        if start_date is None:
            first_forecast_date = last_week_date + timedelta(days=7)
        else:
            # Align explicit start date to the same W-MON bucket end used by forecast periods.
            first_forecast_date = (
                pd.to_datetime(start_date).to_period("W-MON").end_time.normalize()
            )

        forecast_dates = [first_forecast_date + timedelta(days=7 * i) for i in range(0, periods)]
        self.forecast_data = pd.DataFrame(
            {"date": forecast_dates, "forecast": [baseline] * periods, "period": range(1, periods + 1)}
        )
        return self.forecast_data

    def build_demand_plan(self, planned_sales: pd.DataFrame | None = None, periods: int = 12) -> pd.DataFrame:
        """
        Build demand plan using planned sales where available and forecast as fallback.

        Planned sales expected columns: date, planned_quantity
        """
        if self.forecast_data is None:
            self.forecast_weekly(periods=periods)

        demand_plan = self.forecast_data.copy()
        demand_plan["date"] = pd.to_datetime(demand_plan["date"]).dt.normalize()
        first_forecast_date = demand_plan["date"].min()

        if planned_sales is not None and not planned_sales.empty:
            if "date" not in planned_sales.columns or "planned_quantity" not in planned_sales.columns:
                raise ValueError("Planned sales must contain 'date' and 'planned_quantity' columns")

            planned = planned_sales.copy()
            planned["date"] = pd.to_datetime(planned["date"])
            # Align committed/planned sales to the same weekly buckets as forecast (W-MON).
            planned["date"] = planned["date"].dt.to_period("W-MON").dt.end_time.dt.normalize()
            # If committed orders are overdue but still open, treat them as immediate future demand.
            planned.loc[planned["date"] < first_forecast_date, "date"] = first_forecast_date
            planned = planned.groupby("date", as_index=False)["planned_quantity"].sum()
            demand_plan = demand_plan.merge(planned, on="date", how="left")
        else:
            demand_plan["planned_quantity"] = pd.NA

        demand_plan["total_expected_demand"] = demand_plan["planned_quantity"].fillna(demand_plan["forecast"])
        demand_plan["demand_source"] = demand_plan["planned_quantity"].apply(
            lambda x: "planned_sales" if pd.notna(x) else "forecast"
        )
        return demand_plan

    def forecast_simple_moving_average(self, periods: int = 7) -> pd.DataFrame:
        """Backward-compatible wrapper for forecast generation."""
        return self.forecast_weekly(periods=12, ma_weeks=periods)

    def add_planned_sales(self, planned_sales: pd.DataFrame) -> pd.DataFrame:
        """Backward-compatible wrapper to merge planned sales with forecast."""
        return self.build_demand_plan(planned_sales=planned_sales, periods=len(self.forecast_data or []))

    def detect_seasonality(self, min_coefficient: float = 0.2) -> dict:
        """
        Detect seasonal patterns in historical demand data.

        Analyzes month-over-month variation to identify seasonal trends.
        Returns a dictionary with seasonality metrics and the seasonal pattern
        by month for forecast adjustments.

        Args:
            min_coefficient: Minimum seasonality coefficient (0-1) to consider
                           a pattern as significant. Default 0.2 (20% variation).

        Returns:
            Dictionary with keys:
                - 'has_seasonality': Boolean indicating if seasonality detected
                - 'seasonality_strength': Float (0-1) measuring pattern strength
                - 'seasonal_indices': Dict mapping months to adjustment factors
                - 'peak_months': List of months with above-average demand
                - 'low_months': List of months with below-average demand
                - 'annual_pattern': String description of the pattern

        Example:
            planner = DemandPlanner()
            planner.load_historical_sales(df)
            seasonality = planner.detect_seasonality(min_coefficient=0.15)
            if seasonality['has_seasonality']:
                print(f"Peak months: {seasonality['peak_months']}")
        """
        if self.historical_data is None:
            raise ValueError("No historical data loaded")

        data = self.historical_data.copy()
        data["month"] = data["date"].dt.month
        data["year"] = data["date"].dt.year

        # Group by month and year to get monthly totals
        monthly = data.groupby(["year", "month"])["quantity"].sum().reset_index()
        if len(monthly) < 12:
            return {
                "has_seasonality": False,
                "seasonality_strength": 0.0,
                "seasonal_indices": {},
                "peak_months": [],
                "low_months": [],
                "annual_pattern": "Insufficient data (less than 12 months)",
            }

        # Calculate average demand by month across all years
        month_avg = monthly.groupby("month")["quantity"].mean()
        overall_avg = month_avg.mean()

        if overall_avg == 0:
            return {
                "has_seasonality": False,
                "seasonality_strength": 0.0,
                "seasonal_indices": {},
                "peak_months": [],
                "low_months": [],
                "annual_pattern": "No demand recorded",
            }

        # Calculate seasonal indices (ratio of month average to overall average)
        seasonal_indices = (month_avg / overall_avg).to_dict()

        # Calculate seasonality strength using coefficient of variation
        month_avg_array = month_avg.values
        seasonality_strength = float(
            (month_avg_array.std() / month_avg_array.mean()) if month_avg_array.mean() > 0 else 0
        )

        # Identify peak and low months
        threshold_high = 1.0 + (min_coefficient / 2)
        threshold_low = 1.0 - (min_coefficient / 2)

        peak_months = sorted([int(m) for m, idx in seasonal_indices.items() if idx > threshold_high])
        low_months = sorted([int(m) for m, idx in seasonal_indices.items() if idx < threshold_low])

        has_seasonality = seasonality_strength >= min_coefficient

        # Generate pattern description
        if has_seasonality:
            if peak_months and low_months:
                month_names = {
                    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
                }
                peak_str = ", ".join([month_names[m] for m in peak_months])
                low_str = ", ".join([month_names[m] for m in low_months])
                pattern = f"Peak: {peak_str} | Low: {low_str}"
            else:
                pattern = f"Seasonality detected (strength: {seasonality_strength:.1%})"
        else:
            pattern = "No significant seasonality detected"

        return {
            "has_seasonality": has_seasonality,
            "seasonality_strength": seasonality_strength,
            "seasonal_indices": seasonal_indices,
            "peak_months": peak_months,
            "low_months": low_months,
            "annual_pattern": pattern,
        }

    def forecast_with_seasonality(
        self, periods: int = 12, ma_weeks: int = 4, start_date: str | pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """
        Generate forecast with seasonal adjustments if seasonality detected.

        Uses detect_seasonality() to identify patterns and applies monthly
        seasonal indices to the baseline forecast.

        Args:
            periods: Number of weekly periods to forecast
            ma_weeks: Number of historical weeks for moving average
            start_date: Optional start date for forecast

        Returns:
            DataFrame with columns: date, forecast, period, seasonal_factor,
                                   seasonality_adjusted_forecast
        """
        # Generate base forecast
        forecast = self.forecast_weekly(periods=periods, ma_weeks=ma_weeks, start_date=start_date)

        # Detect seasonality
        seasonality = self.detect_seasonality(min_coefficient=0.15)
        forecast["seasonal_factor"] = 1.0

        if seasonality["has_seasonality"]:
            seasonal_indices = seasonality["seasonal_indices"]
            # Apply monthly seasonal factors to weekly forecast
            forecast["month"] = forecast["date"].dt.month
            forecast["seasonal_factor"] = forecast["month"].map(seasonal_indices).fillna(1.0)
        else:
            forecast["seasonal_factor"] = 1.0

        forecast["seasonality_adjusted_forecast"] = forecast["forecast"] * forecast["seasonal_factor"]
        return forecast
    def forecast_exponential_smoothing(
        self, periods: int = 12, alpha: float = 0.3, trend_beta: float | None = None
    ) -> pd.DataFrame:
        """
        Generate forecast using exponential smoothing (Holt's method).

        Better than moving average for trending data. Weights recent observations
        more heavily and can capture trend (upward/downward movement).

        Args:
            periods: Number of periods to forecast
            alpha: Smoothing parameter for level (0-1). Higher = more responsive
                   to recent changes. Default 0.3 is conservative.
            trend_beta: Smoothing parameter for trend (0-1). If None, no trend
                       adjustment. Recommended: 0.1-0.2.

        Returns:
            DataFrame with columns: date, forecast, period, trend_component

        Example:
            planner.load_historical_sales(df)
            forecast = planner.forecast_exponential_smoothing(periods=12, alpha=0.3, trend_beta=0.15)
        """
        if self.historical_data is None:
            raise ValueError("No historical data loaded")

        data = self.historical_data.copy()
        
        # Resample to weekly to match other forecast methods
        weekly = data.set_index("date")["quantity"].resample("W-MON").sum().reset_index()
        
        if len(weekly) < 2:
            return self.forecast_weekly(periods=periods)

        quantities = weekly["quantity"].values
        last_date = pd.to_datetime(weekly["date"].max())

        # Initialize level and trend
        level = float(quantities[0])
        trend = float(quantities[1] - quantities[0]) if len(quantities) > 1 else 0.0

        # Apply exponential smoothing
        smoothed_values = [level]
        trends = [trend]

        for i in range(1, len(quantities)):
            prev_level = level
            level = alpha * quantities[i] + (1 - alpha) * (level + trend)
            
            if trend_beta is not None:
                trend = trend_beta * (level - prev_level) + (1 - trend_beta) * trend
            
            smoothed_values.append(level)
            trends.append(trend)

        # Generate forecast periods
        forecast_dates = [last_date + pd.Timedelta(days=7 * (i + 1)) for i in range(periods)]
        forecast_levels = []

        for period in range(1, periods + 1):
            if trend_beta is not None:
                forecast_value = level + (period * trend)
            else:
                forecast_value = level
            forecast_levels.append(max(0, forecast_value))

        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "forecast": forecast_levels,
            "period": range(1, periods + 1),
            "trend_component": [trend] * periods if trend_beta is not None else [0.0] * periods,
        })

        return forecast_df

    def calculate_forecast_error(
        self, actual: pd.Series | list, forecast: pd.Series | list
    ) -> dict:
        """
        Calculate forecast accuracy metrics.

        Compares actual demand against forecasted values to measure forecast quality.
        Useful for tuning forecast parameters and model selection.

        Args:
            actual: Series or list of actual demand values
            forecast: Series or list of forecasted values (same length as actual)

        Returns:
            Dictionary with metrics:
                - 'mae': Mean Absolute Error
                - 'rmse': Root Mean Squared Error
                - 'mape': Mean Absolute Percentage Error (%)
                - 'mean_actual': Average actual demand
                - 'bias': Forecast bias (positive = overforecast)
                - 'forecast_quality': String rating ('Excellent', 'Good', etc.)

        Example:
            actual = [100, 105, 98, 110]
            forecast = [102, 103, 100, 108]
            metrics = planner.calculate_forecast_error(actual, forecast)
            print(f"MAPE: {metrics['mape']:.1f}%")
        """
        import numpy as np
        
        actual_arr = np.array(actual, dtype=float)
        forecast_arr = np.array(forecast, dtype=float)

        if len(actual_arr) != len(forecast_arr):
            raise ValueError("Actual and forecast arrays must have same length")

        if len(actual_arr) == 0:
            raise ValueError("Arrays cannot be empty")

        # Calculate errors
        errors = actual_arr - forecast_arr
        absolute_errors = np.abs(errors)
        squared_errors = errors ** 2

        # MAE: Mean Absolute Error
        mae = float(np.mean(absolute_errors))

        # RMSE: Root Mean Squared Error
        rmse = float(np.sqrt(np.mean(squared_errors)))

        # MAPE: Mean Absolute Percentage Error
        # Handle zero actual values
        non_zero_mask = actual_arr != 0
        if non_zero_mask.sum() > 0:
            mape = float(
                np.mean(np.abs(errors[non_zero_mask] / actual_arr[non_zero_mask])) * 100
            )
        else:
            mape = float('inf')

        # Bias: Do we tend to over or under forecast?
        bias = float(np.mean(errors))

        # Mean actual demand
        mean_actual = float(np.mean(actual_arr))

        # Quality rating based on MAPE
        if mape <= 5:
            quality = "Excellent"
        elif mape <= 10:
            quality = "Good"
        elif mape <= 15:
            quality = "Fair"
        elif mape <= 25:
            quality = "Poor"
        else:
            quality = "Very Poor"

        return {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2) if mape != float('inf') else mape,
            "mean_actual": round(mean_actual, 2),
            "bias": round(bias, 2),
            "forecast_quality": quality,
            "sample_size": len(actual_arr),
        }

    def detect_trend(self, periods: int = 60) -> dict:
        """
        Detect trend in demand over recent historical data.

        Analyzes slope of demand line to identify long-term direction:
        growing, declining, or stable market.

        Args:
            periods: Number of recent days to analyze (default 60 days = 2 months)

        Returns:
            Dictionary with:
                - 'trend': 'increasing', 'decreasing', or 'stable'
                - 'trend_strength': Float (-1 to 1) indicating magnitude
                  Negative = declining, Positive = increasing
                - 'growth_rate': Daily growth percentage
                - 'forecast_impact': How this should affect forecasts
                - 'recent_avg': Average demand in recent period
                - 'earlier_avg': Average demand in earlier period

        Example:
            planner.load_historical_sales(df)
            trend = planner.detect_trend(periods=90)
            if trend['trend'] == 'increasing':
                print(f"Growth rate: {trend['growth_rate']:.1%}")
        """
        if self.historical_data is None:
            raise ValueError("No historical data loaded")

        import numpy as np

        data = self.historical_data.copy()
        recent = data.tail(periods)

        if len(recent) < 2:
            return {
                "trend": "stable",
                "trend_strength": 0.0,
                "growth_rate": 0.0,
                "forecast_impact": "No trend detected (insufficient data)",
                "recent_avg": float(recent["quantity"].mean()),
                "earlier_avg": 0.0,
            }

        # Split into two halves to compare
        split_point = len(recent) // 2
        earlier_half = recent.iloc[:split_point]["quantity"].values
        recent_half = recent.iloc[split_point:]["quantity"].values

        earlier_avg = float(np.mean(earlier_half))
        recent_avg = float(np.mean(recent_half))

        if earlier_avg == 0:
            growth_rate = 0.0
        else:
            growth_rate = (recent_avg - earlier_avg) / earlier_avg

        # Calculate trend strength using linear regression
        x = np.arange(len(recent))
        y = recent["quantity"].values
        
        # Simple linear regression
        coefficients = np.polyfit(x, y, 1)
        slope = float(coefficients[0])
        
        # Normalize slope to -1 to 1 range
        max_value = np.max(y)
        if max_value > 0:
            trend_strength = float(np.clip(slope / max_value, -1, 1))
        else:
            trend_strength = 0.0

        # Determine trend direction
        threshold = 0.01  # 1% movement per day threshold
        if growth_rate > threshold:
            trend = "increasing"
            forecast_impact = "Consider increasing forecast. Market is growing."
        elif growth_rate < -threshold:
            trend = "decreasing"
            forecast_impact = "Consider decreasing forecast. Market is declining."
        else:
            trend = "stable"
            forecast_impact = "No strong trend. Use baseline forecast."

        return {
            "trend": trend,
            "trend_strength": round(trend_strength, 3),
            "growth_rate": round(growth_rate * 100, 2),  # percentage
            "forecast_impact": forecast_impact,
            "recent_avg": round(recent_avg, 2),
            "earlier_avg": round(earlier_avg, 2),
        }