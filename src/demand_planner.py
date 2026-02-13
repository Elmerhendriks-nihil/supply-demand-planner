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
