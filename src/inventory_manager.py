"""Inventory management module for supply and demand planning."""

from math import ceil

import pandas as pd


class InventoryManager:
    """Manages inventory levels, calculations, and purchase recommendations."""

    _SERVICE_LEVEL_Z = {
        0.80: 0.84,
        0.85: 1.04,
        0.90: 1.28,
        0.95: 1.65,
        0.98: 2.05,
        0.99: 2.33,
    }

    def __init__(
        self,
        current_stock: float,
        safety_stock: float = 0,
        min_order_qty: float = 0,
        order_multiple: float | None = None,
        service_level: float = 0.95,
    ):
        """
        Initialize inventory manager.

        Args:
            current_stock: Current quantity in inventory
            safety_stock: Minimum safety stock to maintain
            min_order_qty: Minimum order quantity per purchase order
            order_multiple: Optional order quantity multiple (e.g., case pack)
            service_level: Target service level for safety stock calculations
        """
        self.current_stock = current_stock
        self.safety_stock = safety_stock
        self.min_order_qty = min_order_qty
        self.order_multiple = order_multiple
        self.service_level = service_level
        self.inventory_log = []

    def _z_value(self, service_level: float | None = None) -> float:
        level = self.service_level if service_level is None else service_level
        rounded = round(level, 2)
        return self._SERVICE_LEVEL_Z.get(rounded, 1.65)

    def _apply_order_constraints(self, quantity: float) -> float:
        if quantity <= 0:
            return 0.0

        constrained = max(quantity, self.min_order_qty)
        if self.order_multiple and self.order_multiple > 0:
            constrained = ceil(constrained / self.order_multiple) * self.order_multiple
        return float(constrained)

    def calculate_safety_stock(
        self, daily_demand_std: float, lead_time_days: int = 7, service_level: float | None = None
    ) -> float:
        """
        Calculate statistical safety stock.

        Formula: z * sigma_demand * sqrt(lead_time_days)
        """
        if daily_demand_std <= 0 or lead_time_days <= 0:
            return float(self.safety_stock)
        z = self._z_value(service_level)
        return float(max(self.safety_stock, z * daily_demand_std * (lead_time_days ** 0.5)))

    def calculate_optimal_stock(
        self,
        average_daily_demand: float,
        lead_time_days: int = 7,
        daily_demand_std: float | None = None,
    ) -> float:
        """
        Calculate optimal stock level.

        Formula: Reorder Point = demand during lead time + safety stock
        """
        safety_stock = self.safety_stock
        if daily_demand_std is not None:
            safety_stock = self.calculate_safety_stock(daily_demand_std, lead_time_days)
        reorder_point = (average_daily_demand * lead_time_days) + safety_stock
        return reorder_point

    def plan_purchases(
        self,
        demand_forecast: pd.DataFrame,
        lead_time_days: int = 7,
        planned_purchases: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Generate purchase recommendations based on demand forecast.

        Args:
            demand_forecast: DataFrame with 'date' and 'total_expected_demand'
            lead_time_days: Lead time for purchase orders
            planned_purchases: Optional DataFrame with 'date' and 'planned_purchase_quantity'

        Returns:
            DataFrame with period-level purchase recommendations
        """
        plan = demand_forecast.copy()

        required_columns = {"date", "total_expected_demand"}
        if not required_columns.issubset(set(plan.columns)):
            raise ValueError("Demand forecast must contain 'date' and 'total_expected_demand' columns")

        plan["date"] = pd.to_datetime(plan["date"])
        plan = plan.sort_values("date").reset_index(drop=True)
        plan["total_expected_demand"] = plan["total_expected_demand"].astype(float)

        date_values = plan["date"].tolist()
        if len(date_values) >= 2:
            period_days = max(1, int((date_values[1] - date_values[0]).days))
        else:
            period_days = 7
        lead_periods = max(1, ceil(lead_time_days / period_days))

        planned_receipts = {}
        if planned_purchases is not None and not planned_purchases.empty:
            if "date" not in planned_purchases.columns or "planned_purchase_quantity" not in planned_purchases.columns:
                raise ValueError("Planned purchases must contain 'date' and 'planned_purchase_quantity' columns")
            receipts_df = planned_purchases.copy()
            receipts_df["date"] = pd.to_datetime(receipts_df["date"])
            grouped = receipts_df.groupby("date", as_index=False)["planned_purchase_quantity"].sum()
            planned_receipts = {
                row["date"]: float(row["planned_purchase_quantity"]) for _, row in grouped.iterrows()
            }

        suggested_receipts = {}
        current_stock = float(self.current_stock)
        records = []

        for idx, row in plan.iterrows():
            date = row["date"]
            demand = float(row["total_expected_demand"])

            opening_stock = current_stock
            planned_receipt = float(planned_receipts.get(date, 0.0))
            suggested_receipt = float(suggested_receipts.get(date, 0.0))
            net_available_before_demand = opening_stock + planned_receipt + suggested_receipt
            projected_stock = net_available_before_demand - demand

            lead_window = plan.loc[idx + 1 : idx + lead_periods, "total_expected_demand"]
            demand_during_lead_time = float(lead_window.sum()) if not lead_window.empty else 0.0
            safety_stock_used = float(self.safety_stock)
            target_stock = safety_stock_used + demand_during_lead_time
            purchase_needed = projected_stock < target_stock

            suggested_purchase = 0.0
            expected_arrival_date = pd.NaT
            if purchase_needed:
                raw_qty = target_stock - projected_stock
                suggested_purchase = self._apply_order_constraints(raw_qty)
                arrival_idx = idx + lead_periods
                if arrival_idx < len(plan):
                    expected_arrival_date = plan.loc[arrival_idx, "date"]
                    arrival_date = plan.loc[arrival_idx, "date"]
                    suggested_receipts[arrival_date] = suggested_receipts.get(arrival_date, 0.0) + suggested_purchase
                else:
                    expected_arrival_date = date + pd.Timedelta(days=lead_time_days)

            stockout_risk = projected_stock < 0
            excess_stock_risk = projected_stock > (target_stock * 2 if target_stock > 0 else self.safety_stock * 3)
            late_po_risk = stockout_risk and purchase_needed and lead_periods > 0
            recommendation_gap = target_stock - projected_stock

            if purchase_needed and stockout_risk:
                reason_code = "stockout_risk"
                reason_detail = "Projected stock falls below zero and target stock."
            elif purchase_needed:
                reason_code = "below_target_stock"
                reason_detail = "Projected stock is below target stock during lead time."
            elif excess_stock_risk:
                reason_code = "excess_stock_risk"
                reason_detail = "Projected stock is materially above target stock."
            else:
                reason_code = "no_action"
                reason_detail = "Projected stock remains within policy range."

            records.append(
                {
                    "date": date,
                    "opening_stock": round(opening_stock, 2),
                    "planned_receipts": round(planned_receipt, 2),
                    "suggested_receipts": round(suggested_receipt, 2),
                    "net_available": round(net_available_before_demand, 2),
                    "total_expected_demand": round(demand, 2),
                    "projected_stock": round(projected_stock, 2),
                    "demand_during_lead_time": round(demand_during_lead_time, 2),
                    "safety_stock_used": round(safety_stock_used, 2),
                    "target_stock": round(target_stock, 2),
                    "purchase_needed": purchase_needed,
                    "suggested_purchase": round(suggested_purchase, 2),
                    "recommendation_gap": round(recommendation_gap, 2),
                    "expected_arrival_date": expected_arrival_date,
                    "stockout_risk": stockout_risk,
                    "excess_stock_risk": excess_stock_risk,
                    "late_po_risk": late_po_risk,
                    "reason_code": reason_code,
                    "reason_detail": reason_detail,
                }
            )

            current_stock = projected_stock

        return pd.DataFrame(records)

    def update_stock(self, quantity_sold: float, quantity_purchased: float = 0):
        """Update current stock after transaction."""
        self.current_stock = self.current_stock - quantity_sold + quantity_purchased
        self.inventory_log.append(
            {"sold": quantity_sold, "purchased": quantity_purchased, "current_stock": self.current_stock}
        )
        return self.current_stock

    def get_stock_status(self) -> dict:
        """Get current inventory status."""
        return {
            "current_stock": self.current_stock,
            "safety_stock": self.safety_stock,
            "status": "Critical" if self.current_stock < self.safety_stock else "Healthy",
        }
