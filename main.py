"""Main script demonstrating supply and demand planning tool usage."""

import re
from pathlib import Path

import pandas as pd

from src import DemandPlanner, ExcelGenerator, InventoryManager


DATA_DIR = Path("data")
DEFAULT_PRODUCT_MASTER_PATH = Path(r"C:\Users\elmer\Downloads\Product range.csv")
DEFAULT_STOCK_PATH = Path(r"C:\Users\elmer\Downloads\Stock.xlsx")
DEFAULT_COMMITTED_SALES_PATH = Path(r"C:\Users\elmer\Downloads\planned_sales.xlsx")
DEFAULT_PLANNED_PURCHASES_PATH = Path(r"C:\Users\elmer\Downloads\planned_purchases.xlsx")
DEFAULT_HISTORICAL_SALES_PATH = Path(
    r"C:\Users\elmer\Downloads\4. Odoo Verkoop History (gefactureerd datum) - data (2).csv"
)


def _normalize_sku(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _read_tabular_file(path_without_ext: Path) -> pd.DataFrame | None:
    """Read .csv or .xlsx file by base path."""
    csv_path = path_without_ext.with_suffix(".csv")
    xlsx_path = path_without_ext.with_suffix(".xlsx")

    if csv_path.exists():
        return pd.read_csv(csv_path)
    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)
    return None


def _read_csv_fallback_encodings(path: Path) -> pd.DataFrame:
    """Read CSV with common encodings to handle supplier exports."""
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def _load_optional_config() -> dict:
    """
    Load optional config from data/config.csv or .xlsx.

    Required columns in config file: parameter, value
    """
    config_df = _read_tabular_file(DATA_DIR / "config")
    if config_df is None:
        return {}

    required = {"parameter", "value"}
    if not required.issubset(config_df.columns):
        raise ValueError(
            "Config file must contain columns: parameter, value. "
            f"Found: {list(config_df.columns)}"
        )
    return dict(zip(config_df["parameter"].astype(str), config_df["value"]))


def _load_product_master(config: dict) -> pd.DataFrame:
    """
    Load product master file and normalize SKU key.

    Expected columns include: ART_NUMBER/SKU, ART_NAME, CATEGORY, SUB CATEGORY
    """
    path_value = str(config.get("product_master_path", DEFAULT_PRODUCT_MASTER_PATH))
    product_master_path = Path(path_value)
    if not product_master_path.exists():
        raise FileNotFoundError(
            f"Product master not found at '{product_master_path}'. "
            "Set 'product_master_path' in data/config.csv or place the file at the default path."
        )

    product_master = _read_csv_fallback_encodings(product_master_path)
    if "ART_NUMBER/SKU" not in product_master.columns:
        raise ValueError(
            "Product master must contain 'ART_NUMBER/SKU' column. "
            f"Found: {list(product_master.columns)}"
        )

    prepared = product_master.copy()
    prepared["sku"] = prepared["ART_NUMBER/SKU"].apply(_normalize_sku)
    prepared = prepared.drop_duplicates(subset=["sku"]).reset_index(drop=True)
    return prepared


def _load_stock_from_file(config: dict) -> pd.DataFrame:
    """
    Load stock file and normalize to columns: sku, current_stock.

    Supports the provided stock export format:
    - Interne referentie
    - Beschikbare voorraad
    """
    path_value = str(config.get("stock_file_path", DEFAULT_STOCK_PATH))
    stock_path = Path(path_value)
    if not stock_path.exists():
        return pd.DataFrame(columns=["sku", "current_stock"])

    if stock_path.suffix.lower() == ".xlsx":
        raw = pd.read_excel(stock_path)
    elif stock_path.suffix.lower() == ".csv":
        raw = _read_csv_fallback_encodings(stock_path)
    else:
        raise ValueError(
            f"Unsupported stock file format '{stock_path.suffix}'. Use .xlsx or .csv."
        )

    sku_candidates = ["sku", "SKU", "Interne referentie", "ART_NUMBER/SKU"]
    stock_candidates = ["current_stock", "Beschikbare voorraad", "Available stock"]

    sku_col = next((col for col in sku_candidates if col in raw.columns), None)
    stock_col = next((col for col in stock_candidates if col in raw.columns), None)

    if not sku_col or not stock_col:
        raise ValueError(
            "Stock file must include a SKU column and a stock quantity column. "
            f"Found columns: {list(raw.columns)}"
        )

    stock_df = raw[[sku_col, stock_col]].copy()
    stock_df.columns = ["sku", "current_stock"]
    stock_df["sku"] = stock_df["sku"].apply(_normalize_sku)
    stock_df["current_stock"] = pd.to_numeric(stock_df["current_stock"], errors="coerce").fillna(0.0)
    stock_df = stock_df.groupby("sku", as_index=False)["current_stock"].sum()
    return stock_df


def _extract_sku_from_product_text(value) -> str | None:
    """Extract SKU from strings like '[107299] Remora Pro HV ...'."""
    if pd.isna(value):
        return None
    match = re.search(r"\[(\d+)\]", str(value))
    if not match:
        return None
    return _normalize_sku(match.group(1))


def _load_committed_sales_from_file(config: dict) -> pd.DataFrame:
    """
    Load confirmed-but-not-shipped sales and map to: sku, date, planned_quantity.

    Expected columns in provided export:
    - Verplaatsingen/Orderregel/Aantal te leveren
    - Verplaatsingen/Geplande datum
    - Verplaatsingen/Orderregel/Product
    """
    path_value = str(config.get("committed_sales_path", DEFAULT_COMMITTED_SALES_PATH))
    committed_path = Path(path_value)
    if not committed_path.exists():
        return pd.DataFrame(columns=["sku", "date", "planned_quantity"])

    if committed_path.suffix.lower() == ".xlsx":
        raw = pd.read_excel(committed_path)
    elif committed_path.suffix.lower() == ".csv":
        raw = _read_csv_fallback_encodings(committed_path)
    else:
        raise ValueError(
            f"Unsupported committed sales file format '{committed_path.suffix}'. Use .xlsx or .csv."
        )

    qty_col = "Verplaatsingen/Orderregel/Aantal te leveren"
    date_col = "Verplaatsingen/Geplande datum"
    product_col = "Verplaatsingen/Orderregel/Product"
    required_cols = {qty_col, date_col, product_col}
    if not required_cols.issubset(raw.columns):
        return pd.DataFrame(columns=["sku", "date", "planned_quantity"])

    committed = raw[[qty_col, date_col, product_col]].copy()
    committed.columns = ["planned_quantity", "date", "product_text"]
    committed["planned_quantity"] = pd.to_numeric(committed["planned_quantity"], errors="coerce").fillna(0.0)
    committed = committed[committed["planned_quantity"] > 0].copy()
    committed["date"] = pd.to_datetime(committed["date"], errors="coerce")
    committed = committed.dropna(subset=["date"])
    committed["sku"] = committed["product_text"].apply(_extract_sku_from_product_text)
    committed = committed.dropna(subset=["sku"])

    if committed.empty:
        return pd.DataFrame(columns=["sku", "date", "planned_quantity"])

    committed["date"] = committed["date"].dt.normalize()
    committed = committed.groupby(["sku", "date"], as_index=False)["planned_quantity"].sum()
    return committed


def _load_planned_purchases_from_file(config: dict) -> pd.DataFrame:
    """
    Load planned purchases and map to: sku, date, planned_purchase_quantity.

    Expected columns in provided export:
    - Orderregels/Product/Interne referentie
    - Orderregels/Hoeveelheid
    - Orderregels/Verwachte levering
    """
    path_value = str(config.get("planned_purchases_path", DEFAULT_PLANNED_PURCHASES_PATH))
    purchases_path = Path(path_value)
    if not purchases_path.exists():
        return pd.DataFrame(columns=["sku", "date", "planned_purchase_quantity"])

    if purchases_path.suffix.lower() == ".xlsx":
        raw = pd.read_excel(purchases_path)
    elif purchases_path.suffix.lower() == ".csv":
        raw = _read_csv_fallback_encodings(purchases_path)
    else:
        raise ValueError(
            f"Unsupported planned purchases file format '{purchases_path.suffix}'. Use .xlsx or .csv."
        )

    sku_col = "Orderregels/Product/Interne referentie"
    qty_col = "Orderregels/Hoeveelheid"
    date_col = "Orderregels/Verwachte levering"
    required_cols = {sku_col, qty_col, date_col}
    if not required_cols.issubset(raw.columns):
        return pd.DataFrame(columns=["sku", "date", "planned_purchase_quantity"])

    purchases = raw[[sku_col, qty_col, date_col]].copy()
    purchases.columns = ["sku", "planned_purchase_quantity", "date"]
    purchases["sku"] = purchases["sku"].apply(_normalize_sku)
    purchases["planned_purchase_quantity"] = pd.to_numeric(
        purchases["planned_purchase_quantity"], errors="coerce"
    ).fillna(0.0)
    purchases = purchases[purchases["planned_purchase_quantity"] > 0].copy()
    purchases["date"] = pd.to_datetime(purchases["date"], errors="coerce")
    purchases = purchases.dropna(subset=["sku", "date"])
    purchases["date"] = purchases["date"].dt.normalize()
    purchases = purchases.groupby(["sku", "date"], as_index=False)["planned_purchase_quantity"].sum()
    return purchases


def _filter_product_master(product_master: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply optional filters from config to product master."""
    filtered = product_master.copy()

    category_filter = str(config.get("product_category", "")).strip()
    sub_category_filter = str(config.get("product_sub_category", "")).strip()
    gender_filter = str(config.get("product_gender", "")).strip()
    sku_filter = str(config.get("product_skus", "")).strip()

    if category_filter:
        filtered = filtered[filtered["CATEGORY"].astype(str).str.upper() == category_filter.upper()]
    if sub_category_filter:
        filtered = filtered[
            filtered["SUB CATEGORY"].astype(str).str.upper() == sub_category_filter.upper()
        ]
    if gender_filter:
        filtered = filtered[filtered["GENDER"].astype(str).str.upper() == gender_filter.upper()]
    if sku_filter:
        sku_values = {_normalize_sku(part) for part in sku_filter.split(",") if part.strip()}
        filtered = filtered[filtered["sku"].isin(sku_values)]

    return filtered.reset_index(drop=True)


def _load_historical_sales_from_odoo_file(path: Path) -> pd.DataFrame:
    """
    Load Odoo invoiced sales history and normalize to: sku, date, quantity.

    Expected columns:
    - dateinvoice
    - prodid
    - qtyinv (fallback qtyorder)
    """
    raw = _read_csv_fallback_encodings(path)
    required = {"dateinvoice", "prodid"}
    if not required.issubset(raw.columns):
        raise ValueError(
            "Historical sales file does not match expected Odoo structure. "
            f"Found: {list(raw.columns)}"
        )

    qty_col = "qtyinv" if "qtyinv" in raw.columns else "qtyorder"
    if qty_col not in raw.columns:
        raise ValueError("Historical sales file must contain 'qtyinv' or 'qtyorder' column.")

    hist = raw[["dateinvoice", "prodid", qty_col]].copy()
    hist.columns = ["date", "sku", "quantity"]
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce", dayfirst=True)
    hist["sku"] = hist["sku"].apply(_normalize_sku)
    hist["quantity"] = pd.to_numeric(hist["quantity"], errors="coerce")
    hist = hist.dropna(subset=["date", "sku", "quantity"])
    hist = hist.groupby(["sku", "date"], as_index=False)["quantity"].sum()
    return hist


def _load_required_historical_sales(config: dict) -> pd.DataFrame:
    """
    Load historical sales from data/historical_sales.csv or .xlsx.

    Required columns: date, quantity
    Optional: sku
    """
    historical_path = Path(str(config.get("historical_sales_path", DEFAULT_HISTORICAL_SALES_PATH)))
    if historical_path.exists():
        return _load_historical_sales_from_odoo_file(historical_path)

    historical = _read_tabular_file(DATA_DIR / "historical_sales")
    if historical is None:
        raise FileNotFoundError(
            "Missing historical sales source. Provide 'historical_sales_path' in config, or "
            "'data/historical_sales.csv' / 'data/historical_sales.xlsx' with columns: date, quantity (optional sku)."
        )
    required = {"date", "quantity"}
    if not required.issubset(historical.columns):
        raise ValueError(
            "Historical sales file must contain columns: date, quantity. "
            f"Found: {list(historical.columns)}"
        )
    prepared = historical.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    prepared["quantity"] = pd.to_numeric(prepared["quantity"], errors="raise")
    if "sku" in prepared.columns:
        prepared["sku"] = prepared["sku"].apply(_normalize_sku)
    return prepared


def _load_optional_dataframe(file_base_name: str, required_columns: set[str]) -> pd.DataFrame:
    """Load optional input table. Returns empty DataFrame if missing."""
    df = _read_tabular_file(DATA_DIR / file_base_name)
    if df is None:
        return pd.DataFrame(columns=list(required_columns))
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{file_base_name} must contain columns: {sorted(required_columns)}. "
            f"Found: {list(df.columns)}"
        )
    prepared = df.copy()
    if "date" in prepared.columns:
        prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    for col in required_columns:
        if col != "date":
            prepared[col] = pd.to_numeric(prepared[col], errors="raise")
    if "sku" in prepared.columns:
        prepared["sku"] = prepared["sku"].apply(_normalize_sku)
    return prepared


def _load_optional_stock_by_sku() -> pd.DataFrame:
    """
    Load optional SKU stock/parameter overrides.

    Optional file: data/stock_by_sku.csv or .xlsx
    Required columns: sku, current_stock
    """
    df = _read_tabular_file(DATA_DIR / "stock_by_sku")
    if df is None:
        return pd.DataFrame(columns=["sku", "current_stock"])

    if "sku" not in df.columns or "current_stock" not in df.columns:
        raise ValueError(
            "stock_by_sku must contain columns: sku, current_stock. "
            f"Found: {list(df.columns)}"
        )

    prepared = df.copy()
    prepared["sku"] = prepared["sku"].apply(_normalize_sku)

    numeric_cols = [
        "current_stock",
        "safety_stock",
        "lead_time_days",
        "min_order_qty",
        "order_multiple",
        "service_level",
    ]
    for col in numeric_cols:
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="raise")
    return prepared


def _run_plan_for_single_series(
    historical_df: pd.DataFrame,
    planned_sales_df: pd.DataFrame,
    planned_purchases_df: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run planning for one demand series (single SKU or aggregate)."""
    planner = DemandPlanner()
    planner.load_historical_sales(historical_df)

    avg_demand = planner.calculate_average_daily_demand(days=30)
    demand_std = planner.calculate_daily_demand_std(days=90)
    forecast = planner.forecast_weekly(
        periods=int(params["forecast_periods"]),
        ma_weeks=int(params["moving_average_weeks"]),
        start_date=f"{params['planning_start_month']}-01",
    )
    demand_plan = planner.build_demand_plan(
        planned_sales=planned_sales_df if not planned_sales_df.empty else None,
        periods=int(params["forecast_periods"]),
    )

    inventory = InventoryManager(
        current_stock=float(params["current_stock"]),
        safety_stock=float(params["safety_stock"]),
        min_order_qty=float(params["min_order_qty"]),
        order_multiple=float(params["order_multiple"]),
        service_level=float(params["service_level"]),
    )
    purchase_plan = inventory.plan_purchases(
        demand_forecast=demand_plan,
        lead_time_days=int(params["lead_time_days"]),
        planned_purchases=planned_purchases_df if not planned_purchases_df.empty else None,
    )
    purchase_plan["avg_daily_demand"] = avg_demand
    purchase_plan["daily_demand_std"] = demand_std
    return forecast, purchase_plan


def _resolve_weekly_periods(planning_start_month: str, planning_end_month: str, configured_periods: int) -> int:
    """
    Ensure forecast periods cover the full planning window in weekly buckets.

    Returns at least `configured_periods`, and increases it when needed to reach planning_end_month.
    """
    start_anchor = pd.to_datetime(f"{planning_start_month}-01")
    first_forecast_week = start_anchor.to_period("W-MON").end_time.normalize()

    end_anchor = pd.to_datetime(f"{planning_end_month}-01") + pd.offsets.MonthEnd(0)
    last_required_week = end_anchor.to_period("W-MON").end_time.normalize()

    required_periods = int(((last_required_week - first_forecast_week).days // 7) + 1)
    return max(configured_periods, required_periods)


def _build_action_list(purchase_plan_df: pd.DataFrame) -> pd.DataFrame:
    """Create one actionable row per SKU (or one aggregate row if no SKU)."""
    has_sku = "sku" in purchase_plan_df.columns
    group_keys = ["sku"] if has_sku else ["_aggregate"]

    working = purchase_plan_df.copy()
    if not has_sku:
        working["_aggregate"] = "ALL_PRODUCTS"
    working = working.sort_values("date")

    action_rows = []
    for key, group in working.groupby(group_keys):
        sku_value = key[0] if isinstance(key, tuple) else key
        group = group.sort_values("date")
        purchase_rows = group[(group["purchase_needed"]) & (group["suggested_purchase"] > 0)]
        first_purchase = purchase_rows.iloc[0] if not purchase_rows.empty else None

        stockout_rows = group[group["stockout_risk"]]
        first_stockout_date = stockout_rows["date"].min() if not stockout_rows.empty else pd.NaT

        buy_now = first_purchase is not None
        order_by_date = first_purchase["date"] if first_purchase is not None else pd.NaT
        initial_recommended_qty = float(first_purchase["suggested_purchase"]) if first_purchase is not None else 0.0
        total_horizon_qty = float(purchase_rows["suggested_purchase"].sum()) if not purchase_rows.empty else 0.0
        primary_reason = first_purchase["reason_code"] if first_purchase is not None else "no_action"

        risk_level = "low"
        if not stockout_rows.empty:
            risk_level = "high"
        elif buy_now:
            risk_level = "medium"

        row = {
            "sku": sku_value if has_sku else "ALL_PRODUCTS",
            "buy_now": buy_now,
            "order_by_date": order_by_date,
            "initial_recommended_qty": round(initial_recommended_qty, 2),
            "total_recommended_qty_horizon": round(total_horizon_qty, 2),
            "expected_stockout_date": first_stockout_date,
            "risk_level": risk_level,
            "primary_reason_code": primary_reason,
        }

        # Bring key product context if available.
        for meta_col in ["ART_NAME", "CATEGORY", "SUB CATEGORY", "GENDER", "COLOR_DESCRIPTION"]:
            if meta_col in group.columns:
                first_val = group[meta_col].dropna()
                row[meta_col] = first_val.iloc[0] if not first_val.empty else None

        action_rows.append(row)

    action_list = pd.DataFrame(action_rows)
    risk_rank = {"high": 0, "medium": 1, "low": 2}
    action_list["_risk_rank"] = action_list["risk_level"].map(risk_rank).fillna(9)
    action_list = action_list.sort_values(
        by=["_risk_rank", "buy_now", "order_by_date", "total_recommended_qty_horizon"],
        ascending=[True, False, True, False],
    ).drop(columns=["_risk_rank"])
    return action_list.reset_index(drop=True)


def _build_monthly_stock_outlook(
    purchase_plan_df: pd.DataFrame, planning_end_month: str | None = None
) -> pd.DataFrame:
    """
    Build one-row-per-SKU table with current stock and future stock by month.

    Future stock is the last projected stock value within each month.
    """
    working = purchase_plan_df.copy()
    has_sku = "sku" in working.columns
    if not has_sku:
        working["sku"] = "ALL_PRODUCTS"

    working["date"] = pd.to_datetime(working["date"])
    if planning_end_month:
        end_date = pd.to_datetime(f"{planning_end_month}-01") + pd.offsets.MonthEnd(0)
        working = working[working["date"] <= end_date].copy()
    working["month"] = working["date"].dt.to_period("M").astype(str)
    working = working.sort_values(["sku", "date"])

    current_stock_df = (
        working.groupby("sku", as_index=False)["opening_stock"].first().rename(columns={"opening_stock": "current_stock"})
    )
    monthly_stock = (
        working.groupby(["sku", "month"], as_index=False)["projected_stock"].last()
        .pivot(index="sku", columns="month", values="projected_stock")
        .reset_index()
    )
    monthly_stock.columns.name = None

    outlook = current_stock_df.merge(monthly_stock, on="sku", how="left")

    # Add key product metadata if available.
    for meta_col in ["ART_NAME", "CATEGORY", "SUB CATEGORY", "GENDER"]:
        if meta_col in working.columns:
            meta_df = working.groupby("sku", as_index=False)[meta_col].first()
            outlook = outlook.merge(meta_df, on="sku", how="left")

    preferred_order = ["sku", "ART_NAME", "CATEGORY", "SUB CATEGORY", "GENDER", "current_stock"]
    month_cols = sorted([col for col in outlook.columns if col[:4].isdigit() and len(col) == 7])
    other_cols = [col for col in outlook.columns if col not in preferred_order + month_cols]
    ordered_cols = [col for col in preferred_order if col in outlook.columns] + month_cols + other_cols
    return outlook[ordered_cols]


def main():
    """Run the supply and demand planning example."""
    print("=" * 60)
    print("Supply and Demand Planning Tool - Product Master Integrated")
    print("=" * 60)

    config = _load_optional_config()
    product_master = _load_product_master(config)
    filtered_products = _filter_product_master(product_master, config)
    allowed_skus = set(filtered_products["sku"].tolist())

    planning_start_month = str(config.get("planning_start_month", "2026-01"))
    planning_end_month = str(config.get("planning_end_month", f"{planning_start_month[:4]}-12"))
    configured_periods = int(config.get("forecast_periods", 12))
    resolved_forecast_periods = _resolve_weekly_periods(
        planning_start_month=planning_start_month,
        planning_end_month=planning_end_month,
        configured_periods=configured_periods,
    )

    default_params = {
        "current_stock": float(config.get("current_stock", 1000)),
        "safety_stock": float(config.get("safety_stock", 150)),
        "lead_time_days": int(config.get("lead_time_days", 14)),
        "min_order_qty": float(config.get("min_order_qty", 100)),
        "order_multiple": float(config.get("order_multiple", 25)),
        "service_level": float(config.get("service_level", 0.95)),
        "forecast_periods": resolved_forecast_periods,
        "moving_average_weeks": int(config.get("moving_average_weeks", 4)),
        "planning_start_month": planning_start_month,
        "planning_end_month": planning_end_month,
    }

    print("\n[1] Initialization")
    print(f"    Product master rows: {len(product_master)}")
    print(f"    Filtered products in scope: {len(filtered_products)}")
    print(f"    Default lead time: {default_params['lead_time_days']} days")
    print(f"    Default service level: {default_params['service_level']:.2f}")
    print(f"    Planning start month: {default_params['planning_start_month']}")
    print(f"    Planning end month: {default_params['planning_end_month']}")
    print(f"    Forecast periods (weeks): {default_params['forecast_periods']}")

    print("\n[2] Loading planning inputs from data/ ...")
    historical_df = _load_required_historical_sales(config)
    planned_sales = _load_optional_dataframe("planned_sales", {"date", "planned_quantity"})
    planned_purchases_data = _load_optional_dataframe(
        "planned_purchases", {"date", "planned_purchase_quantity"}
    )
    planned_purchases_file = _load_planned_purchases_from_file(config)
    committed_sales = _load_committed_sales_from_file(config)
    stock_from_file = _load_stock_from_file(config)
    stock_by_sku = _load_optional_stock_by_sku()

    data_has_sku = (not planned_purchases_data.empty) and ("sku" in planned_purchases_data.columns)
    file_has_sku = not planned_purchases_file.empty
    if data_has_sku and file_has_sku:
        planned_purchases = pd.concat(
            [planned_purchases_data[["sku", "date", "planned_purchase_quantity"]], planned_purchases_file],
            ignore_index=True,
        )
        planned_purchases = (
            planned_purchases.groupby(["sku", "date"], as_index=False)["planned_purchase_quantity"].sum()
        )
    elif file_has_sku:
        planned_purchases = planned_purchases_file.copy()
    else:
        planned_purchases = planned_purchases_data.copy()

    if not stock_by_sku.empty:
        stock_lookup_df = stock_from_file.merge(stock_by_sku, on="sku", how="outer", suffixes=("", "_override"))
        for col in ["current_stock", "safety_stock", "lead_time_days", "min_order_qty", "order_multiple", "service_level"]:
            override_col = f"{col}_override"
            if override_col in stock_lookup_df.columns:
                stock_lookup_df[col] = stock_lookup_df[override_col].combine_first(stock_lookup_df.get(col))
        keep_cols = ["sku", "current_stock", "safety_stock", "lead_time_days", "min_order_qty", "order_multiple", "service_level"]
        stock_lookup_df = stock_lookup_df[[col for col in keep_cols if col in stock_lookup_df.columns]]
    else:
        stock_lookup_df = stock_from_file

    print(f"    Historical data rows: {len(historical_df)}")
    print(f"    Planned sales rows: {len(planned_sales)}")
    print(f"    Committed sales rows: {len(committed_sales)}")
    print(f"    Planned purchases file rows: {len(planned_purchases_file)}")
    print(f"    Planned purchases rows: {len(planned_purchases)}")
    print(f"    Stock file rows: {len(stock_from_file)}")
    print(f"    Stock overrides rows: {len(stock_by_sku)}")

    has_sku = "sku" in historical_df.columns
    if has_sku:
        historical_df = historical_df[historical_df["sku"].isin(allowed_skus)].copy()
        if "sku" in planned_sales.columns:
            planned_sales = planned_sales[planned_sales["sku"].isin(allowed_skus)].copy()
        if "sku" in planned_purchases.columns:
            planned_purchases = planned_purchases[planned_purchases["sku"].isin(allowed_skus)].copy()
        committed_sales = committed_sales[committed_sales["sku"].isin(allowed_skus)].copy()

    # Add committed future sales into planned sales demand.
    if not committed_sales.empty:
        if "sku" in planned_sales.columns:
            merged_sales = pd.concat(
                [planned_sales[["sku", "date", "planned_quantity"]], committed_sales],
                ignore_index=True,
            )
            planned_sales = (
                merged_sales.groupby(["sku", "date"], as_index=False)["planned_quantity"].sum()
            )
        elif planned_sales.empty:
            planned_sales = committed_sales.copy()

        if historical_df.empty:
            raise ValueError(
                "No historical rows remain after product master SKU filtering. "
                "Check sku values in data/historical_sales.csv and product filters in data/config.csv."
            )

    print("\n[3] Demand and inventory planning...")
    all_purchase_plans = []
    all_forecasts = []

    if has_sku:
        stock_lookup = (
            stock_lookup_df.set_index("sku").to_dict("index") if not stock_lookup_df.empty else {}
        )

        for sku, hist_sku in historical_df.groupby("sku"):
            sku_planned_sales = (
                planned_sales[planned_sales["sku"] == sku]
                if "sku" in planned_sales.columns
                else pd.DataFrame(columns=planned_sales.columns)
            )
            sku_planned_purchases = (
                planned_purchases[planned_purchases["sku"] == sku]
                if "sku" in planned_purchases.columns
                else pd.DataFrame(columns=planned_purchases.columns)
            )

            params = default_params.copy()
            if sku in stock_lookup:
                for key in [
                    "current_stock",
                    "safety_stock",
                    "lead_time_days",
                    "min_order_qty",
                    "order_multiple",
                    "service_level",
                ]:
                    if key in stock_lookup[sku] and pd.notna(stock_lookup[sku][key]):
                        params[key] = stock_lookup[sku][key]

            forecast, purchase_plan = _run_plan_for_single_series(
                historical_df=hist_sku[["date", "quantity"]],
                planned_sales_df=sku_planned_sales,
                planned_purchases_df=sku_planned_purchases,
                params=params,
            )
            forecast["sku"] = sku
            purchase_plan["sku"] = sku
            all_forecasts.append(forecast)
            all_purchase_plans.append(purchase_plan)

        forecast_df = pd.concat(all_forecasts, ignore_index=True)
        purchase_plan_df = pd.concat(all_purchase_plans, ignore_index=True)
    else:
        forecast_df, purchase_plan_df = _run_plan_for_single_series(
            historical_df=historical_df[["date", "quantity"]],
            planned_sales_df=planned_sales,
            planned_purchases_df=planned_purchases,
            params=default_params,
        )

    if "sku" in purchase_plan_df.columns:
        enrich_cols = [
            "sku",
            "ART_NAME",
            "CATEGORY",
            "SUB CATEGORY",
            "GENDER",
            "COLOR_DESCRIPTION",
            "TRADE PRICE",
        ]
        available_cols = [col for col in enrich_cols if col in filtered_products.columns]
        purchase_plan_df = purchase_plan_df.merge(
            filtered_products[available_cols], on="sku", how="left"
        )

    purchases_needed = purchase_plan_df[purchase_plan_df["purchase_needed"]]
    stockout_risk_periods = purchase_plan_df[purchase_plan_df["stockout_risk"]]
    late_po_risk_periods = purchase_plan_df[purchase_plan_df["late_po_risk"]]
    action_list_df = _build_action_list(purchase_plan_df)
    stock_outlook_df = _build_monthly_stock_outlook(
        purchase_plan_df, planning_end_month=default_params["planning_end_month"]
    )

    print(f"    Purchase order recommendations: {len(purchases_needed)} periods")
    print(f"    Stockout risk periods: {len(stockout_risk_periods)}")
    print(f"    Late PO risk periods: {len(late_po_risk_periods)}")
    print(f"    Action list rows: {len(action_list_df)}")
    print(f"    Stock outlook rows: {len(stock_outlook_df)}")
    if not purchases_needed.empty:
        total_to_purchase = purchases_needed["suggested_purchase"].sum()
        print(f"    Total suggested purchase quantity: {total_to_purchase:.1f} units")

    print("\n[4] Generating Excel templates...")
    excel_gen = ExcelGenerator(output_dir="templates")
    template_path = excel_gen.create_planning_template()
    report_path = excel_gen.create_forecast_report(
        forecast_df=purchase_plan_df,
        action_list_df=action_list_df,
        stock_outlook_df=stock_outlook_df,
        dashboard_title="Planner Dashboard",
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"[OK] Products in scope: {len(filtered_products)}")
    print(f"[OK] Forecast rows: {len(forecast_df)}")
    print(f"[OK] Purchase plan rows: {len(purchase_plan_df)}")
    print(f"[OK] Excel template generated: {template_path}")
    print(f"[OK] Forecast report generated: {report_path}")
    print("=" * 60)

    return {
        "product_master": filtered_products,
        "historical_data": historical_df,
        "forecast": forecast_df,
        "planned_sales": planned_sales,
        "committed_sales": committed_sales,
        "planned_purchases": planned_purchases,
        "purchase_plan": purchase_plan_df,
        "action_list": action_list_df,
        "stock_outlook": stock_outlook_df,
    }


if __name__ == "__main__":
    results = main()
