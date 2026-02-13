# Supply and Demand Planning Tool

A Python-based tool for managing supply chain inventory and demand planning. Helps forecast demand, track stock levels, and generate purchase recommendations based on historical sales, forecasted demand, and planned purchases.

## Features

- **Weekly Demand Planning**: Combines historical demand forecast with planned sales overrides
- **Lead-Time Aware Purchasing**: Recommends purchase quantities and expected arrival periods
- **Inventory Risk Flags**: Highlights stockout, excess stock, and late PO risk periods
- **Explainable Recommendations**: Adds reason codes and policy metrics behind buy decisions
- **Action List Output**: One-row-per-SKU purchase action plan
- **Planner Dashboard**: KPI summary and top urgent SKUs in the Excel report
- **Stock Outlook Sheet**: One row per SKU with current stock and monthly future stock
- **Planning Constraints**: Supports minimum order quantity and order multiple rules
- **Excel Integration**: Generates template and report files for planning reviews

## Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Usage

```python
from src import DemandPlanner, InventoryManager, ExcelGenerator
import pandas as pd

# 1. Initialize components
planner = DemandPlanner()
inventory = InventoryManager(current_stock=1000, safety_stock=100)
excel_gen = ExcelGenerator()

# 2. Load historical sales data
historical_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=90, freq='D'),
    'quantity': [50 + i % 20 for i in range(90)]
})
planner.load_historical_sales(historical_data)

# 3. Generate weekly forecast + demand plan
forecast = planner.forecast_weekly(periods=12, ma_weeks=4)
planned_sales = pd.DataFrame({
    "date": forecast["date"].head(2),
    "planned_quantity": [500, 550]
})
demand_plan = planner.build_demand_plan(planned_sales=planned_sales)

# 4. Plan purchases with lead-time logic
planned_purchases = pd.DataFrame({
    "date": [forecast["date"].iloc[1]],
    "planned_purchase_quantity": [400]
})
purchase_plan = inventory.plan_purchases(
    demand_forecast=demand_plan,
    lead_time_days=14,
    planned_purchases=planned_purchases
)

# 5. Create Excel template
excel_gen.create_planning_template()
```

### Run with your real data

1. Put your files in `data/` with these names:
- `historical_sales.csv` or `historical_sales.xlsx` (required)
- `planned_sales.csv` or `planned_sales.xlsx` (optional)
- `planned_purchases.csv` or `planned_purchases.xlsx` (optional)
- `stock_by_sku.csv` or `stock_by_sku.xlsx` (optional)
- `config.csv` or `config.xlsx` (optional)
2. Run:
   ```bash
   python main.py
   ```

### Input file schema

- `historical_sales`: `date`, `quantity` (optional `sku` for SKU-level planning)
- `planned_sales`: `date`, `planned_quantity` (optional `sku`)
- `planned_purchases`: `date`, `planned_purchase_quantity` (optional `sku`)
- `stock_by_sku`: `sku`, `current_stock` with optional overrides:
  - `safety_stock`, `lead_time_days`, `min_order_qty`, `order_multiple`, `service_level`
- `config`: `parameter`, `value`

Example `config` parameters:
- `product_master_path` (e.g. `C:\Users\elmer\Downloads\Product range.csv`)
- `stock_file_path` (e.g. `C:\Users\elmer\Downloads\Stock.xlsx`)
- `committed_sales_path` (e.g. `C:\Users\elmer\Downloads\planned_sales.xlsx`)
- `planned_purchases_path` (e.g. `C:\Users\elmer\Downloads\planned_purchases.xlsx`)
- `historical_sales_path` (e.g. `C:\Users\elmer\Downloads\4. Odoo Verkoop History (gefactureerd datum) - data (2).csv`)
- `product_category` (optional filter)
- `product_sub_category` (optional filter)
- `product_gender` (optional filter)
- `product_skus` (optional comma-separated SKU list)
- `current_stock`
- `safety_stock`
- `lead_time_days`
- `min_order_qty`
- `order_multiple`
- `service_level`
- `planning_start_month` (e.g. `2026-01`)
- `planning_end_month` (e.g. `2026-12`)
- `forecast_periods`
- `moving_average_weeks`

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── demand_planner.py       # Forecasting logic
│   ├── inventory_manager.py     # Inventory calculations
│   └── excel_generator.py       # Excel template creation
├── data/                         # Sample datasets
├── templates/                    # Generated Excel outputs
├── requirements.txt
├── main.py                       # Example usage
└── README.md
```

## Key Classes

### DemandPlanner
- `load_historical_sales()`: Load sales history
- `calculate_average_daily_demand()`: Get baseline demand
- `calculate_daily_demand_std()`: Measure demand variability
- `forecast_weekly()`: Generate weekly forecast periods
- `build_demand_plan()`: Use planned sales with forecast fallback

### InventoryManager
- `calculate_safety_stock()`: Compute safety stock from service level and variability
- `calculate_optimal_stock()`: Determine reorder point
- `plan_purchases()`: Generate lead-time-aware purchase recommendations
- `update_stock()`: Track inventory changes
- `get_stock_status()`: Check current status

### ExcelGenerator
- `create_planning_template()`: Generate blank planning template
- `create_forecast_report()`: Export data to Excel

## Example Workflow

1. **Load your data**: Add historical sales in `data/` folder
2. **Run analysis**: Use `main.py` to process your data
3. **Review forecast**: Check the generated Excel templates
4. **Plan purchases**: Use purchase recommendations for procurement

## Configuration

Primary configuration can be maintained in `data/config.csv` or `data/config.xlsx`.

If config file is missing, defaults in `main.py` are used.

Stock file mapping:
- SKU column accepted: `Interne referentie` (or `sku`)
- Current stock column accepted: `Beschikbare voorraad` (or `current_stock`)

Committed sales mapping:
- Quantity column: `Verplaatsingen/Orderregel/Aantal te leveren`
- Planned date column: `Verplaatsingen/Geplande datum`
- Product/SKU source: `Verplaatsingen/Orderregel/Product` (SKU parsed from `[123456] ...`)
- Only rows with quantity > 0 are included

Planned purchases mapping:
- SKU column: `Orderregels/Product/Interne referentie`
- Quantity column: `Orderregels/Hoeveelheid`
- Expected receipt date: `Orderregels/Verwachte levering`
- Only rows with quantity > 0 are included

Historical sales (Odoo invoiced history) mapping:
- Date column: `dateinvoice` (parsed as day-first, e.g. `31-12-2025`)
- SKU column: `prodid`
- Quantity column: `qtyinv` (fallback `qtyorder`)

## Output Files

Generated files are saved in the `templates/` folder:
- `supply_demand_plan.xlsx`: Planning template with formulas
- `forecast_report.xlsx`: Detailed forecast data

## Requirements

- Python 3.8+
- pandas
- openpyxl
- statsmodels
- numpy

## License

MIT

## Support

For questions or issues, refer to the copilot-instructions.md file or update the modules based on your specific needs.
