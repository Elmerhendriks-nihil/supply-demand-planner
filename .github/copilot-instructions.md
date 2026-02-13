# Supply and Demand Planning Tool

## Project Overview
A Python-based tool for supply chain and demand planning that helps manage inventory based on:
- Current stock levels
- Historical sales data
- Planned/forecasted sales
- Planned purchases

## Technology Stack
- **Language**: Python 3.8+
- **Data Processing**: pandas, openpyxl
- **Forecasting**: statsmodels
- **Output**: Excel templates with formulas

## Project Structure
```
.
├── src/
│   ├── __init__.py
│   ├── demand_planner.py      # Core forecasting logic
│   ├── inventory_manager.py    # Inventory calculations
│   └── excel_generator.py      # Excel template creation
├── data/                        # Sample data files
├── templates/                   # Excel template outputs
├── requirements.txt
└── README.md
```

## Development Steps
1. ✅ Project scaffolding
2. Create core demand forecasting module
3. Build inventory management logic
4. Develop Excel template generator
5. Test with sample data
