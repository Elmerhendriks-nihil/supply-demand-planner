"""Streamlit UI for the supply and demand planning tool."""

from pathlib import Path

import pandas as pd
import streamlit as st

import main as planner_main


UPLOAD_DIR = Path("data/ui_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(uploaded_file) -> str | None:
    """Save uploaded file to disk and return absolute path."""
    if uploaded_file is None:
        return None
    target = UPLOAD_DIR / uploaded_file.name
    with target.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(target.resolve())


def _download_file_button(path: str, label: str, mime: str):
    """Render a download button for a file path."""
    file_path = Path(path)
    if not file_path.exists():
        return
    st.download_button(
        label=label,
        data=file_path.read_bytes(),
        file_name=file_path.name,
        mime=mime,
    )


def main():
    st.set_page_config(page_title="Supply & Demand Planner", layout="wide")
    st.title("Supply & Demand Planner")
    st.caption("Upload planning files, run calculation, and download the Excel report.")

    with st.sidebar:
        st.header("Planning Window")
        planning_start = st.text_input("Planning Start Month (YYYY-MM)", value="2026-01")
        planning_end = st.text_input("Planning End Month (YYYY-MM)", value="2026-12")
        st.header("Uploads")
        product_master = st.file_uploader("Product Master", type=["csv"])
        historical_sales = st.file_uploader("Historical Sales", type=["csv", "xlsx"])
        committed_sales = st.file_uploader("Committed Sales", type=["csv", "xlsx"])
        planned_purchases = st.file_uploader("Planned Purchases", type=["csv", "xlsx"])
        stock_file = st.file_uploader("Current Stock", type=["csv", "xlsx"])

    st.subheader("Run Planner")
    run_clicked = st.button("Run Planning")
    if not run_clicked:
        st.info("Upload files and click 'Run Planning'.")
        return

    if product_master is None or historical_sales is None:
        st.error("Product Master and Historical Sales are required.")
        return

    overrides = {
        "planning_start_month": planning_start.strip(),
        "planning_end_month": planning_end.strip(),
    }

    path_mappings = {
        "product_master_path": _save_upload(product_master),
        "historical_sales_path": _save_upload(historical_sales),
        "committed_sales_path": _save_upload(committed_sales),
        "planned_purchases_path": _save_upload(planned_purchases),
        "stock_file_path": _save_upload(stock_file),
    }
    for key, value in path_mappings.items():
        if value:
            overrides[key] = value

    with st.spinner("Running planning engine..."):
        try:
            results = planner_main.main(config_overrides=overrides)
        except Exception as exc:
            st.exception(exc)
            return

    st.success("Planning completed.")

    c1, c2, c3 = st.columns(3)
    c1.metric("SKUs in Scope", int(results["purchase_plan"]["sku"].nunique()) if "sku" in results["purchase_plan"].columns else 1)
    c2.metric("Action List Rows", len(results["action_list"]))
    c3.metric("Stock Outlook Rows", len(results["stock_outlook"]))

    st.subheader("Downloads")
    _download_file_button(
        "templates/forecast_report.xlsx",
        "Download Forecast Report",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    _download_file_button(
        "templates/supply_demand_plan.xlsx",
        "Download Planning Template",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Action List Preview")
    st.dataframe(results["action_list"].head(200), use_container_width=True)

    st.subheader("Stock Outlook Preview")
    st.dataframe(results["stock_outlook"].head(200), use_container_width=True)


if __name__ == "__main__":
    main()
