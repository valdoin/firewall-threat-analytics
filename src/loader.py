import pandas as pd
import numpy as np
import streamlit as st
import os

RAW_LOG_COLUMNS = ["date", "ip_src", "ip_dst", "protocol", "src_port", "dst_port", "rule_id", "action", "in_interface", "out_interface", "fw"]

def parse_raw_log(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw_log column (semicolon-separated) into proper columns."""
    parsed = df["raw_log"].str.strip().str.split(";", expand=True)
    parsed.columns = RAW_LOG_COLUMNS[:len(parsed.columns)]
    parsed["date"] = pd.to_datetime(parsed["date"], errors="coerce")
    for col in ["src_port", "dst_port", "rule_id"]:
        if col in parsed.columns:
            parsed[col] = pd.to_numeric(parsed[col], errors="coerce")
    parsed.drop(columns=["fw"], errors="ignore", inplace=True)
    return parsed

@st.cache_data(show_spinner="Chargement des données...")
def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Format non supporté: {ext}")

    # Handle raw_log format
    if "raw_log" in df.columns:
        df = parse_raw_log(df)
    
    # Normalize columns
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Convert date
    if "date" in df.columns:
        if pd.api.types.is_numeric_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], unit="ms", errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Numeric conversions
    for col in ["src_port", "dst_port", "rule_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop fw column if present
    df.drop(columns=["fw"], errors="ignore", inplace=True)
    
    # Normalize action/protocol
    if "action" in df.columns:
        df["action"] = df["action"].str.upper().str.strip()
    if "protocol" in df.columns:
        df["protocol"] = df["protocol"].str.upper().str.strip()
    
    return df


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if filters.get("date_range"):
        d0, d1 = filters["date_range"]
        df = df[(df["date"] >= pd.Timestamp(d0)) & (df["date"] <= pd.Timestamp(d1))]
    if filters.get("protocols"):
        df = df[df["protocol"].isin(filters["protocols"])]
    if filters.get("actions"):
        df = df[df["action"].isin(filters["actions"])]
    if filters.get("port_range"):
        lo, hi = filters["port_range"]
        df = df[(df["dst_port"] >= lo) & (df["dst_port"] <= hi)]
    if filters.get("rule_ids"):
        df = df[df["rule_id"].isin(filters["rule_ids"])]
    return df
