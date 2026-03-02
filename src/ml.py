import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import streamlit as st


@st.cache_data(show_spinner="Calcul des features ML...")
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-IP features for anomaly detection."""
    grp = df.groupby("ip_src")
    
    feats = pd.DataFrame()
    feats["events"] = grp.size()
    feats["uniq_dst"] = grp["ip_dst"].nunique()
    feats["uniq_ports"] = grp["dst_port"].nunique()
    
    tcp = df[df["protocol"] == "TCP"].groupby("ip_src").size()
    udp = df[df["protocol"] == "UDP"].groupby("ip_src").size()
    feats["tcp_events"] = tcp.reindex(feats.index, fill_value=0)
    feats["udp_events"] = udp.reindex(feats.index, fill_value=0)
    feats["tcp_ratio"] = feats["tcp_events"] / feats["events"].replace(0, np.nan)
    feats["udp_ratio"] = feats["udp_events"] / feats["events"].replace(0, np.nan)
    
    priv = df[df["dst_port"] < 1024].groupby("ip_src").size()
    feats["priv_port_events"] = priv.reindex(feats.index, fill_value=0)
    feats["priv_port_ratio"] = feats["priv_port_events"] / feats["events"].replace(0, np.nan)
    
    # Max events per minute (burst)
    if "date" in df.columns and df["date"].notna().any():
        df2 = df.copy()
        df2["minute"] = df2["date"].dt.floor("min")
        burst = df2.groupby(["ip_src", "minute"]).size().groupby("ip_src").max()
        feats["max_events_per_min"] = burst.reindex(feats.index, fill_value=1)
    else:
        feats["max_events_per_min"] = 1
    
    feats = feats.fillna(0)
    feats.index.name = "ip_src"
    feats = feats.reset_index()
    return feats


def run_isolation_forest(feats: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    feature_cols = ["events", "uniq_dst", "uniq_ports", "tcp_ratio", "udp_ratio",
                    "priv_port_ratio", "max_events_per_min"]
    X = feats[feature_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    clf.fit(Xs)
    
    feats = feats.copy()
    feats["anomaly_score_raw"] = clf.score_samples(Xs)
    feats["is_anomaly"] = clf.predict(Xs) == -1
    return feats
