import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import requests
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from src.loader import load_data, apply_filters
from src.ml import compute_features, run_isolation_forest
from src.llm import generate_report

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Challenge SISE-OPSIE 2026 – Analyse des menaces firewall",
    page_icon="images/logo.png", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — Hacker UI ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

:root{
  --bg: #06080f;
  --panel: rgba(12, 16, 30, 0.75);
  --border: rgba(0, 255, 140, 0.22);
  --neon: #00ff8c;
  --neon2: #00d4ff;
  --text: #d7ffe9;
  --muted: rgba(215, 255, 233, 0.72);
  --danger: #ff3860;
  --warn: #ffcc00;
}

html, body, [data-testid="stApp"] {
  background: radial-gradient(1200px 600px at 10% 10%, rgba(0,255,140,0.08), transparent 60%),
              radial-gradient(900px 500px at 90% 30%, rgba(0,212,255,0.07), transparent 55%),
              var(--bg);
  color: var(--text);
  font-family: 'Share Tech Mono', monospace !important;
}

.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1700px; }

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(0,255,140,0.10) 0%, rgba(0,0,0,0.0) 70%), var(--bg);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

div[data-baseweb="select"] > div{
  background: rgba(10, 14, 26, 0.65) !important;
  border: 1px solid var(--border) !important;
  box-shadow: 0 0 0 1px rgba(0,255,140,0.08), 0 0 18px rgba(0,255,140,0.08);
}

.stButton > button{
  background: linear-gradient(135deg, rgba(0,255,140,0.18) 0%, rgba(0,212,255,0.12) 100%);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.75rem 1.2rem;
  font-weight: 700;
  letter-spacing: 0.5px;
  transition: transform .15s ease, box-shadow .15s ease;
  box-shadow: 0 0 22px rgba(0,255,140,0.10);
}
.stButton > button:hover{
  transform: translateY(-1px);
  box-shadow: 0 0 26px rgba(0,255,140,0.18);
}

.metric-card{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px;
  text-align: center;
  box-shadow: 0 0 0 1px rgba(0,255,140,0.06), 0 0 20px rgba(0,255,140,0.08);
  backdrop-filter: blur(6px);
}
.metric-value{
  font-size: 2.1rem;
  font-weight: 800;
  color: var(--neon);
  text-shadow: 0 0 10px rgba(0,255,140,0.25);
}
.metric-label{
  font-size: .95rem;
  color: var(--muted);
}

button[data-baseweb="tab"]{ color: var(--muted) !important; }
button[data-baseweb="tab"][aria-selected="true"]{
  color: var(--text) !important;
  border-bottom: 2px solid var(--neon) !important;
}

[data-testid="stDataFrame"]{
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}

h1, h2, h3 { text-shadow: 0 0 12px rgba(0,255,140,0.10); }

[data-testid="stApp"]:before{
  content:"";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background: repeating-linear-gradient(
    to bottom,
    rgba(255,255,255,0.02),
    rgba(255,255,255,0.02) 1px,
    transparent 2px,
    transparent 6px
  );
  opacity: 0.14;
  mix-blend-mode: overlay;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="border:1px solid rgba(0,255,140,0.22);
            background:rgba(12,16,30,0.65);
            border-radius:14px;padding:18px;margin-bottom:14px;
            box-shadow:0 0 22px rgba(0,255,140,0.10);">
  <div style="font-size:22px;color:#00ff8c;font-weight:800;">
    Challenge SISE-OPSIE 2026 — Analyse des menaces firewall
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: Holidays (FR) + time
# ─────────────────────────────────────────────────────────────────────────────
def is_holiday_fr(ts: pd.Timestamp) -> bool:
    """Best-effort FR holiday check. Fallback: weekend only."""
    if ts is None or pd.isna(ts):
        return False
    try:
        import holidays  # pip install holidays
        fr = holidays.country_holidays("FR")
        return (ts.date() in fr) or (ts.weekday() >= 5)
    except Exception:
        return ts.weekday() >= 5

def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts, errors="coerce")
    if ts is pd.NaT:
        return ts
    # si naive, on suppose UTC
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

# ─────────────────────────────────────────────────────────────────────────────
# GeoIP (ip-api)
# ─────────────────────────────────────────────────────────────────────────────
PRIVATE_PREFIXES = (
    "10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.", "172.21.",
    "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
    "172.30.", "172.31.", "127.", "169.254."
)

def is_private_ip(ip: str) -> bool:
    ip = str(ip)
    return ip.startswith(PRIVATE_PREFIXES) or ip in ("0.0.0.0",) or ip.lower() == "nan"

@st.cache_data(show_spinner=False)
def geoip_lookup_ip_api(ip: str) -> dict:
    ip = str(ip).strip()
    if not ip or is_private_ip(ip):
        return {"status": "skip"}

    url = "http://ip-api.com/json/" + ip
    params = {"fields": "status,message,country,city,lat,lon,org,isp,query,timezone"}
    try:
        r = requests.get(url, params=params, timeout=4)
        data = r.json()
        if data.get("status") != "success":
            return {"status": "fail", "message": data.get("message", "unknown")}
        return {
            "status": "success",
            "ip": data.get("query", ip),
            "country": data.get("country"),
            "city": data.get("city"),
            "lat": data.get("lat"),
            "lon": data.get("lon"),
            "org": data.get("org"),
            "isp": data.get("isp"),
            "timezone": data.get("timezone"),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def enrich_geo(df_in: pd.DataFrame, ip_col: str, max_ips: int = 200) -> pd.DataFrame:
    ips = df_in[ip_col].dropna().astype(str).unique().tolist()
    ips = [ip for ip in ips if not is_private_ip(ip)][:max_ips]

    rows = []
    for ip in ips:
        info = geoip_lookup_ip_api(ip)
        if info.get("status") == "success":
            rows.append(info)
        time.sleep(0.05)  # limiter le rate-limit

    if not rows:
        return df_in

    geo = pd.DataFrame(rows).drop_duplicates(subset=["ip"]).rename(columns={"ip": ip_col})
    return df_in.merge(
        geo[[ip_col, "lat", "lon", "country", "city", "org", "isp", "timezone"]],
        on=ip_col,
        how="left"
    )

# ─────────────────────────────────────────────────────────────────────────────
# "Sun over Earth" (terminator line - approx)
# ─────────────────────────────────────────────────────────────────────────────
def _solar_declination_approx(dt_utc: datetime) -> float:
    # approx NOAA declination (OK pour UI)
    doy = dt_utc.timetuple().tm_yday
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (dt_utc.hour - 12) / 24.0)
    decl = (0.006918
            - 0.399912 * np.cos(gamma)
            + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2*gamma)
            + 0.000907 * np.sin(2*gamma)
            - 0.002697 * np.cos(3*gamma)
            + 0.00148  * np.sin(3*gamma))
    return float(np.degrees(decl))

def _subsolar_lon_approx(dt_utc: datetime) -> float:
    return float(-15.0 * (dt_utc.hour + dt_utc.minute/60.0 + dt_utc.second/3600.0))

def _terminator_line(dt_utc: datetime, n=181):
    decl = np.radians(_solar_declination_approx(dt_utc))
    sublon = np.radians(_subsolar_lon_approx(dt_utc))
    lons = np.linspace(-np.pi, np.pi, n)
    tan_decl = np.tan(decl)
    if abs(tan_decl) < 1e-6:
        tan_decl = 1e-6
    lats = np.arctan(-np.cos(lons - sublon) / tan_decl)
    return np.degrees(lats).tolist(), np.degrees(lons).tolist()

def earth_globe_figure(points_src=None, points_dst=None, lines=None, dt_utc=None, title=None):
    fig = go.Figure()

    if points_src is not None and len(points_src.get("lat", [])):
        fig.add_trace(go.Scattergeo(
            lat=points_src["lat"], lon=points_src["lon"],
            mode="markers",
            marker=dict(size=8),
            text=points_src.get("text"),
            name="Sources"
        ))

    if points_dst is not None and len(points_dst.get("lat", [])):
        fig.add_trace(go.Scattergeo(
            lat=points_dst["lat"], lon=points_dst["lon"],
            mode="markers",
            marker=dict(size=10),
            text=points_dst.get("text"),
            name="Destinations"
        ))

    if lines:
        for ln in lines:
            fig.add_trace(go.Scattergeo(
                lat=ln["lat"], lon=ln["lon"],
                mode="lines",
                line=dict(width=1.8, color=ln.get("color", "rgba(0,255,140,0.35)")),
                opacity=0.65,
                showlegend=False
            ))

    if dt_utc is not None:
        lats, lons = _terminator_line(dt_utc)
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons,
            mode="lines",
            line=dict(width=2, color="rgba(0,212,255,0.35)"),
            name="Terminator",
            showlegend=False
        ))

    fig.update_layout(
        height=560,
        margin=dict(t=35, b=0, l=0, r=0),
        title=dict(text=title or "", x=0.02, font=dict(size=14)),
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            projection_type="orthographic",
            showland=True,
            landcolor="rgba(255,255,255,0.05)",
            showocean=True,
            oceancolor="rgba(0,0,0,0.15)",
            showcountries=True,
            countrycolor="rgba(0,255,140,0.18)",
            coastlinecolor="rgba(0,255,140,0.12)",
        ),
    )
    return fig

# ─── Sidebar – File loader ───────────────────────────────────────────────────
with st.sidebar:

    # ─── Page config ────────────────────────────────────────────────────────────

    uploaded = st.file_uploader("Charger un fichier (Parquet / CSV)", type=["parquet", "csv"])
    data_path = None

    if uploaded:
        import tempfile
        suffix = ".parquet" if uploaded.name.endswith(".parquet") else ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            data_path = tmp.name
    else:
        for candidate in [
            "data/firewall_clean.csv",
            "data/clean_logs_nov2025_feb2026.parquet",
            "data/logs_export.parquet",
            "/data/clean_logs_nov2025_feb2026.parquet",
            "/data/logs_export.parquet",
        ]:
            if os.path.exists(candidate):
                data_path = candidate
                st.info(f"Fichier auto-détecté: `{os.path.basename(data_path)}`")
                break

    if not data_path:
        st.warning("⚠️ Aucun fichier chargé. Veuillez uploader un fichier Parquet ou CSV.")
        st.stop()

@st.cache_data(show_spinner=False)
def cached_load(path: str) -> pd.DataFrame:
    return load_data(path)

df_raw = cached_load(data_path)

# ─── Sidebar – Filters ───────────────────────────────────────────────────────
with st.sidebar:
    st.subheader(" Filtres globaux")

    if "date" in df_raw.columns and df_raw["date"].notna().any():
        dmin = pd.to_datetime(df_raw["date"], errors="coerce").min().date()
        dmax = pd.to_datetime(df_raw["date"], errors="coerce").max().date()
        date_range = st.date_input("Plage de dates", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        filters_date = date_range if len(date_range) == 2 else (dmin, dmax)
    else:
        filters_date = None

    protocols_avail = sorted(df_raw["protocol"].dropna().unique().tolist()) if "protocol" in df_raw.columns else []
    sel_protocols = st.multiselect("Protocoles", protocols_avail, default=protocols_avail)

    actions_avail = sorted(df_raw["action"].dropna().unique().tolist()) if "action" in df_raw.columns else []
    sel_actions = st.multiselect("Actions", actions_avail, default=actions_avail)

    st.markdown("**Plage de ports (dst)**")
    port_lo, port_hi = st.slider("", 0, 65535, (0, 65535), step=1)

    if "rule_id" in df_raw.columns:
        rule_ids_avail = sorted(df_raw["rule_id"].dropna().unique().astype(int).tolist())
        sel_rules = st.multiselect("Rule IDs", rule_ids_avail, default=rule_ids_avail)
    else:
        sel_rules = []

    top_n = st.slider("Top N", 5, 50, 10)

    st.divider()
    network_plan = st.text_input("Plan d'adressage (préfixe)", value="159.84.")

filters = {
    "date_range": filters_date,
    "protocols": sel_protocols if sel_protocols else None,
    "actions": sel_actions if sel_actions else None,
    "port_range": (port_lo, port_hi),
    "rule_ids": sel_rules if sel_rules else None,
}

df = apply_filters(df_raw, filters)
if df.empty:
    st.error("Aucune donnée après filtrage. Ajustez les filtres.")
    st.stop()

# Normalisation actions
ALLOW = {"PERMIT", "ACCEPT", "ALLOW", "ALLOWED"}
DENY  = {"DENY", "DROP", "REJECT", "BLOCK", "DENIED"}

COLOR_ALLOW = "#00ff8c"   # vert néon
COLOR_DENY  = "#ff3860"   # rouge
ACTION_COLOR_MAP = {
    "ALLOW": COLOR_ALLOW,
    "PERMIT": COLOR_ALLOW,
    "ACCEPT": COLOR_ALLOW,
    "DENY": COLOR_DENY,
    "DROP": COLOR_DENY,
    "REJECT": COLOR_DENY,
    "BLOCK": COLOR_DENY,
    "DENIED": COLOR_DENY,
}

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Vue d'ensemble", "Règles & Ports", "Analyse des IP", "ML / Anomalies", "Rapport IA"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Vue d'ensemble des événements")

    n_events = len(df)
    n_src = df["ip_src"].nunique() if "ip_src" in df.columns else 0
    n_dst = df["ip_dst"].nunique() if "ip_dst" in df.columns else 0
    n_ports = df["dst_port"].nunique() if "dst_port" in df.columns else 0

    action_norm = df["action"].astype(str).str.upper().str.strip() if "action" in df.columns else pd.Series("", index=df.index)
    allow_mask = action_norm.isin(ALLOW)
    deny_mask  = action_norm.isin(DENY)

    n_allow = int(allow_mask.sum())
    n_deny = int(deny_mask.sum())
    ratio = f"{(n_allow/n_events):.1%} / {(n_deny/n_events):.1%}" if n_events > 0 else "N/A"

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, val, label in zip(
        [col1, col2, col3, col4, col5],
        [f"{n_events:,}", f"{n_src:,}", f"{n_dst:,}", f"{n_ports:,}", ratio],
        ["Total Événements", "IP Sources", "IP Destinations", "Ports Uniques", "ALLOW / DENY"]
    ):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("ALLOW vs DENY")
        if "action" in df.columns:
            action_counts = df["action"].astype(str).str.upper().value_counts().reset_index()
            action_counts.columns = ["action", "count"]
            action_counts["action"] = action_counts["action"].astype(str).str.upper().str.strip()
            fig = px.pie(
                action_counts, values="count", names="action", hole=0.4,
                color="action", color_discrete_map=ACTION_COLOR_MAP
            )
            fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Répartition par protocole")
        if "protocol" in df.columns:
            all_protos = ["TCP", "UDP"]
            proto_counts = df["protocol"].astype(str).str.upper().value_counts().reindex(all_protos, fill_value=0).reset_index()
            proto_counts.columns = ["protocol", "count"]
            fig2 = px.bar(proto_counts, x="protocol", y="count", text="count")
            fig2.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig2.update_layout(showlegend=False, margin=dict(t=10, b=0), height=300)
            st.plotly_chart(fig2, use_container_width=True)

    with c3:
        st.subheader(f"Top {top_n} règles utilisées (table)")
        if "rule_id" in df.columns:
            rule_counts = df["rule_id"].value_counts().head(top_n).reset_index()
            rule_counts.columns = ["rule_id", "count"]
            rule_counts["rule_id"] = rule_counts["rule_id"].astype(int)

            st.dataframe(
                rule_counts,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Colonne `rule_id` absente.")

    if "date" in df.columns and df["date"].notna().any():
        st.subheader("Événements dans le temps (par heure)")
        df_time = df.copy()
        df_time["date"] = pd.to_datetime(df_time["date"], errors="coerce")
        df_time = df_time.dropna(subset=["date"])
        df_time["hour"] = df_time["date"].dt.floor("h")
        timeline = df_time.groupby(["hour", "action"]).size().reset_index(name="count")
        timeline["action"] = timeline["action"].astype(str).str.upper().str.strip()
        fig_t = px.line(
            timeline, x="hour", y="count", color="action",
            color_discrete_map=ACTION_COLOR_MAP
        )
        fig_t.update_layout(height=300, margin=dict(t=10))
        st.plotly_chart(fig_t, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – RULES & PORTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Analyse des règles & ports")

    df_tcp = df[df["protocol"].astype(str).str.upper() == "TCP"] if "protocol" in df.columns else df.iloc[0:0]
    df_udp = df[df["protocol"].astype(str).str.upper() == "UDP"] if "protocol" in df.columns else df.iloc[0:0]

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top 5 règles — TCP (table)")
        if not df_tcp.empty and "rule_id" in df_tcp.columns:
            top_tcp = df_tcp["rule_id"].value_counts().head(5).reset_index()
            top_tcp.columns = ["rule_id", "count"]
            top_tcp["rule_id"] = top_tcp["rule_id"].astype(int)

            st.dataframe(top_tcp, use_container_width=True, hide_index=True)
        else:
            st.info("Pas de trafic TCP (ou `rule_id` absent).")

    with col_b:
        st.subheader("Top 10 règles — UDP")
        if not df_udp.empty and "rule_id" in df_udp.columns:
            top_udp = df_udp["rule_id"].value_counts().head(10).reset_index()
            top_udp.columns = ["rule_id", "count"]
            fig_udp = px.bar(top_udp, x="count", y=top_udp["rule_id"].astype(str), orientation="h", text="count")
            fig_udp.update_traces(texttemplate="%{text:,}")
            fig_udp.update_layout(yaxis_title="Rule ID", height=280, margin=dict(t=10))
            st.plotly_chart(fig_udp, use_container_width=True)
            st.dataframe(top_udp, use_container_width=True, hide_index=True)
        else:
            st.info("Pas de trafic UDP (ou rule_id absent).")

    st.divider()
    st.subheader("TCP — Rule ID → Port DST → Action (Sankey)")

    sankey_k = st.slider("Nombre de flux top K", 10, 100, 30, key="sankey_k")
    threshold = st.number_input("Seuil minimal de count", min_value=1, value=10, step=1, key="sankey_thresh")

    if not df_tcp.empty and {"rule_id", "dst_port", "action"}.issubset(df_tcp.columns):
        cross = df_tcp.groupby(["rule_id", "dst_port", "action"]).size().reset_index(name="count")
        cross = cross[cross["count"] >= threshold].nlargest(sankey_k, "count")

        if cross.empty:
            st.info("Pas de données pour le Sankey avec ces filtres.")
        else:
            rule_labels = [f"Rule {r}" for r in cross["rule_id"].unique()]
            port_labels = [f"Port {p}" for p in cross["dst_port"].unique()]
            action_labels = list(cross["action"].unique())
            all_labels = rule_labels + port_labels + action_labels

            rule_map = {r: i for i, r in enumerate(cross["rule_id"].unique())}
            port_map = {p: len(rule_labels) + i for i, p in enumerate(cross["dst_port"].unique())}
            action_map = {a: len(rule_labels) + len(port_labels) + i for i, a in enumerate(cross["action"].unique())}

            src1 = [rule_map[r] for r in cross["rule_id"]]
            tgt1 = [port_map[p] for p in cross["dst_port"]]
            val1 = cross["count"].tolist()
            src2 = [port_map[p] for p in cross["dst_port"]]
            tgt2 = [action_map[a] for a in cross["action"]]
            val2 = cross["count"].tolist()

            colors = ["rgba(255,56,96,0.35)" if str(a).upper() in DENY else "rgba(0,255,140,0.35)" for a in cross["action"]]

            fig_sankey = go.Figure(go.Sankey(
                node=dict(pad=15, thickness=20, line=dict(color="rgba(0,255,140,0.25)", width=0.5), label=all_labels),
                link=dict(source=src1 + src2, target=tgt1 + tgt2, value=val1 + val2, color=colors + colors)
            ))
            fig_sankey.update_layout(height=520, margin=dict(t=10))
            st.plotly_chart(fig_sankey, use_container_width=True)
            st.dataframe(cross.sort_values("count", ascending=False), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – IP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Analyse des IP sources")

    sc1, sc2 = st.columns(2)

    with sc1:
        st.subheader("Top 5 IP sources")
        top5_src = df["ip_src"].value_counts().head(5).reset_index()
        top5_src.columns = ["ip_src", "events"]
        fig_top5 = px.bar(top5_src, x="events", y="ip_src", orientation="h", text="events")
        fig_top5.update_traces(texttemplate="%{text:,}")
        fig_top5.update_layout(height=250, margin=dict(t=10), yaxis_title="")
        st.plotly_chart(fig_top5, use_container_width=True)

    with sc2:
        st.subheader("Top 10 ports <1024 autorisés (ALLOW) — table")
        if {"dst_port", "action"}.issubset(df.columns):
            allow_mask = df["action"].astype(str).str.upper().str.strip().isin(ALLOW)
            priv_allow = df[(df["dst_port"] < 1024) & allow_mask]

            if not priv_allow.empty:
                top_priv = priv_allow["dst_port"].value_counts().head(10).reset_index()
                top_priv.columns = ["dst_port", "count"]
                top_priv["dst_port"] = top_priv["dst_port"].astype(int)

                st.dataframe(top_priv, use_container_width=True, hide_index=True)
            else:
                st.info("Aucun accès ALLOW vers ports <1024.")
        else:
            st.info("Colonnes `dst_port` / `action` absentes.")

    st.subheader(f"IP sources hors plan d'adressage (ne commençant pas par `{network_plan}`)")
    if network_plan and "ip_src" in df.columns:
        out_plan = df[~df["ip_src"].astype(str).str.startswith(network_plan)]
        st.caption(f"{out_plan['ip_src'].nunique():,} IP(s) hors plan sur {df['ip_src'].nunique():,} total")

        out_summary = out_plan.groupby("ip_src").size().reset_index(name="events").sort_values("events", ascending=False)

        if "action" in out_plan.columns:
            allow_mask = out_plan["action"].astype(str).str.upper().isin(ALLOW)
            deny_mask = out_plan["action"].astype(str).str.upper().isin(DENY)
            allow_by_ip = out_plan[allow_mask].groupby("ip_src").size().reset_index(name="allow")
            deny_by_ip = out_plan[deny_mask].groupby("ip_src").size().reset_index(name="deny")
            out_summary = out_summary.merge(allow_by_ip, on="ip_src", how="left").merge(deny_by_ip, on="ip_src", how="left").fillna(0)

        st.dataframe(out_summary.head(50), use_container_width=True, hide_index=True)

        st.markdown("### Analyse IA — Triage des IP hors plan")
        top_out = out_summary.head(20).copy()

        out_events = out_plan[out_plan["ip_src"].isin(top_out["ip_src"])].copy()
        payload_rows = []
        for ip in top_out["ip_src"].tolist():
            ip_e = out_events[out_events["ip_src"] == ip]
            payload_rows.append({
                "ip_src": str(ip),
                "events": int(len(ip_e)),
                "top_ports": (ip_e["dst_port"].value_counts().head(5).to_dict() if "dst_port" in ip_e.columns else {}),
                "top_rules": (ip_e["rule_id"].value_counts().head(5).to_dict() if "rule_id" in ip_e.columns else {}),
                "action_counts": (ip_e["action"].astype(str).str.upper().value_counts().to_dict() if "action" in ip_e.columns else {}),
            })

        if st.button("Générer triage IA (hors plan)", key="llm_outplan"):
            with st.spinner("Génération IA..."):
                report = generate_report(
                    ip="OUT_OF_PLAN",
                    stats={"mode": "triage_out_of_plan", "network_plan": network_plan, "n_out_ips": int(out_summary["ip_src"].nunique())},
                    examples=payload_rows,
                )
            st.markdown(report)

    st.divider()
    st.subheader("Carte géographique — Origine des connexions (GeoIP ip-api)")

    geo_mode = st.radio(
        "Géolocalisation",
        ["Sources (ip_src)", "Destinations (ip_dst)", "Les deux"],
        horizontal=True,
    )
    max_geo = st.slider("Limiter le nombre d'IP géolocalisées (perf)", 20, 500, 200, step=10)

    # ── 1) Construire des tables AGRÉGÉES (petites) au lieu de merge sur df complet ──
    src_points = None
    dst_points = None

    if geo_mode in ["Sources (ip_src)", "Les deux"] and "ip_src" in df.columns:
        src_points = (
            df["ip_src"].dropna().astype(str)
            .value_counts()
            .head(max_geo)
            .reset_index()
        )
        src_points.columns = ["ip_src", "count"]
        src_points = enrich_geo(src_points, "ip_src", max_ips=max_geo).rename(
            columns={"lat": "src_lat", "lon": "src_lon"}
        )

    if geo_mode in ["Destinations (ip_dst)", "Les deux"] and "ip_dst" in df.columns:
        dst_points = (
            df["ip_dst"].dropna().astype(str)
            .value_counts()
            .head(max_geo)
            .reset_index()
        )
        dst_points.columns = ["ip_dst", "count"]
        dst_points = enrich_geo(dst_points, "ip_dst", max_ips=max_geo).rename(
            columns={"lat": "dst_lat", "lon": "dst_lon"}
        )

    # ── 2) Affichage carte ──
    fig_map = go.Figure()

    # Points sources
    if src_points is not None and {"src_lat", "src_lon"}.issubset(src_points.columns) and src_points["src_lat"].notna().any():
        sp = src_points.dropna(subset=["src_lat", "src_lon"]).head(300)
        fig_map.add_trace(go.Scattergeo(
            lat=sp["src_lat"],
            lon=sp["src_lon"],
            mode="markers",
            marker=dict(size=7),
            text=sp["ip_src"].astype(str) + " (" + sp["count"].astype(str) + ")",
            name="Sources"
        ))

    # Points destinations
    if dst_points is not None and {"dst_lat", "dst_lon"}.issubset(dst_points.columns) and dst_points["dst_lat"].notna().any():
        dp = dst_points.dropna(subset=["dst_lat", "dst_lon"]).head(200)
        fig_map.add_trace(go.Scattergeo(
            lat=dp["dst_lat"],
            lon=dp["dst_lon"],
            mode="markers",
            marker=dict(size=9),
            text=dp["ip_dst"].astype(str) + " (" + dp["count"].astype(str) + ")",
            name="Destinations"
        ))

    # ── 3) Lignes src -> dst (uniquement si "Les deux") ──
    if geo_mode == "Les deux" and src_points is not None and dst_points is not None:
        if {"ip_src", "ip_dst", "action"}.issubset(df.columns):
            pairs = (
                df.groupby(["ip_src", "ip_dst", "action"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(150)
            )

            pairs["ip_src"] = pairs["ip_src"].astype(str)
            pairs["ip_dst"] = pairs["ip_dst"].astype(str)

            pairs = pairs.merge(src_points[["ip_src", "src_lat", "src_lon"]], on="ip_src", how="left")
            pairs = pairs.merge(dst_points[["ip_dst", "dst_lat", "dst_lon"]], on="ip_dst", how="left")
            pairs = pairs.dropna(subset=["src_lat", "src_lon", "dst_lat", "dst_lon"])

            for _, r in pairs.iterrows():
                ok = str(r["action"]).upper() in ALLOW
                line_color = "rgba(0,255,140,0.35)" if ok else "rgba(255,56,96,0.35)"
                fig_map.add_trace(go.Scattergeo(
                    lat=[r["src_lat"], r["dst_lat"]],
                    lon=[r["src_lon"], r["dst_lon"]],
                    mode="lines",
                    line=dict(width=1.5, color=line_color),
                    opacity=0.6,
                    showlegend=False,
                ))

    fig_map.update_layout(
        height=650,
        margin=dict(t=10, b=0, l=0, r=0),
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            showland=True,
            landcolor="rgba(255,255,255,0.05)",
            showocean=True,
            oceancolor="rgba(0,0,0,0.1)",
            showcountries=True,
            countrycolor="rgba(0,255,140,0.18)",
            coastlinecolor="rgba(0,255,140,0.12)",
            projection_type="natural earth",
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.caption("Note: ip-api ne géolocalise pas les IP privées (RFC1918).")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – ML ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Détection d'anomalies — IsolationForest")

    col_ml1, col_ml2 = st.columns([1, 3])
    with col_ml1:
        contamination = st.slider("Contamination (% anomalies attendues)", 0.001, 0.20, 0.02, step=0.001, format="%.3f")
        run_ml = st.button("Lancer la détection", type="primary")

    if run_ml or "ml_results" in st.session_state:
        if run_ml:
            with st.spinner("Calcul des features et IsolationForest..."):
                feats = compute_features(df)
                results = run_isolation_forest(feats, contamination=contamination)
                st.session_state["ml_results"] = results

        results = st.session_state["ml_results"]
        anomalies = results[results["is_anomaly"]].sort_values("anomaly_score_raw")
        st.success(f"{len(anomalies)} anomalies détectées sur {len(results)} IP analysées")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.subheader("Top 20 IP suspectes")
            display_cols = ["ip_src", "events", "uniq_dst", "uniq_ports", "priv_port_ratio", "max_events_per_min", "anomaly_score_raw"]
            avail_cols = [c for c in display_cols if c in anomalies.columns]
            st.dataframe(anomalies[avail_cols].head(20), use_container_width=True, hide_index=True)

        with col_s2:
            st.subheader("Score d'anomalie vs événements")
            fig_ml = px.scatter(
                results.head(500), x="events", y="anomaly_score_raw",
                color="is_anomaly",
                hover_name="ip_src",
                hover_data={"events": ":,", "uniq_dst": True, "max_events_per_min": True},
                labels={"anomaly_score_raw": "Score (plus bas = plus suspect)", "events": "Nombre d'événements"}
            )
            fig_ml.update_layout(height=400, margin=dict(t=10))
            st.plotly_chart(fig_ml, use_container_width=True)

        st.subheader("Exploration des features")
        feat_cols = ["events", "uniq_dst", "uniq_ports", "tcp_ratio", "priv_port_ratio", "max_events_per_min"]
        avail_feat_cols = [c for c in feat_cols if c in results.columns]
        fig_parallel = px.parallel_coordinates(
            results.head(300),
            dimensions=avail_feat_cols,
            color="anomaly_score_raw",
            color_continuous_scale=px.colors.diverging.RdYlGn,
        )
        fig_parallel.update_layout(height=420)
        st.plotly_chart(fig_parallel, use_container_width=True)

        if not anomalies.empty:
            st.divider()
            st.subheader("Détail d'une IP suspecte — Globe + Timeline + IA")

            sel_suspect = st.selectbox("Choisir une IP suspecte", anomalies["ip_src"].head(50).astype(str).tolist(), key="ml_ip")
            sus_df = df[df["ip_src"].astype(str) == str(sel_suspect)].copy()

            if sus_df.empty:
                st.warning("Aucun événement pour cette IP.")
            else:
                if "date" in sus_df.columns:
                    sus_df["date"] = pd.to_datetime(sus_df["date"], errors="coerce")
                    sus_df = sus_df.dropna(subset=["date"]).sort_values("date")
                    sus_df["hour"] = sus_df["date"].dt.floor("h")
                else:
                    sus_df["hour"] = pd.NaT

                hours = sus_df["hour"].dropna().unique()
                hours = np.sort(hours) if len(hours) else np.array([])

                if len(hours):
                    idx = st.slider("Heure (défilement)", 0, int(len(hours)-1), int(len(hours)-1))
                    t_sel = pd.Timestamp(hours[idx])
                else:
                    t_sel = None

                left, right = st.columns([1.35, 1.0], vertical_alignment="top")

                with right:
                    st.markdown("### Timeline")
                    if "date" in sus_df.columns and sus_df["date"].notna().any():
                        tl = sus_df.groupby(["hour", "action"]).size().reset_index(name="count")
                        fig_tl = px.line(tl, x="hour", y="count", color="action")
                        fig_tl.update_layout(height=250, margin=dict(t=10))
                        st.plotly_chart(fig_tl, use_container_width=True)

                    st.markdown("### Contexte calendrier (France)")
                    if t_sel is not None:
                        holiday = is_holiday_fr(t_sel)
                        st.write(f"**Heure sélectionnée (UTC)** : `{t_sel}`")
                        st.write(f"**Jour férié / week-end ?** : {' OUI' if holiday else 'NON'}")
                    else:
                        st.write("Pas de date exploitable.")

                    st.markdown("### Derniers événements")
                    cols = [c for c in ["date", "ip_src", "ip_dst", "protocol", "dst_port", "rule_id", "action"] if c in sus_df.columns]
                    st.dataframe(sus_df.sort_values("date", ascending=False)[cols].head(25), use_container_width=True, hide_index=True)

                with left:
                    st.markdown("### Globe tactique (terminator jour/nuit)")
                    slice_df = sus_df[sus_df["hour"] == t_sel].copy() if t_sel is not None else sus_df.copy()

                    slice_geo = slice_df.copy()
                    if "ip_src" in slice_geo.columns:
                        slice_geo = enrich_geo(slice_geo, "ip_src", max_ips=10).rename(columns={"lat": "src_lat", "lon": "src_lon"})
                    if "ip_dst" in slice_geo.columns:
                        slice_geo = enrich_geo(slice_geo, "ip_dst", max_ips=10).rename(columns={"lat": "dst_lat", "lon": "dst_lon"})

                    points_src = None
                    if {"src_lat", "src_lon", "ip_src"}.issubset(slice_geo.columns) and slice_geo["src_lat"].notna().any():
                        s = slice_geo.dropna(subset=["src_lat", "src_lon"]).groupby(["ip_src", "src_lat", "src_lon"]).size().reset_index(name="count")
                        points_src = {"lat": s["src_lat"].tolist(), "lon": s["src_lon"].tolist(),
                                      "text": (s["ip_src"] + " (" + s["count"].astype(str) + ")").tolist()}

                    points_dst = None
                    if {"dst_lat", "dst_lon", "ip_dst"}.issubset(slice_geo.columns) and slice_geo["dst_lat"].notna().any():
                        d = slice_geo.dropna(subset=["dst_lat", "dst_lon"]).groupby(["ip_dst", "dst_lat", "dst_lon"]).size().reset_index(name="count")
                        d = d.sort_values("count", ascending=False).head(5)
                        points_dst = {"lat": d["dst_lat"].tolist(), "lon": d["dst_lon"].tolist(),
                                      "text": (d["ip_dst"] + " (" + d["count"].astype(str) + ")").tolist()}

                    lines = []
                    if {"src_lat", "src_lon", "dst_lat", "dst_lon", "action"}.issubset(slice_geo.columns):
                        pairs = (slice_geo.dropna(subset=["src_lat", "src_lon", "dst_lat", "dst_lon"])
                                 .groupby(["src_lat", "src_lon", "dst_lat", "dst_lon", "action"]).size()
                                 .reset_index(name="count").sort_values("count", ascending=False).head(30))
                        for _, r in pairs.iterrows():
                            ok = str(r["action"]).upper() in ALLOW
                            lines.append({
                                "lat": [r["src_lat"], r["dst_lat"]],
                                "lon": [r["src_lon"], r["dst_lon"]],
                                "color": "rgba(0,255,140,0.45)" if ok else "rgba(255,56,96,0.45)"
                            })

                    dt_utc = to_utc(t_sel).to_pydatetime() if t_sel is not None else None
                    fig_globe = earth_globe_figure(
                        points_src=points_src,
                        points_dst=points_dst,
                        lines=lines,
                        dt_utc=dt_utc,
                        title=f"IP: {sel_suspect} — {'UTC ' + str(t_sel) if t_sel is not None else 'sans date'}"
                    )
                    st.plotly_chart(fig_globe, use_container_width=True)
                    st.caption("Le trait cyan (approx) représente le terminator (jour/nuit) selon l'heure UTC.")

                st.divider()
                st.subheader("IA — Hypothèse de politique firewall (défensif)")

                summary = {
                    "mode": "policy_inference",
                    "ip_src": str(sel_suspect),
                    "total_events": int(len(sus_df)),
                    "uniq_dst": int(sus_df["ip_dst"].nunique()) if "ip_dst" in sus_df.columns else None,
                    "uniq_ports": int(sus_df["dst_port"].nunique()) if "dst_port" in sus_df.columns else None,
                    "top_ports": (sus_df["dst_port"].value_counts().head(10).to_dict() if "dst_port" in sus_df.columns else {}),
                    "top_rules": (sus_df["rule_id"].value_counts().head(10).to_dict() if "rule_id" in sus_df.columns else {}),
                    "action_counts": (sus_df["action"].astype(str).str.upper().value_counts().to_dict() if "action" in sus_df.columns else {}),
                }

                example_cols = [c for c in ["date", "ip_src", "ip_dst", "protocol", "src_port", "dst_port", "rule_id", "action"] if c in sus_df.columns]
                examples = sus_df.sort_values("date", ascending=False)[example_cols].head(40).astype(str).to_dict(orient="records")

                if st.button("Générer analyse IA (politique observée)", key="llm_policy"):
                    with st.spinner("Génération IA..."):
                        report = generate_report(str(sel_suspect), stats=summary, examples=examples)
                    st.markdown(report)

    else:
        st.info("Cliquez sur **Lancer la détection** pour exécuter IsolationForest sur les données filtrées.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – LLM REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Rapport d'incident (LLM / Template)")

    mistral_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if mistral_key:
        st.success("Clé Mistral détectée — mode IA activé (`mistral-small-latest`)")
    else:
        st.info("Pas de clé Mistral (`MISTRAL_API_KEY`) — mode template offline activé")

    ip_list = df["ip_src"].value_counts().head(100).index.astype(str).tolist() if "ip_src" in df.columns else []
    report_ip = st.selectbox("Sélectionner une IP à analyser", ip_list, key="report_ip")

    if st.button("Générer le rapport", type="primary"):
        with st.spinner("Génération du rapport..."):
            ip_df = df[df["ip_src"].astype(str) == str(report_ip)].copy()

            stats = {
                "mode": "incident_report",
                "events": int(len(ip_df)),
                "uniq_dst": int(ip_df["ip_dst"].nunique()) if "ip_dst" in ip_df.columns else 0,
                "uniq_ports": int(ip_df["dst_port"].nunique()) if "dst_port" in ip_df.columns else 0,
                "tcp_ratio": float((ip_df["protocol"].astype(str).str.upper() == "TCP").mean()) if "protocol" in ip_df.columns else 1.0,
                "priv_port_ratio": float((ip_df["dst_port"] < 1024).mean()) if "dst_port" in ip_df.columns else 0.0,
                "permit_count": int(ip_df["action"].astype(str).str.upper().isin(ALLOW).sum()) if "action" in ip_df.columns else 0,
                "deny_count": int(ip_df["action"].astype(str).str.upper().isin(DENY).sum()) if "action" in ip_df.columns else 0,
                "max_events_per_min": int(1),
            }
            if "date" in ip_df.columns and ip_df["date"].notna().any():
                ip_df["date"] = pd.to_datetime(ip_df["date"], errors="coerce")
                ip_df = ip_df.dropna(subset=["date"])
                ip_df["minute"] = ip_df["date"].dt.floor("min")
                stats["max_events_per_min"] = int(ip_df.groupby("minute").size().max())

            example_cols = [c for c in ["date", "ip_src", "ip_dst", "protocol", "src_port", "dst_port", "rule_id", "action"] if c in ip_df.columns]
            examples = ip_df.sort_values("date", ascending=False)[example_cols].head(20).astype(str).to_dict(orient="records")
            report = generate_report(str(report_ip), stats, examples)

        st.markdown(report)