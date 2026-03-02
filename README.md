# SISE-OPSIE 2026 — Firewall Analytics Web App

Application Streamlit pour l'analyse des logs Firewall Iptables (cloud + on-prem).

## 📁 Structure du projet
```
opsie-app/
├── app.py              # Application Streamlit principale
├── src/
│   ├── loader.py       # Chargement & parsing des données (Parquet/CSV/raw_log)
│   ├── ml.py           # Features engineering + IsolationForest
│   └── llm.py          # Rapport d'incident (OpenAI ou template offline)
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Lancement local

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer l'app
streamlit run app.py

# 3. Ouvrir http://localhost:8501
```

Placez vos fichiers Parquet dans `data/` ou uploadez-les directement via l'interface.

## 🐳 Docker

### Build & Run
```bash
# Build
docker build -t opsie-app .

# Run avec fichiers de données montés
docker run -p 8501:8501 \
  -v $(pwd)/data:/data \
  opsie-app

# Avec clé OpenAI (optionnel)
docker run -p 8501:8501 \
  -v $(pwd)/data:/data \
  -e OPENAI_API_KEY=sk-... \
  opsie-app
```

### Depuis Docker Desktop
1. `docker build -t opsie-app .`
2. Images → `opsie-app` → Run → Ports: `8501:8501` → Volume: dossier data → `/data`
3. Ouvrir http://localhost:8501

## 📊 Formats de données supportés

### Format 1 — Parquet structuré (`logs_export.parquet`)
Colonnes : `date, ip_src, ip_dst, protocol, src_port, dst_port, rule_id, action, in_interface, out_interface`

### Format 2 — Parquet avec `raw_log` (`clean_logs_nov2025_feb2026.parquet`)
Colonne `raw_log` au format : `date;ip_src;ip_dst;protocol;src_port;dst_port;rule_id;action;in_interface;out_interface;fw`

### Format 3 — CSV (fallback)
Même colonnes que format 1.

## 🧩 Fonctionnalités

| Page | Contenu |
|------|---------|
| **Overview** | KPIs, PERMIT/DENY, protocoles, top règles, timeline |
| **Rules & Ports** | Top TCP/UDP, Sankey rule→port→action |
| **IP Analysis** | Bubble plot, top émetteurs, ports privés, hors-plan, datatable |
| **ML / Anomalies** | IsolationForest, top suspects, features, parallel coordinates |
| **LLM Report** | Rapport d'incident par IP (OpenAI ou template offline) |

## 🔑 Variable d'environnement (optionnel)
```
OPENAI_API_KEY=sk-...   # Active le mode LLM pour les rapports d'incident
```
Sans clé, un rapport template basé sur des règles heuristiques est généré.
