import os
import json

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "").strip()


def _template_incident_report(ip: str, stats: dict, examples: list) -> str:
    events = stats.get("events", 0)
    uniq_dst = stats.get("uniq_dst", 0)
    uniq_ports = stats.get("uniq_ports", 0)
    priv_ratio = stats.get("priv_port_ratio", 0)
    burst = stats.get("max_events_per_min", 0)
    tcp_ratio = stats.get("tcp_ratio", 0)

    deny_count = stats.get("deny_count", "N/A")
    permit_count = stats.get("permit_count", "N/A")

    hypothesis = []
    if burst and burst > 50:
        hypothesis.append("burst élevé → possible scan, brute-force ou activité automatisée")
    if uniq_ports and uniq_ports > 100:
        hypothesis.append(f"scan de ports ({uniq_ports} ports distincts contactés)")
    if priv_ratio and priv_ratio > 0.7:
        hypothesis.append("ciblage massif de ports privilégiés (<1024) → tentative d'intrusion")
    if uniq_dst <= 1 and events and events > 1000:
        hypothesis.append("forte concentration sur 1 destination → scan vertical / brute-force probable")
    if not hypothesis:
        hypothesis.append("trafic applicatif standard ou légèrement anormal")

    ex_lines = "\n".join([f"  - {e}" for e in examples[:5]])

    report = f"""
## 🔍 Rapport d'Incident — IP Source : {ip}

### Résumé exécutif
L'adresse IP **{ip}** a généré **{events}** événements vers **{uniq_dst}** destination(s),
en ciblant **{uniq_ports}** ports (ratio ports <1024 : {priv_ratio:.1%}).  

### Statistiques clés
| Métrique | Valeur |
|---|---|
| Total événements | {events} |
| Destinations uniques | {uniq_dst} |
| Ports uniques | {uniq_ports} |
| Ratio TCP | {tcp_ratio:.1%} |
| Ratio ports <1024 | {priv_ratio:.1%} |
| Burst max (évts/min) | {burst} |
| Flux ALLOW | {permit_count} |
| Flux DENY | {deny_count} |

### Hypothèse comportementale
{chr(10).join(f'- {h}' for h in hypothesis)}

### Exemples d'événements récents
{ex_lines if ex_lines else "Aucun exemple disponible."}

### Recommandations (défensives)
- Corréler l’IP avec d’autres sources (WAF, IDS, logs applicatifs).
- Vérifier réputation (AbuseIPDB, Shodan, etc.).
- Si le pattern est automatisé: rate-limit / blocage temporaire.
- Vérifier les ports exposés et durcir (WAF, MFA, allowlist stricte).
"""
    return report.strip()


def _template_triage_out_of_plan(stats: dict, examples: list) -> str:
    network_plan = stats.get("network_plan", "")
    n_out = stats.get("n_out_ips", 0)

    lines = []
    for row in examples[:20]:
        ip = row.get("ip_src")
        events = row.get("events", 0)
        ports = row.get("top_ports", {})
        rules = row.get("top_rules", {})
        actions = row.get("action_counts", {})
        lines.append(f"- **{ip}**: {events} évts | actions={actions} | top_ports={list(ports)[:5]} | top_rules={list(rules)[:5]}")

    report = f"""
## 🚨 Triage — IP sources hors plan d'adressage

### Contexte
Plan d'adressage attendu : `{network_plan}`  
Nombre d'IP hors plan (dans la sélection) : **{n_out}**

### Lecture rapide (Top IP)
{chr(10).join(lines) if lines else "- Aucun élément."}

### Hypothèses (défensives)
- Certaines IP peuvent être légitimes (CDN, monitoring, fournisseurs).  
- D'autres peuvent être du bruit Internet (scan, bots) ou une activité ciblée.

### Actions recommandées
1. **Classifier**: légitime vs suspect (ASN/ISP, pays, historique).
2. **Prioriser**: DENY élevé + ports sensibles + burst.
3. **Durcir**: limiter exposition, rate-limit, règles temporaires.
4. **Documenter**: exceptions justifiées / allowlist contrôlée.
"""
    return report.strip()


def _template_policy_inference(stats: dict, examples: list) -> str:
    ip = stats.get("ip_src", "N/A")
    actions = stats.get("action_counts", {})
    top_ports = stats.get("top_ports", {})
    top_rules = stats.get("top_rules", {})

    report = f"""
## 🧠 Hypothèse sur la politique firewall (défensif) — {ip}

### Observations (issues des logs)
- Actions observées (ALLOW/DENY) : **{actions}**
- Ports les plus touchés : **{list(top_ports)[:10]}**
- Rules_id les plus fréquentes : **{list(top_rules)[:10]}**

### Ce que la politique *semble* indiquer (hypothèses)
- Politique typique “deny-by-default” avec exceptions sur ports applicatifs.
- Les `rule_id` récurrents suggèrent des chemins “standard” vs “exceptions”.
- Beaucoup de DENY sur ports sensibles = surface ciblée mais bloquée → surveiller.

### Ce qu’on peut inférer sans procédure offensive
- Quels ports sont le plus souvent autorisés / refusés.
- Si les décisions varient selon le temps (fenêtres d’activité) ou règles spécifiques.

### Recommandations (blue-team)
- Vérifier la nécessité des ports les plus fréquents.
- Rate-limit / anti-scan sur ports ciblés.
- Alerting SIEM sur répétitions DENY + burst.
"""
    return report.strip()


def _build_prompt(ip: str, stats: dict, examples: list) -> str:
    mode = (stats.get("mode") or "incident_report").strip()

    if mode == "triage_out_of_plan":
        instruction = (
            "Tu es analyste SOC. Produis un triage des IP sources hors plan à partir des données fournies. "
            "Classe (probable légitime vs suspect) et propose des actions défensives. "
            "Ne donne aucune procédure offensive."
        )
    elif mode == "policy_inference":
        instruction = (
            "Tu es analyste cybersécurité. À partir des logs (actions, ports, rule_id), propose une hypothèse sur la politique firewall observée. "
            "Explique ce qu'on peut inférer uniquement à partir des logs, sans donner de méthode d'attaque. "
            "Termine par des recommandations défensives."
        )
    else:
        instruction = (
            "Tu es analyste cybersécurité. Génère un rapport d'incident concis en français. "
            "Inclure: résumé, ports, ratio DENY/ALLOW, hypothèse, recommandations. "
            "Basé UNIQUEMENT sur les données fournies, ne pas inventer, et ne pas donner d'instructions offensives."
        )

    return f"""{instruction}

IP/Contexte: {ip}
Statistiques: {json.dumps(stats, ensure_ascii=False)}
Exemples: {json.dumps(examples[:25], ensure_ascii=False)}
"""


def generate_report(ip: str, stats: dict, examples: list) -> str:
    mode = (stats.get("mode") or "incident_report").strip()

    if not MISTRAL_API_KEY:
        if mode == "triage_out_of_plan":
            return _template_triage_out_of_plan(stats, examples)
        if mode == "policy_inference":
            return _template_policy_inference(stats, examples)
        return _template_incident_report(ip, stats, examples)

    try:
        from mistralai import Mistral

        prompt = _build_prompt(ip, stats, examples)
        client = Mistral(api_key=MISTRAL_API_KEY)
        resp = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
        )
        return resp.choices[0].message.content

    except Exception as e:
        fallback = (
            _template_triage_out_of_plan(stats, examples) if mode == "triage_out_of_plan"
            else _template_policy_inference(stats, examples) if mode == "policy_inference"
            else _template_incident_report(ip, stats, examples)
        )
        return fallback + f"\n\n*(Erreur LLM: {e})*"