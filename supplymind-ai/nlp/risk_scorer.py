"""High-accuracy supply chain risk scoring using multi-tier keyword rules,
density normalization, and contextual boosting."""

import re
import math

# ── CRITICAL: Immediate, severe supply chain impact ──────────────────────────

CRITICAL_PATTERNS = [
    (r"\b(earthquake|tsunami|volcano|hurricane|typhoon|cyclone)\b", 4.5),
    (r"\b(war|invasion|military\s+strike|missile|bomb|attack|terrorism)\b", 4.5),
    (r"\b(pandemic|outbreak|quarantine|lockdown|covid)\b", 4.0),
    (r"\b(factory\s*(fire|explosion|shutdown|closed|halt|destroy))\b", 4.0),
    (r"\b(port\s*(closed|shutdown|blockade|congestion|strike))\b", 4.0),
    (r"\b(suez|strait\s+of\s+hormuz|panama\s+canal)\b.*\b(block|stuck|closed|disrupt)", 4.5),
    (r"\b(nuclear|radiation|meltdown|chemical\s+spill)\b", 4.5),
    (r"\b(embargo|sanction|trade\s+ban|blacklist)\b", 3.8),
    (r"\b(bankrupt|insolvency|default|financial\s+collapse)\b", 3.8),
    (r"\b(catastroph|devastat|massive\s+damage|total\s+loss)\b", 4.0),
    (r"\b(complete\s+shutdown|indefinite\s+halt|force\s+majeure)\b", 4.0),
    (r"\b(global\s+shortage|worldwide\s+disrupt|critical\s+supply)\b", 4.2),
]

# ── HIGH: Significant operational disruptions ────────────────────────────────

HIGH_PATTERNS = [
    (r"\b(shortage|scarcity|deficit|stockout|out\s+of\s+stock)\b", 3.2),
    (r"\b(disruption|disrupt|bottleneck|gridlock)\b", 3.0),
    (r"\b(flood|storm|typhoon|hurricane)\b.*\b(disrupt|damag|destroy|halt|shut)\b", 3.4),
    (r"\b(disrupt|damag|destroy|halt|shut)\b.*\b(flood|storm|typhoon|hurricane)\b", 3.4),
    (r"\b(tariff|trade\s+war|import\s+duty|export\s+restrict|retaliatory)\b", 3.0),
    (r"\b(strike|labor\s+dispute|walkout|mass\s+protest)\b", 2.8),
    (r"\b(recall|safety\s+issue|defect|contamina|compliance\s+violation)\b", 2.8),
    (r"\b(chip\s+shortage|semiconductor\s+crisis|supply\s+crunch)\b", 3.4),
    (r"\b(price\s+surge|price\s+spike|cost\s+skyrocket|hyperinflation)\b", 2.8),
    (r"\b(severe\s+delay|massive\s+backlog|lead\s+time\s+tripl|lead\s+time\s+doubl)\b", 3.0),
    (r"\b(wildfire|drought|famine|extreme\s+weather|ice\s+storm)\b", 3.0),
    (r"\b(geopoliti|rising\s+tensions|military\s+buildup|escalat)\b", 2.8),
    (r"\b(production\s+halt|plant\s+closure|capacity\s+cut|output\s+slash)\b", 3.2),
    (r"\b(cyber\s*attack|ransomware|data\s+breach|critical\s+hack)\b", 3.2),
    (r"\b(shipping\s+crisis|container\s+shortage|freight\s+spike)\b", 3.0),
    (r"\b(power\s+outage|blackout|energy\s+crisis|grid\s+failure)\b", 3.0),
    (r"\b(evacuat|forced\s+relocation|refugee|humanitarian\s+crisis)\b", 2.8),
    (r"\b(canal\s+block|route\s+diversion|shipping\s+reroute)\b", 3.0),
]

# ── MEDIUM: Elevated risk, developing situations ─────────────────────────────

MEDIUM_PATTERNS = [
    (r"\b(supply\s+chain|logistics|procurement|sourcing)\b", 2.0),
    (r"\b(shipping|freight|cargo|container|vessel|tanker)\b", 1.8),
    (r"\b(warehouse|inventory|distribution|fulfillment)\b", 1.8),
    (r"\b(manufactur|assembl|production|factory|foundry)\b", 1.8),
    (r"\b(semiconductor|chip|processor|wafer|fab)\b", 2.0),
    (r"\b(crude\s+oil|natural\s+gas|commodity|raw\s+material)\b", 2.0),
    (r"\b(regulation|compliance|policy\s+change|legislation)\b", 1.5),
    (r"\b(downturn|recession|slowdown|contraction|stagflation)\b", 2.2),
    (r"\b(volatil|uncertain|risk\s+exposure|risk\s+assessment)\b", 1.5),
    (r"\b(outsourc|offshoring|nearshoring|reshoring|decoupling)\b", 1.8),
    (r"\b(esg|carbon\s+tax|sustainability|emission\s+regulation)\b", 1.5),
    (r"\b(merger|acquisition|takeover|restructur|divestiture)\b", 1.8),
    (r"\b(layoff|job\s+cut|workforce\s+reduction|downsiz|furlough)\b", 2.0),
    (r"\b(pipeline|transit|rail\s+freight|trucking|intermodal)\b", 1.5),
    (r"\b(demand\s+drop|consumption\s+decline|order\s+cancel)\b", 2.0),
    (r"\b(quality\s+issue|inspection\s+fail|regulatory\s+action)\b", 1.8),
    (r"\b(lithium|cobalt|rare\s+earth|copper|steel|aluminum)\b", 1.8),
    (r"\b(ev\s+battery|battery\s+supply|cathode|anode)\b", 1.8),
    (r"\b(pharma|drug\s+supply|api\s+shortage|vaccine\s+supply)\b", 2.0),
    (r"\b(port\s+congestion|berth\s+delay|vessel\s+queue)\b", 2.2),
    (r"\b(customs\s+delay|border\s+control|import\s+inspection)\b", 1.8),
    (r"\b(inventory\s+build|safety\s+stock|buffer\s+stock)\b", 1.5),
]

# ── Entity boost: higher for supply-chain-critical entities ──────────────────

ENTITY_BOOST = {
    "tsmc": 0.5, "samsung": 0.4, "intel": 0.3, "nvidia": 0.3, "amd": 0.3,
    "qualcomm": 0.3, "broadcom": 0.3, "micron": 0.3, "sk hynix": 0.3,
    "asml": 0.4, "globalfoundries": 0.3, "foxconn": 0.4, "huawei": 0.4,
    "smic": 0.4, "mediatek": 0.3, "infineon": 0.3,
    "apple": 0.3, "microsoft": 0.2, "google": 0.2, "amazon": 0.2,
    "meta": 0.2, "cisco": 0.2, "dell": 0.2, "lenovo": 0.2, "sony": 0.2,
    "toyota": 0.3, "volkswagen": 0.3, "tesla": 0.3, "ford": 0.3,
    "gm": 0.2, "bmw": 0.2, "hyundai": 0.3, "honda": 0.2, "byd": 0.3,
    "catl": 0.4, "stellantis": 0.3, "panasonic": 0.3,
    "boeing": 0.4, "airbus": 0.4, "lockheed martin": 0.3,
    "general electric": 0.3, "rolls-royce": 0.3,
    "walmart": 0.2, "costco": 0.2, "target": 0.2, "nike": 0.2,
    "exxonmobil": 0.3, "shell": 0.2, "bp": 0.2, "chevron": 0.2,
    "saudi aramco": 0.4,
    "maersk": 0.5, "cosco": 0.4, "evergreen": 0.4, "fedex": 0.3,
    "ups": 0.3, "dhl": 0.3, "hapag-lloyd": 0.3, "cma cgm": 0.3,
    "pfizer": 0.3, "moderna": 0.3, "astrazeneca": 0.3,
    "cargill": 0.3, "adm": 0.2, "basf": 0.3,
    "taiwan": 0.5, "china": 0.4, "shenzhen": 0.4, "shanghai": 0.3,
    "rotterdam": 0.3, "suez": 0.5, "singapore": 0.2,
    "strait of hormuz": 0.5, "panama canal": 0.4, "red sea": 0.4,
    "hong kong": 0.3, "vietnam": 0.3,
}

# ── Contextual combos that amplify severity ──────────────────────────────────

COMBO_PATTERNS = [
    (r"\b(semiconductor|chip)\b.*\b(shortage|crisis|halt)\b", 2.0),
    (r"\b(supply\s+chain)\b.*\b(disrupt|crisis|collapse|halt)\b", 2.5),
    (r"\b(factory|plant|production)\b.*\b(shut|halt|clos|fire|explod)\b", 2.5),
    (r"\b(shipping|freight|port)\b.*\b(delay|congestion|crisis|block)\b", 2.0),
    (r"\b(trade|tariff|sanction)\b.*\b(war|escalat|retaliat|ban)\b", 2.0),
    (r"\b(natural\s+disaster|earthquake|flood|typhoon)\b.*\b(damage|destroy|disrupt)\b", 2.5),
    (r"\b(energy|oil|gas)\b.*\b(crisis|shortage|spike|embargo)\b", 2.0),
    (r"\b(cyber|ransomware)\b.*\b(shut|disrupt|halt|crippl)\b", 2.0),
]


def score_risk(text):
    """Score a single text for supply chain risk with density normalization.

    Short, focused texts about real supply chain events score higher than
    long, rambling posts with incidental keyword matches.
    """
    if not text or len(str(text).strip()) < 5:
        return {"label": "low risk", "score": 0.1, "risk_level": 1,
                "raw_score": 0.0, "signal_count": 0}

    t = str(text).lower()
    word_count = max(len(t.split()), 1)

    total = 0.0
    max_severity = 0
    matched_signals = []

    for pattern, weight in CRITICAL_PATTERNS:
        if re.search(pattern, t):
            total += weight
            max_severity = 3
            matched_signals.append("CRITICAL")

    for pattern, weight in HIGH_PATTERNS:
        if re.search(pattern, t):
            total += weight
            max_severity = max(max_severity, 2)
            matched_signals.append("HIGH")

    for pattern, weight in MEDIUM_PATTERNS:
        if re.search(pattern, t):
            total += weight
            max_severity = max(max_severity, 1)
            matched_signals.append("MEDIUM")

    for pattern, weight in COMBO_PATTERNS:
        if re.search(pattern, t):
            total += weight
            matched_signals.append("COMBO")

    entity_boost = 0.0
    for entity, boost in ENTITY_BOOST.items():
        if entity in t:
            entity_boost += boost
    total += min(entity_boost, 2.0)

    n_signals = len(matched_signals)

    if word_count > 60:
        length_penalty = 60.0 / word_count
        total *= length_penalty

    if n_signals >= 2:
        density_bonus = min(1.0 + n_signals * 0.06, 1.5)
        total *= density_bonus

    if total >= 7.0:
        label, level = "critical risk", 4
    elif total >= 3.5:
        label, level = "high risk", 3
    elif total >= 1.0:
        label, level = "medium risk", 2
    else:
        label, level = "low risk", 1

    confidence = min(0.99, 0.3 + total * 0.08)

    return {
        "label": label,
        "score": round(confidence, 4),
        "risk_level": level,
        "raw_score": round(total, 2),
        "signal_count": n_signals,
    }


def score_batch(texts):
    """Score a list of texts."""
    return [score_risk(t) for t in texts]


if __name__ == "__main__":
    tests = [
        "TSMC factory shutdown in Taiwan due to earthquake disrupting chip supply",
        "Apple reports strong quarterly earnings beating expectations",
        "Port congestion at Rotterdam causing severe shipping delays across Europe",
        "Best platform for options trading? I use Robinhood and Fidelity",
        "Semiconductor shortage forces Toyota to halt production at 3 plants",
        "Trade war escalation: new tariffs on Chinese imports threaten supply chains",
        "Flooding in Shenzhen disrupts Samsung and Foxconn manufacturing lines",
        "Suez Canal blocked again by container ship causing global shipping chaos",
        "Maersk reroutes all Red Sea shipping due to Houthi missile attacks",
        "CATL battery factory in Germany faces regulatory delay",
        "Boeing 737 MAX production halted after safety defect found",
        "Random Reddit post about yolo tendies and diamond hands",
    ]
    for t in tests:
        r = score_risk(t)
        print(f"[{r['label'].upper():14s}] score={r['raw_score']:6.2f} "
              f"signals={r['signal_count']:2d} conf={r['score']:.2f} | {t[:65]}")
