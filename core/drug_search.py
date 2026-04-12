"""
core/drug_search.py  --  High-performance fuzzy drug search engine
backed by the FDA NDC SQLite database.
"""

import sqlite3
import os
import re
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pharmacy.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# ---- Common abbreviation map (Indian + US pharmacy shorthand) ----
ABBREVIATIONS = {
    "pcm": "paracetamol",
    "para": "paracetamol",
    "ibu": "ibuprofen",
    "amox": "amoxicillin",
    "azith": "azithromycin",
    "metro": "metronidazole",
    "cefix": "cefixime",
    "diclo": "diclofenac",
    "oflox": "ofloxacin",
    "cipro": "ciprofloxacin",
    "levo": "levofloxacin",
    "dolo": "paracetamol",
    "crocin": "paracetamol",
    "combiflam": "ibuprofen",
    "augmentin": "amoxicillin",
    "zithromax": "azithromycin",
    "flagyl": "metronidazole",
    "brufen": "ibuprofen",
    "disprin": "aspirin",
    "aspirin": "aspirin",
    "omez": "omeprazole",
    "pan": "pantoprazole",
    "rantac": "ranitidine",
    "metformin": "metformin",
    "atorva": "atorvastatin",
    "ecosprin": "aspirin",
    "shelcal": "calcium",
    "montair": "montelukast",
    "allegra": "fexofenadine",
    "cetrizine": "cetirizine",
    "avil": "pheniramine",
    "sinarest": "paracetamol",
}


def expand_abbreviation(token: str) -> str:
    """Expand known abbreviations to full generic name."""
    clean = _normalize(token)
    return ABBREVIATIONS.get(clean, clean)


def fuzzy_score(query: str, target: str) -> float:
    """Return 0-1 similarity between query and target strings."""
    q = _normalize(query)
    t = _normalize(target)

    if q in t or t in q:
        return 0.95

    return SequenceMatcher(None, q, t).ratio()


def search_drugs(
    query: str,
    limit: int = 20,
    min_score: float = 0.4,
) -> List[Dict]:
    """
    Fuzzy-search the drug database by brand or generic name.
    Returns list of dicts sorted by relevance score.
    """
    expanded = expand_abbreviation(query)
    conn = _get_conn()
    cur = conn.cursor()

    like_pattern = f"%{expanded}%"
    cur.execute(
        """SELECT id, brand_name, generic_name, active_ingredient,
                  strength, dosage_form, route, product_type, labeler,
                  pharm_classes
           FROM drugs
           WHERE generic_search LIKE ? OR brand_search LIKE ?
           LIMIT ?""",
        (like_pattern, like_pattern, limit * 3),
    )
    rows = cur.fetchall()

    results = []
    for row in rows:
        brand_score = fuzzy_score(expanded, row["brand_name"] or "")
        generic_score = fuzzy_score(expanded, row["generic_name"] or "")
        score = max(brand_score, generic_score)

        if score >= min_score:
            results.append(
                {
                    "id": row["id"],
                    "brand_name": row["brand_name"],
                    "generic_name": row["generic_name"],
                    "active_ingredient": row["active_ingredient"],
                    "strength": row["strength"],
                    "dosage_form": row["dosage_form"],
                    "route": row["route"],
                    "product_type": row["product_type"],
                    "labeler": row["labeler"],
                    "pharm_classes": row["pharm_classes"],
                    "score": round(score, 3),
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    conn.close()
    return results[:limit]


def get_alternatives(drug_id: int, limit: int = 10) -> List[Dict]:
    """Find alternative medicines for a given drug ID (same active ingredient)."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT d.id, d.brand_name, d.generic_name, d.strength,
                  d.dosage_form, d.labeler, a.reason
           FROM alternatives a
           JOIN drugs d ON d.id = a.alt_drug_id
           WHERE a.drug_id = ?
           LIMIT ?""",
        (drug_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def check_interactions(drug_ids: List[int]) -> List[Dict]:
    """
    Check for potential drug-drug interactions among a list of drug IDs.
    """
    if len(drug_ids) < 2:
        return []

    conn = _get_conn()
    cur = conn.cursor()

    drug_classes = {}
    for did in drug_ids:
        cur.execute("SELECT pharm_classes, brand_name, generic_name FROM drugs WHERE id = ?", (did,))
        row = cur.fetchone()
        if row and row["pharm_classes"]:
            classes = [c.strip() for c in row["pharm_classes"].split(";") if c.strip()]
            drug_classes[did] = classes

    cur.execute("SELECT class_a, class_b, severity, description FROM interactions")
    rules = cur.fetchall()

    alerts = []
    checked_pairs = set()

    for i, id_a in enumerate(drug_ids):
        for id_b in drug_ids[i + 1 :]:
            pair_key = (min(id_a, id_b), max(id_a, id_b))
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)

            classes_a = drug_classes.get(id_a, [])
            classes_b = drug_classes.get(id_b, [])

            for rule in rules:
                a_match = any(rule["class_a"] in c for c in classes_a)
                b_match = any(rule["class_b"] in c for c in classes_b)
                a_match_rev = any(rule["class_b"] in c for c in classes_a)
                b_match_rev = any(rule["class_a"] in c for c in classes_b)

                if (a_match and b_match) or (a_match_rev and b_match_rev):
                    cur.execute("SELECT brand_name, generic_name FROM drugs WHERE id IN (?,?)", (id_a, id_b))
                    names = cur.fetchall()
                    alerts.append(
                        {
                            "drug_a": names[0]["brand_name"] or names[0]["generic_name"],
                            "drug_b": names[1]["brand_name"] or names[1]["generic_name"] if len(names) > 1 else "Unknown",
                            "severity": rule["severity"],
                            "description": rule["description"],
                        }
                    )

    conn.close()
    return alerts


def get_drug_stats() -> Dict:
    """Get summary statistics of the drug database."""
    conn = _get_conn()
    cur = conn.cursor()

    stats = {}
    stats["total_drugs"] = cur.execute("SELECT COUNT(*) FROM drugs").fetchone()[0]
    stats["unique_generics"] = cur.execute(
        "SELECT COUNT(DISTINCT generic_search) FROM drugs"
    ).fetchone()[0]
    stats["unique_brands"] = cur.execute(
        "SELECT COUNT(DISTINCT brand_search) FROM drugs"
    ).fetchone()[0]
    stats["total_alternatives"] = cur.execute(
        "SELECT COUNT(*) FROM alternatives"
    ).fetchone()[0]
    stats["interaction_rules"] = cur.execute(
        "SELECT COUNT(*) FROM interactions"
    ).fetchone()[0]

    cur.execute(
        "SELECT dosage_form, COUNT(*) as cnt FROM drugs GROUP BY dosage_form ORDER BY cnt DESC LIMIT 10"
    )
    stats["top_dosage_forms"] = [{"form": r[0], "count": r[1]} for r in cur.fetchall()]

    cur.execute(
        "SELECT product_type, COUNT(*) as cnt FROM drugs GROUP BY product_type ORDER BY cnt DESC LIMIT 5"
    )
    stats["top_product_types"] = [{"type": r[0], "count": r[1]} for r in cur.fetchall()]

    conn.close()
    return stats
