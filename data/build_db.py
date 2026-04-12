"""
build_db.py  --  Pre-process the FDA NDC dataset into a compact SQLite database
for fast fuzzy search, alternative lookups, and interaction checks.

Run once:
    python data/build_db.py
"""

import json
import sqlite3
import os
import re
import sys

DB_PATH = os.path.join(os.path.dirname(__file__), "pharmacy.db")
JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "drug-ndc-0001-of-0001.json",
)


def _clean(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def build():
    if not os.path.exists(JSON_PATH):
        print(f"ERROR: FDA NDC file not found at {JSON_PATH}")
        sys.exit(1)

    print("Loading FDA NDC dataset ...")
    with open(JSON_PATH, "r") as f:
        raw = json.load(f)

    records = raw["results"]
    print(f"  -> {len(records)} product records loaded.")

    # ---- create database ----
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE drugs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            product_ndc   TEXT,
            brand_name    TEXT,
            generic_name  TEXT,
            active_ingredient TEXT,
            strength      TEXT,
            dosage_form   TEXT,
            route         TEXT,
            product_type  TEXT,
            labeler       TEXT,
            pharm_classes TEXT,
            brand_search  TEXT,
            generic_search TEXT
        );
        CREATE INDEX idx_brand   ON drugs(brand_search);
        CREATE INDEX idx_generic ON drugs(generic_search);

        CREATE TABLE alternatives (
            drug_id       INTEGER,
            alt_drug_id   INTEGER,
            reason        TEXT,
            FOREIGN KEY (drug_id) REFERENCES drugs(id),
            FOREIGN KEY (alt_drug_id) REFERENCES drugs(id)
        );

        CREATE TABLE interactions (
            class_a TEXT,
            class_b TEXT,
            severity TEXT,
            description TEXT
        );
        """
    )

    # ---- insert drugs ----
    print("Inserting drug records ...")
    seen = set()
    insert_count = 0

    for r in records:
        brand = r.get("brand_name") or ""
        generic = r.get("generic_name") or ""
        if not brand and not generic:
            continue

        active_parts = []
        strength_parts = []
        for ai in r.get("active_ingredients", []):
            active_parts.append(ai.get("name", ""))
            strength_parts.append(ai.get("strength", ""))

        active = "; ".join(active_parts)
        strength = "; ".join(strength_parts)
        dosage = r.get("dosage_form", "")
        route = ", ".join(r.get("route", []))
        prod_type = r.get("product_type", "")
        labeler = r.get("labeler_name", "")
        pharm = "; ".join(r.get("pharm_class", []))
        ndc = r.get("product_ndc", "")

        # De-duplicate on brand+generic+strength+dosage
        key = (_clean(brand), _clean(generic), _clean(strength), _clean(dosage))
        if key in seen:
            continue
        seen.add(key)

        cur.execute(
            """INSERT INTO drugs
            (product_ndc, brand_name, generic_name, active_ingredient,
             strength, dosage_form, route, product_type, labeler,
             pharm_classes, brand_search, generic_search)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ndc,
                brand,
                generic,
                active,
                strength,
                dosage,
                route,
                prod_type,
                labeler,
                pharm,
                _clean(brand),
                _clean(generic),
            ),
        )
        insert_count += 1

    print(f"  -> {insert_count} unique drug entries stored.")

    # ---- build alternatives table ----
    print("Building alternatives index (same active ingredient) ...")
    cur.execute(
        """
        INSERT INTO alternatives (drug_id, alt_drug_id, reason)
        SELECT a.id, b.id, 'Same active ingredient'
        FROM drugs a
        JOIN drugs b ON a.active_ingredient = b.active_ingredient
                    AND a.dosage_form = b.dosage_form
                    AND a.id != b.id
        WHERE a.active_ingredient != ''
        LIMIT 500000
        """
    )
    alt_count = cur.execute("SELECT COUNT(*) FROM alternatives").fetchone()[0]
    print(f"  -> {alt_count} alternative links created.")

    # ---- seed common interaction data ----
    print("Seeding known drug interaction rules ...")
    INTERACTIONS = [
        ("Nonsteroidal Anti-inflammatory Drug [EPC]",
         "Nonsteroidal Anti-inflammatory Drug [EPC]",
         "HIGH",
         "Concurrent use of multiple NSAIDs increases risk of GI bleeding and renal impairment."),
        ("Nonsteroidal Anti-inflammatory Drug [EPC]",
         "Anticoagulant [EPC]",
         "HIGH",
         "NSAIDs combined with anticoagulants significantly increase bleeding risk."),
        ("ACE Inhibitor [EPC]",
         "Potassium Salt [EPC]",
         "MODERATE",
         "ACE Inhibitors combined with potassium can cause dangerous hyperkalemia."),
        ("Selective Serotonin Reuptake Inhibitor [EPC]",
         "Monoamine Oxidase Inhibitor [EPC]",
         "CRITICAL",
         "SSRI + MAOI can cause serotonin syndrome, a potentially fatal condition."),
        ("Beta Adrenergic Blocker [EPC]",
         "Calcium Channel Blocker [EPC]",
         "MODERATE",
         "Combined use may cause severe bradycardia and hypotension."),
        ("Statin [EPC]",
         "Fibrate [EPC]",
         "HIGH",
         "Co-administration increases risk of rhabdomyolysis."),
        ("Opioid Agonist [EPC]",
         "Benzodiazepine [EPC]",
         "CRITICAL",
         "Combined use of opioids and benzodiazepines can cause fatal respiratory depression."),
        ("Corticosteroid [EPC]",
         "Nonsteroidal Anti-inflammatory Drug [EPC]",
         "MODERATE",
         "Concurrent use increases risk of GI ulceration and bleeding."),
        ("Anticoagulant [EPC]",
         "Antiplatelet Agent [EPC]",
         "HIGH",
         "Combined use significantly increases risk of hemorrhage."),
        ("Proton Pump Inhibitor [EPC]",
         "Antiplatelet Agent [EPC]",
         "MODERATE",
         "PPIs may reduce the efficacy of certain antiplatelet agents like clopidogrel."),
    ]
    cur.executemany(
        "INSERT INTO interactions (class_a, class_b, severity, description) VALUES (?,?,?,?)",
        INTERACTIONS,
    )
    print(f"  -> {len(INTERACTIONS)} interaction rules seeded.")

    conn.commit()
    conn.close()
    print(f"\nDatabase saved to: {DB_PATH}")
    db_size = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"Database size: {db_size:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    build()
