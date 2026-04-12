"""
core/prescription_parser.py  --  Hybrid prescription text parser.

Extraction pipeline:
  1. Regex-based entity extraction (fast, no API cost)
  2. LLM fallback for complex / ambiguous text
  3. Returns structured list: medicine name, dosage, frequency
"""

import re
import os
from typing import List, Dict, Optional
from openai import OpenAI

# ---- Regex patterns for common prescription formats ----
# Matches patterns like:
#   Tab PCM 500mg 1-0-1
#   Cap Amoxicillin 250mg TDS x 5 days
#   Inj Ceftriaxone 1g IV BD
MEDICINE_PATTERN = re.compile(
    r"(?:Tab\.?|Cap\.?|Inj\.?|Syp\.?|Oint\.?|Drops?|Susp\.?|Cream|Gel|Spray)?\s*"
    r"([A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)\s*"
    r"(\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|iu|%)?)\s*"
    r"((?:OD|BD|TDS|QDS|SOS|PRN|HS|AC|PC|stat|once|twice|thrice|"
    r"1-0-0|0-1-0|0-0-1|1-1-0|1-0-1|0-1-1|1-1-1|1-1-1-1)?\s*"
    r"(?:x\s*\d+\s*days?|for\s*\d+\s*days?)?)",
    re.IGNORECASE,
)

# Simple line-by-line extraction for cleaner prescriptions
LINE_PATTERN = re.compile(
    r"^\s*\d*\.?\s*"
    r"(?:Tab\.?|Cap\.?|Inj\.?|Syp\.?|Oint\.?|Drops?|Susp\.?|Cream|Gel|Spray)?\s*"
    r"(.+)",
    re.IGNORECASE,
)

# Dosage patterns
DOSAGE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|iu|%)", re.IGNORECASE)
FREQUENCY_PATTERN = re.compile(
    r"(OD|BD|TDS|QDS|SOS|PRN|HS|stat|once daily|twice daily|"
    r"1-0-0|0-1-0|0-0-1|1-1-0|1-0-1|0-1-1|1-1-1|1-1-1-1|"
    r"morning|evening|night|before food|after food|empty stomach)",
    re.IGNORECASE,
)
DURATION_PATTERN = re.compile(
    r"(?:x|for)\s*(\d+)\s*(days?|weeks?|months?)", re.IGNORECASE
)

# Words to skip during extraction
STOP_WORDS = {
    "tab", "cap", "inj", "syp", "oint", "drops", "susp", "cream", "gel", "spray",
    "take", "give", "the", "use", "apply", "with", "and", "or", "if", "not",
    "available", "alternative", "prescribed", "daily", "twice", "thrice", "once",
    "mg", "ml", "gm", "mcg",
}


def _extract_medicine_tokens(text: str) -> List[str]:
    """
    Extract potential medicine name tokens from text using heuristics.
    """
    tokens = []
    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove leading numbers (line items like "1. ...")
        line = re.sub(r"^\d+\.\s*", "", line)

        # Remove dosage form prefixes
        cleaned = re.sub(
            r"\b(Tab\.?|Cap\.?|Inj\.?|Syp\.?|Oint\.?|Drops?|Susp\.?|Cream|Gel|Spray)\b",
            "",
            line,
            flags=re.IGNORECASE,
        )

        # Remove dosage numbers: "500mg", "1g"
        cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|iu|%)\b", "", cleaned, flags=re.IGNORECASE)

        # Remove frequency patterns
        cleaned = re.sub(
            r"\b(OD|BD|TDS|QDS|SOS|PRN|HS|stat|1-0-0|0-1-0|0-0-1|1-1-0|1-0-1|0-1-1|1-1-1)\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        # Remove duration
        cleaned = re.sub(r"(?:x|for)\s*\d+\s*(?:days?|weeks?|months?)", "", cleaned, flags=re.IGNORECASE)

        # Extract remaining meaningful words
        words = cleaned.split()
        for w in words:
            w_clean = re.sub(r"[^a-zA-Z]", "", w).lower()
            if len(w_clean) >= 3 and w_clean not in STOP_WORDS:
                tokens.append(w)

    return tokens


def parse_prescription_local(text: str) -> List[Dict]:
    """
    Parse prescription text using regex and heuristics.
    Returns list of extracted entities.
    """
    results = []
    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue

        # Extract medicine tokens
        tokens = _extract_medicine_tokens(line)
        medicine_name = " ".join(tokens) if tokens else line.strip()

        # Extract dosage
        dosage_match = DOSAGE_PATTERN.search(line)
        dosage = dosage_match.group(0) if dosage_match else None

        # Extract frequency
        freq_match = FREQUENCY_PATTERN.search(line)
        frequency = freq_match.group(0) if freq_match else None

        # Extract duration
        dur_match = DURATION_PATTERN.search(line)
        duration = dur_match.group(0) if dur_match else None

        if medicine_name:
            results.append(
                {
                    "medicine": medicine_name.strip(),
                    "dosage": dosage,
                    "frequency": frequency,
                    "duration": duration,
                    "raw_text": line,
                    "method": "regex",
                }
            )

    return results


def parse_prescription_llm(text: str) -> List[Dict]:
    """
    Parse prescription text using LLM for complex / ambiguous cases.
    Falls back to local extraction on API failure.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return parse_prescription_local(text)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a pharmacist assistant. Extract all medicines from the "
                        "prescription text. For each medicine, return:\n"
                        "- medicine: the medicine name (generic or brand)\n"
                        "- dosage: amount and unit (e.g., 500mg)\n"
                        "- frequency: how often (e.g., BD, TDS, 1-0-1)\n"
                        "- duration: for how long (e.g., 5 days)\n\n"
                        "Return a JSON array. If a field is not specified, use null.\n"
                        "Example: [{\"medicine\": \"Paracetamol\", \"dosage\": \"500mg\", "
                        "\"frequency\": \"TDS\", \"duration\": \"5 days\"}]"
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        import json
        result = json.loads(response.choices[0].message.content)

        # Handle both {"medicines": [...]} and direct [...]
        if isinstance(result, dict):
            medicines = result.get("medicines", result.get("data", []))
        elif isinstance(result, list):
            medicines = result
        else:
            medicines = []

        for m in medicines:
            m["method"] = "llm"
            m["raw_text"] = text

        return medicines

    except Exception as e:
        print(f"LLM extraction failed ({e}), falling back to local parser.")
        return parse_prescription_local(text)


def parse_prescription(text: str, use_llm: bool = False) -> List[Dict]:
    """
    Main entry point. Extracts structured medicine data from prescription text.
    Uses local regex by default; set use_llm=True for complex prescriptions.
    """
    if use_llm:
        return parse_prescription_llm(text)
    return parse_prescription_local(text)
