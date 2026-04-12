"""
env/noise.py -- Prescription noise generator.

Simulates real-world prescription input challenges:
- Handwriting OCR errors (character substitution, deletion)
- Medical abbreviations (PCM, IBU, BD, SOS, TDS)
- Missing dosage information
- Inconsistent formatting
- Conflicting instructions
"""

import random
import re
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Abbreviation Expansion Table  (Indian + international pharmacy shorthand)
# ---------------------------------------------------------------------------

ABBREVIATIONS: Dict[str, str] = {
    # Medicine abbreviations
    "pcm": "paracetamol",
    "para": "paracetamol",
    "dolo": "paracetamol",
    "crocin": "paracetamol",
    "ibu": "ibuprofen",
    "brufen": "ibuprofen",
    "combiflam": "ibuprofen + paracetamol",
    "amox": "amoxicillin",
    "augmentin": "amoxicillin + clavulanate",
    "azith": "azithromycin",
    "zithromax": "azithromycin",
    "metro": "metronidazole",
    "flagyl": "metronidazole",
    "cefix": "cefixime",
    "cipro": "ciprofloxacin",
    "levo": "levofloxacin",
    "oflox": "ofloxacin",
    "diclo": "diclofenac",
    "ecosprin": "aspirin",
    "disprin": "aspirin",
    "atorva": "atorvastatin",
    "omez": "omeprazole",
    "pan": "pantoprazole",
    "rantac": "ranitidine",
    "montair": "montelukast",
    "allegra": "fexofenadine",

    # Frequency abbreviations
    "od": "once daily",
    "bd": "twice daily",
    "tds": "three times daily",
    "qds": "four times daily",
    "sos": "as needed",
    "prn": "as needed",
    "hs": "at bedtime",
    "ac": "before food",
    "pc": "after food",
    "stat": "immediately",
    "qid": "four times daily",
    "bid": "twice daily",
    "tid": "three times daily",

    # Dosage form abbreviations
    "tab": "tablet",
    "cap": "capsule",
    "inj": "injection",
    "syp": "syrup",
    "oint": "ointment",
    "susp": "suspension",

    # Route abbreviations
    "po": "oral",
    "iv": "intravenous",
    "im": "intramuscular",
    "sc": "subcutaneous",
    "sl": "sublingual",
}


# ---------------------------------------------------------------------------
# OCR Noise Patterns  (simulates handwriting recognition errors)
# ---------------------------------------------------------------------------

# Character-level substitutions that mimic OCR misreads
OCR_SUBSTITUTIONS: Dict[str, List[str]] = {
    "a": ["o", "e"],
    "c": ["e", "o"],
    "e": ["c", "a"],
    "i": ["l", "1"],
    "l": ["i", "1"],
    "m": ["n", "rn"],
    "n": ["m", "u"],
    "o": ["0", "a"],
    "r": ["n", "i"],
    "t": ["f", "l"],
    "u": ["v", "n"],
    "0": ["o"],
    "1": ["l", "i"],
}

# Common medicine misspellings that simulate OCR errors
OCR_MISSPELLINGS: Dict[str, str] = {
    "paracetamol": "paracetmol",
    "ibuprofen": "ibuprofn",
    "amoxicillin": "amoxicilin",
    "azithromycin": "azithromycn",
    "metronidazole": "metronidazol",
    "ciprofloxacin": "ciprofloxacn",
    "diclofenac": "diclofenec",
    "atorvastatin": "atorvastatim",
    "metformin": "metformin",
    "omeprazole": "omeprazol",
    "pantoprazole": "pantoprazol",
    "warfarin": "warfarin",
    "losartan": "losarton",
    "clopidogrel": "clopidogrl",
    "prednisolone": "prednisolon",
}


def expand_abbreviation(token: str) -> str:
    """Expand a known medical abbreviation or correct OCR misspelling."""
    clean = token.lower().strip().rstrip(".")
    # Check abbreviation first
    if clean in ABBREVIATIONS:
        return ABBREVIATIONS[clean]
    # Check OCR misspelling correction
    for correct, misspelled in OCR_MISSPELLINGS.items():
        if clean == misspelled:
            return correct
    # Fuzzy: 1-char difference from a known correct spelling
    for correct in OCR_MISSPELLINGS:
        if abs(len(clean) - len(correct)) <= 1:
            diffs = sum(1 for a, b in zip(clean, correct) if a != b)
            extra = abs(len(clean) - len(correct))
            if diffs + extra == 1:
                return correct
    return token


def is_abbreviation(token: str) -> bool:
    """Check if a token is a known medical abbreviation."""
    clean = token.lower().strip().rstrip(".")
    return clean in ABBREVIATIONS


def apply_ocr_noise(text: str, noise_level: float = 0.15) -> str:
    """
    Apply OCR-like noise to prescription text.

    Args:
        text: Clean prescription text
        noise_level: Probability of introducing an error per word (0.0-1.0)

    Returns:
        Noisy prescription text
    """
    lines = text.split("\n")
    noisy_lines = []

    for line in lines:
        words = line.split()
        noisy_words = []
        for word in words:
            if random.random() > noise_level:
                noisy_words.append(word)
                continue

            lower_word = word.lower()

            # Strategy 1: Known misspelling (50% chance)
            if random.random() < 0.5 and lower_word in OCR_MISSPELLINGS:
                replacement = OCR_MISSPELLINGS[lower_word]
                if word[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                noisy_words.append(replacement)
                continue

            # Strategy 2: Character substitution (30% chance)
            if random.random() < 0.6:
                chars = list(word)
                if len(chars) > 3:
                    idx = random.randint(1, len(chars) - 2)
                    char = chars[idx].lower()
                    if char in OCR_SUBSTITUTIONS:
                        chars[idx] = random.choice(OCR_SUBSTITUTIONS[char])
                        noisy_words.append("".join(chars))
                        continue

            # Strategy 3: Character deletion (20% chance)
            if len(word) > 4:
                idx = random.randint(1, len(word) - 2)
                noisy_words.append(word[:idx] + word[idx + 1:])
            else:
                noisy_words.append(word)
        noisy_lines.append(" ".join(noisy_words))

    return "\n".join(noisy_lines)


def add_formatting_noise(text: str) -> str:
    """
    Add formatting inconsistencies to prescription text.
    Simulates real-world prescription formatting issues.
    """
    lines = text.strip().split("\n")
    noisy_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Random leading whitespace
        if random.random() < 0.3:
            line = "  " + line

        # Random case changes
        if random.random() < 0.2:
            line = line.upper()
        elif random.random() < 0.1:
            line = line.lower()

        # Missing periods/dots
        if random.random() < 0.3:
            line = line.replace(".", "")

        noisy_lines.append(line)

    return "\n".join(noisy_lines)


def generate_noisy_prescription(
    clean_text: str,
    noise_level: float = 0.15,
    apply_formatting: bool = True,
) -> str:
    """
    Generate a noisy version of a clean prescription text.
    Combines OCR noise and formatting noise.
    """
    noisy = apply_ocr_noise(clean_text, noise_level=noise_level)
    if apply_formatting:
        noisy = add_formatting_noise(noisy)
    return noisy
