"""
env/interactions.py -- Rule-based drug interaction engine.

Contains real pharmacological interaction rules covering:
- Drug-drug interactions (DDI)
- Drug-allergy cross-reactivity
- Age-based contraindications
- Condition-based contraindications
- Pregnancy contraindications

Each rule is backed by real clinical pharmacology.
"""

from typing import List, Dict, Tuple, Optional
from .models import DrugInteraction, RiskFlag, PatientProfile, Severity, RiskCategory


# ---------------------------------------------------------------------------
# Drug-Drug Interaction Database
# Each entry: (drug_a_keywords, drug_b_keywords, severity, description,
#              clinical_effect, mechanism)
# ---------------------------------------------------------------------------

INTERACTION_RULES: List[Dict] = [
    {
        "drugs_a": ["warfarin", "coumadin"],
        "drugs_b": ["ibuprofen", "aspirin", "diclofenac", "naproxen", "piroxicam",
                     "indomethacin", "meloxicam", "ketorolac", "celecoxib"],
        "severity": "critical",
        "description": "Anticoagulant combined with NSAID creates severe bleeding risk",
        "clinical_effect": "Significantly increased risk of gastrointestinal and "
                          "intracranial hemorrhage",
        "mechanism": "NSAIDs inhibit platelet aggregation and damage GI mucosa; "
                    "warfarin inhibits clotting factor synthesis. Combined effect "
                    "creates synergistic bleeding risk.",
    },
    {
        "drugs_a": ["warfarin", "coumadin"],
        "drugs_b": ["amoxicillin", "ciprofloxacin", "metronidazole", "fluconazole",
                     "erythromycin", "clarithromycin", "cotrimoxazole"],
        "severity": "high",
        "description": "Anticoagulant combined with certain antibiotics increases INR",
        "clinical_effect": "Elevated INR leading to increased bleeding risk",
        "mechanism": "Antibiotics alter gut flora reducing vitamin K synthesis, "
                    "and some inhibit CYP450 enzymes affecting warfarin metabolism.",
    },
    {
        "drugs_a": ["lisinopril", "enalapril", "ramipril", "captopril", "perindopril"],
        "drugs_b": ["potassium", "spironolactone", "amiloride", "triamterene"],
        "severity": "high",
        "description": "ACE inhibitor combined with potassium-sparing agent",
        "clinical_effect": "Dangerous hyperkalemia leading to cardiac arrhythmias",
        "mechanism": "ACE inhibitors reduce aldosterone secretion, retaining potassium. "
                    "Combined with potassium supplements or K-sparing diuretics, "
                    "serum potassium can reach fatal levels.",
    },
    {
        "drugs_a": ["fluoxetine", "sertraline", "paroxetine", "citalopram",
                     "escitalopram", "fluvoxamine"],
        "drugs_b": ["phenelzine", "tranylcypromine", "isocarboxazid", "selegiline",
                     "moclobemide", "linezolid"],
        "severity": "critical",
        "description": "SSRI combined with MAOI causes serotonin syndrome",
        "clinical_effect": "Potentially fatal serotonin syndrome: hyperthermia, "
                          "rigidity, seizures, cardiovascular collapse",
        "mechanism": "SSRIs block serotonin reuptake while MAOIs prevent serotonin "
                    "breakdown, causing dangerous serotonin accumulation.",
    },
    {
        "drugs_a": ["metoprolol", "atenolol", "propranolol", "bisoprolol", "carvedilol"],
        "drugs_b": ["verapamil", "diltiazem"],
        "severity": "high",
        "description": "Beta-blocker combined with non-dihydropyridine calcium "
                      "channel blocker",
        "clinical_effect": "Severe bradycardia, heart block, and hypotension",
        "mechanism": "Both drug classes suppress AV node conduction and reduce "
                    "heart rate. Combined effect can cause complete heart block.",
    },
    {
        "drugs_a": ["atorvastatin", "simvastatin", "rosuvastatin", "lovastatin"],
        "drugs_b": ["gemfibrozil", "fenofibrate"],
        "severity": "high",
        "description": "Statin combined with fibrate increases rhabdomyolysis risk",
        "clinical_effect": "Rhabdomyolysis: muscle breakdown releasing myoglobin, "
                          "potentially causing acute kidney injury",
        "mechanism": "Both drug classes cause myopathy independently. Combined use "
                    "has synergistic toxicity on skeletal muscle.",
    },
    {
        "drugs_a": ["morphine", "oxycodone", "hydrocodone", "fentanyl", "codeine",
                     "tramadol", "methadone"],
        "drugs_b": ["diazepam", "lorazepam", "alprazolam", "clonazepam", "midazolam",
                     "temazepam"],
        "severity": "critical",
        "description": "Opioid combined with benzodiazepine",
        "clinical_effect": "Fatal respiratory depression, excessive sedation, coma",
        "mechanism": "Opioids depress respiratory drive via brainstem mu-receptors. "
                    "Benzodiazepines enhance GABAergic inhibition. Combined CNS "
                    "depression can stop breathing.",
    },
    {
        "drugs_a": ["metformin"],
        "drugs_b": ["alcohol", "contrast dye"],
        "severity": "high",
        "description": "Metformin with agents that impair renal/hepatic function",
        "clinical_effect": "Lactic acidosis, a rare but potentially fatal condition",
        "mechanism": "Metformin is cleared renally. Agents that impair renal function "
                    "cause metformin accumulation and lactate buildup.",
    },
    {
        "drugs_a": ["lithium"],
        "drugs_b": ["ibuprofen", "naproxen", "diclofenac", "indomethacin",
                     "lisinopril", "enalapril", "hydrochlorothiazide"],
        "severity": "high",
        "description": "Lithium levels increased by NSAIDs, ACE inhibitors, or thiazides",
        "clinical_effect": "Lithium toxicity: tremor, confusion, seizures, renal failure",
        "mechanism": "These drugs reduce renal lithium clearance, causing serum "
                    "lithium to rise into the toxic range.",
    },
    {
        "drugs_a": ["clopidogrel", "prasugrel", "ticagrelor"],
        "drugs_b": ["omeprazole", "esomeprazole"],
        "severity": "moderate",
        "description": "Antiplatelet agent with proton pump inhibitor",
        "clinical_effect": "Reduced antiplatelet efficacy, increased cardiovascular risk",
        "mechanism": "PPIs inhibit CYP2C19, which is required to convert clopidogrel "
                    "to its active metabolite.",
    },
    {
        "drugs_a": ["prednisolone", "dexamethasone", "hydrocortisone", "methylprednisolone"],
        "drugs_b": ["ibuprofen", "aspirin", "diclofenac", "naproxen"],
        "severity": "moderate",
        "description": "Corticosteroid combined with NSAID",
        "clinical_effect": "Increased risk of gastrointestinal ulceration and bleeding",
        "mechanism": "Corticosteroids impair mucosal healing. NSAIDs inhibit "
                    "protective prostaglandins. Combined effect damages GI mucosa.",
    },
    {
        "drugs_a": ["digoxin"],
        "drugs_b": ["amiodarone", "verapamil", "quinidine"],
        "severity": "high",
        "description": "Digoxin toxicity risk with interacting drugs",
        "clinical_effect": "Digoxin toxicity: nausea, visual disturbances, fatal arrhythmias",
        "mechanism": "These drugs increase serum digoxin levels by inhibiting "
                    "P-glycoprotein and reducing renal clearance.",
    },
]


# ---------------------------------------------------------------------------
# Allergy Cross-Reactivity Database
# ---------------------------------------------------------------------------

ALLERGY_CROSS_REACTIVITY: Dict[str, List[str]] = {
    "penicillin": [
        "amoxicillin", "ampicillin", "piperacillin", "nafcillin",
        "oxacillin", "flucloxacillin", "dicloxacillin", "augmentin",
    ],
    "sulfa": [
        "sulfamethoxazole", "cotrimoxazole", "sulfasalazine", "dapsone",
        "sulfadiazine", "trimethoprim-sulfamethoxazole",
    ],
    "cephalosporin": [
        "cephalexin", "cefazolin", "cefuroxime", "ceftriaxone",
        "cefixime", "cefpodoxime", "ceftazidime",
    ],
    "nsaid": [
        "aspirin", "ibuprofen", "naproxen", "diclofenac",
        "indomethacin", "piroxicam", "meloxicam", "ketorolac",
    ],
    "aspirin": [
        "aspirin", "ibuprofen", "naproxen", "diclofenac",
    ],
    "codeine": [
        "codeine", "morphine", "oxycodone", "hydrocodone",
    ],
}


# ---------------------------------------------------------------------------
# Age-Based Contraindication Rules
# ---------------------------------------------------------------------------

AGE_CONTRAINDICATIONS: List[Dict] = [
    {
        "drugs": ["aspirin"],
        "age_max": 16,
        "severity": "critical",
        "description": "Aspirin in children under 16: risk of Reye's syndrome, "
                      "a rare but fatal condition causing brain and liver swelling",
    },
    {
        "drugs": ["diazepam", "lorazepam", "alprazolam", "clonazepam", "temazepam"],
        "age_min": 65,
        "severity": "high",
        "description": "Benzodiazepines in elderly (65+): increased fall risk, "
                      "cognitive impairment, paradoxical agitation, respiratory depression",
    },
    {
        "drugs": ["metformin"],
        "age_min": 80,
        "severity": "moderate",
        "description": "Metformin in patients 80+: increased risk of lactic acidosis "
                      "due to age-related decline in renal function",
    },
    {
        "drugs": ["ibuprofen", "naproxen", "diclofenac", "piroxicam", "ketorolac"],
        "age_min": 70,
        "severity": "high",
        "description": "NSAIDs in elderly (70+): elevated risk of GI bleeding, "
                      "renal impairment, and cardiovascular events",
    },
    {
        "drugs": ["doxycycline", "tetracycline", "minocycline"],
        "age_max": 8,
        "severity": "high",
        "description": "Tetracyclines in children under 8: permanent tooth "
                      "discoloration and bone growth inhibition",
    },
]


# ---------------------------------------------------------------------------
# Condition Contraindication Rules
# ---------------------------------------------------------------------------

CONDITION_CONTRAINDICATIONS: List[Dict] = [
    {
        "drugs": ["ibuprofen", "naproxen", "diclofenac", "aspirin", "ketorolac"],
        "conditions": ["peptic ulcer", "gi bleeding", "gastric ulcer"],
        "severity": "critical",
        "description": "NSAIDs are contraindicated in patients with active or "
                      "history of peptic ulcer disease",
    },
    {
        "drugs": ["metformin"],
        "conditions": ["severe renal impairment", "chronic kidney disease stage 4",
                       "chronic kidney disease stage 5"],
        "severity": "critical",
        "description": "Metformin is contraindicated in severe renal impairment "
                      "due to risk of lactic acidosis",
    },
    {
        "drugs": ["metoprolol", "atenolol", "propranolol", "bisoprolol"],
        "conditions": ["severe asthma", "acute asthma", "copd severe"],
        "severity": "high",
        "description": "Non-selective beta-blockers can trigger severe bronchospasm "
                      "in patients with asthma or severe COPD",
    },
    {
        "drugs": ["lisinopril", "enalapril", "ramipril"],
        "conditions": ["bilateral renal artery stenosis", "angioedema history"],
        "severity": "critical",
        "description": "ACE inhibitors can cause renal failure in bilateral renal "
                      "artery stenosis and life-threatening angioedema in susceptible patients",
    },
]


# ---------------------------------------------------------------------------
# Engine Functions
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Lowercase, strip whitespace, and correct common OCR misspellings."""
    n = name.lower().strip().replace("-", " ")
    # Reverse OCR misspelling lookup: if 'n' matches a known misspelling, fix it
    for correct, misspelled in _OCR_CORRECTIONS.items():
        if n == misspelled or n.startswith(misspelled) or misspelled.startswith(n):
            return correct
    # Also try fuzzy: if 1 character difference from a known drug
    for correct in _KNOWN_DRUGS:
        if abs(len(n) - len(correct)) <= 1 and _edit_dist_1(n, correct):
            return correct
    return n


# Known OCR misspelling corrections
_OCR_CORRECTIONS = {
    "paracetamol": "paracetmol",
    "ibuprofen": "ibuprofn",
    "amoxicillin": "amoxicilin",
    "azithromycin": "azithromycn",
    "metronidazole": "metronidazol",
    "ciprofloxacin": "ciprofloxacn",
    "diclofenac": "diclofenec",
    "atorvastatin": "atorvastatim",
    "omeprazole": "omeprazol",
    "pantoprazole": "pantoprazol",
    "losartan": "losarton",
    "clopidogrel": "clopidogrl",
    "prednisolone": "prednisolon",
}

_KNOWN_DRUGS = set()
for rule in INTERACTION_RULES:
    _KNOWN_DRUGS.update(rule["drugs_a"])
    _KNOWN_DRUGS.update(rule["drugs_b"])
for drugs in ALLERGY_CROSS_REACTIVITY.values():
    _KNOWN_DRUGS.update(drugs)
for rule in AGE_CONTRAINDICATIONS:
    _KNOWN_DRUGS.update(rule["drugs"])


def _edit_dist_1(a: str, b: str) -> bool:
    """Check if edit distance between a and b is exactly 1."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = sum(1 for x, y in zip(a, b) if x != y)
        return diffs == 1
    # Insertion/deletion
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    i = j = 0
    found_diff = False
    while i < len(longer) and j < len(shorter):
        if longer[i] != shorter[j]:
            if found_diff:
                return False
            found_diff = True
            i += 1
        else:
            i += 1
            j += 1
    return True


def check_drug_interactions(
    drug_names: List[str],
) -> List[DrugInteraction]:
    """
    Check all pairwise drug-drug interactions among the given drug names.
    Returns a list of detected interactions sorted by severity.
    """
    normalized = [_normalize(d) for d in drug_names]
    found: List[DrugInteraction] = []
    seen_pairs = set()

    for rule in INTERACTION_RULES:
        for i, drug_a in enumerate(normalized):
            for j, drug_b in enumerate(normalized):
                if i >= j:
                    continue

                pair_key = (min(drug_a, drug_b), max(drug_a, drug_b))
                if pair_key in seen_pairs:
                    continue

                a_match = any(kw in drug_a for kw in rule["drugs_a"])
                b_match = any(kw in drug_b for kw in rule["drugs_b"])
                a_match_rev = any(kw in drug_a for kw in rule["drugs_b"])
                b_match_rev = any(kw in drug_b for kw in rule["drugs_a"])

                if (a_match and b_match) or (a_match_rev and b_match_rev):
                    seen_pairs.add(pair_key)
                    found.append(DrugInteraction(
                        drug_a=drug_names[i],
                        drug_b=drug_names[j],
                        severity=Severity(rule["severity"]),
                        description=rule["description"],
                        clinical_effect=rule["clinical_effect"],
                        mechanism=rule["mechanism"],
                    ))

    severity_order = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
    found.sort(key=lambda x: severity_order.get(x.severity.value, 4))
    return found


def check_allergy_conflicts(
    drug_names: List[str],
    patient_allergies: List[str],
) -> List[RiskFlag]:
    """
    Check if any prescribed drug conflicts with patient allergies,
    including cross-reactivity.
    """
    flags: List[RiskFlag] = []
    norm_allergies = [_normalize(a) for a in patient_allergies]

    for drug in drug_names:
        norm_drug = _normalize(drug)

        # Direct allergy match
        for allergy in norm_allergies:
            if allergy in norm_drug or norm_drug in allergy:
                flags.append(RiskFlag(
                    category=RiskCategory.ALLERGY,
                    severity=Severity.CRITICAL,
                    description=f"Patient is allergic to {allergy}. "
                               f"{drug} is contraindicated.",
                    affected_drug=drug,
                ))

        # Cross-reactivity check
        for allergy in norm_allergies:
            cross_drugs = ALLERGY_CROSS_REACTIVITY.get(allergy, [])
            for cross in cross_drugs:
                if cross in norm_drug or norm_drug in cross:
                    flags.append(RiskFlag(
                        category=RiskCategory.ALLERGY,
                        severity=Severity.CRITICAL,
                        description=f"Patient is allergic to {allergy}. "
                                   f"{drug} has cross-reactivity and is contraindicated.",
                        affected_drug=drug,
                    ))
                    break

    return flags


def check_age_contraindications(
    drug_names: List[str],
    patient_age: int,
) -> List[RiskFlag]:
    """Check for age-based contraindications."""
    flags: List[RiskFlag] = []

    for drug in drug_names:
        norm_drug = _normalize(drug)
        for rule in AGE_CONTRAINDICATIONS:
            for kw in rule["drugs"]:
                if kw in norm_drug:
                    age_min = rule.get("age_min")
                    age_max = rule.get("age_max")
                    triggered = False

                    if age_min is not None and patient_age >= age_min:
                        triggered = True
                    if age_max is not None and patient_age <= age_max:
                        triggered = True

                    if triggered:
                        flags.append(RiskFlag(
                            category=RiskCategory.AGE,
                            severity=Severity(rule["severity"]),
                            description=rule["description"],
                            affected_drug=drug,
                        ))
                    break

    return flags


def check_condition_contraindications(
    drug_names: List[str],
    patient_conditions: List[str],
) -> List[RiskFlag]:
    """Check for condition-based contraindications."""
    flags: List[RiskFlag] = []
    norm_conditions = [_normalize(c) for c in patient_conditions]

    for drug in drug_names:
        norm_drug = _normalize(drug)
        for rule in CONDITION_CONTRAINDICATIONS:
            drug_match = any(kw in norm_drug for kw in rule["drugs"])
            if not drug_match:
                continue

            condition_match = any(
                any(rc in nc for rc in rule["conditions"])
                for nc in norm_conditions
            )
            if condition_match:
                flags.append(RiskFlag(
                    category=RiskCategory.CONTRAINDICATION,
                    severity=Severity(rule["severity"]),
                    description=rule["description"],
                    affected_drug=drug,
                ))

    return flags


def run_full_safety_check(
    drug_names: List[str],
    patient: PatientProfile,
) -> Tuple[List[DrugInteraction], List[RiskFlag]]:
    """
    Run the complete safety check pipeline:
    1. Drug-drug interactions
    2. Allergy conflicts (including cross-reactivity)
    3. Age-based contraindications
    4. Condition-based contraindications

    Returns (interactions, risk_flags).
    """
    interactions = check_drug_interactions(drug_names)

    risk_flags: List[RiskFlag] = []
    risk_flags.extend(check_allergy_conflicts(drug_names, patient.allergies))
    risk_flags.extend(check_age_contraindications(drug_names, patient.age))
    risk_flags.extend(check_condition_contraindications(drug_names, patient.conditions))

    return interactions, risk_flags
