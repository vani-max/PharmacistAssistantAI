"""
env/tasks.py -- Strategic task definitions for PharmacistEnv.

Three difficulty levels, each requiring progressively deeper clinical reasoning:

EASY:   Simple single-drug prescription. No interactions, no risks.
MEDIUM: Abbreviations, OCR noise, stock shortage, alternative needed.
HARD:   Multi-drug with critical DDI, allergy conflict, age risk,
        missing dosage, stock issue, conflicting instructions.
        Requires multi-step reasoning across state.
"""

from .models import (
    TaskDefinition, PatientProfile, InventoryItem,
    GroundTruthMedicine, DrugInteraction, RiskFlag,
    Severity, RiskCategory,
)
from .noise import generate_noisy_prescription


# ---------------------------------------------------------------------------
# TASK: EASY
# ---------------------------------------------------------------------------

def _build_easy() -> TaskDefinition:
    """
    Simple prescription: single medicine, healthy patient, full stock.
    Optimal path: extract -> search_inventory -> finalize (3 steps).
    """
    clean_text = "Tab Paracetamol 500mg 1-1-1 for 5 days"

    return TaskDefinition(
        name="easy",
        difficulty="easy",
        description="Simple single-drug prescription for a healthy adult patient.",
        prescription_text=generate_noisy_prescription(clean_text, noise_level=0.0),
        prescription_clean=clean_text,
        patient_profile=PatientProfile(
            age=30,
            weight_kg=70.0,
            gender="male",
            allergies=[],
            conditions=[],
            current_medications=[],
        ),
        inventory=[
            InventoryItem(
                name="Paracetamol 500mg",
                generic_name="paracetamol",
                stock=100,
                price=2.50,
                dosage_form="tablet",
                strength="500mg",
                category="analgesic",
            ),
            InventoryItem(
                name="Ibuprofen 400mg",
                generic_name="ibuprofen",
                stock=50,
                price=4.00,
                dosage_form="tablet",
                strength="400mg",
                category="nsaid",
            ),
            InventoryItem(
                name="Amoxicillin 250mg",
                generic_name="amoxicillin",
                stock=80,
                price=6.00,
                dosage_form="capsule",
                strength="250mg",
                category="antibiotic",
            ),
        ],
        ground_truth_medicines=[
            GroundTruthMedicine(
                name="Paracetamol",
                generic_name="paracetamol",
                dosage="500mg",
                frequency="1-1-1",
                duration="5 days",
                is_safe=True,
            ),
        ],
        expected_interactions=[],
        expected_risks=[],
        optimal_action_count=3,
        notes="Baseline task. Agent should extract, verify availability, and dispense.",
    )


# ---------------------------------------------------------------------------
# TASK: MEDIUM
# ---------------------------------------------------------------------------

def _build_medium() -> TaskDefinition:
    """
    Abbreviations + OCR noise + stock shortage + alternative needed.
    Optimal path: extract -> search_inventory (find shortage)
                  -> suggest_alternative -> finalize (4-5 steps).
    """
    clean_text = (
        "Tab PCM 500mg BD for 3 days\n"
        "Cap Amoxicillin 250mg TDS for 7 days"
    )

    return TaskDefinition(
        name="medium",
        difficulty="medium",
        description=(
            "Two-drug prescription with abbreviations. "
            "Amoxicillin is out of stock; agent must identify shortage "
            "and suggest an appropriate alternative."
        ),
        prescription_text=generate_noisy_prescription(clean_text, noise_level=0.10),
        prescription_clean=clean_text,
        patient_profile=PatientProfile(
            age=25,
            weight_kg=60.0,
            gender="female",
            allergies=["sulfa"],
            conditions=["urinary tract infection"],
            current_medications=[],
        ),
        inventory=[
            InventoryItem(
                name="Paracetamol 500mg",
                generic_name="paracetamol",
                stock=100,
                price=2.50,
                dosage_form="tablet",
                strength="500mg",
                category="analgesic",
            ),
            InventoryItem(
                name="Amoxicillin 250mg",
                generic_name="amoxicillin",
                stock=0,  # OUT OF STOCK
                price=6.00,
                dosage_form="capsule",
                strength="250mg",
                category="antibiotic",
            ),
            InventoryItem(
                name="Cephalexin 250mg",
                generic_name="cephalexin",
                stock=60,
                price=7.00,
                dosage_form="capsule",
                strength="250mg",
                category="antibiotic",
            ),
            InventoryItem(
                name="Azithromycin 500mg",
                generic_name="azithromycin",
                stock=40,
                price=12.00,
                dosage_form="tablet",
                strength="500mg",
                category="antibiotic",
            ),
            InventoryItem(
                name="Ibuprofen 400mg",
                generic_name="ibuprofen",
                stock=50,
                price=4.00,
                dosage_form="tablet",
                strength="400mg",
                category="nsaid",
            ),
        ],
        ground_truth_medicines=[
            GroundTruthMedicine(
                name="Paracetamol",
                generic_name="paracetamol",
                dosage="500mg",
                frequency="BD",
                duration="3 days",
                is_safe=True,
            ),
            GroundTruthMedicine(
                name="Amoxicillin",
                generic_name="amoxicillin",
                dosage="250mg",
                frequency="TDS",
                duration="7 days",
                is_safe=True,
                requires_alternative=True,
                alternative_reason="Out of stock",
            ),
        ],
        expected_interactions=[],
        expected_risks=[],
        optimal_action_count=5,
        notes=(
            "Agent must expand abbreviations (PCM, BD, TDS), detect stock "
            "shortage for Amoxicillin, and suggest Cephalexin or Azithromycin "
            "as safe alternative. Sulfa allergy should not affect alternatives."
        ),
    )


# ---------------------------------------------------------------------------
# TASK: HARD (CRITICAL)
# ---------------------------------------------------------------------------

def _build_hard() -> TaskDefinition:
    """
    Multi-drug with cascading clinical dangers:
    1. Warfarin + Ibuprofen = CRITICAL bleeding interaction
    2. Amoxicillin + Penicillin allergy = CRITICAL allergy conflict
    3. Age 72 + NSAID = elevated GI bleeding risk
    4. OCR noise on 'Ibuprofen' spelling
    5. Missing duration for Warfarin
    6. Amoxicillin requires alternative due to allergy

    Optimal path: extract -> check_interaction -> ask_patient_info
                  -> risk_assessment -> suggest_alternative (x2)
                  -> search_inventory -> finalize (7-8 steps).
    """
    clean_text = (
        "Tab Warfarin 5mg OD\n"
        "Tab Ibuprofen 400mg TDS SOS\n"
        "Cap Amoxicillin 500mg BD for 5 days\n"
        "Tab Metformin 500mg BD"
    )

    # Apply significant noise to simulate real-world handwriting OCR
    noisy_text = (
        "Tab Warfarin 5mg OD\n"
        "Tab Ibuprofn 400mg TDS SOS\n"
        "Cap Amoxicilin 500mg BD x 5 days\n"
        "Tab Metformin 500mg BD"
    )

    return TaskDefinition(
        name="hard",
        difficulty="hard",
        description=(
            "Complex multi-drug prescription for an elderly patient on "
            "anticoagulation therapy. Contains a critical drug-drug interaction "
            "(Warfarin + NSAID), an allergy conflict (Penicillin allergy + "
            "Amoxicillin), age-related NSAID risk, and OCR noise."
        ),
        prescription_text=noisy_text,
        prescription_clean=clean_text,
        patient_profile=PatientProfile(
            age=72,
            weight_kg=68.0,
            gender="male",
            allergies=["penicillin"],
            conditions=[
                "atrial fibrillation",
                "type 2 diabetes",
                "hypertension",
            ],
            current_medications=["lisinopril 10mg"],
            renal_function="impaired",
        ),
        inventory=[
            InventoryItem(
                name="Warfarin 5mg",
                generic_name="warfarin",
                stock=30,
                price=8.00,
                dosage_form="tablet",
                strength="5mg",
                category="anticoagulant",
            ),
            InventoryItem(
                name="Ibuprofen 400mg",
                generic_name="ibuprofen",
                stock=50,
                price=4.00,
                dosage_form="tablet",
                strength="400mg",
                category="nsaid",
            ),
            InventoryItem(
                name="Amoxicillin 500mg",
                generic_name="amoxicillin",
                stock=40,
                price=8.00,
                dosage_form="capsule",
                strength="500mg",
                category="antibiotic",
            ),
            InventoryItem(
                name="Metformin 500mg",
                generic_name="metformin",
                stock=100,
                price=3.00,
                dosage_form="tablet",
                strength="500mg",
                category="antidiabetic",
            ),
            InventoryItem(
                name="Paracetamol 500mg",
                generic_name="paracetamol",
                stock=100,
                price=2.50,
                dosage_form="tablet",
                strength="500mg",
                category="analgesic",
            ),
            InventoryItem(
                name="Azithromycin 500mg",
                generic_name="azithromycin",
                stock=35,
                price=15.00,
                dosage_form="tablet",
                strength="500mg",
                category="antibiotic",
            ),
            InventoryItem(
                name="Pantoprazole 40mg",
                generic_name="pantoprazole",
                stock=80,
                price=5.00,
                dosage_form="tablet",
                strength="40mg",
                category="ppi",
            ),
        ],
        ground_truth_medicines=[
            GroundTruthMedicine(
                name="Warfarin",
                generic_name="warfarin",
                dosage="5mg",
                frequency="OD",
                duration=None,
                is_safe=True,
            ),
            GroundTruthMedicine(
                name="Ibuprofen",
                generic_name="ibuprofen",
                dosage="400mg",
                frequency="TDS SOS",
                duration=None,
                is_safe=False,  # UNSAFE: interacts with Warfarin + age risk
                requires_alternative=True,
                alternative_reason=(
                    "Critical interaction with Warfarin (bleeding risk) "
                    "and age-related NSAID contraindication (72 years old)"
                ),
            ),
            GroundTruthMedicine(
                name="Amoxicillin",
                generic_name="amoxicillin",
                dosage="500mg",
                frequency="BD",
                duration="5 days",
                is_safe=False,  # UNSAFE: penicillin allergy
                requires_alternative=True,
                alternative_reason="Patient has penicillin allergy; amoxicillin "
                                  "is a penicillin-class antibiotic",
            ),
            GroundTruthMedicine(
                name="Metformin",
                generic_name="metformin",
                dosage="500mg",
                frequency="BD",
                duration=None,
                is_safe=True,
            ),
        ],
        expected_interactions=[
            DrugInteraction(
                drug_a="Warfarin",
                drug_b="Ibuprofen",
                severity=Severity.CRITICAL,
                description="Anticoagulant combined with NSAID creates severe "
                           "bleeding risk",
                clinical_effect="Significantly increased risk of gastrointestinal "
                               "and intracranial hemorrhage",
                mechanism="NSAIDs inhibit platelet aggregation and damage GI mucosa; "
                         "warfarin inhibits clotting factor synthesis.",
            ),
        ],
        expected_risks=[
            RiskFlag(
                category=RiskCategory.ALLERGY,
                severity=Severity.CRITICAL,
                description="Patient is allergic to penicillin. Amoxicillin has "
                           "cross-reactivity and is contraindicated.",
                affected_drug="Amoxicillin",
            ),
            RiskFlag(
                category=RiskCategory.AGE,
                severity=Severity.HIGH,
                description="NSAIDs in elderly (70+): elevated risk of GI bleeding, "
                           "renal impairment, and cardiovascular events",
                affected_drug="Ibuprofen",
            ),
        ],
        optimal_action_count=8,
        notes=(
            "This task requires the agent to: "
            "(1) Extract 4 medicines despite OCR noise, "
            "(2) Detect Warfarin+Ibuprofen interaction (critical), "
            "(3) Detect Penicillin allergy vs Amoxicillin conflict (critical), "
            "(4) Identify age-based NSAID risk (72yo), "
            "(5) Replace Ibuprofen with Paracetamol (safe for Warfarin patients), "
            "(6) Replace Amoxicillin with Azithromycin (no penicillin cross-reactivity), "
            "(7) Verify Metformin is appropriate (monitor renal function), "
            "(8) Make a safe final decision. "
            "Failing to detect the Warfarin+NSAID interaction is a critical safety failure."
        ),
    )


# ---------------------------------------------------------------------------
# Task Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "easy": _build_easy,
    "medium": _build_medium,
    "hard": _build_hard,
}


def get_task(name: str) -> TaskDefinition:
    """Get a task definition by name. Rebuilds each time for noise variation."""
    builder = TASK_REGISTRY.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown task: {name}. Available: {list(TASK_REGISTRY.keys())}"
        )
    return builder()


def list_tasks() -> list:
    """Return list of available task names."""
    return list(TASK_REGISTRY.keys())