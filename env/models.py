"""
env/models.py -- Pydantic models for PharmacistEnv.

Defines the complete type system for observations, actions, rewards,
patient profiles, inventory, drug interactions, and risk flags.
All models are strongly typed with validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Strongly typed action space for the pharmacist agent."""
    EXTRACT_MEDICINE = "extract_medicine"
    CHECK_INTERACTION = "check_interaction"
    ASK_PATIENT_INFO = "ask_patient_info"
    SEARCH_INVENTORY = "search_inventory"
    SUGGEST_ALTERNATIVE = "suggest_alternative"
    RISK_ASSESSMENT = "risk_assessment"
    FINALIZE = "finalize"


class Severity(str, Enum):
    """Severity levels for interactions, risks, and alerts."""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class RiskCategory(str, Enum):
    """Categories of clinical risk."""
    ALLERGY = "allergy"
    INTERACTION = "interaction"
    AGE = "age"
    CONTRAINDICATION = "contraindication"
    DOSAGE = "dosage"
    PREGNANCY = "pregnancy"
    RENAL = "renal"


# ---------------------------------------------------------------------------
# Medicine / Extraction
# ---------------------------------------------------------------------------

class ExtractedMedicine(BaseModel):
    """A medicine extracted from the prescription text by the agent."""
    name: str = Field(..., description="Medicine name as extracted")
    dosage: Optional[str] = Field(None, description="Dosage amount and unit")
    frequency: Optional[str] = Field(None, description="Dosing frequency")
    duration: Optional[str] = Field(None, description="Duration of treatment")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Extraction confidence")


class GroundTruthMedicine(BaseModel):
    """Ground truth medicine data for task validation."""
    name: str
    generic_name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    is_safe: bool = True
    requires_alternative: bool = False
    alternative_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------

class PatientProfile(BaseModel):
    """Complete patient profile for clinical decision-making."""
    age: int = Field(..., ge=0, le=120)
    weight_kg: Optional[float] = Field(None, ge=0)
    gender: str = Field("unknown", description="male / female / unknown")
    allergies: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    pregnancy: bool = False
    renal_function: str = Field("normal", description="normal / impaired / severe")


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

class InventoryItem(BaseModel):
    """A single item in the pharmacy inventory. Stock is mutable."""
    name: str
    generic_name: str
    stock: int = Field(..., ge=0)
    price: float = Field(..., ge=0)
    dosage_form: str = "tablet"
    strength: str = ""
    category: str = ""


# ---------------------------------------------------------------------------
# Interactions and Risks
# ---------------------------------------------------------------------------

class DrugInteraction(BaseModel):
    """A detected drug-drug interaction."""
    drug_a: str
    drug_b: str
    severity: Severity
    description: str
    clinical_effect: str
    mechanism: str = ""


class RiskFlag(BaseModel):
    """A clinical risk flag raised during the simulation."""
    category: RiskCategory
    severity: Severity
    description: str
    affected_drug: str


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    Strongly typed action submitted by the agent.
    Each action_type requires specific parameters.
    """
    action_type: ActionType
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v, info):
        return v


# ---------------------------------------------------------------------------
# Step Log
# ---------------------------------------------------------------------------

class StepLog(BaseModel):
    """Record of a single environment step for audit trail."""
    step: int
    action_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reward: float = 0.0
    reward_components: Dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""
    state_changes: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Observation (full state)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Complete environment state exposed to the agent.
    Evolves after every action.
    """
    prescription_text: str = Field(..., description="Raw prescription input (may contain noise)")
    extracted_medicines: List[ExtractedMedicine] = Field(default_factory=list)
    patient_profile: PatientProfile
    inventory: List[InventoryItem] = Field(default_factory=list)
    detected_interactions: List[DrugInteraction] = Field(default_factory=list)
    risk_flags: List[RiskFlag] = Field(default_factory=list)
    action_history: List[StepLog] = Field(default_factory=list)
    step_count: int = 0
    max_steps: int = 15
    available_actions: List[str] = Field(default_factory=list)
    clinical_notes: List[str] = Field(default_factory=list)
    final_decision: Optional[Dict[str, Any]] = None
    done: bool = False


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Detailed reward breakdown for a single step."""
    value: float
    components: Dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Task Definition
# ---------------------------------------------------------------------------

class TaskDefinition(BaseModel):
    """Complete definition of a simulation task."""
    name: str
    difficulty: str
    description: str
    prescription_text: str
    prescription_clean: str  # ground truth clean text
    patient_profile: PatientProfile
    inventory: List[InventoryItem]
    ground_truth_medicines: List[GroundTruthMedicine]
    expected_interactions: List[DrugInteraction] = Field(default_factory=list)
    expected_risks: List[RiskFlag] = Field(default_factory=list)
    optimal_action_count: int = 5
    notes: str = ""