"""
env/environment.py -- PharmacistEnv: Multi-step Clinical Decision Environment.

This is a reinforcement-learning-style environment where each agent action:
- mutates the environment state
- produces a dense step-wise reward
- influences all subsequent observations and outcomes

Incorrect decisions create cascading failures:
- Failing to check interactions before finalizing leads to safety penalties
- Dispensing allergy-triggering drugs is penalized at -1.0
- Hallucinated medicines are penalized at -0.5

The environment evaluates not only correctness, but clinical safety,
decision-making under uncertainty, and real-world pharmacist reasoning.
"""

from typing import Tuple, Dict, Any, List, Optional
from copy import deepcopy

from .models import (
    Observation, Action, ActionType, Reward, StepLog,
    ExtractedMedicine, DrugInteraction, RiskFlag,
    Severity, RiskCategory, TaskDefinition,
)
from .tasks import get_task
from .interactions import (
    check_drug_interactions,
    check_allergy_conflicts,
    check_age_contraindications,
    check_condition_contraindications,
    run_full_safety_check,
)
from .noise import expand_abbreviation


class PharmacistEnv:
    """
    Multi-step clinical decision environment.

    Implements the OpenEnv interface:
        reset(task_name) -> Observation
        step(action)     -> (Observation, float, bool, dict)
        state()          -> Observation

    The agent navigates a sequence of clinical decisions to safely
    process a prescription. State evolves after every action.
    """

    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self._task: Optional[TaskDefinition] = None
        self._state: Optional[Observation] = None
        self._done: bool = False
        self._total_reward: float = 0.0

        # Internal tracking for grading
        self._checked_interactions: bool = False
        self._checked_allergies: bool = False
        self._checked_age_risks: bool = False
        self._searched_drugs: set = set()
        self._suggested_alternatives: Dict[str, str] = {}
        self._asked_info: set = set()
        self._risk_assessments_made: List[Dict] = []

    # -------------------------------------------------------------------
    # OpenEnv Interface
    # -------------------------------------------------------------------

    def reset(self, task_name: Optional[str] = None, custom_task: Optional['TaskDefinition'] = None) -> Observation:
        """Initialize the environment with a task. Returns initial observation."""
        if custom_task is not None:
            self._task = custom_task
            self.task_name = custom_task.name
        else:
            if task_name is not None:
                self.task_name = task_name
            self._task = get_task(self.task_name)
        self._done = False
        self._total_reward = 0.0
        self._checked_interactions = False
        self._checked_allergies = False
        self._checked_age_risks = False
        self._searched_drugs = set()
        self._suggested_alternatives = {}
        self._asked_info = set()
        self._risk_assessments_made = []

        self._state = Observation(
            prescription_text=self._task.prescription_text,
            patient_profile=deepcopy(self._task.patient_profile),
            inventory=deepcopy(self._task.inventory),
            extracted_medicines=[],
            detected_interactions=[],
            risk_flags=[],
            action_history=[],
            step_count=0,
            max_steps=15,
            available_actions=[a.value for a in ActionType],
            clinical_notes=[],
            final_decision=None,
            done=False,
        )

        return deepcopy(self._state)

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Execute an action in the environment.
        Returns (observation, reward, done, info).
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        # Validate action type
        if action.action_type not in ActionType.__members__.values():
            raise ValueError(f"Invalid action type: {action.action_type}")

        # Dispatch to handler
        handler = {
            ActionType.EXTRACT_MEDICINE: self._handle_extract,
            ActionType.CHECK_INTERACTION: self._handle_check_interaction,
            ActionType.ASK_PATIENT_INFO: self._handle_ask_patient_info,
            ActionType.SEARCH_INVENTORY: self._handle_search_inventory,
            ActionType.SUGGEST_ALTERNATIVE: self._handle_suggest_alternative,
            ActionType.RISK_ASSESSMENT: self._handle_risk_assessment,
            ActionType.FINALIZE: self._handle_finalize,
        }

        reward_obj = handler[action.action_type](action.parameters)

        # Apply redundant action penalty
        # Exempt actions that are expected to be repeated for different parameters
        if action.action_type not in (ActionType.SEARCH_INVENTORY, ActionType.SUGGEST_ALTERNATIVE):
            recent_actions = [
                log.action_type for log in self._state.action_history[-3:]
            ]
            if action.action_type.value in recent_actions:
                redundancy_penalty = -0.30
                reward_obj.value += redundancy_penalty
                reward_obj.components["redundant_action"] = redundancy_penalty
                reward_obj.reasoning += " Penalty: redundant repeated action."

        # Update step count
        self._state.step_count += 1

        # Check step limit
        if self._state.step_count >= self._state.max_steps:
            self._done = True
            self._state.done = True
            if self._state.final_decision is None:
                # Auto-penalize for running out of steps without deciding
                reward_obj.value -= 0.50
                reward_obj.components["timeout"] = -0.50
                reward_obj.reasoning += " Timeout: exceeded maximum steps without finalizing."

        # Log the step
        step_log = StepLog(
            step=self._state.step_count,
            action_type=action.action_type.value,
            parameters=action.parameters,
            reward=round(reward_obj.value, 4),
            reward_components=reward_obj.components,
            reasoning=reward_obj.reasoning,
            state_changes=self._compute_state_changes(action),
        )
        self._state.action_history.append(step_log)

        # Track total reward
        self._total_reward += reward_obj.value

        info = {
            "step_reward": round(reward_obj.value, 4),
            "total_reward": round(self._total_reward, 4),
            "reward_components": reward_obj.components,
            "reasoning": reward_obj.reasoning,
        }

        return deepcopy(self._state), round(reward_obj.value, 4), self._done, info

    def state(self) -> Observation:
        """Return the current observation state."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return deepcopy(self._state)

    # -------------------------------------------------------------------
    # Action Handlers
    # -------------------------------------------------------------------

    def _handle_extract(self, params: Dict[str, Any]) -> Reward:
        """
        Handle extract_medicine action.
        Expected params: {"medicines": [{"name": str, "dosage": str, ...}]}
        """
        medicines_data = params.get("medicines", [])
        if not medicines_data:
            return Reward(
                value=-0.20,
                components={"empty_extraction": -0.20},
                reasoning="No medicines provided in extraction.",
            )

        reward_value = 0.0
        components = {}
        reasoning_parts = []
        gt_names = [
            m.generic_name.lower()
            for m in self._task.ground_truth_medicines
        ]

        for med_data in medicines_data:
            name = med_data.get("name", "")
            expanded = expand_abbreviation(name).lower()

            # Check if this is a real medicine from the task
            matched = False
            for gt in self._task.ground_truth_medicines:
                if (expanded in gt.generic_name.lower()
                        or gt.generic_name.lower() in expanded
                        or expanded in gt.name.lower()
                        or gt.name.lower() in expanded):
                    matched = True
                    break

            if self._task.difficulty == "custom":
                reward_value += 0.10
                components[f"custom_extract_{name}"] = 0.10
                reasoning_parts.append(f"Extracted custom medicine: {name}.")
            elif matched:
                reward_value += 0.15
                components[f"correct_extract_{name}"] = 0.15
                reasoning_parts.append(f"Correctly extracted {name}.")
            else:
                reward_value -= 0.50
                components[f"hallucinated_{name}"] = -0.50
                reasoning_parts.append(
                    f"Hallucinated medicine: {name} not in prescription."
                )

            self._state.extracted_medicines.append(
                ExtractedMedicine(
                    name=name,
                    dosage=med_data.get("dosage"),
                    frequency=med_data.get("frequency"),
                    duration=med_data.get("duration"),
                    confidence=med_data.get("confidence", 0.8),
                )
            )

        # Bonus for extracting all medicines
        if self._task.difficulty != "custom" and len(self._task.ground_truth_medicines) > 0:
            extracted_names = [
                expand_abbreviation(m.name).lower()
                for m in self._state.extracted_medicines
            ]
            all_found = all(
                any(
                    gt.generic_name.lower() in en or en in gt.generic_name.lower()
                    for en in extracted_names
                )
                for gt in self._task.ground_truth_medicines
            )
            if all_found:
                reward_value += 0.10
                components["complete_extraction"] = 0.10
                reasoning_parts.append("All medicines correctly extracted.")

        self._state.clinical_notes.append(
            f"Extracted {len(medicines_data)} medicine(s) from prescription."
        )

        return Reward(
            value=round(reward_value, 4),
            components=components,
            reasoning=" ".join(reasoning_parts),
        )

    def _handle_check_interaction(self, params: Dict[str, Any]) -> Reward:
        """
        Handle check_interaction action.
        Expected params: {"drug_names": [str, ...]}
        """
        drug_names = params.get("drug_names", [])
        if not drug_names:
            # Use extracted medicines if no specific drugs provided
            drug_names = [m.name for m in self._state.extracted_medicines]

        if len(drug_names) < 2:
            return Reward(
                value=-0.10,
                components={"insufficient_drugs": -0.10},
                reasoning="Need at least 2 drugs to check interactions.",
            )

        interactions = check_drug_interactions(drug_names)
        reward_value = 0.0
        components = {}
        reasoning_parts = []

        # Add found interactions to state
        for intr in interactions:
            if intr not in self._state.detected_interactions:
                self._state.detected_interactions.append(intr)

        # Check against expected interactions
        expected = self._task.expected_interactions
        found_expected = 0

        for exp in expected:
            for det in interactions:
                exp_a = exp.drug_a.lower()
                exp_b = exp.drug_b.lower()
                det_a = det.drug_a.lower()
                det_b = det.drug_b.lower()

                if (exp_a in det_a or det_a in exp_a) and \
                   (exp_b in det_b or det_b in exp_b):
                    found_expected += 1
                    break
                if (exp_a in det_b or det_b in exp_a) and \
                   (exp_b in det_a or det_a in exp_b):
                    found_expected += 1
                    break

        if found_expected > 0:
            reward_value += 0.25 * found_expected
            components["correct_interaction"] = 0.25 * found_expected
            reasoning_parts.append(
                f"Detected {found_expected} expected interaction(s)."
            )
        elif len(expected) > 0:
            reasoning_parts.append("Expected interactions not detected.")
        elif len(interactions) == 0:
            reward_value += 0.10
            components["correct_no_interaction"] = 0.10
            reasoning_parts.append("Correctly confirmed no interactions.")
            self._state.clinical_notes.append("Checked for interactions: None found.")

        self._checked_interactions = True

        for intr in interactions:
            sev = intr.severity.value.upper()
            self._state.clinical_notes.append(
                f"[{sev}] Interaction: {intr.drug_a} + {intr.drug_b} -- "
                f"{intr.description}"
            )

        return Reward(
            value=round(reward_value, 4),
            components=components,
            reasoning=" ".join(reasoning_parts),
        )

    def _handle_ask_patient_info(self, params: Dict[str, Any]) -> Reward:
        """
        Handle ask_patient_info action.
        Expected params: {"query": str}
        """
        query = params.get("query", "").lower()

        if not query:
            return Reward(
                value=-0.10,
                components={"empty_query": -0.10},
                reasoning="No query provided.",
            )

        # Check if this query is redundant
        if query in self._asked_info:
            return Reward(
                value=-0.30,
                components={"redundant_query": -0.30},
                reasoning=f"Already asked about: {query}. Redundant action.",
            )

        self._asked_info.add(query)
        reward_value = 0.0
        components = {}
        reasoning_parts = []
        patient = self._task.patient_profile

        # Determine if the query is useful
        useful_keywords = {
            "allergy": patient.allergies,
            "allergies": patient.allergies,
            "condition": patient.conditions,
            "conditions": patient.conditions,
            "medication": patient.current_medications,
            "medications": patient.current_medications,
            "current": patient.current_medications,
            "age": [str(patient.age)],
            "weight": [str(patient.weight_kg)],
            "pregnancy": [str(patient.pregnancy)],
            "renal": [patient.renal_function],
            "kidney": [patient.renal_function],
        }

        found_useful = False
        for kw, info in useful_keywords.items():
            if kw in query and info:
                found_useful = True
                note = f"Patient info ({kw}): {', '.join(str(i) for i in info)}"
                self._state.clinical_notes.append(note)

                if kw in ("allergy", "allergies"):
                    self._checked_allergies = True
                    # Check allergy conflicts with extracted medicines
                    drug_names = [m.name for m in self._state.extracted_medicines]
                    allergy_flags = check_allergy_conflicts(
                        drug_names, patient.allergies
                    )
                    for flag in allergy_flags:
                        if flag not in self._state.risk_flags:
                            self._state.risk_flags.append(flag)

                reasoning_parts.append(f"Retrieved useful patient info: {kw}.")
                break

        if found_useful:
            reward_value += 0.15
            components["useful_info"] = 0.15
        else:
            reward_value -= 0.05
            components["irrelevant_query"] = -0.05
            reasoning_parts.append("Query did not yield useful information.")

        return Reward(
            value=round(reward_value, 4),
            components=components,
            reasoning=" ".join(reasoning_parts),
        )

    def _handle_search_inventory(self, params: Dict[str, Any]) -> Reward:
        """
        Handle search_inventory action.
        Expected params: {"drug_name": str}
        """
        drug_name = params.get("drug_name", "")
        if not drug_name:
            return Reward(
                value=-0.10,
                components={"empty_search": -0.10},
                reasoning="No drug name provided for inventory search.",
            )

        expanded = expand_abbreviation(drug_name).lower()

        # Check redundancy
        if expanded in self._searched_drugs:
            return Reward(
                value=-0.15,
                components={"redundant_search": -0.15},
                reasoning=f"Already searched for {drug_name}. Redundant.",
            )

        self._searched_drugs.add(expanded)
        reward_value = 0.0
        components = {}
        reasoning_parts = []

        # Search inventory
        found = False
        for item in self._state.inventory:
            if (expanded in item.generic_name.lower()
                    or expanded in item.name.lower()):
                found = True
                status = "IN STOCK" if item.stock > 0 else "OUT OF STOCK"
                note = (
                    f"Inventory: {item.name} ({item.strength}) -- "
                    f"{status} (qty: {item.stock}, price: {item.price})"
                )
                self._state.clinical_notes.append(note)
                reasoning_parts.append(f"{item.name}: {status}.")

                if item.stock == 0:
                    reward_value += 0.10
                    components["found_shortage"] = 0.10
                    reasoning_parts.append("Identified stock shortage.")
                else:
                    reward_value += 0.10
                    components["confirmed_stock"] = 0.10
                break

        if not found:
            self._state.clinical_notes.append(
                f"Inventory: {drug_name} -- NOT FOUND in inventory."
            )
            reward_value += 0.05
            components["search_performed"] = 0.05
            reasoning_parts.append(f"{drug_name} not found in inventory.")

        return Reward(
            value=round(reward_value, 4),
            components=components,
            reasoning=" ".join(reasoning_parts),
        )

    def _handle_suggest_alternative(self, params: Dict[str, Any]) -> Reward:
        """
        Handle suggest_alternative action.
        Expected params: {"original": str, "alternative": str, "reason": str}
        """
        original = params.get("original", "")
        alternative = params.get("alternative", "")
        reason = params.get("reason", "")

        if not original or not alternative:
            return Reward(
                value=-0.10,
                components={"incomplete_suggestion": -0.10},
                reasoning="Must specify both original and alternative drug.",
            )

        reward_value = 0.0
        components = {}
        reasoning_parts = []

        original_lower = expand_abbreviation(original).lower()
        alt_lower = expand_abbreviation(alternative).lower()

        # Check if original actually needs an alternative
        needs_alt = False
        for gt in self._task.ground_truth_medicines:
            if (gt.generic_name.lower() in original_lower
                    or original_lower in gt.generic_name.lower()):
                if gt.requires_alternative:
                    needs_alt = True
                    break

        if needs_alt or self._task.difficulty == "custom":
            # Validate alternative is in inventory and in stock
            alt_available = False
            for item in self._state.inventory:
                if (alt_lower in item.generic_name.lower()
                        or alt_lower in item.name.lower()):
                    if item.stock > 0:
                        alt_available = True
                    break

            # Validate alternative is safe for patient
            patient = self._task.patient_profile
            alt_allergy_flags = check_allergy_conflicts(
                [alternative], patient.allergies
            )
            alt_safe = len(alt_allergy_flags) == 0

            if alt_available and alt_safe:
                reward_value += 0.20 # Higher reward for proactive clinical safety
                components["valid_alternative"] = 0.20
                reasoning_parts.append(
                    f"Valid alternative: {alternative} for {original}."
                )
                self._suggested_alternatives[original_lower] = alt_lower

                # Decrement stock of alternative
                for item in self._state.inventory:
                    if (alt_lower in item.generic_name.lower()
                            or alt_lower in item.name.lower()):
                        item.stock = max(0, item.stock - 1)
                        break
            elif not alt_safe:
                reward_value -= 0.50
                components["unsafe_alternative"] = -0.50
                reasoning_parts.append(
                    f"UNSAFE: {alternative} conflicts with patient allergies."
                )
                for flag in alt_allergy_flags:
                    self._state.risk_flags.append(flag)
            elif not alt_available:
                reward_value -= 0.10
                components["unavailable_alternative"] = -0.10
                reasoning_parts.append(
                    f"{alternative} is not available in inventory."
                )
        else:
            reward_value -= 0.20
            components["unnecessary_alternative"] = -0.20
            reasoning_parts.append(
                f"{original} does not require an alternative."
            )

        self._state.clinical_notes.append(
            f"Alternative suggestion: {original} -> {alternative}. "
            f"Reason: {reason}"
        )

        return Reward(
            value=round(reward_value, 4),
            components=components,
            reasoning=" ".join(reasoning_parts),
        )

    def _handle_risk_assessment(self, params: Dict[str, Any]) -> Reward:
        """
        Handle risk_assessment action.
        Expected params: {"assessments": [{"drug": str, "risk_type": str,
                          "severity": str, "description": str}]}
        """
        assessments = params.get("assessments", [])
        if not assessments:
            return Reward(
                value=-0.10,
                components={"empty_assessment": -0.10},
                reasoning="No risk assessments provided.",
            )

        reward_value = 0.0
        components = {}
        reasoning_parts = []

        expected_risks = self._task.expected_risks
        matched_expected = 0

        for assessment in assessments:
            drug = assessment.get("drug", "")
            risk_type = assessment.get("risk_type", "")
            severity = assessment.get("severity", "moderate")
            description = assessment.get("description", "")

            # Check against expected risks
            matched = False
            for exp_risk in expected_risks:
                drug_match = (
                    drug.lower() in exp_risk.affected_drug.lower()
                    or exp_risk.affected_drug.lower() in drug.lower()
                )
                type_match = (
                    risk_type.lower() in exp_risk.category.value
                    or exp_risk.category.value in risk_type.lower()
                )
                if drug_match and type_match:
                    matched = True
                    matched_expected += 1
                    break

            if matched:
                reward_value += 0.25
                components[f"correct_risk_{drug}"] = 0.25
                reasoning_parts.append(
                    f"Correct risk assessment for {drug}: {risk_type}."
                )
            elif self._task.difficulty == "custom":
                reward_value += 0.20
                components[f"custom_risk_{drug}"] = 0.20
                reasoning_parts.append(
                    f"Risk assessment recorded for {drug}: {risk_type}."
                )
            else:
                reward_value -= 0.10
                components[f"incorrect_risk_{drug}"] = -0.10
                reasoning_parts.append(
                    f"Risk assessment for {drug} ({risk_type}) does not match "
                    f"expected clinical risks."
                )

            # Add to risk flags
            try:
                cat = RiskCategory(risk_type.lower())
            except ValueError:
                cat = RiskCategory.CONTRAINDICATION

            try:
                sev = Severity(severity.lower())
            except ValueError:
                sev = Severity.MODERATE

            flag = RiskFlag(
                category=cat,
                severity=sev,
                description=description or f"Risk: {risk_type} for {drug}",
                affected_drug=drug,
            )
            self._state.risk_flags.append(flag)

            self._risk_assessments_made.append(assessment)
            self._checked_age_risks = True

        self._state.clinical_notes.append("Risk assessment completed.")

        return Reward(
            value=round(reward_value, 4),
            components=components,
            reasoning=" ".join(reasoning_parts),
        )

    def _handle_finalize(self, params: Dict[str, Any]) -> Reward:
        """
        Handle finalize action. This is the terminal action.
        Expected params: {
            "decision": str,
            "medications": [{"name": str, "dosage": str, ...}],
            "reasoning": str,
            "confidence": float
        }
        """
        self._done = True
        self._state.done = True

        decision = params.get("decision", "")
        medications = params.get("medications", [])
        reasoning = params.get("reasoning", "")
        confidence = params.get("confidence", 0.0)

        self._state.final_decision = {
            "decision": decision,
            "medications": medications,
            "reasoning": reasoning,
            "confidence": confidence,
        }

        reward_value = 0.0
        components = {}
        reasoning_parts = []

        # -- SAFETY EVALUATION --

        # 1. Check if critical interactions were detected before finalizing
        expected_interactions = self._task.expected_interactions
        if expected_interactions and not self._checked_interactions:
            reward_value -= 0.80
            components["unchecked_interactions"] = -0.80
            reasoning_parts.append(
                "UNSAFE: Finalized without checking drug interactions."
            )

        # 2. Check if allergy was considered
        if self._task.patient_profile.allergies and not self._checked_allergies:
            reward_value -= 0.80
            components["unchecked_allergies"] = -0.80
            reasoning_parts.append(
                "UNSAFE: Finalized without verifying patient allergies."
            )

        # 3. Check final medication list for safety
        final_med_names = [
            m.get("name", "").lower() for m in medications
        ]

        for gt in self._task.ground_truth_medicines:
            if not gt.is_safe:
                # This medicine should NOT be dispensed
                dispensed_unsafe = any(
                    gt.generic_name.lower() in fm or fm in gt.generic_name.lower()
                    for fm in final_med_names
                )
                if dispensed_unsafe:
                    reward_value -= 1.00
                    components[f"unsafe_dispensing_{gt.name}"] = -1.00
                    reasoning_parts.append(
                        f"CRITICAL: Dispensed unsafe drug {gt.name}. "
                        f"Reason: {gt.alternative_reason}"
                    )

        # 4. Check if safe medicines are included
        for gt in self._task.ground_truth_medicines:
            if gt.is_safe:
                included = any(
                    gt.generic_name.lower() in fm or fm in gt.generic_name.lower()
                    for fm in final_med_names
                )
                if included:
                    reward_value += 0.15
                    components[f"correct_dispensing_{gt.name}"] = 0.15
                    reasoning_parts.append(f"Correctly included {gt.name}.")

        # 5. Check if alternatives were provided for unsafe/unavailable drugs
        for gt in self._task.ground_truth_medicines:
            if gt.requires_alternative:
                alt_provided = gt.generic_name.lower() in self._suggested_alternatives
                alt_in_final = any(
                    alt_name in fm
                    for fm in final_med_names
                    for alt_name in self._suggested_alternatives.values()
                )
                if alt_provided or alt_in_final:
                    reward_value += 0.15
                    components[f"alternative_provided_{gt.name}"] = 0.15
                    reasoning_parts.append(
                        f"Provided alternative for {gt.name}."
                    )

        # 6. Efficiency bonus
        optimal = self._task.optimal_action_count
        actual = self._state.step_count
        if actual <= optimal:
            reward_value += 0.10
            components["efficiency_bonus"] = 0.10
            reasoning_parts.append("Efficient: completed in optimal steps.")
        elif actual <= optimal * 1.5:
            reward_value += 0.05
            components["efficiency_bonus"] = 0.05
        else:
            components["efficiency_penalty"] = 0.0
            reasoning_parts.append("Suboptimal step count.")

        # 7. Reasoning quality bonus
        if reasoning and len(reasoning) > 50:
            reward_value += 0.05
            components["reasoning_quality"] = 0.05

        # 8. Safe decision bonus (only if no safety violations)
        safety_violations = [
            k for k, v in components.items()
            if ("unsafe" in k or "unchecked" in k) and v < 0
        ]
        if not safety_violations:
            reward_value += 0.25
            components["safe_decision"] = 0.25
            reasoning_parts.append("Overall safe clinical decision.")

        self._state.clinical_notes.append(
            f"Final decision: {decision}. Confidence: {confidence}."
        )

        return Reward(
            value=round(reward_value, 4),
            components=components,
            reasoning=" ".join(reasoning_parts),
        )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _compute_state_changes(self, action: Action) -> List[str]:
        """Compute human-readable list of state changes from an action."""
        changes = []
        at = action.action_type.value
        changes.append(f"Action: {at}")
        changes.append(f"Step: {self._state.step_count}")

        if self._state.extracted_medicines:
            changes.append(
                f"Extracted medicines: {len(self._state.extracted_medicines)}"
            )
        if self._state.detected_interactions:
            changes.append(
                f"Detected interactions: {len(self._state.detected_interactions)}"
            )
        if self._state.risk_flags:
            changes.append(f"Risk flags: {len(self._state.risk_flags)}")
        if self._state.final_decision:
            changes.append("Final decision made.")

        return changes