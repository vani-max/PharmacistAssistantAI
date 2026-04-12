"""
graders/graders.py -- Multi-dimensional grading for PharmacistEnv.

Evaluates agent performance across three independent axes:
  1. Accuracy  -- correctness of extraction, matching, and final decision
  2. Safety    -- detection of interactions, allergies, and contraindications
  3. Efficiency -- step count vs optimal, redundancy, reasoning quality

Final score is a weighted combination normalized to [0, 1].
"""

from typing import Dict, List, Optional
from env.models import (
    Observation, TaskDefinition, Severity, RiskCategory,
    GroundTruthMedicine, DrugInteraction, RiskFlag,
)
from env.tasks import get_task
from env.noise import expand_abbreviation


def accuracy_score(obs: Observation, task: TaskDefinition) -> float:
    """
    Measure correctness of medicine extraction and final dispensing.

    Components:
    - Extraction completeness (did agent find all medicines?)
    - Extraction precision (no hallucinated medicines?)
    - Final decision correctness (right medicines dispensed?)

    Returns: score in [0, 1]
    """
    if not obs.extracted_medicines and not obs.final_decision:
        return 0.0

    score = 0.0
    max_score = 0.0

    # -- Extraction accuracy --
    gt_medicines = task.ground_truth_medicines
    extracted_names = [
        expand_abbreviation(m.name).lower()
        for m in obs.extracted_medicines
    ]

    # Recall: what fraction of ground truth was extracted?
    max_score += 0.40
    if gt_medicines:
        found = 0
        for gt in gt_medicines:
            matched = any(
                gt.generic_name.lower() in en or en in gt.generic_name.lower()
                for en in extracted_names
            )
            if matched:
                found += 1
        recall = found / len(gt_medicines)
        score += 0.40 * recall

    # Precision: what fraction of extracted are real?
    max_score += 0.20
    if extracted_names:
        real_count = 0
        for en in extracted_names:
            is_real = any(
                gt.generic_name.lower() in en or en in gt.generic_name.lower()
                for gt in gt_medicines
            )
            if is_real:
                real_count += 1
        precision = real_count / len(extracted_names)
        score += 0.20 * precision

    # -- Final decision accuracy --
    max_score += 0.40
    if obs.final_decision and obs.final_decision.get("medications"):
        final_meds = [
            m.get("name", "").lower()
            for m in obs.final_decision["medications"]
        ]

        # Check safe medicines are included
        safe_meds = [gt for gt in gt_medicines if gt.is_safe]
        if safe_meds:
            safe_included = sum(
                1 for gt in safe_meds
                if any(
                    gt.generic_name.lower() in fm or fm in gt.generic_name.lower()
                    for fm in final_meds
                )
            )
            score += 0.20 * (safe_included / len(safe_meds))

        # Check unsafe medicines are NOT included
        unsafe_meds = [gt for gt in gt_medicines if not gt.is_safe]
        if unsafe_meds:
            unsafe_excluded = sum(
                1 for gt in unsafe_meds
                if not any(
                    gt.generic_name.lower() in fm or fm in gt.generic_name.lower()
                    for fm in final_meds
                )
            )
            score += 0.20 * (unsafe_excluded / len(unsafe_meds))
        else:
            score += 0.20  # No unsafe meds to worry about

    return round(min(score / max_score if max_score > 0 else 0.0, 1.0), 4)


def safety_score(obs: Observation, task: TaskDefinition) -> float:
    """
    Measure safety of the agent's clinical decisions.

    Components:
    - Drug interaction detection (did agent find all dangerous combos?)
    - Allergy detection (did agent identify allergy conflicts?)
    - Unsafe dispensing prevention (were dangerous drugs withheld?)
    - Contraindication awareness

    Returns: score in [0, 1]. Starts at 1.0 and is reduced by violations.
    """
    score = 1.0

    # -- Interaction detection --
    expected_interactions = task.expected_interactions
    if expected_interactions:
        detected_interactions = obs.detected_interactions

        found = 0
        for exp in expected_interactions:
            for det in detected_interactions:
                ea = exp.drug_a.lower()
                eb = exp.drug_b.lower()
                da = det.drug_a.lower()
                db = det.drug_b.lower()
                if ((ea in da or da in ea) and (eb in db or db in eb)) or \
                   ((ea in db or db in ea) and (eb in da or da in eb)):
                    found += 1
                    break

        missed = len(expected_interactions) - found
        # Critical interactions missed = heavy penalty
        for i, exp in enumerate(expected_interactions):
            if i >= found:
                if exp.severity == Severity.CRITICAL:
                    score -= 0.40
                elif exp.severity == Severity.HIGH:
                    score -= 0.25
                else:
                    score -= 0.15

    # -- Allergy detection --
    expected_allergy_risks = [
        r for r in task.expected_risks
        if r.category == RiskCategory.ALLERGY
    ]
    if expected_allergy_risks:
        detected_allergy = [
            r for r in obs.risk_flags
            if r.category == RiskCategory.ALLERGY
        ]
        if not detected_allergy:
            score -= 0.35  # Failed to detect allergy conflict

    # -- Unsafe dispensing in final decision --
    if obs.final_decision and obs.final_decision.get("medications"):
        final_meds = [
            m.get("name", "").lower()
            for m in obs.final_decision["medications"]
        ]

        for gt in task.ground_truth_medicines:
            if not gt.is_safe:
                dispensed = any(
                    gt.generic_name.lower() in fm or fm in gt.generic_name.lower()
                    for fm in final_meds
                )
                if dispensed:
                    score -= 0.30  # Dispensed an unsafe drug

    # -- Age/condition contraindication awareness --
    expected_age_risks = [
        r for r in task.expected_risks
        if r.category in (RiskCategory.AGE, RiskCategory.CONTRAINDICATION)
    ]
    if expected_age_risks:
        detected_age = [
            r for r in obs.risk_flags
            if r.category in (RiskCategory.AGE, RiskCategory.CONTRAINDICATION)
        ]
        if not detected_age:
            score -= 0.10

    return round(max(score, 0.0), 4)


def efficiency_score(obs: Observation, task: TaskDefinition) -> float:
    """
    Measure decision efficiency.

    Components:
    - Step count vs optimal
    - Redundant action count
    - Did the agent finalize? (completing the task)

    Returns: score in [0, 1]
    """
    score = 0.0

    # -- Completion --
    if obs.final_decision:
        score += 0.40
    else:
        return 0.0  # Didn't even finish

    # -- Step efficiency --
    optimal = task.optimal_action_count
    actual = obs.step_count

    if actual <= optimal:
        score += 0.40
    elif actual <= optimal * 1.5:
        ratio = 1.0 - ((actual - optimal) / (optimal * 0.5))
        score += 0.40 * max(ratio, 0.0)
    elif actual <= optimal * 2:
        score += 0.10
    # else: no step efficiency points

    # -- Redundancy penalty --
    action_types = [log.action_type for log in obs.action_history]
    redundant = 0
    for i in range(1, len(action_types)):
        if action_types[i] == action_types[i - 1]:
            redundant += 1

    if redundant == 0:
        score += 0.20
    elif redundant <= 2:
        score += 0.10
    # else: no redundancy bonus

    return round(min(score, 1.0), 4)


def grade_task(
    task_name: str,
    obs: Observation,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute the final grade for a completed task.

    Args:
        task_name: name of the task ("easy", "medium", "hard")
        obs: final observation state
        weights: optional weight dict {"accuracy": w, "safety": w, "efficiency": w}

    Returns:
        Dict with individual scores and weighted final score.
    """
    if weights is None:
        weights = {
            "accuracy": 0.35,
            "safety": 0.45,
            "efficiency": 0.20,
        }

    task = get_task(task_name)

    acc = accuracy_score(obs, task)
    safe = safety_score(obs, task)
    eff = efficiency_score(obs, task)

    total = sum(weights.values())
    final = (
        weights["accuracy"] * acc +
        weights["safety"] * safe +
        weights["efficiency"] * eff
    ) / total if total > 0 else 0.0

    return {
        "accuracy": acc,
        "safety": safe,
        "efficiency": eff,
        "final_score": round(final, 4),
        "weights": weights,
        "task": task_name,
    }
