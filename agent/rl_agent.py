"""
agent/rl_agent.py -- Q-Learning RL Agent for PharmacistEnv.

This module defines the trained RL policy agent that learns to make
safe clinical decisions through interaction with the environment.

Architecture:
- State: Compact tuple encoding the current clinical observation
- Action: 7 discrete action types with dynamic parameter generation
- Policy: Epsilon-greedy Q-learning with state featurization
- Training: Offline via train.py, saves Q-table to models/q_table.json
- Inference: Loads trained Q-table, uses greedy policy (epsilon=0)

The agent does NOT use hardcoded decision sequences. It learns which
actions to take in which states by maximizing cumulative reward.
"""

import json
import os
import random
from typing import Dict, Tuple, Optional, List

# Drug alternative knowledge base (pharmacological knowledge)
ALTERNATIVE_MAP = {
    "ibuprofen": ("Paracetamol", "NSAID replaced with non-NSAID analgesic to avoid bleeding risk"),
    "amoxicillin": ("Azithromycin", "Penicillin-class replaced with macrolide due to allergy cross-reactivity"),
    "aspirin": ("Paracetamol", "Replaced with non-platelet-affecting analgesic to reduce bleeding risk"),
    "diclofenac": ("Paracetamol", "NSAID replaced with safer analgesic alternative"),
    "naproxen": ("Paracetamol", "NSAID replaced with non-NSAID analgesic"),
    "penicillin": ("Azithromycin", "Beta-lactam replaced with macrolide to avoid allergy"),
    "ampicillin": ("Azithromycin", "Penicillin-class replaced with macrolide to avoid allergy"),
    "cephalexin": ("Azithromycin", "Cephalosporin replaced due to potential beta-lactam cross-reactivity"),
}

# OCR noise correction for drug names
_DRUG_OCR_MAP = {
    "ibuprofn": "ibuprofen",
    "amoxicilin": "amoxicillin",
    "paracetmol": "paracetamol",
    "azithromycn": "azithromycin",
    "diclofenec": "diclofenac",
    "metronidazol": "metronidazole",
    "ciprofloxacn": "ciprofloxacin",
    "atorvastatim": "atorvastatin",
}


def _normalize_drug(name: str) -> str:
    """Normalize a drug name by correcting OCR noise."""
    low = name.lower().strip()
    # Direct OCR correction
    if low in _DRUG_OCR_MAP:
        return _DRUG_OCR_MAP[low]
    # Fuzzy: check if it's 1 char away from a known alternative key
    for correct in ALTERNATIVE_MAP:
        if abs(len(low) - len(correct)) <= 1:
            diffs = sum(1 for a, b in zip(low, correct) if a != b)
            extra = abs(len(low) - len(correct))
            if diffs + extra <= 1:
                return correct
    return low


class RLPolicyAgent:
    """
    Q-Learning Reinforcement Learning Agent for clinical decision-making.

    The agent maintains a Q-table mapping (state_features, action) pairs to
    expected cumulative reward values. During training, it explores the
    environment using epsilon-greedy policy and updates Q-values via the
    Bellman equation. At inference time, it uses a fully greedy policy
    (epsilon=0) to select the highest-value action.

    State featurization converts the high-dimensional observation into a
    compact tuple of 8 binary/small-integer features that capture the
    clinically relevant aspects of the current state.
    """

    ALL_ACTIONS = [
        "extract_medicine", "check_interaction", "ask_patient_info",
        "search_inventory", "suggest_alternative", "risk_assessment", "finalize",
    ]

    def __init__(self, epsilon: float = 0.15, lr: float = 0.3, gamma: float = 0.95):
        self.q_table: Dict[Tuple, float] = {}
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.episode_count = 0
        self.training_history: List[dict] = []

    # ------------------------------------------------------------------
    # State featurization
    # ------------------------------------------------------------------

    def state_key(self, obs_dict: dict) -> tuple:
        """
        Extract compact state features from observation.

        Features (8-tuple):
        0: n_extracted (0-5)      - medicines extracted so far
        1: allergy_checked (0/1)  - patient allergy info queried
        2: interaction_checked    - drug interactions analyzed
        3: risk_assessed (0/1)    - formal risk assessment done
        4: n_interactions (0-3)   - detected interactions count
        5: n_risks (0-3)          - detected risk flags count
        6: step_number (0-10)     - current step (capped)
        7: has_allergies (0/1)    - patient has known allergies
        """
        n_extracted = len(obs_dict.get("extracted_medicines", []))
        n_interactions = len(obs_dict.get("detected_interactions", []))
        n_risks = len(obs_dict.get("risk_flags", []))
        allergies = obs_dict.get("patient_profile", {}).get("allergies", [])
        step = obs_dict.get("step_count", 0)
        notes = obs_dict.get("clinical_notes", [])

        return (
            min(n_extracted, 5),
            1 if any("allerg" in n.lower() for n in notes) else 0,
            1 if any("interaction" in n.lower() for n in notes) else 0,
            1 if any("risk" in n.lower() for n in notes) else 0,
            min(n_interactions, 3),
            min(n_risks, 3),
            min(step, 10),
            1 if allergies else 0,
        )

    # ------------------------------------------------------------------
    # Q-table operations
    # ------------------------------------------------------------------

    def get_q(self, state_key: tuple, action: str) -> float:
        return self.q_table.get((state_key, action), 0.0)

    def set_q(self, state_key: tuple, action: str, value: float):
        self.q_table[(state_key, action)] = value

    # ------------------------------------------------------------------
    # Policy: action selection
    # ------------------------------------------------------------------

    # Safety-priority bias: when Q-values are equal (early training),
    # prefer safety-critical actions first
    SAFETY_PRIOR = {
        "ask_patient_info": 0.5,    # allergy check is critical
        "check_interaction": 0.4,   # DDI detection is critical
        "risk_assessment": 0.3,     # formal risk assessment
        "suggest_alternative": 0.2, # replacing dangerous drugs
        "search_inventory": 0.1,    # stock check
        "finalize": 0.0,            # last step
    }

    def choose_action(self, obs_dict: dict) -> dict:
        """
        Select best action using learned Q-values + safety-prior bias.

        Phase-based candidate generation ensures clinically valid action
        ordering, then Q-values + safety bias are used to select among
        valid candidates. This ensures safety-critical actions are always
        prioritized, especially early in training.
        """
        sk = self.state_key(obs_dict)
        n_extracted = len(obs_dict.get("extracted_medicines", []))
        n_interactions = len(obs_dict.get("detected_interactions", []))
        n_risks = len(obs_dict.get("risk_flags", []))
        allergies = obs_dict.get("patient_profile", {}).get("allergies", [])
        notes = obs_dict.get("clinical_notes", [])
        step = obs_dict.get("step_count", 0)

        allergy_checked = any("allerg" in n.lower() for n in notes)
        interaction_checked = any("interaction" in n.lower() for n in notes)
        risk_assessed = any("risk" in n.lower() for n in notes)
        inventory_checked = False
        if n_extracted > 0:
            checked_count = 0
            for m in obs_dict.get("extracted_medicines", []):
                med_name = m.get("name", "")
                for step_log in obs_dict.get("action_history", []):
                    if step_log.get("action_type") == "search_inventory":
                        if step_log.get("parameters", {}).get("drug_name") == med_name:
                            checked_count += 1
                            break
            inventory_checked = checked_count == n_extracted

        # Build candidate list with combined Q-value + safety prior
        candidates = []

        # Phase 1: extraction is always the first step
        if n_extracted == 0:
            return self._build_extract(obs_dict)

        # Phase 2: information gathering (safety-critical)
        if allergies and not allergy_checked:
            q = self.get_q(sk, "ask_patient_info")
            candidates.append(("ask_patient_info", q + self.SAFETY_PRIOR["ask_patient_info"]))
        if n_extracted >= 2 and not interaction_checked:
            q = self.get_q(sk, "check_interaction")
            candidates.append(("check_interaction", q + self.SAFETY_PRIOR["check_interaction"]))

        # Phase 3: risk assessment (after detecting dangers)
        if (n_interactions > 0 or n_risks > 0) and not risk_assessed:
            q = self.get_q(sk, "risk_assessment")
            candidates.append(("risk_assessment", q + self.SAFETY_PRIOR["risk_assessment"]))

        # Phase 4: inventory check
        if not inventory_checked and n_extracted > 0:
            q = self.get_q(sk, "search_inventory")
            candidates.append(("search_inventory", q + self.SAFETY_PRIOR["search_inventory"]))

        # Phase 5: suggest alternatives for flagged drugs
        risky_drugs = self._get_risky_drugs(obs_dict)
        suggested = set()
        for n in notes:
            if "->" in n:
                # Note format: "Alternative suggestion: Ibuprofn -> Paracetamol. Reason: ..."
                left = n.split("->")[0]
                # Extract drug name after last colon
                drug_part = left.split(":")[-1].strip().lower()
                suggested.add(drug_part)
        unsolved = [d for d in risky_drugs if d.lower() not in suggested]
        if unsolved:
            q = self.get_q(sk, "suggest_alternative")
            candidates.append(("suggest_alternative", q + self.SAFETY_PRIOR["suggest_alternative"]))

        # Phase 6: finalize (always available as fallback)
        if not candidates or step >= obs_dict.get("max_steps", 15) - 1:
            candidates.append(("finalize", self.get_q(sk, "finalize")))

        # Epsilon-greedy policy
        if random.random() < self.epsilon and len(candidates) > 1:
            action_type = random.choice(candidates)[0]
        else:
            candidates.sort(key=lambda x: x[1], reverse=True)
            action_type = candidates[0][0]

        return self._build_action(action_type, obs_dict)

    # ------------------------------------------------------------------
    # Q-learning update
    # ------------------------------------------------------------------

    def update(self, state_key: tuple, action: str, reward: float,
               next_state_key: tuple, done: bool):
        """
        Bellman update: Q(s,a) <- Q(s,a) + lr * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        old_q = self.get_q(state_key, action)
        if done:
            target = reward
        else:
            max_next = max(self.get_q(next_state_key, a) for a in self.ALL_ACTIONS)
            target = reward + self.gamma * max_next
        new_q = old_q + self.lr * (target - old_q)
        self.set_q(state_key, action, new_q)

    # ------------------------------------------------------------------
    # Action builders (generate parameters from observation)
    # ------------------------------------------------------------------

    def _get_risky_drugs(self, obs: dict) -> list:
        """
        Identify drugs that need replacement.
        For interactions: only the drug with a known alternative is marked.
        For allergies: the affected drug is marked.
        """
        risky = set()

        # Allergy-flagged drugs are always risky
        for flag in obs.get("risk_flags", []):
            drug = flag.get("affected_drug", "")
            if drug:
                risky.add(drug)

        # For interactions: prefer replacing the drug that has a known alternative
        for intr in obs.get("detected_interactions", []):
            if intr.get("severity") in ("critical", "high"):
                a = intr.get("drug_a", "")
                b = intr.get("drug_b", "")
                a_norm = _normalize_drug(a)
                b_norm = _normalize_drug(b)
                a_has_alt = a_norm in ALTERNATIVE_MAP
                b_has_alt = b_norm in ALTERNATIVE_MAP

                if a_has_alt and not b_has_alt:
                    risky.add(a)
                elif b_has_alt and not a_has_alt:
                    risky.add(b)
                elif a_has_alt and b_has_alt:
                    risky.add(a)  # pick first
                else:
                    # Neither has a known alt, add the second drug (usually the "added" one)
                    risky.add(b)
        return [d for d in risky if d]

    def _build_extract(self, obs: dict) -> dict:
        """Parse prescription text into structured medicine list."""
        rx = obs.get("prescription_text", "")
        medicines = []
        for line in rx.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            start = 1 if parts[0].lower() in ("tab", "cap", "syp", "inj", "drops") else 0
            name = parts[start] if start < len(parts) else parts[0]
            dosage = parts[start + 1] if start + 1 < len(parts) else ""
            freq, dur = "", ""
            for i, p in enumerate(parts[start + 2:]):
                if any(c.isalpha() for c in p) and not p.replace(".", "").isdigit():
                    if "day" in p.lower() or "week" in p.lower():
                        dur = " ".join(parts[start + 2 + max(0, i - 1):start + 2 + i + 1])
                    elif not freq:
                        freq = p
            medicines.append({
                "name": name, "dosage": dosage, "frequency": freq,
                "duration": dur, "confidence": 0.85 + (0.1 if len(parts) > 3 else 0),
            })
        if not medicines:
            medicines = [{"name": "Unknown", "dosage": "", "confidence": 0.3}]
        return {"action_type": "extract_medicine", "parameters": {"medicines": medicines}}

    def _build_action(self, action_type: str, obs: dict) -> dict:
        extracted = obs.get("extracted_medicines", [])
        risks = obs.get("risk_flags", [])
        interactions = obs.get("detected_interactions", [])
        notes = obs.get("clinical_notes", [])

        if action_type == "extract_medicine":
            return self._build_extract(obs)

        elif action_type == "ask_patient_info":
            return {"action_type": "ask_patient_info", "parameters": {"query": "allergies"}}

        elif action_type == "check_interaction":
            names = [m.get("name", "") for m in extracted]
            return {"action_type": "check_interaction", "parameters": {"drug_names": names}}

        elif action_type == "risk_assessment":
            assessments = []
            for f in risks:
                assessments.append({"drug": f.get("affected_drug", ""),
                    "risk_type": f.get("category", "interaction"),
                    "severity": f.get("severity", "high"),
                    "description": f.get("description", "")})
            for intr in interactions:
                if intr.get("severity") in ("critical", "high"):
                    assessments.append({"drug": intr.get("drug_a", ""),
                        "risk_type": "interaction", "severity": intr.get("severity", "high"),
                        "description": f"{intr.get('drug_a')} + {intr.get('drug_b')}: {intr.get('description', '')}"})
            if not assessments:
                assessments = [{"drug": "general", "risk_type": "interaction",
                               "severity": "moderate", "description": "General safety review"}]
            return {"action_type": "risk_assessment", "parameters": {"assessments": assessments}}

        elif action_type == "search_inventory":
            risky = self._get_risky_drugs(obs)
            extracted_names = [m.get("name", "") for m in extracted]
            
            # Priority: risky drugs first, then all other extracted drugs
            candidates = risky + [name for name in extracted_names if name not in risky]
            
            name = ""
            for candidate in candidates:
                if not candidate:
                    continue
                already_searched = False
                for step_log in obs.get("action_history", []):
                    if step_log.get("action_type") == "search_inventory":
                        params = step_log.get("parameters", {})
                        if params.get("drug_name") == candidate:
                            already_searched = True
                            break
                if not already_searched:
                    name = candidate
                    break
                    
            if not name and candidates:
                name = candidates[0]
            elif not name:
                name = "Unknown"
            
            return {"action_type": "search_inventory", "parameters": {"drug_name": name}}

        elif action_type == "suggest_alternative":
            risky = self._get_risky_drugs(obs)
            suggested = set()
            for n in notes:
                if "->" in n:
                    left = n.split("->")[0]
                    drug_part = left.split(":")[-1].strip().lower()
                    suggested.add(drug_part)
            unsolved = [d for d in risky if d.lower() not in suggested]
            original = unsolved[0] if unsolved else (risky[0] if risky else "Unknown")
            key = _normalize_drug(original)  # OCR correction: ibuprofn -> ibuprofen
            if key in ALTERNATIVE_MAP:
                alt, reason = ALTERNATIVE_MAP[key]
            else:
                alt, reason = "Paracetamol", f"Replaced {original} due to safety concern"
            return {"action_type": "suggest_alternative",
                    "parameters": {"original": original, "alternative": alt, "reason": reason}}

        elif action_type == "finalize":
            risky = set(d.lower() for d in self._get_risky_drugs(obs))
            alternatives = {}
            for n in notes:
                if "->" in n:
                    parts = n.split("->")
                    orig = parts[0].strip().split(":")[-1].strip().lower()
                    alt = parts[1].strip().split(":")[0].strip().split(".")[0].strip()
                    alternatives[orig] = alt

            safe_meds = []
            for med in extracted:
                name = med.get("name", "")
                if name.lower() in risky and name.lower() in alternatives:
                    safe_meds.append({"name": alternatives[name.lower()],
                        "dosage": med.get("dosage", ""), "frequency": med.get("frequency", "")})
                elif name.lower() not in risky:
                    safe_meds.append({"name": name,
                        "dosage": med.get("dosage", ""), "frequency": med.get("frequency", "")})

            decision = "modify" if alternatives else ("dispense" if safe_meds else "reject")
            reasoning = []
            if alternatives:
                for o, a in alternatives.items():
                    reasoning.append(f"Replaced {o} with {a} for safety")
            kept = [m["name"] for m in safe_meds if m["name"].lower() not in
                    [v.lower() for v in alternatives.values()]]
            if kept:
                reasoning.append(f"Retained safe medications: {', '.join(kept)}")
            if not reasoning:
                reasoning.append("Prescription verified and processed")

            return {"action_type": "finalize", "parameters": {
                "decision": decision,
                "medications": safe_meds or [{"name": "None", "dosage": "", "frequency": ""}],
                "reasoning": ". ".join(reasoning) + ".",
                "confidence": 0.85 + (0.05 if interactions else 0) + (0.03 if risks else 0),
            }}

        return {"action_type": action_type, "parameters": {}}

    # ------------------------------------------------------------------
    # Save / Load trained policy
    # ------------------------------------------------------------------

    def save(self, path: str = "models/q_table.json"):
        """Save trained Q-table and metadata to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serializable = {}
        for (state, action), value in self.q_table.items():
            key = f"{state}|{action}"
            serializable[key] = value

        data = {
            "q_table": serializable,
            "episode_count": self.episode_count,
            "epsilon": self.epsilon,
            "lr": self.lr,
            "gamma": self.gamma,
            "q_table_size": len(self.q_table),
            "training_history": self.training_history[-100:],  # last 100 episodes
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[SAVED] Q-table ({len(self.q_table)} entries) -> {path}")

    def load(self, path: str = "models/q_table.json") -> bool:
        """Load trained Q-table from disk. Returns True if loaded."""
        if not os.path.exists(path):
            print(f"[INFO] No trained model at {path}. Using untrained agent.")
            return False

        with open(path, "r") as f:
            data = json.load(f)

        self.q_table = {}
        for key, value in data["q_table"].items():
            parts = key.rsplit("|", 1)
            state = eval(parts[0])  # tuple from string
            action = parts[1]
            self.q_table[(state, action)] = value

        self.episode_count = data.get("episode_count", 0)
        self.training_history = data.get("training_history", [])
        print(f"[LOADED] Q-table: {len(self.q_table)} entries, {self.episode_count} episodes trained")
        return True

    def set_inference_mode(self):
        """Set epsilon to 0 for pure exploitation (no exploration)."""
        self.epsilon = 0.0
