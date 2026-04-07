from .models import Observation, Action, Reward
from .tasks import get_task
import random

SYNONYMS = {
    "pcm": "paracetamol",
    "crocin": "paracetamol"
}

def normalize(text):
    text = text.lower()
    for k, v in SYNONYMS.items():
        text = text.replace(k, v)
    return text


def match_medicine(extracted, available):
    extracted = extracted.lower()

    # Handle common abbreviations
    mapping = {
        "pcm": "paracetamol",
        "para": "paracetamol"
    }

    if extracted in mapping:
        extracted = mapping[extracted]

    for med in available:
        if extracted in med.name.lower():
            return med.name

    return None


def suggest_alternative(extracted, available):
    extracted = normalize(extracted)

    for med in available:
        if normalize(med.name).split()[0] in extracted:
            return med.name
    return None

class PharmacyEnv:
    def __init__(self, task_name="easy"):
        self.task_name = task_name
        self.state = None
        self.steps = 0
        self.max_steps = 5
        self.previous_actions = []

    def reset(self):
        self.state = get_task(self.task_name)
        self.steps = 0
        self.previous_actions = []   # reset actions each episode
        return self.state

    def step(self, action: Action):
        reward = 0.0
        done = False

        # 🔹 Penalize repeated actions
        if action.action_type in self.previous_actions:
            reward -= 0.2
        self.previous_actions.append(action.action_type)

        # 🔹 Extraction
        # Extraction
        if action.action_type == "extract":
            if action.value:
                self.state.extracted = action.value
                reward += 0.3
            else:
                reward -= 0.2

        # 🔹 Matching
        elif action.action_type == "match":
            if self.state.extracted:
                match = match_medicine(
                    self.state.extracted[0],
                    self.state.available_medicines
                )

                if match:
                    self.state.matched = [match]
                    reward += 0.4
                else:
                    reward -= 0.3

        # 🔹 Suggest Alternative (for hard cases)
        elif action.action_type == "suggest_alternative":
            if self.state.extracted:
                alt = suggest_alternative(
                    self.state.extracted[0],
                    self.state.available_medicines
                )

                if alt:
                    self.state.matched = [alt]
                    reward += 0.4
                else:
                    reward -= 0.3

        # 🔹 Finalize
        elif action.action_type == "finalize":
            done = True
            if self.state.matched:
                reward += 0.3
            else:
                reward -= 0.5

        # 🔹 Step limit
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        # 🔹 Difficulty scaling
        if self.task_name == "hard":
            reward *= 1.2

        # 🔹 Bonus for correct workflow
        if self.state.extracted and self.state.matched:
            reward += 0.2

        return self.state, reward, done, {}
    
    def state_fn(self):
        return self.state