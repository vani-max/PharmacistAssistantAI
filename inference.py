import os
import sys
import json
import textwrap
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from env.environment import PharmacistEnv
from env.models import Action, ActionType

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("PHARMACIST_ENV_TASK", "easy")
# In openenv, older runners might pass the task via just TASK_NAME or PHARMACIST_ENV_TASK.
BENCHMARK = os.getenv("PHARMACIST_ENV_BENCHMARK", "pharmacist_env")

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a clinical pharmacist AI agent operating in a simulated pharmacy environment.
    You must make safe, accurate decisions about prescription processing.

    ENVIRONMENT STATE is provided to you as JSON. You must respond with EXACTLY ONE action per turn as a JSON object.

    AVAILABLE ACTIONS:
    1. extract_medicine
       Parameters: {"medicines": [{"name": str, "dosage": str, "frequency": str, "duration": str, "confidence": float}]}
    2. check_interaction
       Parameters: {"drug_names": ["drug1", "drug2", ...]}
    3. ask_patient_info
       Parameters: {"query": "allergies" | "conditions" | "medications" | "age" | "renal" | "pregnancy"}
    4. search_inventory
       Parameters: {"drug_name": "medicine_name"}
    5. suggest_alternative
       Parameters: {"original": "original_drug", "alternative": "replacement_drug", "reason": "explanation"}
    6. risk_assessment
       Parameters: {"assessments": [{"drug": str, "risk_type": "allergy"|"interaction"|"age"|"contraindication", "severity": "critical"|"high"|"moderate"|"low", "description": str}]}
    7. finalize
       THIS ENDS THE EPISODE. Parameters: {"decision": "dispense"|"modify"|"reject", "medications": [{"name": str, "dosage": str, "frequency": str}], "reasoning": str, "confidence": float}

    RULES:
    - You MUST check interactions before finalizing if multiple drugs are prescribed.
    - You MUST check patient allergies if the patient has known allergies.
    - You MUST NOT dispense drugs that interact dangerously or trigger allergies.
    - Suggest safe alternatives for any drug that cannot be dispensed.
    - Be efficient: minimize redundant actions.
    - Your response must be ONLY a valid JSON object with "action_type" and "parameters" keys. No other text.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error.replace("\n", " ") if error else "null"
    action_val = action.replace("\n", " ")
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_val} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def build_user_prompt(obs_dict: dict, step: int) -> str:
    trimmed = {
        "prescription_text": obs_dict["prescription_text"],
        "extracted_medicines": obs_dict["extracted_medicines"],
        "patient_profile": obs_dict["patient_profile"],
        "inventory": [
            {"name": i["name"], "stock": i["stock"], "strength": i["strength"]}
            for i in obs_dict["inventory"]
        ],
        "detected_interactions": obs_dict["detected_interactions"],
        "risk_flags": obs_dict["risk_flags"],
        "clinical_notes": obs_dict["clinical_notes"][-5:],
        "step_count": obs_dict["step_count"],
        "max_steps": obs_dict["max_steps"],
    }
    return f"Step {step}. Current environment state:\n{json.dumps(trimmed, indent=2)}\n\nDecide your next action. Respond with ONLY a JSON object."


def get_model_action(client: OpenAI, obs_dict: dict, step: int) -> Action:
    user_prompt = build_user_prompt(obs_dict, step)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1000,
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    try:
        data = json.loads(content)
        return Action(action_type=ActionType(data["action_type"]), parameters=data.get("parameters", {}))
    except Exception as e:
        raise ValueError(f"Failed to parse action: {e} | Raw JSON: {content}")


def main():
    if not API_KEY:
        print("[DEBUG] No API credentials provided. Set HF_TOKEN or OPENAI_API_KEY.", file=sys.stderr)
        
    client = OpenAI(api_key=API_KEY or "dummy", base_url=API_BASE_URL)
    env = PharmacistEnv(TASK_NAME)
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    success = False
    
    try:
        obs = env.reset()
        for step in range(1, obs.max_steps + 1):
            if obs.done:
                break
                
            obs_dict = obs.model_dump()
            action_str = ""
            reward_val = 0.0
            error_msg = None
            done_val = False
            
            try:
                action = get_model_action(client, obs_dict, step)
                
                # Make the action string single line and easy to read
                action_str = f"{action.action_type.value}({json.dumps(action.parameters, separators=(',', ':'))})"
                
                obs, reward_val, done_val, info = env.step(action)
            except Exception as e:
                error_msg = str(e)
                done_val = True
                
            rewards.append(reward_val)
            steps_taken = step
            
            log_step(step=step, action=action_str or "error", reward=reward_val, done=done_val, error=error_msg)
            
            if done_val:
                break
                
        final_obs = env.state()
        from graders.graders import grade_task
        grades = grade_task(TASK_NAME, final_obs)
        score = float(grades['final_score'])
        score = max(0.0, min(1.0, score))
        success = (score >= 0.5)
        
    except Exception as e:
        print(f"[DEBUG] Fatal error: {e}", file=sys.stderr)
        score = 0.0
        success = False
        
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()