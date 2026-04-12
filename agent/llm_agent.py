"""
agent/llm_agent.py

Encapsulates OpenAI-compatible LLM inference logic (Hugging Face via router).
Can be used by the FastAPI backend to run intelligent interactive sessions.
"""
import os
import json
import textwrap
from typing import Optional
from openai import OpenAI

from env.models import Action, ActionType

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


# Secondary and free fallback models (Hugging Face)
MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

class LLMAgent:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the LLM agent with multi-provider support.
        Prioritizes OpenAI (if key available) and Fallbacks to Hugging Face.
        """
        self.providers = []
        
        # 1. Try to set up OpenAI Provider
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.providers.append({
                "name": "openai",
                "client": OpenAI(api_key=openai_key, base_url="https://api.openai.com/v1"),
                "models": ["gpt-4o", "gpt-4o-mini"]
            })
            
        # 2. Try to set up Hugging Face Provider
        hf_token = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        if hf_token:
            self.providers.append({
                "name": "huggingface",
                "client": OpenAI(api_key=hf_token, base_url="https://router.huggingface.co/v1"),
                "models": MODELS
            })
            
        if not self.providers:
            raise ValueError("No API keys found for LLMAgent (checked OPENAI_API_KEY and HF_TOKEN)")
            
        self.is_ready = True

    def _build_user_prompt(self, obs_dict: dict, step: int) -> str:
        trimmed = {
            "prescription_text": obs_dict.get("prescription_text", ""),
            "extracted_medicines": obs_dict.get("extracted_medicines", []),
            "patient_profile": obs_dict.get("patient_profile", {}),
            "inventory": [
                {"name": i["name"], "stock": i["stock"], "strength": i["strength"]}
                for i in obs_dict.get("inventory", [])
            ],
            "detected_interactions": obs_dict.get("detected_interactions", []),
            "risk_flags": obs_dict.get("risk_flags", []),
            "clinical_notes": obs_dict.get("clinical_notes", [])[-5:],
            "step_count": obs_dict.get("step_count", step),
            "max_steps": obs_dict.get("max_steps", 15),
        }
        return f"Step {step}. Current environment state:\n{json.dumps(trimmed, indent=2)}\n\nDecide your next action. Respond with ONLY a JSON object."

    def _call_llm(self, messages: list, temperature: float = 0, max_tokens: int = 1000) -> str:
        """Helper to call LLM with multi-provider failover (OpenAI -> HF)."""
        last_error = None
        
        for provider in self.providers:
            client = provider["client"]
            name = provider["name"]
            
            for model in provider["models"]:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    err_msg = str(e).lower()
                    # Check for rate limit / quota / capacity / support errors
                    is_retryable = any(x in err_msg for x in [
                        "429", "400", "limit", "quota", "capacity", 
                        "overloaded", "credit", "not supported", "unavailable", "insufficient_quota"
                    ])
                    
                    if is_retryable:
                        print(f"[{name.upper()}] Model {model} failed, trying fallback... Error: {e}")
                        last_error = e
                        continue
                    else:
                        # For major auth errors, jump to next provider immediately
                        print(f"[{name.upper()}] Fatal provider error, trying next provider... Error: {e}")
                        last_error = e
                        break # Break model loop, continue provider loop
        
        raise ValueError(f"All providers (OpenAI & HF) failed or hit quotas. Last error: {last_error}")

    def choose_action(self, obs_dict: dict, step: int = 1) -> dict:
        """
        Returns a dictionary representing the action to take.
        Mirrors the RL agent's `choose_action()` output but generates via LLM.
        """
        user_prompt = self._build_user_prompt(obs_dict, step)
        content = self._call_llm(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        
        # Cleanup potential markdown codeblock formatting 
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.lower().startswith("json"):
                content = content[4:]
            content = content.strip()
            
        try:
            data = json.loads(content)
            return {
                "action_type": data["action_type"],
                "parameters": data.get("parameters", {})
            }
        except Exception as e:
            raise ValueError(f"Failed to parse LLM action: {e} | Raw JSON: {content}")

    def generate_report(self, steps: list, patient_profile: dict) -> dict:
        """
        Generates a structured clinical report summarizing the AI's logic and decisions.
        """
        report_prompt = textwrap.dedent(
            f"""
            You are a senior clinical pharmacist.
            Review the following patient profile and the actions taken by an AI agent.
            Respond with ONLY a JSON object containing the clinical summary.

            PATIENT PROFILE:
            {json.dumps(patient_profile, indent=2)}
            
            ACTIONS TAKEN:
            {json.dumps([{"step": s["step"], "action": s["action_type"], "params": s["parameters"]} for s in steps], indent=2)}
            
            JSON FORMAT (MANDATORY):
            {{
                "verdict": "Safe | Warning | Critical",
                "summary": "One sentence high-level observation.",
                "patient_analysis": {{
                    "risk_level": "Low | Moderate | High",
                    "details": "Clinical vitals and allergy context."
                }},
                "reasoning_steps": [
                    {{ "step": "Workflow Step Name", "observation": "What the AI did and why" }}
                ],
                "final_recommendation": "Final professional clinical advice."
            }}
            """
        ).strip()

        try:
            content = self._call_llm(
                messages=[
                    {"role": "system", "content": "You are a senior clinical pharmacist. Respond with JSON ONLY."},
                    {"role": "user", "content": report_prompt},
                ],
                temperature=0.3
            )
            
            # Cleanup potential markdown codeblock formatting 
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.lower().startswith("json"):
                    content = content[4:]
                content = content.strip()
                
            return json.loads(content)
        except Exception as e:
            print(f"[LLM Agent] Report generation failed: {e}")
            return {
                "verdict": "Unknown",
                "summary": "Clinical synthesis failed to generate.",
                "patient_analysis": {"risk_level": "Unknown", "details": "N/A"},
                "reasoning_steps": [],
                "final_recommendation": "Error in clinical report generation. Please review raw simulation logs."
            }
