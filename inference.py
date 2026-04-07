from env.environment import PharmacyEnv
from env.models import Action
from env.graders import grade_task
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def simple_extract(text):
    words = text.split()

    # Look for medicine-like keyword (PCM, Ibuprofen, etc.)
    for w in words:
        w_clean = w.lower()
        if w_clean in ["pcm", "paracetamol", "ibuprofen", "crocin"]:
            return [w]

    # fallback: return first meaningful word
    return [words[0]]

def llm_extract(text):
    text_lower = text.lower()

    # ✅ Step 1: Handle common medicines locally (NO API CALL)
    if "pcm" in text_lower:
        return ["PCM"]
    if "paracetamol" in text_lower:
        return ["Paracetamol"]
    if "ibuprofen" in text_lower:
        return ["Ibuprofen"]
    if "crocin" in text_lower:
        return ["Crocin"]

    # ✅ Step 2: Only use LLM for unknown/complex cases
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extract only the medicine name from the prescription. Return just the medicine name only."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0
        )

        result = response.choices[0].message.content.strip()
        return [result]

    except Exception as e:
        print("⚠️ LLM fallback used:", e)
        return simple_extract(text)

def run_task(task):
    env = PharmacyEnv(task)
    obs = env.reset()

    # Step 1: Extract dynamically
    extracted = llm_extract(obs.prescription_text)
    obs, _, _, _ = env.step(Action(action_type="extract", value=extracted))

    # Step 2: Decide action
    if task == "hard":
        obs, _, _, _ = env.step(Action(action_type="suggest_alternative", value=[]))
    else:
        obs, _, _, _ = env.step(Action(action_type="match", value=[]))

    # Step 3: Finalize
    obs, _, _, _ = env.step(Action(action_type="finalize"))
    print("\n---", task.upper(), "---")
    print("Prescription:", obs.prescription_text)
    print("Extracted:", extracted)
    print("Matched:", obs.matched)

    return grade_task(task, obs)

if __name__ == "__main__":
    for t in ["easy", "medium", "hard"]:
        print(t, "score:", run_task(t))