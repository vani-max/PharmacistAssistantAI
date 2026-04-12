---
title: PharmacistAssistantAI
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
license: mit
app_port: 8000
---

# PharmacistAssistantAI: Clinical Decision Simulation Platform

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-brightgreen)](https://openenv.ai)

PharmacistAssistantAI is a **multi-step reinforcement learning environment** that simulates and evaluates clinical prescription validation workflows. It uses a hybrid architecture of Reinforcement Learning (RL) and Large Language Models (LLM) to perform safety-critical decision tasks in a controlled pharmacy setting.

> **Hackathon Submission** — OpenEnv Challenge

---

## Mandatory Environment Variables

> These must be set before running the inference script or deploying the space:

| Variable | Purpose |
|---|---|
| `API_BASE_URL` | The API endpoint for the LLM (e.g. `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | The model identifier (e.g. `Qwen/Qwen2.5-72B-Instruct`) |
| `HF_TOKEN` | Your Hugging Face API key (used to authenticate LLM calls) |

---

## OpenEnv Specification

- **Environment**: `env.environment:PharmacistEnv`
- **Tasks**: `easy`, `medium`, `hard`
- **Endpoints**: `/reset`, `/step`, `/state`
- **Action Space**: `extract_medicine`, `check_interaction`, `ask_patient_info`, `search_inventory`, `suggest_alternative`, `risk_assessment`, `finalize`
- **Scores**: Graded `[0, 1]` with weighted accuracy (35%), safety (45%), efficiency (20%)

---

## Technical Architecture

The system is a multi-layered application: a stateful MDP environment, a REST API backend, and a React monitoring dashboard.

### Core Components

| Layer | Component | Description |
|---|---|---|
| **Environment** | `env/environment.py` | `PharmacistEnv` — Gym-compatible MDP modeling pharmacy state, patient records, and clinical logs |
| **RL Policy** | `agent/rl_agent.py` | Q-Learning agent trained on 1,500 clinical episodes (weights in `rl_weights/q_table.json`) |
| **LLM Agent** | `agent/llm_agent.py` | OpenAI-compatible multi-provider agent with OpenAI → HuggingFace failover |
| **Backend** | `server/app.py` | FastAPI server implementing the OpenEnv interface endpoints |
| **Inference** | `inference.py` | Standalone inference script with `[START]`, `[STEP]`, `[END]` structured logging |

### Grading Dimensions

| Dimension | Weight | Measures |
|---|---|---|
| **Accuracy** | 35% | Extraction precision/recall, final dispensing correctness |
| **Safety** | 45% | Interaction detection, allergy prevention, unsafe drug withholding |
| **Efficiency** | 20% | Step count vs optimal, redundancy, task completion |

### Multi-Provider LLM Failover

The `LLMAgent` implements a sequential failover circuit:
1. **Primary**: OpenAI API (GPT-4o / GPT-4o-mini)
2. **Fallback**: Hugging Face Inference API (Qwen, Llama, Mistral)
3. Monitors HTTP `402` and `429` status codes to trigger automatic provider switching

### Clinical Safety Engine (`env/interactions.py`)

- **Drug-Drug Interactions (DDI)**: Evaluates interactions across 12 pharmacological classes
- **Allergy Detection**: Cross-references drugs against patient allergy profiles (penicillin cross-reactivities, etc.)
- **Contraindication Checks**: Flags risks based on patient age, renal function, pregnancy, and conditions

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | `POST` | Initialize a new clinical scenario (`easy`, `medium`, `hard`) |
| `/step` | `POST` | Execute a single action and return updated observation + reward |
| `/state` | `GET` | Get the current environment observation |
| `/auto-run` | `POST` | Run a full episode with the trained RL agent |
| `/custom-run` | `POST` | Process arbitrary user prescriptions dynamically |
| `/tasks` | `GET` | List all available tasks |
| `/health` | `GET` | Health check + model status |

---

## Quick Start

### Local (Python)

```bash
git clone https://github.com/vani-max/PharmacistAssistantAI.git
cd PharmacistAssistantAI

pip install -r requirements.txt

export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Start the backend server
python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run the inference script
python3 inference.py
```

### Docker

```bash
docker build -t pharmacist-env .
docker run -p 8000:8000 \
  -e HF_TOKEN="your_hf_token" \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  pharmacist-env
```

### Run Inference

```bash
# Default task: easy
python3 inference.py

# Specific task
PHARMACIST_ENV_TASK=hard python3 inference.py
```

Expected output format:
```
[START] task=easy env=pharmacist_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=extract_medicine({...}) reward=0.25 done=false error=null
[STEP] step=2 action=check_interaction({...}) reward=0.25 done=false error=null
[STEP] step=3 action=finalize({...}) reward=0.55 done=true error=null
[END] success=true steps=3 score=0.87 rewards=0.25,0.25,0.55
```

---

## Project Structure

```
PharmacistAssistantAI/
├── inference.py          # Inference script (required by OpenEnv)
├── openenv.yaml          # OpenEnv specification
├── Dockerfile            # Docker build for HF Spaces deployment
├── requirements.txt      # Python dependencies
├── env/                  # PharmacistEnv MDP implementation
│   ├── environment.py    # Core step/reset/state logic
│   ├── models.py         # Pydantic typed models
│   ├── tasks.py          # easy/medium/hard task definitions
│   ├── interactions.py   # Drug interaction & safety engine
│   └── noise.py          # OCR noise & abbreviation expansion
├── graders/
│   └── graders.py        # Accuracy, safety, efficiency graders → [0, 1]
├── agent/
│   ├── rl_agent.py       # Q-Learning policy agent
│   └── llm_agent.py      # Multi-provider LLM agent
├── server/
│   └── app.py            # FastAPI REST API (OpenEnv interface)
├── rl_weights/
│   └── q_table.json      # Pre-trained Q-table (1500 episodes)
└── frontend/             # React dashboard
```

---

## Clinical Disclaimer

This platform is a simulation tool for educational and research purposes. All pharmacological logic is codified as rules for environment interactions and does not constitute medical advice or a verified clinical diagnostic tool.
