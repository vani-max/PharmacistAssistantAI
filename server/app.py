"""
api/server.py -- FastAPI backend for PharmacistEnv.

On startup:
    1. Loads the trained RL Q-table from rl_weights/q_table.json
    2. Sets inference mode (epsilon=0, pure exploitation)

Endpoints:
    POST /reset     -- Initialize/reset the environment
    POST /step      -- Execute a single manual action
    GET  /state     -- Get current observation
    GET  /tasks     -- List available tasks
    GET  /health    -- Health check
    POST /auto-run  -- Run the TRAINED agent on a task
"""

import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.environment import PharmacistEnv
from env.models import Action, ActionType, Observation
from env.tasks import list_tasks as _list_tasks, get_task
from graders.graders import grade_task
from agent.rl_agent import RLPolicyAgent
from agent.llm_agent import LLMAgent


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "easy"

class StepRequest(BaseModel):
    action_type: str
    parameters: dict = {}

class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict

class AutoRunRequest(BaseModel):
    task_name: str = "easy"

class CustomRunRequest(BaseModel):
    prescription_text: str
    age: int
    gender: str
    weight_kg: float = 0.0
    renal_function: str = "normal"
    allergies: list[str] = []
    conditions: list[str] = []

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "2.0.0"
    environment: str = "PharmacistEnv"
    model_loaded: bool = False
    episodes_trained: int = 0


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PharmacistEnv API",
    description="Clinical Decision Intelligence Environment -- REST API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
env: Optional[PharmacistEnv] = None
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rl_weights", "q_table.json")

# Load trained RL agent on startup
trained_agent = RLPolicyAgent()
model_loaded = trained_agent.load(MODEL_PATH)
if model_loaded:
    trained_agent.set_inference_mode()  # epsilon=0 for pure exploitation
    print(f"[SERVER] Trained agent loaded: {trained_agent.episode_count} episodes, "
          f"{len(trained_agent.q_table)} Q-values, epsilon={trained_agent.epsilon}")
else:
    print("[SERVER] No trained model found. Run 'python train.py' first.")
    print(f"[SERVER] Agent will use exploration policy (epsilon={trained_agent.epsilon})")

# Load LLM agent
try:
    llm_agent = LLMAgent()
    llm_agent_loaded = True
    print("[SERVER] LLMAgent initialized with API token successfully.")
except Exception as e:
    llm_agent = None
    llm_agent_loaded = False
    print(f"[SERVER] LLMAgent unavailable: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "environment": "PharmacistEnv",
        "model_loaded": model_loaded,
        "episodes_trained": trained_agent.episode_count,
        "q_table_size": len(trained_agent.q_table),
    }

@app.get("/training-history")
def get_training_history():
    import json
    if not os.path.exists(MODEL_PATH):
        return {"history": []}
    try:
        with open(MODEL_PATH, "r") as f:
            data = json.load(f)
            return {"history": data.get("history", [])}
    except Exception as e:
        return {"error": str(e), "history": []}


@app.get("/tasks")
def list_tasks_endpoint():
    tasks = []
    for name in _list_tasks():
        task = get_task(name)
        tasks.append({
            "name": task.name,
            "difficulty": task.difficulty,
            "description": task.description,
            "optimal_steps": task.optimal_action_count,
        })
    return {"tasks": tasks}


@app.post("/reset")
def reset(req: ResetRequest):
    global env
    try:
        env = PharmacistEnv(req.task_name)
        obs = env.reset()
        return {"observation": obs.model_dump(), "message": f"Environment reset with task: {req.task_name}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")

    try:
        action_type = ActionType(req.action_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid action_type: {req.action_type}")

    action = Action(action_type=action_type, parameters=req.parameters)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(observation=obs.model_dump(), reward=reward, done=done, info=info)


@app.get("/state")
def get_state():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return {"observation": env.state().model_dump()}


@app.post("/auto-run")
def auto_run(req: AutoRunRequest):
    """
    Run the TRAINED RL agent on a task.

    The agent uses its learned Q-table (loaded from rl_weights/q_table.json)
    to make decisions. No hardcoded action sequences. Every decision
    is made by looking up learned Q-values for the current state.
    """
    run_env = PharmacistEnv(req.task_name)
    obs = run_env.reset()
    obs_dict = obs.model_dump()

    steps = []
    total_reward = 0.0

    for step_num in range(1, obs.max_steps + 1):
        if obs_dict.get("done", False):
            break

        # Trained agent decides
        action_def = trained_agent.choose_action(obs_dict)

        try:
            action = Action(
                action_type=ActionType(action_def["action_type"]),
                parameters=action_def["parameters"],
            )
            obs, reward, done, info = run_env.step(action)
            obs_dict = obs.model_dump()
            total_reward += reward

            steps.append({
                "step": step_num,
                "action_type": action_def["action_type"],
                "parameters": action_def["parameters"],
                "reward": reward,
                "cumulative_reward": total_reward,
                "observation": obs_dict,
                "info": info,
            })

            if done:
                break
        except Exception as e:
            steps.append({
                "step": step_num,
                "action_type": action_def["action_type"],
                "error": str(e),
                "reward": 0,
            })
            continue

    # Grade
    final_obs = run_env.state()
    grades = grade_task(req.task_name, final_obs)

    # Generate professional clinical report using the LLM (even for RL runs)
    clinical_report = "Clinical synthesis currently unavailable for RL model."
    if llm_agent_loaded:
        try:
            # Get the profile from the environment's current task
            task_profile = run_env._task.patient_profile.model_dump()
            clinical_report = llm_agent.generate_report(steps, task_profile)
        except Exception as e:
            clinical_report = f"Safety review interrupted: {str(e)}"

    return {
        "task_name": req.task_name,
        "steps": steps,
        "total_reward": total_reward,
        "step_count": len(steps),
        "grades": grades,
        "clinical_report": clinical_report,
        "agent": "trained_rl",
        "model_loaded": model_loaded,
        "episodes_trained": trained_agent.episode_count,
        "q_table_size": len(trained_agent.q_table),
    }

@app.post("/custom-run")
def custom_run(req: CustomRunRequest):
    """
    Run the TRAINED RL agent on a dynamic, custom user prescription.
    """
    from env.models import TaskDefinition, PatientProfile

    # Build an ad-hoc task
    custom_prof = PatientProfile(
        age=req.age,
        gender=req.gender,
        weight_kg=req.weight_kg if req.weight_kg > 0 else None,
        renal_function=req.renal_function,
        allergies=req.allergies,
        conditions=req.conditions,
        current_medications=[],
        pregnancy=False
    )
    
    custom_task = TaskDefinition(
        name="custom_user_task",
        difficulty="custom",
        description="A real-world custom clinical request.",
        prescription_text=req.prescription_text,
        prescription_clean=req.prescription_text,
        patient_profile=custom_prof,
        inventory=get_task("easy").inventory, # reuse typical inventory
        ground_truth_medicines=[], # won't be graded
        expected_interactions=[],
        expected_risks=[],
        optimal_action_count=10
    )

    run_env = PharmacistEnv("custom")
    obs = run_env.reset(custom_task=custom_task)
    obs_dict = obs.model_dump()

    steps = []
    
    for step_num in range(1, obs.max_steps + 1):
        if obs_dict.get("done", False):
            break

        try:
            if llm_agent_loaded:
                action_def = llm_agent.choose_action(obs_dict, step_num)
            else:
                action_def = trained_agent.choose_action(obs_dict)
                
            action = Action(
                action_type=ActionType(action_def["action_type"]),
                parameters=action_def["parameters"],
            )
            obs, reward, done, info = run_env.step(action)
            obs_dict = obs.model_dump()

            steps.append({
                "step": step_num,
                "action_type": action_def["action_type"],
                "parameters": action_def["parameters"],
                "observation": obs_dict,
                "info": info,
            })
            if done:
                break
        except Exception as e:
            err_str = str(e)
            steps.append({
                "step": step_num,
                "action_type": "error_recovery",
                "error": err_str,
                "reward": 0,
            })
            # If it's a fatal LLM error (all models hit limits), stop the simulation
            if any(x in err_str.lower() for x in ["quota", "limit", "failed", "unsupported", "key"]):
                break
            continue

    # Generate professional clinical report
    clinical_report = ""
    if llm_agent_loaded:
        try:
            clinical_report = llm_agent.generate_report(steps, req.model_dump())
        except Exception as e:
            clinical_report = f"Could not generate report: {e}"

    return {
        "task_name": "custom_user_task",
        "steps": steps,
        "step_count": len(steps),
        "agent": "llm" if llm_agent_loaded else "trained_rl",
        "model_loaded": model_loaded,
        "llm_loaded": llm_agent_loaded,
        "clinical_report": clinical_report
    }

# ---------------------------------------------------------------------------
# Serve React Frontend
# ---------------------------------------------------------------------------
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")

if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
    app.mount("/vite.svg", StaticFiles(directory=frontend_dist, html=False), name="vite_svg")
    
    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        # We only catch-all for paths that don't match specific backend endpoints.
        # Ensure we don't accidentally swallow API errors with an HTML page, but here it's fine.
        return FileResponse(os.path.join(frontend_dist, "index.html"))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
