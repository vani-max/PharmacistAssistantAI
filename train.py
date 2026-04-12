"""
train.py -- Offline RL Training for PharmacistEnv.

Trains the Q-learning agent across many episodes on all task difficulties.
The agent learns from environment reward signals which action sequences
lead to safe clinical outcomes.

Training loop:
  1. For each episode, reset a random task
  2. Agent selects actions using epsilon-greedy policy
  3. Environment returns reward for each step
  4. Agent updates Q-values via Bellman equation
  5. Epsilon decays over episodes (explore -> exploit)

After training, the Q-table is saved to rl_weights/q_table.json.
The API server loads this trained model on startup for inference.

Usage:
    python train.py                     # Train 100 episodes (default)
    python train.py --episodes 500      # Train 500 episodes
    python train.py --episodes 200 --task hard  # Train only on hard
"""

import sys
import os
import time
import argparse
import random

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(__file__))

from env.environment import PharmacistEnv
from env.models import Action, ActionType
from env.tasks import list_tasks
from graders.graders import grade_task
from agent.rl_agent import RLPolicyAgent


def train(
    episodes: int = 100,
    task_filter: str = "all",
    model_path: str = "rl_weights/q_table.json",
    verbose: bool = True,
):
    """
    Train the RL agent and save the learned policy.

    Args:
        episodes: Number of training episodes
        task_filter: "all", "easy", "medium", or "hard"
        model_path: Path to save the trained Q-table
        verbose: Print per-episode metrics
    """
    agent = RLPolicyAgent(epsilon=0.3, lr=0.3, gamma=0.95)

    # Try to load existing model for incremental training
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"[INFO] Continuing training from {agent.episode_count} episodes")
    else:
        print("[INFO] Starting fresh training")

    task_names = list_tasks() if task_filter == "all" else [task_filter]
    start_time = time.time()

    print("=" * 70)
    print(f"  PharmacistEnv RL Training")
    print(f"  Episodes: {episodes} | Tasks: {task_names}")
    print(f"  Hyperparameters: lr={agent.lr}, gamma={agent.gamma}, epsilon_start=0.3")
    print("=" * 70)
    print()

    best_scores = {t: 0.0 for t in task_names}
    all_results = []

    for episode in range(1, episodes + 1):
        # Epsilon decay: high exploration early, exploitation later
        agent.epsilon = max(0.05, 0.3 * (1 - (episode - 1) / episodes))

        for task_name in task_names:
            env = PharmacistEnv(task_name)
            obs = env.reset()
            obs_dict = obs.model_dump()

            episode_reward = 0.0
            step_count = 0
            actions_taken = []

            for _ in range(obs.max_steps):
                if obs_dict.get("done", False):
                    break

                state_key = agent.state_key(obs_dict)
                action_def = agent.choose_action(obs_dict)

                try:
                    action = Action(
                        action_type=ActionType(action_def["action_type"]),
                        parameters=action_def["parameters"],
                    )
                    obs, reward, done, info = env.step(action)
                    obs_dict = obs.model_dump()

                    next_key = agent.state_key(obs_dict)
                    agent.update(state_key, action_def["action_type"], reward, next_key, done)

                    episode_reward += reward
                    step_count += 1
                    actions_taken.append(action_def["action_type"])

                    if done:
                        break
                except Exception as e:
                    if verbose and episode <= 5:
                        print(f"  [WARN] Step error: {e}")
                    break

            # Grade the episode
            final_obs = env.state()
            grades = grade_task(task_name, final_obs)
            agent.episode_count += 1

            result = {
                "episode": episode,
                "task": task_name,
                "reward": round(episode_reward, 4),
                "steps": step_count,
                "accuracy": grades["accuracy"],
                "safety": grades["safety"],
                "efficiency": grades["efficiency"],
                "final_score": grades["final_score"],
                "actions": actions_taken,
            }
            all_results.append(result)
            agent.training_history.append(result)

            # Track best
            if grades["final_score"] > best_scores[task_name]:
                best_scores[task_name] = grades["final_score"]

        # Print progress
        if verbose and (episode % 10 == 0 or episode <= 5 or episode == episodes):
            latest = {t: [r for r in all_results if r["task"] == t][-1] for t in task_names if any(r["task"] == t for r in all_results)}
            line_parts = [f"Episode {episode:4d}"]
            for t in task_names:
                if t in latest:
                    r = latest[t]
                    line_parts.append(
                        f"{t}: score={r['final_score']:.3f} "
                        f"(acc={r['accuracy']:.2f} saf={r['safety']:.2f} eff={r['efficiency']:.2f}) "
                        f"rew={r['reward']:+.2f} steps={r['steps']}"
                    )
            print(" | ".join(line_parts))

    elapsed = time.time() - start_time

    # Save trained model
    agent.save(model_path)

    # Print summary
    print()
    print("=" * 70)
    print(f"  Training Complete")
    print(f"  Total episodes: {agent.episode_count}")
    print(f"  Q-table size: {len(agent.q_table)} state-action pairs")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Model saved: {model_path}")
    print()

    for task_name in task_names:
        task_results = [r for r in all_results if r["task"] == task_name]
        if task_results:
            first_5 = task_results[:5]
            last_5 = task_results[-5:]
            avg_first = sum(r["final_score"] for r in first_5) / len(first_5)
            avg_last = sum(r["final_score"] for r in last_5) / len(last_5)
            improvement = avg_last - avg_first
            print(f"  {task_name.upper():8s} | Best: {best_scores[task_name]:.4f} "
                  f"| First 5 avg: {avg_first:.4f} | Last 5 avg: {avg_last:.4f} "
                  f"| Improvement: {improvement:+.4f}")

    print("=" * 70)

    return agent, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for PharmacistEnv")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--task", type=str, default="all", help="Task to train on: all, easy, medium, hard")
    parser.add_argument("--model", type=str, default="rl_weights/q_table.json", help="Path to save model")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-episode output")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        task_filter=args.task,
        model_path=args.model,
        verbose=not args.quiet,
    )
