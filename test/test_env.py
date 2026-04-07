from env.environment import PharmacyEnv
from env.models import Action

for task in ["easy", "medium", "hard"]:
    print("\n---", task, "---")
    
    env = PharmacyEnv(task)
    obs = env.reset()
    print("Initial:", obs)

    obs, reward, done, _ = env.step(Action(action_type="extract", value=["PCM 500mg"]))
    print("Extract:", reward)

    if task == "hard":
        # ✅ correct behavior for hard
        obs, reward, done, _ = env.step(Action(action_type="suggest_alternative", value=["Crocin 500mg"]))
        print("Alternative:", reward)
    else:
        obs, reward, done, _ = env.step(Action(action_type="match", value=["Paracetamol 500mg"]))
        print("Match:", reward)

    obs, reward, done, _ = env.step(Action(action_type="finalize"))
    print("Final:", reward, done)