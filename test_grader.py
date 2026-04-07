from env.environment import PharmacyEnv
from env.models import Action
from env.graders import grade_easy, grade_medium, grade_hard

for task in ["easy", "medium", "hard"]:
    env = PharmacyEnv(task)
    obs = env.reset()

    obs.extracted = ["PCM 500mg"]
    obs.matched = ["Paracetamol 500mg"]

    if task == "easy":
        score = grade_easy(obs)
    elif task == "medium":
        score = grade_medium(obs)
    else:
        score = grade_hard(obs)

    print(task, "score:", score)