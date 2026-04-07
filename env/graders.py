def grade_easy(state):
    # Just check if something was extracted and matched
    if state.extracted and state.matched:
        return 1.0
    return 0.0


def grade_medium(state):
    score = 0.0

    # extraction done
    if state.extracted:
        score += 0.5

    # correct matching done
    if state.matched:
        score += 0.5

    return score


def grade_hard(state):
    score = 0.0

    if state.extracted:
        score += 0.3

    if state.matched:
        score += 0.4

    return score


def grade_task(task_name, state):
    if task_name == "easy":
        return grade_easy(state)
    elif task_name == "medium":
        return grade_medium(state)
    elif task_name == "hard":
        return grade_hard(state)
    else:
        return 0.0