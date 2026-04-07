from .models import Observation, Medicine

def get_task(level="easy"):
    if level == "easy":
        return Observation(
            prescription_text="Tab PCM 500mg",
            available_medicines=[
                Medicine(name="Paracetamol 500mg", stock=10),
                Medicine(name="Crocin 500mg", stock=5)
            ]
        )

    elif level == "medium":
        return Observation(
            prescription_text="Take PCM twice daily",
            available_medicines=[
                Medicine(name="Paracetamol 650mg", stock=2),  # slight variation
                Medicine(name="Ibuprofen 200mg", stock=10)
            ]
        )

    else:  # hard
        return Observation(
            prescription_text="PCM 500mg, if not available use alternative",
            available_medicines=[
                Medicine(name="Crocin 500mg", stock=5)
            ]
        )