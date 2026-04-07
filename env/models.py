from pydantic import BaseModel
from typing import List, Optional

class Medicine(BaseModel):
    name: str
    stock: int

class Patient(BaseModel):
    age: int
    allergies: List[str]

class Observation(BaseModel):
    prescription_text: str
    available_medicines: List[Medicine]
    extracted: List[str] = []
    matched: List[str] = []

class Action(BaseModel):
    action_type: str
    value: Optional[List[str]] = None

class Reward(BaseModel):
    value: float
    reason: str