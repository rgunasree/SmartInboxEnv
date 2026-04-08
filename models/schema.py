from pydantic import BaseModel
from typing import List, Optional

class Email(BaseModel):
    subject: str
    body: str
    sender: str
    timestamp: float
    priority: str

class State(BaseModel):
    emails: List[Email]
    current_index: int
    user_preference: str
    elapsed_time: float
    history: List[float]

class Action(BaseModel):
    action_type: str = "archive" # reply, archive, escalate
    email_class: Optional[str] = "work" # spam, work, personal
    priority_level: Optional[str] = "low" # low, medium, high
    response: Optional[str] = ""
    reasoning: Optional[str] = ""

class Observation(BaseModel):
    subject: str
    body: str
    sender: str
    timestamp: float
    priority: str
    # Fixed Schema: Added context_history for OpenEnv validator compliance
    context_history: Optional[List[str]] = None
