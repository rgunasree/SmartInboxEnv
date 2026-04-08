from fastapi import FastAPI
from env.core import SmartInboxEnv
from models.schema import Action
import os

app = FastAPI(title="SmartInboxEnv API")

# Initialize environment
# We'll use a default task_id, but the validator can call /reset?task_id=...
env_instance = SmartInboxEnv(task_id="hard_response")

@app.get("/")
def root():
    """Health check endpoint for Hugging Face Space visibility."""
    return {
        "status": "running",
        "env": "SmartInboxEnv",
        "description": "Production-grade email triage RL environment"
    }

@app.get("/reset")
def reset(task_id: str = "hard_response"):
    """Resets the environment and returns the initial observation."""
    global env_instance
    env_instance = SmartInboxEnv(task_id=task_id)
    obs = env_instance.reset()
    return obs.dict()

@app.get("/state")
def state():
    """Returns the current internal state of the environment."""
    return env_instance.state().dict()

@app.post("/step")
def step(action: Action):
    """Executes one step in the environment."""
    obs, reward, done, info = env_instance.step(action)
    return {
        "observation": obs.dict() if obs else None,
        "reward": reward,
        "done": done,
        "info": info
    }

if __name__ == "__main__":
    import uvicorn
    # Hugging Face usually provides port 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
