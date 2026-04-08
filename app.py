from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from env.core import SmartInboxEnv
from models.schema import Action
import os

app = FastAPI(
    title="SmartInboxEnv API",
    description="Production-grade RL environment for email triage benchmarking.",
    version="1.0.0"
)

# Initialize environment (Global for simplicity in this hackathon context)
env_instance = SmartInboxEnv(task_id="hard_response")

@app.get("/", response_class=HTMLResponse)
def root():
    """Live interactive preview for judges."""
    test_env = SmartInboxEnv("easy_classification")
    obs = test_env.reset()
    
    # Run a single illustrative step
    action = Action(
        action_type="archive",
        email_class="spam",
        priority_level="low",
        reasoning="Initial system health check"
    )
    res_obs, reward, done, info = test_env.step(action)
    
    return f"""
    <html>
        <head>
            <title>SmartInboxEnv | Live</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
            <style>
                body {{ font-family: 'Inter', sans-serif; padding: 40px; line-height: 1.6; background: #f4f7f6; }}
                .card {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); max-width: 900px; margin: auto; }}
                h1 {{ color: #2c3e50; margin-top: 0; }}
                .status-badge {{ background: #2ecc71; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; vertical-align: middle; }}
                .demo-box {{ background: #2d3436; color: #ecf0f1; padding: 20px; border-radius: 8px; font-family: monospace; overflow-x: auto; }}
                .nav-links {{ margin-top: 20px; }}
                .btn {{ background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 6px; font-weight: 600; margin-right: 10px; }}
                .btn:hover {{ background: #2980b9; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>📧 SmartInboxEnv <span class="status-badge">ONLINE</span></h1>
                <p>Environment is active and compliant with <strong>OpenEnv</strong> specifications.</p>
                
                <h3>🚀 Live System Probe (Direct)</h3>
                <div class="demo-box">
                    <strong>Input Subject:</strong> {obs.subject}<br>
                    <strong>Action Taken:</strong> {action.action_type}<br>
                    <strong>Computed Reward:</strong> {reward}<br>
                    <strong>Env State:</strong> {"Active" if not done else "Finished"}
                </div>

                <div class="nav-links">
                    <a href="/run-task" class="btn">▶️ Run Full Triage Demo</a>
                    <a href="/docs" class="btn" style="background: #9b59b6;">📖 API Documentation</a>
                </div>
                
                <p style="margin-top: 30px; font-size: 0.9em; color: #7f8c8d;">
                    *Note: This Space runs a FastAPI backend. For automated evaluation, please use the API endpoints or see /docs.
                </p>
            </div>
        </body>
    </html>
    """

@app.get("/run-task")
def run_task(task_id: str = "hard_response"):
    """Demonstrates a full episode execution."""
    test_env = SmartInboxEnv(task_id)
    obs = test_env.reset()
    
    total_reward = 0
    steps = []
    done = False
    
    while not done:
        # Heuristic-based demo agent
        current_subject = obs.subject
        action_type = "reply" if obs.priority in ["high", "medium"] else "archive"
        if "ads" in obs.sender or "fakebank" in obs.sender:
            action_type = "archive"
            
        action = Action(
            action_type=action_type,
            email_class="spam" if action_type == "archive" else "work",
            priority_level=obs.priority,
            response="Automated demo response.",
            reasoning="Context-aware priority triage."
        )
        
        obs, reward, done, info = test_env.step(action)
        total_reward += reward
        steps.append({
            "subject": current_subject,
            "action": action.action_type,
            "reward": round(reward, 2)
        })

    return {
        "status": "success",
        "task_id": task_id,
        "final_score": info.get("final_score"),
        "total_reward": round(total_reward, 2),
        "steps_completed": len(steps),
        "details": steps
    }

@app.get("/reset")
def reset(task_id: str = "hard_response"):
    """Resets the environment."""
    global env_instance
    env_instance = SmartInboxEnv(task_id=task_id)
    return env_instance.reset().dict()

@app.get("/state")
def state():
    """Returns current state."""
    return env_instance.state().dict()

@app.post("/step")
def step(action: Action):
    """Executes environment step."""
    obs, reward, done, info = env_instance.step(action)
    return {
        "observation": obs.dict() if obs else None,
        "reward": round(reward, 2),
        "done": done,
        "info": info
    }
