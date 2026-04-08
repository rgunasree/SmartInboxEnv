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
    """Interactive Landing Page for Judges."""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>SmartInboxEnv | AI-Powered Triage</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
            <style>
                body { font-family: 'Inter', sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 40px auto; padding: 0 20px; background-color: #f4f7f6; }
                .container { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                h3 { color: #2980b9; margin-top: 30px; }
                .btn { display: inline-block; background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 6px; transition: background 0.3s; margin-right: 10px; }
                .btn:hover { background: #2980b9; }
                .btn-secondary { background: #2ecc71; }
                .btn-secondary:hover { background: #27ae60; }
                pre { background: #2d3436; color: #dfe6e9; padding: 15px; border-radius: 6px; overflow-x: auto; }
                code { font-family: 'Courier New', Courier, monospace; }
                .badge { background: #e0e0e0; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📧 SmartInboxEnv</h1>
                <p>A production-grade <strong>OpenEnv</strong> compliant Reinforcement Learning environment for autonomous email triage. This environment simulates real-world challenges like strategic tradeoffs, thread memory, and deceptive phishing.</p>

                <h3>🚀 Test & Explore</h3>
                <p>Explore the interactive API and run a full environment simulation with one click.</p>
                <a href="/docs" class="btn">Swagger API Docs</a>
                <a href="/demo" class="btn btn-secondary">Run Automated Demo</a>
                <a href="/reset" class="btn">Reset Env</a>

                <h3>📌 Integration Example</h3>
                <p>Send a <code>POST</code> request to <code>/step</code> with a JSON payload:</p>
                <pre><code>{
  "action_type": "reply",
  "email_class": "work",
  "priority_level": "high",
  "response": "Sure, let's schedule the meeting for 3 PM.",
  "reasoning": "High priority request from the executive team."
}</code></pre>
                
                <h3>🔗 Useful Endpoints</h3>
                <ul>
                    <li><code>GET /health</code> - System status</li>
                    <li><code>GET /state</code> - Current environment state</li>
                    <li><code>GET /reset?task_id=hard_response</code> - Re-initialize environment</li>
                </ul>
            </div>
        </body>
    </html>
    """

@app.get("/health")
def health():
    """Production health check."""
    return {"status": "ok", "version": "1.0.0"}

@app.get("/demo")
def demo():
    """Automated 1-click demo for judges to see the environment in action."""
    test_env = SmartInboxEnv("hard_response")
    obs = test_env.reset()
    
    steps = []
    done = False
    
    while not done:
        # Simulated 'Good' Agent Logic for Demo
        action_type = "reply" if obs.priority in ["high", "medium"] else "archive"
        if "ads" in obs.sender or "fakebank" in obs.sender:
            action_type = "archive"
            
        action = Action(
            action_type=action_type,
            email_class="spam" if "ads" in obs.sender or "fakebank" in obs.sender else "work",
            priority_level=obs.priority,
            response="This is an automated demo response based on detected priority.",
            reasoning="Evaluating email based on sender reputation and subject context."
        )
        
        current_subject = obs.subject
        obs, reward, done, info = test_env.step(action)
        
        steps.append({
            "subject": current_subject,
            "action": action.action_type,
            "reward": round(reward, 2),
            "done": done
        })

    return {
        "message": "Demo completed successfully",
        "final_score": round(info.get("final_score", 0.0), 2),
        "total_steps": len(steps),
        "steps_detail": steps
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
