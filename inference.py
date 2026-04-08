import os
import json
from openai import OpenAI
from env.core import SmartInboxEnv
from models.schema import Action
from tasks.tasks import TASKS

def run_task(task_id: str):
    client = OpenAI(
        base_url=os.getenv("API_BASE_URL"),
        api_key=os.getenv("HF_TOKEN")
    )
    model = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8b-instruct") 
    
    env = SmartInboxEnv(task_id=task_id)
    obs = env.reset()
    
    print(f"[START]")
    print(f"Task: {task_id}")
    
    final_episode_score = 0.0
    done = False
    
    while not done:
        # 🔴 FIXED SPAM SIGNAL
        is_spam_hint = (
            "ads" in obs.sender.lower() or
            "sale" in obs.subject.lower() or
            "off" in obs.body.lower() or
            "discount" in obs.body.lower()
        )
        
        # HARD DECISION PROMPT
        prompt = (
            f"You are an expert professional email assistant.\n"
            f"Current Task: {task_id}\n\n"
            f"Email Data:\n- Subject: {obs.subject}\n- Body: {obs.body}\n- From: {obs.sender}\n"
            f"Automated Spam Detection: {'YES' if is_spam_hint else 'NO'}\n\n"
            
            "STRICT DECISION RULES:\n"
            "1. If 'Automated Spam Detection' is YES OR message is promotional → action_type MUST be 'archive'.\n"
            "2. If sender contains 'boss' OR subject contains 'meeting', 'project', 'demo', 'confirmed' → action_type MUST be 'reply'.\n"
            "3. If sender is unknown and content is unclear/unimportant → action_type MUST be 'archive'.\n"
            "4. DO NOT default to 'reply'. You are penalized for over-replying to trash.\n"
            "5. You MUST use different actions across emails if the context changes.\n"
            "6. First decide the type of email (spam/work/personal), then choose the hard action.\n\n"
            
            "FEW-SHOT EXAMPLES:\n"
            "Email: Subject: 90% OFF Sale | From: ads@deals.com\n"
            "→ action_type: archive, email_class: spam\n\n"
            "Email: Subject: Meeting Request | From: boss@corp.com\n"
            "→ action_type: reply, email_class: work\n\n"
            
            "Respond ONLY in JSON:\n"
            "{\n"
            "  \"action_type\": \"reply/archive/escalate\",\n"
            "  \"email_class\": \"spam/work/personal\",\n"
            "  \"priority_level\": \"low/medium/high\",\n"
            "  \"response\": \"A professional reply text\",\n"
            "  \"reasoning\": \"A detailed explanation (>15 words) of your hard decision\"\n"
            "}"
        )
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.0
            )
            
            llm_output = response.choices[0].message.content.strip()
            parsed = json.loads(llm_output)
            
            # --- SOFT BIAS SIGNALING (REMOVED HARD OVERRIDE FOR AGENT AUTONOMY) ---
            # We trust the LLM's reasoning but ensure the Action model catches keys.
            action = Action(
                action_type=parsed.get("action_type", "reply"),
                email_class=parsed.get("email_class", "work"),
                priority_level=parsed.get("priority_level", "medium"),
                response=parsed.get("response", "Thank you for the update."),
                reasoning=parsed.get("reasoning", "Autonomous decision based on email context.")
            )
        except Exception:
            action = Action(
                action_type="reply" if obs.priority == "high" else "archive",
                email_class="work" if obs.priority != "low" else "spam",
                priority_level="high" if obs.priority == "high" else "low",
                response="I have received your email.",
                reasoning="Safe fallback triggered."
            )
            
        obs, reward, done, info = env.step(action)
        
        if done:
            final_episode_score = info.get("final_score", 0.0)
        
        print(f"\n[STEP]")
        print(f"Action: {action.action_type}")
        print(f"Reward: {reward:.2f}")

    print(f"\n[END]")
    print(f"Final Score: {final_episode_score:.2f}")

if __name__ == "__main__":
    for task_id in TASKS.keys():
        run_task(task_id)
