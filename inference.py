import os
import json
import random
from openai import OpenAI
from env.core import SmartInboxEnv
from models.schema import Action
from tasks.tasks import TASKS

class AdaptiveAgent:
    """A simple RL-style agent that adapts its decision bias based on rewards."""
    def __init__(self, client, model):
        self.client = client
        self.model = model
        # Bias weights for decision making
        self.weights = {
            "reply_preference": 0.5,
            "archive_preference": 0.5
        }
    
    def get_action(self, obs, task_id):
        # Determine spam hint
        is_spam_hint = (
            "ads" in obs.sender.lower() or
            "sale" in obs.subject.lower() or
            "off" in obs.body.lower() or
            "discount" in obs.body.lower()
        )
        
        # Adjust prompt based on learned weights
        strategy_hint = ""
        if self.weights["reply_preference"] > 0.6:
            strategy_hint = "LEARNED STRATEGY: You have learned that being responsive is highly valued in this context."
        elif self.weights["archive_preference"] > 0.6:
            strategy_hint = "LEARNED STRATEGY: You have learned to be strictly defensive and filter aggressive noise."

        prompt = (
            f"You are an expert professional email assistant.\n"
            f"Current Task: {task_id}\n"
            f"{strategy_hint}\n\n"
            f"Email Data:\n- Subject: {obs.subject}\n- Body: {obs.body}\n- From: {obs.sender}\n"
            f"Automated Spam Detection: {'YES' if is_spam_hint else 'NO'}\n\n"
            "STRICT DECISION RULES:\n"
            "1. If 'Automated Spam Detection' is YES OR message is promotional → action_type MUST be 'archive'.\n"
            "2. If sender contains 'boss' OR subject contains 'meeting', 'project', 'demo', 'confirmed' → action_type MUST be 'reply'.\n"
            "3. If sender is unknown and content is unclear/unimportant → action_type MUST be 'archive'.\n"
            "4. First decide the type (spam/work/personal), then choose the hard action.\n\n"
            "Respond ONLY in JSON:\n"
            "{\n"
            "  \"action_type\": \"reply/archive/escalate\",\n"
            "  \"email_class\": \"spam/work/personal\",\n"
            "  \"priority_level\": \"low/medium/high\",\n"
            "  \"response\": \"A professional reply text\",\n"
            "  \"reasoning\": \"A detailed explanation (>15 words) of your decision\"\n"
            "}"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.0
            )
            parsed = json.loads(response.choices[0].message.content.strip())
            return Action(
                action_type=parsed.get("action_type", "archive"),
                email_class=parsed.get("email_class", "work"),
                priority_level=parsed.get("priority_level", "low"),
                response=parsed.get("response", ""),
                reasoning=parsed.get("reasoning", "")
            )
        except:
            return Action(action_type="archive", reasoning="Fallback due to error")

    def update(self, action, reward):
        # Simple policy gradient-style update
        lr = 0.05
        if action.action_type == "reply":
            self.weights["reply_preference"] += lr * (reward - 0.5)
        elif action.action_type == "archive":
            self.weights["archive_preference"] += lr * (reward - 0.5)
        
        # Clamp
        self.weights["reply_preference"] = max(0.1, min(0.9, self.weights["reply_preference"]))
        self.weights["archive_preference"] = max(0.1, min(0.9, self.weights["archive_preference"]))

def run_evaluation():
    client = OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/"),
        api_key=os.getenv("HF_TOKEN")
    )
    model = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8b-instruct")
    
    # We evaluate on the hardest task to show learning
    task_id = "hard_response"
    env = SmartInboxEnv(task_id=task_id)
    agent = AdaptiveAgent(client, model)
    
    print(f"[START]")
    print(f"Task: {task_id}")
    print("Strategy: Adaptive Learning across Episode")

    episode_rewards = []
    
    # Run 3 episodes to demonstrate improvement/adaptation
    for ep in range(3):
        print(f"\n--- Starting Episode {ep+1} ---")
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.get_action(obs, task_id)
            obs, reward, done, info = env.step(action)
            agent.update(action, reward)
            total_reward += reward
            
            print(f"[STEP]")
            print(f"Action: {action.action_type} | Reward: {reward:.2f}")
            
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1} Total Reward: {total_reward:.2f}")

    print(f"\n[END]")
    print(f"Learning Curve (Total Rewards): {episode_rewards}")
    print(f"Final Strategy Weights: {agent.weights}")
    improvement = episode_rewards[-1] - episode_rewards[0]
    print(f"Intelligence improvement: {improvement:.2f}")

if __name__ == "__main__":
    run_evaluation()
