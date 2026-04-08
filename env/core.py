from models.schema import Email, State, Action, Observation
from graders.grader import grade_classification, grade_priority, grade_response
from typing import List, Tuple
import random

class SmartInboxEnv:
    def __init__(self, task_id: str = "easy_classification"):
        self.task_id = task_id
        # Winner-Tier Email Corpus: High Context, Conflict Scenarios, and Trade-offs
        self._emails = [
            Email(subject="Meeting", body="Discuss project", sender="boss@corp.com", timestamp=100.0, priority="high"),
            Email(subject="Sale", body="90% off now", sender="spam@ads.com", timestamp=101.0, priority="low"),
            Email(subject="Flight", body="Ticket confirmed", sender="travel@fly.com", timestamp=102.0, priority="medium"),
            Email(subject="Conflict", body="Reschedule demo", sender="client@outsider.com", timestamp=103.0, priority="high"),
            Email(subject="Incomplete", body="Sent doc?", sender="peer@corp.com", timestamp=104.0, priority="medium"),
            Email(subject="Quick check", body="Can you look into this?", sender="unknown@random.com", timestamp=105.0, priority="low"),
            # 🧨 Conflict Scenario: High-Impact Trade-off
            Email(
                subject="CRITICAL: Client escalation vs leadership review",
                body="Major client system is down, but you have the mandatory 2026 Strategy Review with leadership in 10 minutes. Both require your presence.",
                sender="ops@corp.com",
                timestamp=106.0,
                priority="high"
            ),
            # 🧨 Deceptive Spam / Phishing Case (New): Tests robustness against false urgency
            Email(
                subject="Important: Account verification required",
                body="Please confirm your credentials immediately to avoid system suspension.",
                sender="security@fakebank.com",
                timestamp=107.0,
                priority="high" # Marked high to trick agent
            )
        ]
        
        random.seed(42)
        random.shuffle(self._emails)
        
        self.reset()
        self._user_preference = "detailed"

    def reset(self) -> Observation:
        self._current_index = 0
        self._done = False
        self._elapsed_time = 0.0
        self._history_actions = []
        self._history_rewards = []
        self._conversation_context = [] 
        return self._get_observation()

    def state(self) -> State:
        return State(
            emails=self._emails, 
            current_index=self._current_index,
            user_preference=self._user_preference,
            elapsed_time=self._elapsed_time,
            history=self._history_rewards
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        if self._done:
            raise RuntimeError("Environment finished. Call reset().")

        email = self._emails[self._current_index]
        reward = 0.0
        
        # Ground Truths
        expected_action = "reply" if email.priority in ["high", "medium"] else "archive"
        
        if "ads.com" in email.sender or "fakebank.com" in email.sender:
            expected_class = "spam"
            expected_action = "archive" # Deceptive spam MUST be archived
        elif "corp.com" in email.sender or "fly.com" in email.sender: # Fixed logical bug
            expected_class = "work"
        else:
            expected_class = "personal"

        # Tradeoff Scenario Logic
        if "Client escalation vs leadership review" in email.subject:
            if action.action_type == "escalate":
                reward += 1.0 
            elif action.action_type == "reply":
                reward += 0.4 
            else:
                reward -= 0.8 

        # Deceptive Phishing Logic
        if "Account verification" in email.subject:
            if action.email_class == "spam" and action.action_type == "archive":
                reward += 1.0 # Perfection: spotted the phish
            elif action.action_type == "reply":
                reward -= 1.0 # Critical Failure: engaged with phish

        # Context Memory
        self._conversation_context.append({"subject": email.subject, "action": action.action_type})
        if len(self._conversation_context) > 1:
            prev = self._conversation_context[-2]
            if "meeting" in prev["subject"].lower() and action.action_type == "archive":
                reward -= 0.3

        # Task Differentiation Core Logic
        if self.task_id == "easy_classification":
            reward += 1.0 * grade_classification(action.email_class or "", expected_class)
        
        elif self.task_id == "medium_prioritization":
            reward += 0.4 * grade_classification(action.email_class or "", expected_class)
            reward += 0.6 * grade_priority(action.priority_level or "low", email.priority)
            
        elif self.task_id == "hard_response":
            reward += 0.1 * grade_classification(action.email_class or "", expected_class)
            reward += 0.1 * grade_priority(action.priority_level or "", email.priority)
            reward += 0.2 * grade_classification(action.action_type, expected_action)
            
            if action.action_type == "reply" and action.response:
                required = ["escalate", "client", "priority"] if "conflict" in email.subject.lower() else ["confirm", "regards"]
                response_quality = grade_response(action.response, required)
                reward += 0.4 * response_quality
                reward += 0.1 * (1.0 if action.reasoning and len(action.reasoning.split()) > 15 else 0.0)

        # Behavioral Bonuses
        consistency_bonus = 0.0
        if len(self._history_actions) > 1:
            unique_actions = len(set(self._history_actions[-3:]))
            if unique_actions > 1:
                consistency_bonus += 0.1  # Reward variety/non-repetitive behavior
        reward += consistency_bonus

        if len(self._history_rewards) > 2:
            if reward > (sum(self._history_rewards[-2:]) / 2):
                reward += 0.1 

        self._history_actions.append(action.action_type)
        self._history_rewards.append(reward)
        self._current_index += 1
        
        if self._current_index >= len(self._emails):
            self._done = True
            avg_reward = sum(self._history_rewards) / len(self._history_rewards)
            final_score = max(0.0, min(1.0, avg_reward))
            return None, reward, True, {"final_score": final_score}

        obs = self._get_observation()
        obs.context_history = [c["action"] for c in self._conversation_context[-3:]]
        return obs, reward, False, {}

    def _get_observation(self) -> Observation:
        email = self._emails[self._current_index]
        return Observation(
            subject=email.subject,
            body=email.body,
            sender=email.sender,
            timestamp=email.timestamp,
            priority=email.priority
        )
