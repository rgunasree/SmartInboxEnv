---
title: SmartInboxEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 📧 SmartInboxEnv

A production-grade reinforcement learning environment for **decision-making under uncertainty** in email triage.

Built for the **OpenEnv Hackathon**.

---

## 🚀 What This Actually Tests

This is NOT a spam classifier.

This environment evaluates whether an agent can:

- Choose the **right action** (reply / archive / escalate)
- Handle **conflicting priorities**
- Detect **deceptive urgency (phishing)**
- Maintain **context across multiple steps**
- Optimize for **long-term reward**

---

## ⚡ Live Demo

- `/` → Instant system probe (1-step environment execution)
- `/run-task` → Full episode (multi-step decision sequence)
- `/train` → Watch an agent improve via reward feedback
- `/docs` → Interactive API (Swagger)

---

## 🧠 Example Output

```json
{
  "final_score": 0.69,
  "total_reward": 5.55,
  "steps_completed": 8
}
```

👉 Not perfect → realistic
👉 Not random → learnable

---

## 🏗️ Environment Design

### 1. Multi-Step Decision Process
- Episodes contain 7+ diverse emails
- Each action affects future rewards
- Context is preserved across steps

### 2. Action Space
```json
{
  "action_type": "reply | archive | escalate",
  "email_class": "work | spam | personal",
  "priority_level": "low | medium | high",
  "response": "text",
  "reasoning": "explanation"
}
```

### 3. Reward Function (Core Innovation)
Reward is multi-factor and dynamic:

| Component | Weight |
| --- | --- |
| Classification | 0.1 |
| Priority alignment | 0.1 |
| Action correctness | 0.2 |
| Response quality | 0.4 |
| Reasoning depth | 0.1 |
| Behavioral consistency | dynamic |

---

## ⚔️ Advanced Scenarios

### 🧨 Conflict Scenario
**Client escalation vs leadership review**
- **escalate** → optimal
- **reply** → partial reward
- **archive** → penalty

👉 Tests strategic decision-making

### 🎣 Adversarial Phishing
**Fake high-priority email (malicious intent)**
- **Reply** → ❌ heavy penalty
- **Archive correctly** → ✅ full reward

👉 Tests robustness against deception

### 🧠 Context Memory
- Agents are penalized for:
- Breaking conversation flow
- Inconsistent decisions across steps

---

## 🛡 Anti-Exploitation Design
- ❌ Keyword stuffing does NOT work
- ❌ Repetitive actions are penalized
- ❌ Blind “always reply” fails

✔ Reward depends on structured correctness + behavior

---

## 📈 Learning Capability

Includes `/train` endpoint demonstrating:
- **ε-greedy exploration**
- **Reward-based policy updates**
- **Measurable improvement across episodes**

Example:
```json
{
  "episode_rewards": [3.2, 4.1, 5.3, 5.8, 6.0],
  "max_improvement_delta": 2.8
}
```

---

## 🌍 Real-World Applications
- Executive assistant automation
- Incident response prioritization
- Customer support triage
- AI email copilots

---

## ⚙️ Architecture
- FastAPI backend
- Custom RL environment
- Reward shaping engine
- Heuristic + learning agents
- Hugging Face Spaces deployment

---

## 🧠 Key Insight

Intelligence is not choosing the correct label — it is choosing the correct action under uncertainty.

---

## ✅ OpenEnv Compliance
- Implements `step()`, `reset()`, `state()`
- Uses structured Pydantic schemas
- Includes evaluation-ready inference logic
- Fully Dockerized for reproducibility

---

## 🏁 Final Note

SmartInboxEnv moves beyond static evaluation and introduces:
- **Decision intelligence**
- **Contextual reasoning**
- **Adversarial robustness**
- **Learning-based adaptation**

---

👉 Start with `/run-task` or `/train` to see it in action.
