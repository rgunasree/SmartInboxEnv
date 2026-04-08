---
title: SmartInboxEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SmartInboxEnv 📧

A production-grade RL environment for benchmarking AI agents in real-world email triage and decision-making. Built for the OpenEnv Hackathon.

## 🌍 Real-World Impact
SmartInboxEnv models real enterprise workflows such as:
- **Executive assistant triage**: Filtering noise from high-signal requests.
- **Incident response**: Prioritizing outages and customer complaints under pressure.
- **Customer success**: Generating context-aware responses with reasoning traces.

## 🧠 Environment Design & Innovation
- **Grounded Reward Signaling**: Rewards are dynamically calculated using multi-factor signals including structural intent and contextual consistency.
- **Decision Tradeoff Modeling**: Includes conflicting priorities (e.g., Client Escalation vs. Internal Review) where strategic delegation is the optimal path.
- **Adversarial Robustness**: Features deceptive phishing-style emails with false urgency to test agent resilience against malicious or deceptive signals.
- **Behavioral Evolution**: SmartInboxEnv evaluates not just correctness, but behavioral consistency and adaptive decision-making over time.
- **Thread Memory**: Models cross-step dependencies; agents are penalized for breaking consistency across a conversation thread.

## 🛡 Anti-Exploitation Design
Unlike naive environments, SmartInboxEnv is resistant to reward hacking:
- **Keyword Hacking Prevention**: Keyword stuffing is penalized via length-normalized, set-based scoring.
- **Behavioral Consistency**: Repetitive actions or context-breaking decisions incur dynamic penalties.
- **Ambiguity Resilience**: Edge cases with unknown senders reward cautious, defensive decision-making.

## ⚙️ Why This Environment is Hard
Unlike standard RL environments, SmartInboxEnv implements:
- **Strategic Tradeoffs**: Forces agents to prioritize between conflicting high-stakes objectives.
- **Multi-step Reasoning**: Actions are not binary and require cross-step contextual awareness.
- **Natural Language Rewards**: Scoring is grounded in response quality, tone, and intent markers.

## 🔄 Episode Structure
- Sequences of **8 diverse emails** (shuffled) including 🧨 Trade-off and 🎣 Phishing scenarios.
- **Final score**: Normalized average reward (0.0 - 1.0).

## ✅ OpenEnv Compliance
- Implements `step()`, `reset()`, and `state()` APIs using Pydantic data contracts.
- Compatible with OpenEnv validation pipeline.
