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
- **Keyword Hacking Prevention**: Keyword stuffing is penalized via length-normalized, set-based scoring (intersections).
- **Behavioral Consistency**: Repetitive actions or context-breaking decisions incur dynamic penalties.
- **Sanitized Outputs**: Validates and coerces agent outputs into strict schema-compliant categories.

## 🎮 Interactive Demo & API

The environment is deployed on Hugging Face Spaces with an interactive UI and API.

- **Landing Page**: Access the root URL to see a live system probe.
- **Adaptive Training Demo**: Visit `/train` to watch a learning agent improve its weights via a feedback loop over multiple episodes.
- **Automated Execution Demo**: Visit `/run-task` to watch a full episode run with a high-performance agent.
- **Interactive API Documentation**: Open `/docs` to test different actions manually via the Swagger UI.

## ⚙️ Why This Environment is Hard
Unlike standard RL environments, SmartInboxEnv implements:
- **Strategic Tradeoffs**: Forces agents to prioritize between conflicting high-stakes objectives.
- **Multi-step Reasoning**: Actions are not binary and require cross-step contextual awareness.
- **Natural Language Rewards**: Scoring is grounded in response quality, tone, and intent markers.

## 🔄 Episode Structure
- Sequences of **7 diverse emails** (shuffled) including 🧨 Trade-off and 🎣 Phishing scenarios.
- **Reward Signals**: Includes ground-truth matching, cross-step consistency bonuses, and behavioral growth rewards.
- **Final score**: Normalized average reward (0.0 - 1.0) returned in the `info` metadata.

## 📈 Proven Learnability
Unlike static benchmarks, SmartInboxEnv is proven to be learnable. The included `inference.py` demonstrates:
- **Policy Gradient Adaptation**: An agent that adjusts its decision-making based on reward signals.
- **Observed Improvement**: Clear convergence trends over multiple episodes, showing measurable intelligence growth.

## ✅ OpenEnv Compliance
- Implements `step()`, `reset()`, and `state()` APIs using Pydantic data contracts.
- Provides a standard `inference.py` for automated evaluation.
- Fully Dockerized and ready for scalable benchmarking.
