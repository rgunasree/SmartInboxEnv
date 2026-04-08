TASKS = {
    "easy_classification": {
        "goal": "Identify correctly if an email is WORK or SPAM.",
        "expected_outcome": "High accuracy in the action_type selection (e.g., reply for work, archive for spam).",
        "difficulty": 1
    },
    "medium_prioritization": {
        "goal": "Correctly prioritize emails based on urgency and sender.",
        "expected_outcome": "Priority matches the ground truth (e.g., boss emails are high priority).",
        "difficulty": 2
    },
    "hard_response": {
        "goal": "Generate appropriate responses and correctly escalate critical issues.",
        "expected_outcome": "Response tone and content align with the email context and sender reputation.",
        "difficulty": 3
    }
}
