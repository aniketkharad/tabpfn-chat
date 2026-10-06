"""Generate customer support feedback dataset with natural language text and numeric features."""

from pathlib import Path
import numpy as np
import pandas as pd

def generate():
    np.random.seed(42)
    demo_dir = Path("demos/06_text_integration")
    demo_dir.mkdir(parents=True, exist_ok=True)

    non_escalated_phrases = [
        "Smooth setup and prompt answer",
        "Quick response from agent",
        "Appreciated the helpful guidance",
        "Resolved my issue within minutes",
        "Polite representative and clear steps",
        "Works as expected after update",
        "Great onboarding session",
        "Very clear instructions provided",
        "Feature works nicely thank you",
        "Helpful documentation and easy fix",
        "Satisfied with the support resolution",
        "Fast turnaround on my inquiry",
    ]

    escalated_phrases = [
        "Critical outage billing failed twice",
        "Service is completely down urgently fix",
        "Charged incorrect amount no refund",
        "Agent was extremely rude and unhelpful",
        "System crashed during migration urgent",
        "Data loss detected please call me now",
        "Unable to log in for two days emergency",
        "Unacceptable downtime demand manager",
        "Security issue unauthorized login alert",
        "Database connection dropped lost data",
        "Production API broken emergency assistance",
        "Recurring failures zero communication",
    ]

    n_total = 120
    n_escalated = 48
    n_normal = n_total - n_escalated

    records = []
    
    # Generate normal records
    for i in range(n_normal):
        text = np.random.choice(non_escalated_phrases)
        tier = np.random.choice(["Basic", "Pro", "Enterprise"], p=[0.5, 0.35, 0.15])
        spend = np.random.uniform(30, 400) if tier == "Basic" else (np.random.uniform(200, 800) if tier == "Pro" else np.random.uniform(600, 1500))
        tenure = np.random.randint(2, 48)
        records.append({
            "feedback_text": text,
            "account_tier": tier,
            "monthly_spend": np.round(spend, 2),
            "tenure_months": tenure,
            "needs_escalation": 0,
        })

    # Generate escalated records
    for i in range(n_escalated):
        text = np.random.choice(escalated_phrases)
        tier = np.random.choice(["Basic", "Pro", "Enterprise"], p=[0.2, 0.4, 0.4])
        spend = np.random.uniform(50, 500) if tier == "Basic" else (np.random.uniform(350, 950) if tier == "Pro" else np.random.uniform(800, 2500))
        tenure = np.random.randint(1, 36)
        records.append({
            "feedback_text": text,
            "account_tier": tier,
            "monthly_spend": np.round(spend, 2),
            "tenure_months": tenure,
            "needs_escalation": 1,
        })

    # Shuffle
    np.random.shuffle(records)
    for idx, r in enumerate(records):
        r["ticket_id"] = f"TCK_{idx+1:03d}"

    df = pd.DataFrame(records)
    cols = ["ticket_id", "feedback_text", "account_tier", "monthly_spend", "tenure_months", "needs_escalation"]
    df = df[cols]

    csv_path = demo_dir / "customer_feedback_escalation.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} rows ({df['needs_escalation'].sum()} escalated) -> {csv_path}")

    # Also save to static samples
    static_samples = Path("src/tabchat/frontend/static/samples")
    static_samples.mkdir(parents=True, exist_ok=True)
    df.to_csv(static_samples / "customer_feedback_escalation.csv", index=False)

if __name__ == "__main__":
    generate()
