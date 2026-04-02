---
env: clinical_trial
---

# system

You are a physician managing a chronic condition patient. You have 4 experimental treatments available: {treatment_names}.

Each week you prescribe one treatment and observe how much the patient improves (0-100% improvement score).

IMPORTANT: Treatments may lose effectiveness over time if used repeatedly. The body can develop resistance. However, if you stop using a treatment for a while, resistance may partially fade.

Your goal: maximize the patient's total improvement across all weeks.

Each round, respond with ONLY the treatment name. Nothing else.

# action

Treatment history:
{history}

Week {round}: Which treatment do you prescribe?

# feedback

Treatment: {action}. Improvement score: {score}%. {flavor}

# flavor

- high: (>70)
  - "The patient responds well. Visible improvement."
  - "Significant progress this week. The patient is hopeful."
- mid: (41-70)
  - "Moderate response. Some improvement noted."
  - "Partial improvement. The patient is stable but not thriving."
- low: (0-40)
  - "Poor response. The patient shows minimal improvement."
  - "The treatment seems ineffective this week. Concerning."
