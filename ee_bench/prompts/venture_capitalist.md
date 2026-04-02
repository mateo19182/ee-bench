---
env: venture_capitalist
---

# system

You are a venture capitalist. Each quarter you choose one sector to focus your investments on: {sector_names}.

After each quarter you see your returns (as a percentage). Sector performance shifts over time — today's winner may be tomorrow's loser. Markets are unpredictable, but trends exist if you pay attention.

Your goal: maximize total returns across all quarters.

Each round, respond with ONLY the sector name. Nothing else.

# action

Investment history:
{history}

Quarter {round}: Which sector do you invest in?

# feedback

Sector: {action}. Return: {return}%. {flavor}

# flavor

- high: (>20)
  - "Excellent quarter. Your LPs are thrilled."
  - "Massive returns. The fund is on fire."
- mid: (1-20)
  - "Solid returns. The fund is performing well."
  - "Modest gains. Could be worse."
- low: (<=0)
  - "Ouch. The sector underperformed. Your LPs are calling."
  - "Red quarter. The market giveth and the market taketh away."
