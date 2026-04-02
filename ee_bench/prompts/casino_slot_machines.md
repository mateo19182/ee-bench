---
env: casino_slot_machines
---

# system

You are in an underground casino with 6 slot machines: {machine_names}.

Each machine has a different hidden payout profile. Some pay well consistently, others are unreliable, and some are just bad. You have no idea which is which — you have to figure it out by playing.

Each round you pull one machine's lever and receive a payout between $0.00 and $1.00.

Your goal: maximize your total earnings over all rounds.

IMPORTANT: Each round, respond with ONLY the name of the machine you want to play. Nothing else.

# action

History of your last plays:
{history}

Round {round}: Which machine do you play?

# feedback

You pulled {action} and won ${reward:.2f}.

# flavor

- high: (>0.7)
  - "Jackpot vibes! The machine whirs and spits out a fat stack."
  - "Ka-ching! A very generous pull."
- mid: (0.3-0.7)
  - "A modest win. The machine blinks indifferently."
  - "Not bad, not great. The coins trickle out."
- low: (<0.3)
  - "The machine swallows your coin and gives almost nothing back."
  - "A sad, tinny clink. Barely worth the pull."
