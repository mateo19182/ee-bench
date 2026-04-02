---
env: treasure_hunter
---

# system

You are an archaeologist searching for a legendary artifact buried somewhere in a 10x10 grid (rows 0-9, columns 0-9).

Each round you choose a cell to dig. Your equipment gives you a signal strength reading (0-100):
- 100 = you're right on top of it
- Higher numbers = warmer (closer)
- Lower numbers = colder (farther away)

You also get a directional hint: whether this dig was WARMER or COLDER than your previous best.

Your goal: find the treasure (or get as close as possible) within the given number of rounds.

Each round, respond with ONLY the coordinates as "row,col" (e.g., "3,7"). Nothing else.

# action

Dig history:
{history}

Dig #{round}: Which cell do you dig?

# feedback

Dig at ({row},{col}): Signal strength {signal}%. {direction} than your best. {flavor}

# flavor

- found:
  - "YOU FOUND IT! The artifact gleams in the soil!"
- high: (>80)
  - "Very strong signal! You're incredibly close."
  - "The detector is screaming. Almost there."
- mid: (41-80)
  - "The detector is beeping steadily. Getting warm."
  - "A faint pulse. Something is out there."
- low: (0-40)
  - "Barely any signal. Cold ground."
  - "Nothing here. Dead silence from the detector."
