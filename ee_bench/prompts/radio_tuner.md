---
env: radio_tuner
---

# system

You are tuning an old analog radio. The dial goes from 0 to 100. Somewhere on the dial is a station broadcasting a signal.

Each round you set the dial to a position and read the signal strength (0-100%). There's static, so readings fluctuate a bit.

WARNING: There might be more than one signal source. A weaker station could mislead you if you settle on it too early.

Your goal: lock onto the strongest possible signal.

Each round, respond with ONLY a number between 0 and 100 (can be a decimal, e.g., "42.5"). Nothing else.

# action

Tuning log:
{history}

Tune #{round}: Where do you set the dial?

# feedback

Dial position: {freq}. Signal strength: {signal}%. {flavor}

# flavor

- high: (>85)
  - "Crystal clear! You can hear every note."
  - "Perfect reception. The music fills the room."
- mid_high: (61-85)
  - "Decent reception. Music comes through with some crackle."
  - "Getting there. The voice is intelligible through the static."
- mid: (31-60)
  - "Faint. You can tell something's there but can't make it out."
  - "Ghostly whispers in the static. A signal, maybe."
- low: (0-30)
  - "Just static and white noise."
  - "Nothing but the hiss of empty airwaves."
