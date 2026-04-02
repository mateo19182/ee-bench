---
env: restaurant_picker
---

# system

You just moved to a new city and don't know any of the restaurants. Every evening you pick one place to eat dinner. After each meal you rate your experience from 1 to 10.

Available restaurants:
{restaurant_list}

Some places look fancy but disappoint. Others look unassuming but are incredible. The only way to know is to try them.

Your goal: have the best dining experiences possible over all your evenings out.

Each round, respond with ONLY the restaurant name. Nothing else.

# action

Your dining history:
{history}

Evening {round}: Where do you eat tonight?

# feedback

You dined at {action}. Your rating: {score}/10. {flavor}

# flavor

- high: (8-10)
  - "Absolutely divine. You lick the plate clean."
  - "The chef outdid themselves. A spiritual experience."
  - "You consider writing a love letter to the kitchen."
- mid: (5-7)
  - "Solid meal. Nothing to complain about."
  - "Decent. You'd come back if nothing else was open."
  - "Fine. The ambiance carried it."
- low: (1-4)
  - "Disappointing. You regret not cooking at home."
  - "The menu promised more than the kitchen delivered."
  - "You leave hungry and slightly offended."
