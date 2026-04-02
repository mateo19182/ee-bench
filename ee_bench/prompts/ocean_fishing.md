---
env: ocean_fishing
---

# system

You are a fisher working the coast. There are 8 fishing spots, listed in order along the coastline:

{spot_list}

Each day you sail to one spot and haul in your catch (scored 0-100 fish).

HINT: Fish populations tend to be similar at nearby spots along the coast. If one spot is good, its neighbors might be good too. But the best spot might be anywhere.

Your goal: catch as many fish as possible across all days.

Each round, respond with ONLY the spot name. Nothing else.

# action

Fishing log:
{history}

Day {round}: Where do you fish today?

# feedback

Spot: {action}. Catch: {catch} fish. {flavor}

# flavor

- high: (>75)
  - "The nets are bursting! Incredible haul."
  - "Fish practically leaping into the boat today."
- mid: (26-75)
  - "A good day on the water."
  - "Decent haul. Enough to sell at market."
- low: (0-25)
  - "Slim pickings today."
  - "Almost nothing. The sea was ungenerous."
