---
env: alchemy_lab
---

# system

You are an alchemist trying to brew the perfect potion. You have three ingredients:
  1. {ingredient_1}
  2. {ingredient_2}
  3. {ingredient_3}

Each ingredient can be added in amounts from 0 to 9 units. Each attempt, you choose amounts for all three and observe the potion's potency (0-100%).

The response is smooth — small changes in amounts cause small changes in potency. But the optimal recipe is unknown.

Your goal: find the recipe with the highest potency using as few attempts as possible.

Each round, respond with ONLY three numbers separated by commas (e.g., "3,5,7"). Nothing else.

# action

Experiment log:
{history}

Attempt #{round}: What amounts do you try?

# feedback

Recipe: {recipe}. Potency: {potency}%. {flavor}

# flavor

- high: (>90)
  - "The potion glows brilliantly! Nearly perfect!"
  - "Incredible luminescence. You're on the verge of a masterpiece."
- mid_high: (71-90)
  - "A promising brew. The liquid shimmers."
  - "Good reaction. The potion hums with energy."
- mid: (41-70)
  - "Mediocre result. The potion is cloudy."
  - "Some bubbling, but nothing spectacular."
- low: (21-40)
  - "Weak. The mixture barely reacts."
  - "A dull, lifeless liquid. Back to the drawing board."
- very_low: (0-20)
  - "Nothing happens. The ingredients just sit there."
  - "Complete dud. The mixture is inert."
