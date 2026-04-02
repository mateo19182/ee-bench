# ee-bench

Explore/Exploit benchmark for LLMs. Measures how well language models navigate the exploration vs. exploitation tradeoff across varied scenarios — from slot machines to treasure hunts.

## Environments

### Bandits (stationary)
| Environment | Actions | Challenge |
|---|---|---|
| **Casino Slot Machines** | 6 machines | One jackpot, one high-variance trap |
| **Restaurant Picker** | 5 restaurants | Tourist trap with flashy name, hidden gem elsewhere |
| **Ocean Fishing** | 8 spots | Spatial correlation — nearby spots yield similarly |

### Bandits (non-stationary)
| Environment | Actions | Challenge |
|---|---|---|
| **Clinical Trial** | 4 treatments | Resistance buildup — cycling is optimal |
| **Venture Capitalist** | 5 sectors | Market regime shifts mid-run |

### Search / Optimization
| Environment | Space | Challenge |
|---|---|---|
| **Treasure Hunter** | 10×10 grid | Proximity hints (warmer/colder) |
| **Alchemy Lab** | 10³ combos | Smooth 3D surface, gradient-like reasoning |
| **Radio Tuner** | 0–100 dial | Decoy signal creates a local optimum trap |

## Metrics

Each run tracks the following metrics per episode. After a run completes, results, analysis, and graphs are saved to a timestamped directory (e.g. `results/sweep_20260402_143000/`).

### Per-step metrics

| Metric | What it measures | Better |
|---|---|---|
| **Cumulative regret curve** | Running sum of `(optimal_reward - actual_reward)` at each step. Shows how quickly the model converges on good actions. | Lower |
| **Exploration ratio curve** | Rolling-window fraction of actions that differ from the empirically best action so far. Tracks the explore/exploit balance over time. | Depends — should start high (exploring) and decay (exploiting) |

### Per-episode summary metrics

| Metric | What it measures | Better |
|---|---|---|
| **Total reward** | Sum of all rewards across the episode. | Higher |
| **Mean reward** | Average reward per step (`total_reward / total_steps`). Comparable across different horizon lengths. | Higher |
| **Total regret** | Final value of the cumulative regret curve — total opportunity cost vs. always picking optimally. | Lower |
| **Final exploration ratio** | Average exploration ratio over the last 10 steps. Shows whether the model settled on an action or kept wandering. | Lower (means the model exploited what it learned) |
| **Unique actions tried** | Count of distinct actions taken during the episode. | Depends — too few means the model never explored, too many means it never committed |
| **Adaptation speed** | Average number of steps to switch strategy after an environment shift. Only computed for non-stationary environments (Clinical Trial, Venture Capitalist). | Lower (faster adaptation) |
| **Adaptation events** | List of detected environment shifts and how the model responded. Each event records the shift step and steps until the model changed its action. | More events detected = longer horizon capturing more shifts |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an [OpenRouter](https://openrouter.ai/) API key.

```bash
# Install dependencies
uv sync

# Set your API key
export OPENROUTER_API_KEY="sk-or-..."
```

## Usage

### List environments

```bash
uv run ee-bench list
```

### Quick test

```bash
uv run python -m ee_bench.cli run --model qwen/qwen3.6-plus:free --env casino_slot_machines --horizons 20 --repetitions 1
```

### Single run

```bash
uv run python -m ee_bench.cli run \
  --model anthropic/claude-sonnet-4 \
  --env casino_slot_machines \
  --temperature 0.7 \
  --horizons 20 100 \
  --repetitions 3
```

### Verbosity levels

```bash
# Default: progress bars only
uv run python -m ee_bench.cli run --model ... --env ...

# Show each action + reward per step
uv run python -m ee_bench.cli -vv run --model ... --env ...

# Full debug: prompts sent to the model + raw LLM responses
uv run python -m ee_bench.cli -vvv run --model ... --env ...

# Silent: no output except errors
uv run python -m ee_bench.cli -q run --model ... --env ...
```

### Parameter sweep

Edit a sweep config in `sweeps/` to configure models, temperatures, environments, and horizons, then:

```bash
uv run ee-bench sweep sweeps/sweep_example.json
```

### Analyze results

Analysis and graphs are generated automatically after each run. To re-analyze an existing results file:

```bash
# Full analysis + graphs (saved to same directory as results file)
uv run ee-bench-analyze results/sweep_20260402_143000/results.json --all

# Specific analyses (printed to stdout only)
uv run ee-bench-analyze results/sweep_20260402_143000/results.json --compare-models
uv run ee-bench-analyze results/sweep_20260402_143000/results.json --compare-temps
uv run ee-bench-analyze results/sweep_20260402_143000/results.json --regret-curves
uv run ee-bench-analyze results/sweep_20260402_143000/results.json --adaptation
```

## Research questions

- Do models adapt based on feedback, or follow fixed heuristics?
- Does temperature produce directed exploration or just noise?
- How does behavior vary across model families and sizes?
- Do non-stationary environments expose qualitatively different failures?
- Does the task framing (narrative) affect exploration strategy?

### Run output

Each run creates a timestamped directory with everything in one place:

```
results/sweep_20260402_143000/
├── results.json                            # Raw results data
├── analysis.txt                            # Full text analysis
└── graphs/
    ├── regret_casino_slot_machines_h20.png  # Cumulative regret curves
    ├── exploration_casino_slot_machines_h20.png  # Exploration ratio over time
    ├── model_comparison.png                 # Reward & regret bar charts
    └── temperature_effects.png              # Temperature vs performance
```

## Project structure

```
ee_bench/
├── environments/       # Bandit + search environments
├── providers/          # OpenRouter adapter
├── metrics/            # Regret, exploration ratio, adaptation speed
├── analysis.py         # Analysis functions + CLI (ee-bench-analyze)
├── graphs.py           # Matplotlib graph generation
├── runner.py           # Experiment orchestration
├── config.py           # Experiment & sweep configs
└── cli.py              # CLI entry point (ee-bench)
sweeps/                 # Sweep configuration files
├── sweep_example.json
└── sweep_test.json
```

## Relevant Literature

- **[Should You Use Your Large Language Model to Explore or Exploit?](https://arxiv.org/abs/2502.00225)** (Harris & Slivkins, 2026) — Evaluates LLMs as separate exploration or exploitation oracles in contextual bandit tasks. Finds reasoning models show promise for exploitation with tool use, while LLMs help explore large action spaces with inherent semantics.

- **[Disentangling Exploration of Large Language Models by Optimal Exploitation](https://arxiv.org/abs/2501.08925)** (Grams et al., 2025) — Proposes measuring exploration via optimal exploitation oracles. Shows most LLMs struggle with state space exploration, with stronger exploration correlating to reasoning capabilities.

- **[Beyond the Exploration-Exploitation Trade-off: A Hidden State Approach for LLM Reasoning in RLVR](https://arxiv.org/abs/2509.23808)** (Huang et al., 2025) — Introduces VERL, which uses Effective Rank and its derivatives to decouple exploration and exploitation at the hidden-state level, achieving up to 21.4% accuracy improvement on reasoning benchmarks.

- **[Comparing Exploration-Exploitation Strategies of LLMs and Humans](https://arxiv.org/abs/2505.09901)** (Zhang et al., 2025) — Uses multi-armed bandit experiments to compare LLM and human E&E strategies. Finds thinking-enabled LLMs show more human-like behavior but struggle with directed exploration in non-stationary settings.

- **[Large Language Model-Enhanced Multi-Armed Bandits](https://arxiv.org/abs/2502.01118)** (Sun et al., 2025) — Proposes combining classical MAB algorithms (Thompson Sampling, SquareCB) with LLMs for reward prediction rather than direct arm selection, achieving better exploration-exploitation balance.

- **[Multi-Armed Bandits Meet Large Language Models](https://arxiv.org/abs/2505.13355)** (Bouneffouf & Feraud, 2025) — Survey exploring bidirectional synergy: bandit algorithms improve LLM training/prompt optimization, while LLMs enhance bandits through contextual understanding and natural language feedback.
