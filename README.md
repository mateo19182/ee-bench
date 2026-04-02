# ee-bench

Explore/Exploit benchmark for LLMs. Measures how well language models navigate the exploration vs. exploitation tradeoff across varied scenarios — from slot machines to treasure hunts.

## Why

LLMs default to over-exploitation (greedy behavior), and increasing temperature gives random variation, not directed exploration. This benchmark quantifies that gap across models, temperatures, and task types.

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

- **Cumulative regret** — gap between LLM's rewards and the theoretical optimum
- **Exploration ratio** — % of actions that deviate from the empirically best option over time
- **Adaptation speed** — how fast the LLM shifts strategy after environment changes (non-stationary only)

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

Edit `sweep_example.json` to configure models, temperatures, environments, and horizons, then:

```bash
uv run ee-bench sweep sweep_example.json
```

### Analyze results

```bash
# Full analysis
uv run python analyze.py results/sweep.json --all

# Specific analyses
uv run python analyze.py results/sweep.json --compare-models
uv run python analyze.py results/sweep.json --compare-temps
uv run python analyze.py results/sweep.json --regret-curves
uv run python analyze.py results/sweep.json --adaptation
```

## Research questions

- Do models adapt based on feedback, or follow fixed heuristics?
- Does temperature produce directed exploration or just noise?
- How does behavior vary across model families and sizes?
- Do non-stationary environments expose qualitatively different failures?
- Does the task framing (narrative) affect exploration strategy?

## Project structure

```
ee_bench/
├── environments/       # Bandit + search environments
├── providers/          # OpenRouter adapter
├── metrics/            # Regret, exploration ratio, adaptation speed
├── runner.py           # Experiment orchestration
├── config.py           # Experiment & sweep configs
└── cli.py              # CLI entry point
analyze.py              # Post-hoc analysis scripts
sweep_example.json      # Example sweep configuration
```
