"""Load and parse prompt template files.

Each .md file has YAML frontmatter and sections delimited by `# heading`.
Sections: system, action, feedback, flavor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


@dataclass
class FlavorEntry:
    label: str
    condition: str  # e.g. ">0.7", "8-10", "found"
    texts: list[str] = field(default_factory=list)


@dataclass
class PromptTemplate:
    env: str
    system: str
    action: str
    feedback: str
    flavors: list[FlavorEntry] = field(default_factory=list)

    def pick_flavor(self, value: float, rng, *, found: bool = False) -> str:
        """Pick a random flavor string matching the value."""
        for f in self.flavors:
            if _matches(f.condition, value, found=found):
                return rng.choice(f.texts)
        return ""


def _matches(condition: str, value: float, *, found: bool = False) -> bool:
    """Check if a value matches a flavor condition string."""
    condition = condition.strip()
    if condition == "found":
        return found

    # range: "8-10", "0.3-0.7"
    m = re.match(r'^\(?([\d.]+)\s*-\s*([\d.]+)\)?$', condition)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return lo <= value <= hi

    # comparison: ">0.7", "<=0", ">=80"
    m = re.match(r'^([<>]=?)\s*([\d.]+)$', condition)
    if m:
        op, num = m.group(1), float(m.group(2))
        if op == '>': return value > num
        if op == '>=': return value >= num
        if op == '<': return value < num
        if op == '<=': return value <= num

    # parenthesized comparison: (>0.7)
    m = re.match(r'^\(([<>]=?)\s*([\d.]+)\)$', condition)
    if m:
        op, num = m.group(1), float(m.group(2))
        if op == '>': return value > num
        if op == '>=': return value >= num
        if op == '<': return value < num
        if op == '<=': return value <= num

    return False


def load_prompt(env_name: str) -> PromptTemplate:
    """Load a prompt template by environment name."""
    path = PROMPTS_DIR / f"{env_name}.md"
    if not path.exists():
        raise FileNotFoundError(f"No prompt file for environment '{env_name}' at {path}")

    text = path.read_text()

    # strip YAML frontmatter
    text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)

    # split into sections by "# heading"
    sections: dict[str, str] = {}
    current_section = None
    current_lines: list[str] = []

    for line in text.split('\n'):
        m = re.match(r'^#\s+(\w+)\s*$', line)
        if m:
            if current_section:
                sections[current_section] = '\n'.join(current_lines).strip()
            current_section = m.group(1).lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_section:
        sections[current_section] = '\n'.join(current_lines).strip()

    # parse flavor section
    flavors = _parse_flavors(sections.get('flavor', ''))

    return PromptTemplate(
        env=env_name,
        system=sections.get('system', ''),
        action=sections.get('action', ''),
        feedback=sections.get('feedback', ''),
        flavors=flavors,
    )


def _parse_flavors(text: str) -> list[FlavorEntry]:
    """Parse the flavor section into structured entries.

    Format:
    - label: (condition)
      - "text 1"
      - "text 2"
    """
    flavors = []
    current: FlavorEntry | None = None

    for line in text.split('\n'):
        # top-level entry: "- high: (>0.7)"
        m = re.match(r'^-\s+(\w+):\s*(.+)$', line)
        if m:
            if current:
                flavors.append(current)
            label = m.group(1)
            condition = m.group(2).strip()
            current = FlavorEntry(label=label, condition=condition)
            continue

        # sub-entry: '  - "text"'
        m = re.match(r'^\s+-\s+"(.+)"$', line)
        if m and current:
            current.texts.append(m.group(1))

    if current:
        flavors.append(current)

    return flavors
