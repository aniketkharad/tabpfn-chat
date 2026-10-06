"""Skill recipes registry for tabchat."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

RECIPES_DIR = Path(__file__).parent / "recipes"


def _parse_yaml_frontmatter(file_content: str) -> tuple[dict[str, Any], str]:
    """Extract and parse YAML frontmatter and markdown body without external dependencies."""
    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(pattern, file_content, re.DOTALL)
    if not match:
        return {}, file_content

    frontmatter_raw, body = match.groups()
    metadata: dict[str, Any] = {}

    for line in frontmatter_raw.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            # Parse simple list [a, b, c]
            if val.startswith("[") and val.endswith("]"):
                items = [item.strip().strip("'\"") for item in val[1:-1].split(",") if item.strip()]
                metadata[key] = items
            # Parse quoted or plain string
            elif (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                metadata[key] = val[1:-1]
            else:
                metadata[key] = val

    return metadata, body


class SkillRegistry:
    """Registry maintaining skill recipes metadata and providing on-demand content retrieval."""

    def __init__(self, recipes_dir: Path | None = None) -> None:
        self.recipes_dir = recipes_dir or RECIPES_DIR
        self._skills: dict[str, dict[str, Any]] = {}
        self._file_paths: dict[str, Path] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Scan recipes directory and load ONLY frontmatter metadata on boot."""
        self._skills.clear()
        self._file_paths.clear()

        if not self.recipes_dir.exists():
            return

        for path in sorted(self.recipes_dir.glob("*.md")):
            content = path.read_text(encoding="utf-8")
            meta, _ = _parse_yaml_frontmatter(content)
            skill_name = meta.get("name", path.stem)
            self._skills[skill_name] = {
                "name": skill_name,
                "description": meta.get("description", ""),
                "tags": meta.get("tags", []),
            }
            self._file_paths[skill_name] = path

    def get_skills_index(self) -> str:
        """Format names and descriptions into a compact string (< 150 tokens) for system prompts."""
        lines = ["Available Modeling Recipes:"]
        for name, meta in sorted(self._skills.items()):
            desc = meta.get("description", "")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def get_skill_content(self, name: str) -> str:
        """Load full markdown text on demand."""
        clean_name = name.removesuffix(".md")
        if clean_name not in self._file_paths:
            raise KeyError(
                f"Skill '{name}' not found. Available skills: {sorted(self._skills.keys())}"
            )
        path = self._file_paths[clean_name]
        return path.read_text(encoding="utf-8")

    def list_skills(self) -> list[dict[str, Any]]:
        """Return list of skill metadata summaries."""
        return list(self._skills.values())


# Global singleton registry instance
_registry: SkillRegistry | None = None


def get_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def get_skills_index() -> str:
    """Format skill names and descriptions into a compact string (< 150 tokens) for system prompts."""
    return get_registry().get_skills_index()


def get_skill_content(name: str) -> str:
    """Load full markdown text for the given skill recipe on demand."""
    return get_registry().get_skill_content(name)
