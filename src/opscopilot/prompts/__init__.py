"""Prompt loader. The prompts themselves are the .txt files in this folder."""
from importlib.resources import files


def load_prompt(name: str) -> str:
    return (files("opscopilot") / "prompts" / f"{name}.txt").read_text(encoding="utf-8").strip()
