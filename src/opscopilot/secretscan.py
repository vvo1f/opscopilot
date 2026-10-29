"""Look for API keys and other secrets before pushing to a public repository."""
import re
from pathlib import PurePath

KEY_PATTERNS = {
    "Google API key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "Groq key": re.compile(r"gsk_[0-9A-Za-z]{20,}"),
    "OpenAI-style key": re.compile(r"sk-[0-9A-Za-z]{20,}"),
}
# NAME = value, where the name looks secret and the value is not an obvious placeholder
ASSIGNMENT = re.compile(r"""(?i)\b\w*(?:api_key|secret|token|password)\w*\s*[:=]\s*['"]?([^\s'"#]{8,})""")
# "NAME = value" is only checked in config-like files; in source code it matches
# harmless things like api_key=api_key
CONFIG_SUFFIXES = {".yml", ".yaml", ".toml", ".ini", ".cfg", ".json", ".txt", ".md", ".sh", ".ps1"}
PLACEHOLDERS = {"put-your-key-here", "change_me", "opscopilot", "geo_admin", "ollama"}


def scan_text(name: str, text: str, keys_only: bool = False) -> list[str]:
    problems = []
    for label, pattern in KEY_PATTERNS.items():
        if pattern.search(text):
            problems.append(f"{name}: looks like a {label}")
    path = PurePath(name)
    is_config = path.name.startswith(".env") or path.suffix in CONFIG_SUFFIXES
    if is_config and not keys_only:
        for match in ASSIGNMENT.finditer(text):
            value = match.group(1)
            if value.lower() not in PLACEHOLDERS and not value.startswith(("<", "$", "{")):
                problems.append(f"{name}: secret-looking assignment ({match.group(0)[:40]}...)")
    return problems
