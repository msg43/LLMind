#!/usr/bin/env python3
"""
Automatic patch version bump for LLMind.

- Increments app_version in config.py (x.y.z -> x.y.(z+1))
- Updates static/js/app.js occurrences of app_version: 'x.y.z'

Intended to be invoked from a git pre-commit hook.
This script is idempotent per run and will only modify files when needed.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.py"
APP_JS_PATH = PROJECT_ROOT / "static" / "js" / "app.js"


def bump_patch(version: str) -> str:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise ValueError(f"Invalid version string: {version}")
    major, minor, patch = map(int, match.groups())
    return f"{major}.{minor}.{patch + 1}"


def update_config_version() -> tuple[bool, str, str]:
    content = CONFIG_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r'app_version:\s*str\s*=\s*"(\d+\.\d+\.\d+)"')
    m = pattern.search(content)
    if not m:
        return False, "", ""
    current = m.group(1)
    new_version = bump_patch(current)
    new_content = pattern.sub(lambda _m: _m.group(0).replace(current, new_version), content, count=1)
    if new_content != content:
        CONFIG_PATH.write_text(new_content, encoding="utf-8")
        return True, current, new_version
    return False, current, current


def update_app_js_version(new_version: str) -> bool:
    if not APP_JS_PATH.exists():
        return False
    content = APP_JS_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r"app_version:\s*'(?P<v>\d+\.\d+\.\d+)'")
    m = pattern.search(content)
    if not m:
        return False
    current = m.group("v")
    if current == new_version:
        return False
    new_content = pattern.sub(f"app_version: '{new_version}'", content)
    APP_JS_PATH.write_text(new_content, encoding="utf-8")
    return True


def main() -> int:
    if not CONFIG_PATH.exists():
        print("config.py not found; skip version bump")
        return 0

    changed, old_v, new_v = update_config_version()
    if not changed:
        print("Version unchanged; skip")
        return 0

    # Try to reflect in app.js if present
    js_changed = update_app_js_version(new_v)

    # Stage files if running inside git repo (best-effort)
    try:
        import subprocess

        paths = [str(CONFIG_PATH)]
        if js_changed:
            paths.append(str(APP_JS_PATH))
        subprocess.run(["git", "add", *paths], cwd=str(PROJECT_ROOT), check=False)
    except Exception:
        pass

    print(f"Version bumped: {old_v} -> {new_v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


