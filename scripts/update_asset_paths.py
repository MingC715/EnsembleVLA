#!/usr/bin/env python3
"""Update RoboTwin asset config paths after linking or downloading assets."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PATH_LINE = re.compile(r"^(\s*(?:urdf_path|collision_spheres):\s*)([^#\s]+)(.*)$")


def normalize_path(value: str, assets_root: Path) -> str:
    if "/assets/" in value:
        suffix = value.split("/assets/", 1)[1]
        return str(assets_root / suffix)
    if value.startswith("assets/"):
        return str(assets_root / value[len("assets/") :])
    if value.startswith("./assets/"):
        return str(assets_root / value[len("./assets/") :])
    return value


def update_file(path: Path, assets_root: Path, dry_run: bool) -> bool:
    original = path.read_text()
    changed = False
    updated_lines: list[str] = []

    for line in original.splitlines(keepends=True):
        newline = ""
        body = line
        if line.endswith("\r\n"):
            body = line[:-2]
            newline = "\r\n"
        elif line.endswith("\n"):
            body = line[:-1]
            newline = "\n"

        match = PATH_LINE.match(body)
        if not match:
            updated_lines.append(line)
            continue

        prefix, value, suffix = match.groups()
        new_value = normalize_path(value, assets_root)
        if new_value != value:
            changed = True
        updated_lines.append(f"{prefix}{new_value}{suffix}{newline}")

    if changed and not dry_run:
        path.write_text("".join(updated_lines))
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print files that would be updated.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    assets_root = repo_root / "assets"

    if not assets_root.exists():
        print(f"Assets directory not found: {assets_root}")
        print("Download or link RoboTwin assets first, then rerun this script.")
        return 1

    config_files = sorted((assets_root / "embodiments").glob("**/curobo*.yml"))
    if not config_files:
        print(f"No curobo asset config files found under: {assets_root / 'embodiments'}")
        return 1

    changed_files = [path for path in config_files if update_file(path, assets_root, args.dry_run)]

    if args.dry_run:
        print(f"Files that would be updated: {len(changed_files)}")
    else:
        print(f"Updated files: {len(changed_files)}")

    for path in changed_files:
        print(path.relative_to(repo_root))

    return 0


if __name__ == "__main__":
    sys.exit(main())
