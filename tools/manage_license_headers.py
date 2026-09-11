# ===== Standard Library Imports =====
from pathlib import Path
from typing import List


#############################################################################################################
# CONFIGURATION
#
# Edit the values in this block to control the behavior of the script.
# Then run the script from VS Code as usual.

# Action to perform:
#   "add"    -> prepend the license header to all Python files that do not have it
#   "remove" -> strip the license header from all Python files that have it
ACTION = "add"

# If True, only report which files would be modified without actually modifying them.
# Recommended: Set to True first, review the output in the VS Code terminal,
# then set to False and run again to apply the changes.
DRY_RUN = False

# Header lines to prepend (must be a list of comment lines starting with "#")
HEADER_LINES = [
    "# Diffusion-Based Phenotypic Extrapolation",
    "# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>",
    "# SPDX-License-Identifier: MIT",
]

# Marker used to detect whether a file already has the header
HEADER_MARKER = "SPDX-License-Identifier: MIT"

# Folders to scan (relative to the project root)
TARGET_FOLDERS = [
    ".",
    "single_training",
    "cross_validation",
    "analysis",
    "utilities",
    "preprocessing",
    "export_plotting",
    "diffusion",
]

# Folder names to skip entirely (anywhere in the tree)
SKIP_FOLDERS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".vscode",
    ".idea",
    "output",
    "input",
    "data",
    "checkpoints",
    "dataset_gen",
}

# Filenames to skip (this script itself)
SKIP_FILENAMES = {
    Path(__file__).name,
}

#############################################################################################################


# Check if a file starts with the header marker.
# Args:
#   filepath (Path): Path to the file to check
# Returns:
#   bool: True if the header is present
def has_header(filepath: Path) -> bool:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            first_lines = "".join(f.readline() for _ in range(10))
        return HEADER_MARKER in first_lines
    except (UnicodeDecodeError, OSError):
        return False


# Read the original content of a file, preserving line endings.
# Args:
#   filepath (Path): Path to the file
# Returns:
#   tuple: (content, line_ending)
def read_file_preserving_line_endings(filepath: Path):
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        content = f.read()

    if "\r\n" in content:
        line_ending = "\r\n"
    elif "\r" in content:
        line_ending = "\r"
    else:
        line_ending = "\n"

    return content, line_ending


# Write content back to a file, preserving the line ending.
# Args:
#   filepath (Path): Path to the file
#   content (str): Content to write
def write_file(filepath: Path, content: str) -> None:
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        f.write(content)


# Build the header string using the given line ending.
# Args:
#   line_ending (str): Line ending to use
# Returns:
#   str: The formatted header followed by a blank line
def build_header(line_ending: str) -> str:
    header = line_ending.join(HEADER_LINES)
    return header + line_ending + line_ending


# Find the project root.
#
# If this script lives in a "tools/" folder, the project root is the parent.
# Otherwise the script is assumed to be in the project root.
# Returns:
#   Path: Project root directory
def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    if script_path.parent.name == "tools":
        return script_path.parent.parent
    return script_path.parent


# Find all Python files in the target folders.
# Args:
#   project_root (Path): Root directory of the project
# Returns:
#   list: Sorted list of Path objects
def find_python_files(project_root: Path) -> List[Path]:
    files = []

    for folder_name in TARGET_FOLDERS:
        folder = project_root / folder_name

        if not folder.exists():
            print(f"Warning: Folder not found, skipping: {folder}")
            continue

        for path in folder.rglob("*.py"):
            if any(part in SKIP_FOLDERS for part in path.parts):
                continue
            if path.name in SKIP_FILENAMES:
                continue
            files.append(path)

    return sorted(set(files))


# Add the license header to a single file.
# Args:
#   filepath (Path): Path to the file
# Returns:
#   str: Status message: "added", "skipped", or "error"
def add_header(filepath: Path) -> str:
    if has_header(filepath):
        return "skipped"

    try:
        content, line_ending = read_file_preserving_line_endings(filepath)
        header = build_header(line_ending)
        new_content = header + content

        if not DRY_RUN:
            write_file(filepath, new_content)

        return "added"

    except Exception as e:
        print(f"  ERROR processing {filepath}: {e}")
        return "error"


# Remove the license header from a single file.
#
# Only removes the header if all three expected lines are present at the top
# of the file. Otherwise the file is left untouched.
# Args:
#   filepath (Path): Path to the file
# Returns:
#   str: Status message: "removed", "skipped", or "error"
def remove_header(filepath: Path) -> str:
    try:
        content, line_ending = read_file_preserving_line_endings(filepath)
        lines = content.split(line_ending)

        if len(lines) < 3:
            return "skipped"

        if (
            "Diffusion-Based Phenotypic Extrapolation" not in lines[0]
            or "Copyright (C)" not in lines[1]
            or HEADER_MARKER not in lines[2]
        ):
            return "skipped"

        lines_to_remove = 3
        if len(lines) > 3 and lines[3].strip() == "":
            lines_to_remove = 4

        new_lines = lines[lines_to_remove:]
        new_content = line_ending.join(new_lines)

        if not DRY_RUN:
            write_file(filepath, new_content)

        return "removed"

    except Exception as e:
        print(f"  ERROR processing {filepath}: {e}")
        return "error"


#############################################################################################################
# MAIN

# Main entry point for the license header manager.
def main() -> None:
    project_root = find_project_root()

    print(f"Project root: {project_root}")
    print(f"Action:       {ACTION.upper()}")
    print(f"Dry run:      {DRY_RUN}")
    print("=" * 70)

    files = find_python_files(project_root)

    if not files:
        print("No Python files found.")
        return

    print(f"Found {len(files)} Python files.")
    print("=" * 70)

    changed = 0
    skipped = 0
    errors = 0

    for filepath in files:
        rel_path = filepath.relative_to(project_root)

        if ACTION == "add":
            status = add_header(filepath)
            action_label = "ADDED  "
        elif ACTION == "remove":
            status = remove_header(filepath)
            action_label = "REMOVED"
        else:
            print(f"Unknown ACTION: {ACTION}. Must be 'add' or 'remove'.")
            return

        if status in ("added", "removed"):
            changed += 1
            print(f"  [{action_label}] {rel_path}")
        elif status == "skipped":
            skipped += 1
            print(f"  [SKIPPED] {rel_path}")
        elif status == "error":
            errors += 1

    print("=" * 70)
    print(f"Summary: {changed} changed, {skipped} skipped, {errors} errors.")

    if DRY_RUN and changed > 0:
        print("\nThis was a dry run. Set DRY_RUN = False at the top of the script to apply the changes.")


if __name__ == "__main__":
    main()