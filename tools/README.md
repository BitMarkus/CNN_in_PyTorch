# Developer Tools

This folder contains utility scripts for repository maintenance.
These scripts are **not part of the CNN or diffusion pipelines** and are not
required to run any of the main functionality.

---

## manage_license_headers.py

Adds or removes SPDX license headers on all Python source files in the repository.

### Usage

1. Open the script in VS Code
2. Edit the configuration block at the top:
   - Set `ACTION = "add"` to prepend headers, or `"remove"` to strip them
   - Set `DRY_RUN = True` to preview changes, or `False` to apply them
3. Run the script (F5 or the Run button in VS Code)

### Behavior

- Scans all folders listed in `TARGET_FOLDERS`
- Skips folders listed in `SKIP_FOLDERS` (e.g., `output/`, `input/`, `__pycache__/`)
- Skips itself automatically
- Preserves line endings (`\r\n` on Windows, `\n` on Linux/macOS)
- Safe to run multiple times — the add operation is idempotent

### Header Format

```python
# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT
```

The `remove` action only strips the header if all three lines are present at
the top of the file, preventing accidental removal of unrelated content.

### Configuration Reference

| Variable | Purpose |
|----------|---------|
| `ACTION` | `"add"` or `"remove"` |
| `DRY_RUN` | `True` to preview, `False` to apply |
| `HEADER_LINES` | The three comment lines to prepend |
| `HEADER_MARKER` | String used to detect existing headers |
| `TARGET_FOLDERS` | Folders scanned for `.py` files |
| `SKIP_FOLDERS` | Folders to ignore during scanning |
| `SKIP_FILENAMES` | Files to ignore by name |