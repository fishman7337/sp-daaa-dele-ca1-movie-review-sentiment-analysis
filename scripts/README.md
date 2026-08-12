# Scripts

This folder contains maintenance and automation scripts.

- `split_notebook.py` regenerates chapter notebooks from the original notebook; `--check` verifies that committed chapters are current.
- `validate_project.py` checks that key repository paths exist.

Scripts should be lightweight and safe to run in CI.
