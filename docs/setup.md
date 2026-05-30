# Setup

This repository is configured to work with a local editable checkout of AdaptiveResonanceLib.

## Python Version

AdaptiveResonanceLib declares support for Python 3.9 and newer. This workspace was verified with Python 3.11.9.

## Virtual Environment

Create a local environment inside `artlib-studio`:

```bash
cd /Users/ognjencadovski/science/projects/art-exploration/artlib-studio
python3 -m venv .venv
source .venv/bin/activate
python --version
```

## Install Dependencies

Upgrade packaging tools and install AdaptiveResonanceLib in editable mode from the sibling checkout:

```bash
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ../AdaptiveResonanceLib
```

If you prefer to run from the workspace root, the equivalent editable install is:

```bash
pip install -e ./AdaptiveResonanceLib
```

## Run Examples

Run the minimal Fuzzy ART verification example:

```bash
source .venv/bin/activate
python examples/minimal_fuzzy_art.py
```

Run the environment check:

```bash
source .venv/bin/activate
python scripts/check_environment.py
```