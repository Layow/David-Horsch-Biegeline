# David-Horsch-Biegeline
Mathematica formulas from David Horsch’s Diplomarbeit ported to Python.

## What’s Included
- `numeric_main.py`: Recommended numeric evaluation of bending energy (fast, robust).
- `main1.py`: Original fully symbolic approach (slow, may not finish).
- `symbolic_main_parallel.py`: Purely symbolic with expression simplifications and optional parallel term integration.
- Helper scripts: `first-run.sh` (Linux/macOS), `first-run.bat` (Windows).
- Runner scripts (Linux/macOS): `run_numeric_main.sh`, `run_symbolic_main_parallel.sh`, `run_main1.sh`.
- Runner scripts (Windows): `run_numeric_main.bat`, `run_symbolic_main_parallel.bat`, `run_main1.bat`.

## Quick Start — Automatic (recommended)
These scripts set up the virtual environment and install dependencies. Then use the runner scripts to execute.

- Linux/macOS
  - First time (setup): `chmod +x first-run.sh && ./first-run.sh`
  - Run: `./run_numeric_main.sh` (or see Runner Scripts below)

- Windows (Command Prompt)
  - First time (setup): `first-run.bat`
  - Run: `run_numeric_main.bat` (or see Runner Scripts below)

## Runner Scripts
- Linux/macOS:
  - Numeric: `./run_numeric_main.sh`
  - Symbolic optimized: `./run_symbolic_main_parallel.sh --print-stats --parallel`
  - Original symbolic: `./run_main1.sh`

- Windows (Command Prompt):
  - Numeric: `run_numeric_main.bat`
  - Symbolic optimized: `run_symbolic_main_parallel.bat --print-stats --parallel`
  - Original symbolic: `run_main1.bat`

  ### Automatic Requirements
- Python 3.x with virtualenv support


## Manual Setup (alternative)
- Linux/macOS
  - First time: `python3 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -r requirements.txt`
  - Run numeric: `source .venv/bin/activate && python3 numeric_main.py`
  - Run symbolic optimized: `source .venv/bin/activate && ./run_symbolic_main_parallel.sh [--print-stats] [--parallel]`
  - Run original symbolic: `source .venv/bin/activate && ./run_main1.sh`

- Windows
  - First time: `py -3 -m venv .venv` (or `python -m venv .venv`), then `.\\.venv\Scripts\activate`, then `python -m pip install -U pip && python -m pip install -r requirements.txt`
  - Run numeric: `run_numeric_main.bat` (or `python numeric_main.py`)
  - Run symbolic optimized: `run_symbolic_main_parallel.bat [--print-stats] [--parallel]`
  - Run original symbolic: `run_main1.bat`

  ### Manual Requirements
- Python 3.x with virtualenv support
- Install via `pip install -r requirements.txt` (SymPy, mpmath, NumPy)

Notes
- The numeric method integrates with high precision (mpmath) and avoids the singularity at `r=0` by starting from a tiny epsilon.
