# Tic-Tac-Toe (Reinforcement Learning)

This project supports both GUI and terminal modes, with train/load/save support for the Q-value model.

## Setup

From the repo root:

```bash
cd reinforcement_learning/tic_tac_toe
```

No external packages are required beyond standard Python (`tkinter` needed for GUI).

## Which file should I run?

### 1) Main launcher (recommended)

```bash
python game.py
```

- Opens the Tkinter GUI by default.

### 2) Terminal mode (CLI)

```bash
python game.py --cli
```

CLI menu options:
- Player vs CPU
- Watch CPU vs CPU
- Test CPU playing itself

### 3) Direct GUI entry point

```bash
python gui.py
```

## GUI workflow

1. Set `Gen1`, `Gen2`, and `Test` game counts.
2. Click `Train` to train/update Q-values.
3. Click `Save` to store a model (`.pkl`), or `Load` to open one.
4. Use `Play First (X)` or `Play Second (O)` to pick turn order.
5. Click `New Game` (or either play-order button, which also starts a fresh game).

Default model file in this folder:
- `tictactoe_qvalues.pkl`

## Files at a glance

- `game.py`: compatibility launcher (GUI by default, CLI with `--cli`)
- `gui.py`: Tkinter app for training + playing
- `game_logic.py`: board logic, Q-learning data, CLI, model serialization
- `tictactoe_qvalues.pkl`: saved model data
