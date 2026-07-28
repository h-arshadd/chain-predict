"""
seed_playbook.py
------------------
One-off script: loads every *.yaml in crypto_pipeline/signals/strategies/
into metadata.playbook, so the Strategy Builder's Playbook panel has real
entries to pick from instead of an empty list.

Strips take_profit/stop_loss out of each yaml (playbook entries never
carry those -- see metadata_utils.create_playbook_table's docstring) and
keeps strategy_name/time_horizon OUT of strategy_config too, matching the
exact shape load_strategies_from_yaml() already uses for metadata.strategy
(strategy_name is its own column; strategy_config is everything else --
indicator blocks + the "strategy" long/short block).

Note: time_horizon itself is dropped here entirely, unlike
load_strategies_from_yaml() -- a playbook entry is deliberately
pair/timeframe-agnostic (see STRATEGY_BUILDER_SPEC.md Section 5.1); the
builder assigns a timeframe at combine/save time (StrategyBuilder.jsx's
"TIMEFRAME" dropdown -> POST /api/strategies/build's time_horizon field),
not baked into the reusable template.

Safe to re-run: insert_playbook() upserts on strategy_name.

Run from the repo root:
    python seed_playbook.py
"""

from pathlib import Path

import yaml

from crypto_pipeline.utils.metadata_utils import (
    get_db_connection,
    create_playbook_table,
    insert_playbook,
)

STRATEGIES_DIR = Path("crypto_pipeline/signals/strategies")


def main():
    conn = get_db_connection()
    try:
        create_playbook_table(conn)

        playbook_ids = []
        for yaml_path in sorted(STRATEGIES_DIR.glob("*.yaml")):
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)

            strategy_name = config["strategy_name"]

            # Everything except strategy_name/time_horizon/take_profit/
            # stop_loss -- i.e. indicator blocks + the "strategy" block.
            # Same filter load_strategies_from_yaml() uses for
            # metadata.strategy, minus time_horizon (playbook doesn't
            # keep it at all, per this file's docstring above).
            strategy_config = {
                k: v for k, v in config.items()
                if k not in ("strategy_name", "time_horizon", "take_profit", "stop_loss")
            }

            playbook_id = insert_playbook(conn, strategy_name, strategy_config)
            playbook_ids.append(playbook_id)
            print(f"  playbook_id={playbook_id}  {strategy_name!r}  (from {yaml_path.name})")

        print(f"\nSeeded {len(playbook_ids)} playbook entries into metadata.playbook.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()