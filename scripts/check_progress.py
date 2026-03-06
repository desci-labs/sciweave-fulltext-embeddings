#!/usr/bin/env python3
"""Print pipeline progress statistics."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.state import PipelineState


def main():
    parser = argparse.ArgumentParser(description="Check pipeline progress")
    parser.add_argument(
        "--state-db", type=str, default="pipeline_state.db",
        help="Path to SQLite state database"
    )
    args = parser.parse_args()

    state = PipelineState(db_path=args.state_db)
    stats = state.get_stats()

    print("\n=== Pipeline Progress ===\n")
    for stage, count in sorted(stats.items()):
        if stage == "total":
            continue
        bar = "#" * min(count // 100, 50)
        print(f"  {stage:20s}: {count:>8,d}  {bar}")
    print(f"\n  {'TOTAL':20s}: {stats.get('total', 0):>8,d}")
    print()

    state.close()


if __name__ == "__main__":
    main()
