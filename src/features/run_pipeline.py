from __future__ import annotations

from pathlib import Path

from .pipeline import build_feature_table, build_windows


def main() -> None:
    build_windows()
    build_feature_table()
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
