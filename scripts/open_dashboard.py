"""Build and open the local HTML dashboard in the default browser."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the local NBA Kalshi dashboard.")
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "data" / "reports"))
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--no-rebuild", action="store_true", help="Open the existing HTML file without rebuilding it.")
    return parser.parse_args()


def _build_dashboard(reports_dir: Path, output_path: Path) -> None:
    from reports.dashboard import write_dashboard

    write_dashboard(reports_dir, output_path)


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    dashboard_path = Path(args.output_path) if args.output_path else reports_dir / "dashboard.html"

    if not args.no_rebuild:
        try:
            _build_dashboard(reports_dir, dashboard_path)
        except Exception as exc:
            if not dashboard_path.exists():
                raise SystemExit(f"Could not build dashboard: {type(exc).__name__}: {exc}") from exc
            print(f"Opening existing dashboard because rebuild failed: {type(exc).__name__}: {exc}")

    if not dashboard_path.exists():
        raise SystemExit(f"Dashboard file does not exist yet: {dashboard_path}")

    dashboard_uri = dashboard_path.resolve().as_uri()
    opened = webbrowser.open(dashboard_uri)
    print(f"Dashboard: {dashboard_path.resolve()}")
    print(f"Opened in browser: {'yes' if opened else 'requested'}")


if __name__ == "__main__":
    main()
