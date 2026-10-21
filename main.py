"""Run the Koopman experiments.

Examples
--------
    python main.py                       # pendulum + Lorenz, full settings
    python main.py --system lorenz       # only the Lorenz experiment
    python main.py --quick               # tiny smoke-test run (seconds, not minutes)

Figures are written to ``<out>/figures`` and metrics to ``<out>/*_metrics.json``.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved, never shown

from experiments import lorenz, pendulum  # noqa: E402

SYSTEMS = {"pendulum": pendulum, "lorenz": lorenz}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=[*SYSTEMS, "all"], default="all")
    parser.add_argument("--out", default="results", help="output directory (default: results)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="tiny settings for a fast smoke test")
    args = parser.parse_args(argv)

    chosen = SYSTEMS if args.system == "all" else {args.system: SYSTEMS[args.system]}
    for name, module in chosen.items():
        print(f"\n===== {name} =====")
        module.run(out_dir=args.out, quick=args.quick, seed=args.seed)
    print(f"\nDone. Figures and metrics are in '{args.out}/'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
