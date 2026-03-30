from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


class WorkflowError(RuntimeError):
    pass

def _run(cmd: Sequence[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    printable = " ".join(str(x) for x in cmd)
    print(f"\n[run] {printable}")
    print(f"[cwd] {cwd}")
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise WorkflowError(f"Command failed with exit code {result.returncode}: {printable}")


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _prepare_r_input(project_root: Path, source_data: Path) -> tuple[Path, bool]:
    """
    GARCHX.r currently hardcodes data/model_data.csv relative to the working directory.
    To keep that script unchanged, ensure that file exists before invoking R.

    Returns
    -------
    (target_path, created_by_pipeline)
    """
    target = project_root / "data" / "model_data.csv"
    target.parent.mkdir(parents=True, exist_ok=True)

    # Reuse the existing file if the user is already pointing at the hardcoded location.
    try:
        if source_data.resolve() == target.resolve():
            return target, False
    except FileNotFoundError:
        pass

    if target.exists():
        # Keep the hardcoded location intact rather than overwriting an unrelated file.
        raise WorkflowError(
            "GARCHX.r expects data/model_data.csv, but that file already exists and is different "
            f"from the requested --data path. Existing: {target}, requested: {source_data}"
        )

    try:
        os.symlink(source_data, target)
        print(f"[info] Created symlink for GARCHX.r input: {target} -> {source_data}")
    except OSError:
        shutil.copy2(source_data, target)
        print(f"[info] Copied input for GARCHX.r to hardcoded path: {target}")

    return target, True


def _cleanup_r_input(path: Path, created_by_pipeline: bool) -> None:
    if not created_by_pipeline:
        return
    try:
        if path.is_symlink() or path.exists():
            path.unlink()
            print(f"[info] Removed temporary GARCHX.r input: {path}")
    except OSError as exc:
        print(f"[warn] Failed to remove temporary GARCHX.r input {path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full VaR workflow: base runner + base plots, extension runner + extension plots, RiskMetrics, and GARCHX.r"
    )
    parser.add_argument("--project-root", default=None, help="Directory containing the existing scripts. Defaults to the directory of this pipeline script.")
    parser.add_argument("--data", default="data/model_data.csv", help="Path to model_data.csv for the Python runners.")
    parser.add_argument("--outputs-root", default="outputs", help="Root output directory for the Python scripts.")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--initial-train", type=float, default=0.7)
    parser.add_argument("--hs-window", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--rscript", default="Rscript", help="Rscript executable to use for GARCHX.r.")
    parser.add_argument("--skip-garchx", action="store_true", help="Skip the GARCHX.r step.")
    args = parser.parse_args()

    pipeline_path = Path(__file__).resolve()
    project_root = Path(args.project_root).resolve() if args.project_root else pipeline_path.parent.resolve()
    data_path = Path(args.data).expanduser().resolve()
    outputs_root = Path(args.outputs_root)
    if not outputs_root.is_absolute():
        outputs_root = (project_root / outputs_root).resolve()

    _ensure_file(data_path, "Input data")

    script_names = [
        "run_var_project.py",
        "plot_var_results.py",
        "run_var_ext.py",
        "plot_var_ext.py",
        "run_var_riskmetrics.py",
    ]
    if not args.skip_garchx:
        script_names.append("GARCHX.r")

    for name in script_names:
        _ensure_file(project_root / name, f"Required script {name}")

    outputs_root.mkdir(parents=True, exist_ok=True)

    python_cmd = args.python
    rscript_cmd = args.rscript

    # 1) Base models
    base_cmd = [
        python_cmd,
        str(project_root / "run_var_project.py"),
        "--data", str(data_path),
        "--outdir", str(outputs_root),
        "--alpha", str(args.alpha),
        "--initial-train", str(args.initial_train),
        "--hs-window", str(args.hs_window),
    ]
    if args.max_steps is not None:
        base_cmd.extend(["--max-steps", str(args.max_steps)])
    _run(base_cmd, cwd=project_root)

    plot_base_cmd = [
        python_cmd,
        str(project_root / "plot_var_results.py"),
        "--forecasts", str(outputs_root / "var_forecasts.csv"),
        "--outdir", str(outputs_root / "plots"),
    ]
    _run(plot_base_cmd, cwd=project_root)

    # 2) Extension models
    ext_cmd = [
        python_cmd,
        str(project_root / "run_var_ext.py"),
        "--data", str(data_path),
        "--outdir", str(outputs_root),
        "--alpha", str(args.alpha),
        "--initial-train", str(args.initial_train),
    ]
    if args.max_steps is not None:
        ext_cmd.extend(["--max-steps", str(args.max_steps)])
    _run(ext_cmd, cwd=project_root)

    plot_ext_cmd = [
        python_cmd,
        str(project_root / "plot_var_ext.py"),
        "--forecasts", str(outputs_root / "var_forecasts_ext.csv"),
        "--outdir", str(outputs_root / "plots_ext"),
    ]
    _run(plot_ext_cmd, cwd=project_root)

    # 3) RiskMetrics
    riskmetrics_cmd = [
        python_cmd,
        str(project_root / "run_var_riskmetrics.py"),
        "--data", str(data_path),
        "--outdir", str(outputs_root / "riskmetrics"),
        "--alpha", str(args.alpha),
        "--initial-train", str(args.initial_train),
    ]
    _run(riskmetrics_cmd, cwd=project_root)

    # 4) GARCHX.r
    created_temp_input = False
    temp_input_path: Path | None = None
    if not args.skip_garchx:
        temp_input_path, created_temp_input = _prepare_r_input(project_root, data_path)
        try:
            _run([rscript_cmd, str(project_root / "GARCHX.r")], cwd=project_root)
        finally:
            if temp_input_path is not None:
                _cleanup_r_input(temp_input_path, created_temp_input)

    print("\n[done] Full VaR workflow completed successfully.")
    print(f"[info] Base outputs: {outputs_root}")
    print(f"[info] RiskMetrics outputs: {outputs_root / 'riskmetrics'}")
    if not args.skip_garchx:
        print(f"[info] GARCHX outputs: {project_root / 'outputs' / 'garchx_only'}")


if __name__ == "__main__":
    main()
