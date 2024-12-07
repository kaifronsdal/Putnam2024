#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import wandb


def load_json_logs(file_path: Path) -> Dict[str, Any]:
    """Load and parse a JSON log file."""
    with open(file_path, "r") as f:
        return json.load(f)


def process_log_directory(log_dir: Path) -> Dict[str, Any]:
    """
    Process all log files in the directory.
    Returns a dictionary mapping file names to their contents.
    """
    logs = {}

    # Process all files in the directory
    for file_path in log_dir.glob("*"):
        if file_path.is_file():
            try:
                # Try to parse as JSON first
                logs[file_path.name] = load_json_logs(file_path)
            except json.JSONDecodeError:
                # If not JSON, read as text
                with open(file_path, "r") as f:
                    logs[file_path.name] = f.read()

    return logs


def upload_to_wandb(
        eval_dir: str,
        solutions_dir: str,
        project: str,
        entity: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
) -> None:
    """
    Upload eval logs and solution files to Weights & Biases.

    Args:
        eval_dir: Directory containing eval logs
        solutions_dir: Directory containing solution JSON files
        project: W&B project name
        entity: W&B entity (username or team name)
        name: Run name (defaults to timestamp)
        tags: List of tags to apply to the run
    """
    # Generate default run name if none provided
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize W&B run
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        tags=tags,
        job_type="eval",
    )

    # Process and upload eval logs
    eval_logs = process_log_directory(Path(eval_dir))
    for log_name, log_content in eval_logs.items():
        wandb.save(os.path.join(eval_dir, log_name))

    # Process and upload solution files
    solutions = process_log_directory(Path(solutions_dir))
    for solution_name, solution_content in solutions.items():
        wandb.save(os.path.join(solutions_dir, solution_name))

    # Log basic stats about the data
    wandb.log({
        "num_eval_logs": len(eval_logs),
        "num_solution_files": len(solutions)
    })

    # Finish the run
    run.finish()


def main():
    parser = argparse.ArgumentParser(description="Upload eval logs and solutions to Weights & Biases")
    parser.add_argument(
        "eval_dir",
        type=str,
        help="Directory containing the eval logs",
    )
    parser.add_argument(
        "solutions_dir",
        type=str,
        help="Directory containing the solution JSON files",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="W&B project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="W&B entity (username or team name)",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Run name (defaults to timestamp)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="Tags to apply to the run",
    )

    args = parser.parse_args()

    # Process log directory
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists() or not eval_dir.is_dir():
        raise ValueError(f"Invalid eval logs directory: {eval_dir}")

    solutions_dir = Path(args.solutions_dir)
    if not solutions_dir.exists() or not solutions_dir.is_dir():
        raise ValueError(f"Invalid solutions directory: {solutions_dir}")

    # Upload to W&B
    upload_to_wandb(
        eval_dir=str(eval_dir),
        solutions_dir=str(solutions_dir),
        project=args.project,
        entity=args.entity,
        name=args.name,
        tags=args.tags,
    )


if __name__ == "__main__":
    main()