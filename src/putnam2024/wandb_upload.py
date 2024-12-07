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
        logs: Dict[str, Any],
        project: str,
        entity: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
) -> None:
    """
    Upload logs to Weights & Biases.

    Args:
        logs: Dictionary of log data
        project: W&B project name
        entity: W&B entity (username or team name)
        name: Run name (defaults to timestamp if not provided)
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

    # Log files and metrics
    for log_name, log_content in logs.items():
        if isinstance(log_content, (dict, list)):
            # For structured data, log as config/metrics
            if "metrics" in log_name.lower():
                wandb.log(log_content)
            else:
                wandb.config.update({log_name: log_content})
        else:
            # For text content, save as files
            with run.dir.join(log_name).open("w") as f:
                f.write(str(log_content))

    # Finish the run
    run.finish()


def main():
    parser = argparse.ArgumentParser(description="Upload inspect-ai logs to Weights & Biases")
    parser.add_argument(
        "log_dir",
        type=str,
        help="Directory containing the logs",
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
    log_dir = Path(args.log_dir)
    if not log_dir.exists() or not log_dir.is_dir():
        raise ValueError(f"Invalid log directory: {log_dir}")

    # Load logs
    logs = process_log_directory(log_dir)
    if not logs:
        raise ValueError(f"No logs found in directory: {log_dir}")

    # Upload to W&B
    upload_to_wandb(
        logs=logs,
        project=args.project,
        entity=args.entity,
        name=args.name,
        tags=args.tags,
    )


if __name__ == "__main__":
    main()