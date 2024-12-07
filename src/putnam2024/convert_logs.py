#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from inspect_ai.log import list_eval_logs, read_eval_log


def extract_answer_from_response(text: str) -> str:
    """Extract the model's solution from its response."""
    # Return the full response text since we want to keep all the reasoning
    return text.strip()


def convert_eval_logs_to_putnam(log_dir: str, output_dir: str) -> None:
    """Convert eval logs to Putnam format with model solutions."""
    # Get all eval logs in the directory
    eval_logs = list_eval_logs(log_dir)

    # putnam_problems: List[Dict[str, Any]] = []
    model_problems: Dict[str, List[Dict[str, Any]]] = {}

    for log_info in eval_logs:
        # Read the full log
        eval_log = read_eval_log(log_info.name)

        if not eval_log.samples:
            continue

        model_name = eval_log.eval.model
        print(model_name)

        # Initialize list for this model if not exists
        if model_name not in model_problems:
            model_problems[model_name] = []

        for sample in eval_log.samples:
            # Get the original problem data
            problem_data = {
                "year": sample.metadata.get("year", "unknown"),
                "id": str(sample.id),
                "problem": sample.input,
                # "answer_type": sample.metadata.get("answer_type", "numerical"),
            }

            # Add the model's solution
            if sample.output and sample.output.completion:
                problem_data["model_solution"] = extract_answer_from_response(sample.output.completion)
                # problem_data["model_chat_history"] = [m.model_dump(mode="json") for m in sample.messages]

            # putnam_problems.append(problem_data)
            model_problems[model_name].append(problem_data)

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write separate file for each model
    for model_name, problems in model_problems.items():
        # Create a safe filename from the model name
        # print(model_name)
        safe_model_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
        output_file = output_path / f"putnam_solutions_{safe_model_name}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
            f.write('\n')  # Add newline at end of file

        print(f"Wrote solutions for {model_name} to {output_file}")

    # Write output JSON file
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(putnam_problems, f, indent=2, ensure_ascii=False)
    #     f.write('\n')  # Add newline at end of file


def main():
    parser = argparse.ArgumentParser(description="Convert eval logs to Putnam format with model solutions")
    parser.add_argument(
        "log_dir",
        type=str,
        help="Directory containing the eval logs",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for JSON files",
    )

    args = parser.parse_args()

    # Ensure log directory exists
    log_dir = Path(args.log_dir)
    if not log_dir.exists() or not log_dir.is_dir():
        raise ValueError(f"Invalid log directory: {log_dir}")

    # Convert logs
    convert_eval_logs_to_putnam(str(log_dir), args.output_dir)

    print(f"Successfully converted logs to {args.output_dir}")


if __name__ == "__main__":
    main()
