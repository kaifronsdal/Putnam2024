from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, json_dataset
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate, system_message, prompt_template, basic_agent

SYSTEM_PROMPT = """
You are a mathematician solving a Putnam competition problem. Think through the problem step by step, showing your work clearly.
""".strip()

USER_PROMPT = """
{prompt}
""".strip()


@task
def putnam(dataset_path: str) -> Task:
    """Inspect Task implementation for Putnam math problems"""

    # Load the Putnam dataset
    dataset = json_dataset(
        dataset_path,
        FieldSpec(
            input="problem",
            # target="solution",
            id="id",
            metadata=["year"],
        ),
    )

    return Task(
        dataset=dataset,
        solver=[
            system_message(SYSTEM_PROMPT),
            prompt_template(USER_PROMPT),
            # generate()
            # basic_agent()
        ],
        scorer=model_graded_qa()
    )
