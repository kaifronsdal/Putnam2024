from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, json_dataset
from inspect_ai.model import CachePolicy
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate, system_message, prompt_template, basic_agent, use_tools
from inspect_ai.tool import python, bash

SYSTEM_PROMPT = """
You are a mathematician solving a Putnam competition problem. Think through the problem step by step, showing your work clearly.
""".strip()

# USER_PROMPT = """
# Solve the following math problem step by step, clearly explaining your reasoning:
# Problem: {prompt}
# """.strip()

USER_PROMPT = """
Given a mathematics question, compose a detailed solution. Simplify your answer as much as possible. Always give the final answer inside a \\boxed{{answer}}. Format:
Final Answer: The final answer is $\\boxed{{answer}}$. I hope it is correct. Here are some examples:

Problem:
Let $ABC$ be a triangle with angle $A < \\angle C < 90^\\circ < \\angle B$. Consider the bisectors of the external angles at $A$ and $B$, each measured from the vertex to the opposite side (extended). Suppose both of these line-segments are equal to $AB$. Compute the angle $A$.

Solution: Suppose the bisector of the exterior angle at $A$ intersects line $BC$ at $X$ and the bisector of the exterior angle at $B$ meets the line $AC$ at $Y$. The assumption that $C$ is between $B$ and $X$ contradicts the fact that $\\angle B > \\angle C$, so we may assume that $B$ is between $X$ and $C$. Similarly, we conclude that $C$ is between $A$ and $Y$ because $\\angle A < \\angle C$.

If $Z$ is a point on line $AB$ with $B$ between $A$ and $Z$, we have from triangle $ABY$ that $\\angle ZBY = 2A$. Hence, $\\angle BXA = \\angle ABX = \\angle ZBC = 2 \\angle ZBY = 4A$, and the angle sum of triangle $ABX$ is $90^\\circ - \\frac{{1}}{{2}}A + 8A$. Thus, $A = \\boxed{{12}}^\\circ$.

Problem:
Given any positive integer $n$ find the value of
\\[
\\sum_{{r=0}}^{{\\lfloor (n-1)/2 \\rfloor}} \\left\\{{\\frac{{n - 2r}}{{n}}\\binom{{n}}{{r}}\\right\\}}^2,
\\]
where $\\lfloor x \\rfloor$ means the greatest integer not exceeding $x$, and $\\binom{{n}}{{r}}$ is the binomial coefficient "$n$ choose $r$," with the convention $\\binom{{0}}{{0}} = 1$. Return your final answer in binomial coefficient form with all other multiplicants reduced to the lowest form.

Solution: Substituting $s=n-r$ in the given summation reveals that twice this sum is equal to:
\\[
\\sum_{{r=0}}^n \\left(\\frac{{n-2r}}{{n}} \\binom{{n}}{{r}}\\right)^2 = \\sum \\left(1 - 2\\frac{{r}}{{n}}\\right)^2 \\binom{{n}}{{r}}^2 = \\sum \\binom{{n}}{{r}}^2 - 4\\sum \\frac{{r}}{{n}} \\binom{{n}}{{r}}^2 + 4\\sum \\left(\\frac{{r}}{{n}}\\right)^2 \\binom{{n}}{{r}}^2.
\\]
\\[
= \\binom{{2n}}{{n}} - 4 \\sum_{{r=1}}^n \\binom{{n-1}}{{r-1}} \\binom{{n}}{{r}} + 4 \\sum_{{r=1}}^n \\binom{{n-1}}{{r-1}}^2.
\\]
\\[
= \\binom{{2n}}{{n}} - 4\\binom{{2n-1}}{{n-1}} + 4\\binom{{2n-2}}{{n-1}}.
\\]
\\[
= \\frac{{2n(2n-1)}}{{n^2}} - \\frac{{4(n-1)}}{{n}}\\binom{{2n-2}}{{n-1}} = \\boxed{{\\frac{{1}}{{n}}\\binom{{2n-2}}{{n-1}}}}.
\\]

Problem:
Find the sum of all sides of all the right-angled triangles whose sides are integers while the area is numerically equal to twice the perimeter.

Solution: All Pythagorean triples can be obtained from $x = \\lambda(p^2 - q^2)$, $y = 2\\lambda pq$, $z = \\lambda(p^2 + q^2)$ where $0 < q < p$, $(p, q) = 1$ and $p \\not\\equiv q \\pmod{{2}}$, $\\lambda$ being any natural number.

The problem requires that $\\frac{{1}}{{2}}xy = 2(x+y+z)$. This condition can be written as $\\lambda^2(p^2-q^2)(pq) = 2\\lambda(p^2-q^2+2pq+p^2+q^2)$ or simply $\\lambda(p-q)q = 4$. Since $p-q$ is odd it follows that $p-q = 1$ and the only possibilities for $q$ are $1, 2, 4$.

- If $q = 1$, $p = 2$, $\\lambda = 4$, $x = 12$, $y = 16$, $z = 20$.
- If $q = 2$, $p = 3$, $\\lambda = 2$, $x = 10$, $y = 24$, $z = 26$.
- If $q = 4$, $p = 5$, $\\lambda = 1$, $x = 9$, $y = 40$, $z = 41$. This gives us the final answer as $12+16+20+10+24+26+9+40+41 = \\boxed{{198}}$.

Problem:
Evaluate
\\[
\\lim_{{n \\to \\infty}} \\int_0^1 \\int_0^1 \\cdots \\int_0^1 \\cos^2 \\left(\\frac{{\\pi}}{{2n}}(x_1 + x_2 + \\cdots + x_n)\\right) dx_1 dx_2 \\cdots dx_n.
\\]

Solution: The change of variables $x_k \\to 1 - x_k$ yields
\\[
\\int_0^1 \\int_0^1 \\cdots \\int_0^1 \\cos^2 \\left(\\frac{{\\pi}}{{2n}}(x_1 + x_2 + \\cdots + x_n)\\right) dx_1 dx_2 \\cdots dx_n \\\\
= \\int_0^1 \\int_0^1 \\cdots \\int_0^1 \\sin^2 \\left(\\frac{{\\pi}}{{2n}}(x_1 + x_2 + \\cdots + x_n)\\right) dx_1 dx_2 \\cdots dx_n.
\\]
Each of these expressions, being equal to half their sum, must equal $\\frac{{1}}{{2}}$. The limit is also $\\boxed{{\\frac{{1}}{{2}}}}$.

Problem:
{prompt}

Solution:
"""


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
            # system_message(SYSTEM_PROMPT),
            prompt_template(USER_PROMPT),
            generate(cache=CachePolicy("3M"))
            # basic_agent()
        ],
        # scorer=model_graded_qa()
    )

# @task
# def putnam_tools(dataset_path: str) -> Task:
#     """Inspect Task implementation for Putnam math problems"""
#
#     # Load the Putnam dataset
#     dataset = json_dataset(
#         dataset_path,
#         FieldSpec(
#             input="problem",
#             # target="solution",
#             id="id",
#             metadata=["year"],
#         ),
#     )
#
#     return Task(
#         dataset=dataset,
#         solver=[
#             system_message(SYSTEM_PROMPT),
#             prompt_template(USER_PROMPT),
#             use_tools([python(), bash()]),
#             generate(cache=CachePolicy("3M"))
#             # basic_agent()
#         ],
#         sandbox="docker",
#         # scorer=model_graded_qa()
#     )
