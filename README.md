To setup this project, you need to install the following dependencies:

```
pip install inspect_ai openai anthropic google-generativeai vllm wandb
```

and set the following environment variables:

```
export OPENAI_API_KEY=
export ANTHROPIC_API_KEY=
export GOOGLE_API_KEY=
export GROK_API_KEY=
```

Then simply run to run evaluations (make sure to change DATASET and CUDA_VISIBLE_DEVICES within eval.sh to the appropriate values):

```
./src/putnam2024/eval.sh
```
