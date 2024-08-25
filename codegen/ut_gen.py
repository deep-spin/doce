import argparse
import os
from os import PathLike
from typing import List, Any

import json
import pickle

from model import DecoderBase, make_model
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from vllm import SamplingParams

##TODO:
# 2. we should include the post-processing functions that process generated unit tests.

def load_json(dir_path: str):
    with open(dir_path, "r") as f:
        return json.load(f)

def load_jsonl(dir_path: str):
    # notice that we use the jsonl file to store the data
    with open(dir_path, "r") as f:
        return [json.loads(line) for line in f]

def ut_gen(
    model: DecoderBase,
    prompt: str, 
    do_sample: bool = True, 
    num_samples: int = 50
) -> List[str]:
    if do_sample:
        assert model.temperature > 0, "Temperature must be greater than 0!"
    batch_size = min(model.batch_size, num_samples)

    vllm_outputs = model.llm.generate(
        [prompt] * batch_size,
        SamplingParams(
            temperature=model.temperature,
            max_tokens=model.max_new_tokens,
            top_p=0.95 if do_sample else 1.0,
            stop=model.eos,
        ),
        use_tqdm=False,
    )

    gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
    return gen_strs

def construct_ut_gen_prompt(prompt: str, entry_point: str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
I have this function stub, please generate 50 test cases for this function. The function stub is as follow:
```python
{prompt}
```
- Each test case is in the form of assertion statement, for example: assert {entry_point}(...) == ...
- Each test case is in a single line
- The length of each test case should be too long, ideally less than or equal to 150 letters
- The test input should not be too long
- The inputs of test cases should be diverse and cover corner cases of the function
- Test cases should not be repeated

### Response:
Here are 50 test cases for function `{entry_point}`:
```python
assert {entry_point}("""

def code_generate(args, workdir: PathLike, model: DecoderBase, id_range=None):
    with Progress(
        TextColumn(
            f"{args.dataset} •" + "[progress.percentage]{task.percentage:>3.0f}%"
        ),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
        if args.dataset == "humaneval":
            from evalplus.data import get_human_eval_plus

            dataset = get_human_eval_plus()
            prompts = load_jsonl("/mnt/scratch-artemis/haausing/code_reranking/code/evalplus/other_data/HumanEval_for_test_case_generation.jsonl")
            prompts = {prompt["task_id"]: prompt for prompt in prompts}
            assert all(task_id in dataset for task_id in prompts)
            
        elif args.dataset == "mbpp":
            from evalplus.data import get_mbpp_plus

            dataset = get_mbpp_plus()
            prompts = load_jsonl("/mnt/scratch-artemis/haausing/code_reranking/code/evalplus/other_data/mbpp_sanitized_for_test_case_generation.jsonl")
            prompts = {prompt["task_id"].replace("MbppEval", "Mbpp"): prompt for prompt in prompts}
            assert all(task_id in prompts for task_id in dataset)

        #final_logprobs = {} # this is a dictionary that will store the logprobs of the generated code

        for task_id, task in p.track(dataset.items()):
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            p_name = task_id.replace("/", "_")
            if args.contract_type != "none" and task["contract"] == "":
                continue

            os.makedirs(os.path.join(workdir, p_name), exist_ok=True)
            log = f"TestCaseGen: {p_name} @ {model}"
            n_existing = 0
            if args.resume:
                # count existing .py files
                n_existing = len(
                    [
                        f
                        for f in os.listdir(os.path.join(workdir, p_name))
                        if f.endswith(".py")
                    ]
                )
                if n_existing > 0:
                    log += f" (resuming from {n_existing})"

            nsamples = args.n_samples - n_existing
            p.console.print(log)

            sidx = args.n_samples - nsamples
            
            # I added the the replace method, just to make sure that the prompt is in the same format as the samples as the samples also include the prompt.
            prompt = task["prompt"].replace("    ", "\t")
            entry_point = task["entry_point"]
            if dataset=="mbpp":
                prompt = prompt.split("\n")[:-2]
                prompt = "\n".join(prompt)
            prompt = construct_ut_gen_prompt(prompt, entry_point)

            while sidx < args.n_samples:
                outputs = ut_gen(model, prompt, do_sample=not args.greedy, num_samples=args.n_samples - sidx)
                assert outputs, "No outputs from model!"
                for idx, impl in enumerate(outputs):
                    try:
                        with open(
                            os.path.join(workdir, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            if model.conversational:
                                f.write(impl)
                            else:
                                f.write(task["prompt"] + impl)
                    except UnicodeEncodeError:
                        continue
                    sidx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument(
        "--dataset", required=True, type=str, choices=["humaneval", "mbpp"]
    )
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--contract-type",
        default="none",
        type=str,
        choices=["none", "code", "docstring"],
    )
    parser.add_argument("--greedy", action="store_true")
    # id_range is list
    parser.add_argument("--id-range", default=None, nargs="+", type=int)
    args = parser.parse_args()

    if args.greedy and (args.temperature != 0 or args.bs != 1 or args.n_samples != 1):
        args.temperature = 0
        args.bs = 1
        args.n_samples = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if args.id_range is not None:
        assert len(args.id_range) == 2, "id_range must be a list of length 2"
        assert args.id_range[0] < args.id_range[1], "id_range must be increasing"
        args.id_range = tuple(args.id_range)

    # Make project dir
    os.makedirs(args.root, exist_ok=True)
    # Make dataset dir
    os.makedirs(os.path.join(args.root, args.dataset), exist_ok=True)
    # Make dir for codes generated by each model
    args.model = args.model.lower()
    model = make_model(
        name=args.model, batch_size=args.bs, temperature=args.temperature
    )
    workdir = os.path.join(
        args.root,
        args.dataset+"_unit_tests",
        args.model
        + f"_temp_{args.temperature}"
        + ("" if args.contract_type == "none" else f"-contract-{args.contract_type}"),
    )
    if args.greedy:
        workdir = workdir.replace("temp_0.0", "temp_0")
    
    os.makedirs(workdir, exist_ok=True)

    with open(os.path.join(workdir, "args.txt"), "w") as f:
        f.write(str(args))

    code_generate(args, workdir=workdir, model=model, id_range=args.id_range)


if __name__ == "__main__":
    main()
