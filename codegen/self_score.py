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

def load_json(dir_path: str):
    with open(dir_path, "r") as f:
        return json.load(f)

def load_jsonl(dir_path: str):
    # notice that we use the jsonl file to store the data
    with open(dir_path, "r") as f:
        return [json.loads(line) for line in f]

def refinement(
    model: DecoderBase, 
    prompts: List[str], 
) -> List[str]:
    assert len(prompts) <= model.batch_size, "Number of prompts must be no greater than batch size!"

    vllm_outputs = model.llm.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            max_tokens=model.max_new_tokens,
            top_p=1.0,
            stop=model.eos,
        ),
        use_tqdm=False,
    )
    
    gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
    return gen_strs

def construct_refinement_prompt(prompt: str, unit_test: str, sample: str, base_detail: bool, exec_info: str) -> str:
    return f"""### instruction
{prompt}
```python
{unit_test}
```

### code
```python
{sample}
```

### execution information
{exec_info}

Please return the quality score of the generated code as the following format:
```python
quality_score = score
```
where the score should be a number between 0 and 100.

### quality score
```python
#assert quality_score >= 0 and quality_score <= 100
quality_score ="""

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
        elif args.dataset == "mbpp":
            from evalplus.data import get_mbpp_plus

            dataset = get_mbpp_plus()
        
        
        if args.debugging_turn == 1:
            eval_results = load_json(
                os.path.join(workdir, 
                         "eval_results.json")
                         #"eval_results_base_plus.json")
            )
        elif args.change_positive:
            starting_turn = args.debugging_turn - 1
            eval_results = load_json(
                os.path.join(workdir+f"_debug{starting_turn}", 
                         "eval_results.json")
                         #"eval_results_base_plus.json")
            )
        else:
            starting_turn = args.debugging_turn - 1
            eval_results = load_json(
                os.path.join(workdir+f"_debug{starting_turn}"+"_not_change_positive", 
                         "eval_results.json")
                         #"eval_results_base_plus.json")
            )
        
        for task_id in eval_results["eval"]:
            eval_results["eval"][task_id] = sorted(eval_results["eval"][task_id], key=lambda x: int(x["solution_id"]))

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

            ### start: loading all generated samples from the directory ###
            # load all py files from the directory os.path.join(workdir, p_name)
            samples = {}
            if args.debugging_turn == 1:
                starting_dir = os.path.join(workdir, p_name)
            else:
                starting_turn = args.debugging_turn - 1
                if args.change_positive:
                    starting_dir = os.path.join(workdir+f"_debug{starting_turn}", p_name)
                else:
                    starting_dir = os.path.join(workdir+f"_debug{starting_turn}"+"_not_change_positive", p_name)
            for f in os.listdir(starting_dir):
                if f.endswith(".py"):
                    with open(os.path.join(starting_dir, f), "r", encoding="utf-8") as file:
                        # delete the suffix .py from f and convert it to interger:
                        samples[int(f.split(".")[0])] = file.read()
            # sort the samples by their keys in ascending order
            samples = dict(sorted(samples.items()))
            # we want to make sure the number of samples is equal to args.n_samples
            assert len(samples) == args.n_samples, f"Number of samples is not equal to {args.n_samples}"
            ### end ###

            if args.change_positive:
                debugged_dir = workdir+f"_debug{args.debugging_turn}"
            else:
                debugged_dir = workdir+f"_debug{args.debugging_turn}"+"_not_change_positive"
            os.makedirs(os.path.join(debugged_dir, p_name), exist_ok=True)
            log = f"Codegen: {p_name} @ {model}"
            n_existing = 0
            if args.resume:
                # count existing .py files
                n_existing = len(
                    [
                        f
                        for f in os.listdir(os.path.join(debugged_dir, p_name))
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
            if not model.conversational:
                assert all([x.startswith(prompt) for x in samples.values()]), "All samples must start with the prompt!"
                for idx in samples:
                    samples[idx] = samples[idx].replace(prompt, "").strip() # notice that I stripped the spaces and line breaks
                
            prompt, unit_test = prompt.replace('"""', '').strip().split('\n')
            
            base_details = []
            for i, solution in enumerate(eval_results["eval"][task_id]):
                assert i == int(solution["solution_id"])
                if len(solution["base_details"]) == 0 or solution["base_details"][0] == 0:
                    base_details.append(False)
                else:
                    base_details.append(True)
            
            prompts = {}
            for idx in samples:
                prompts[idx] = construct_refinement_prompt(prompt, unit_test, samples[idx], base_details[idx])

            while sidx < args.n_samples:
                range_to_use = list(range(sidx, min(sidx + model.batch_size, args.n_samples)))
                outputs = refinement(model, [prompts[i] for i in range_to_use])
                assert outputs, "No outputs from model!"
                for idx, impl in enumerate(outputs):
                    try:
                        with open(
                            os.path.join(debugged_dir, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            if model.conversational:
                                if args.change_positive or (not base_details[sidx]):
                                    f.write(impl)
                                else:
                                    f.write(samples[sidx])
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
    parser.add_argument("--debugging_turn", default=1, type=int)
    parser.add_argument("--change_positive", action="store_true")
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
        args.dataset,
        args.model
        + f"_temp_{args.temperature}"
        + ("" if args.contract_type == "none" else f"-contract-{args.contract_type}"),
    )
    if args.greedy:
        workdir = workdir.replace("temp_0.0", "temp_0")
    if args.change_positive:
        debugged_dir = workdir+f"_debug{args.debugging_turn}"
    else:
        debugged_dir = workdir+f"_debug{args.debugging_turn}"+"_not_change_positive"
    
    os.makedirs(debugged_dir, exist_ok=True)

    with open(os.path.join(debugged_dir, "args.txt"), "w") as f:
        f.write(str(args))

    code_generate(args, workdir=workdir, model=model, id_range=args.id_range)


if __name__ == "__main__":
    main()
