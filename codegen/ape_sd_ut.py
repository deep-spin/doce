import argparse
import os
from os import PathLike
from typing import List, Any, Dict
import time

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

from evalplus.prompt import MBPP_UT_FEW_SHOT, HUMANEVAL_UT_FEW_SHOT
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

from evalplus.data.utils import CACHE_DIR
from evalplus.gen.util import trusted_exec

from vllm import SamplingParams

def get_groundtruth(problems, hashcode, tasks_only_output_not_none):
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
    if os.path.exists(cache_file):
        print(f"Load from ground-truth from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("Computing expected output...")
    tbegin = time.time()
    expected_output = {}
    for task_id, problem in problems.items():
        oracle = {}
        oracle["base"], oracle["base_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["base_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )

        oracle["plus"], oracle["plus_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["plus_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        expected_output[task_id] = oracle
    print(f"Expected outputs computed in {time.time() - tbegin:.2f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(expected_output, f)

    return expected_output

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

def mbpp_construct_refinement_prompt(prompt: str, unit_test: str, sample: str, error: Dict[str, Any], exp_out: Any) -> str:
    if error["status"] == "pass":
        return MBPP_UT_FEW_SHOT+f"""### instruction
{prompt}
```python
{unit_test}
```

### code
```python
{sample}
```

### feedback
With the above function, the assertion is `{unit_test}` and the real execution output is `{exp_out}`. The code passes the assertion.
The code above is """
    elif error["error"].startswith("With the above function"):
        return MBPP_UT_FEW_SHOT+f"""### instruction
{prompt}
```python
{unit_test}
```

### code
```python
{sample}
```

### feedback
{error["error"]}
The code above is """
    elif error["error"].strip() == "":
        return MBPP_UT_FEW_SHOT+f"""### instruction
{prompt}
```python
{unit_test}
```

### code
```python
{sample}
```

### feedback
With the above function, there's following error:
'''
An error occurred.
'''
The code above is """
    else:
        return MBPP_UT_FEW_SHOT+f"""### instruction
{prompt}
```python
{unit_test}
```

### code
```python
{sample}
```

### feedback
With the above function, there's following error:
'''
{error["error"]}
'''
The code above is """

def humaneval_construct_refinement_prompt(prompt: str, unit_test: List[str], sample: str, error: Dict[str, Any], exp_out: List[Any]) -> str:
    if error["status"] == "pass":
        start_prompt = HUMANEVAL_UT_FEW_SHOT+f'''### prompt
```python
{prompt}
```

### code
```python
{sample}
```

### feedback
'''
        for t, e in zip(unit_test, exp_out):
            start_prompt += f"With the above function, the assertion is `{t}` and the real execution output is `{e}`. The code passes the assertion.\n"
        return start_prompt + "The code above is "
    elif error["error"].startswith("With the above function"):
        return HUMANEVAL_UT_FEW_SHOT+f'''### prompt
```python
{prompt}
```

### code
```python
{sample}
```

### feedback
{error["error"]}
The code above is '''
    elif error["error"].strip() == "":
        return HUMANEVAL_UT_FEW_SHOT+f'''### prompt
```python
{prompt}
```

### code
```python
{sample}
```

### feedback
With the above function, there's following error:
"""
An error occurred.
"""
The code above is '''
    else:
        return HUMANEVAL_UT_FEW_SHOT+f'''### prompt
```python
{unit_test}
```

### code
```python
{sample}
```

### feedback
With the above function, there's following error:
"""
{error["error"]}
"""
The code above is '''

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
            unit_tests = load_jsonl("other_data/trial_tests.jsonl")
            unit_tests = {x["task_id"]: x["given_tests"] for x in unit_tests}
            with open("other_data/trial_expected_output.pkl", "rb") as f:
                expected_output = pickle.load(f)
        elif args.dataset == "mbpp":
            from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash

            dataset = get_mbpp_plus()
            dataset_hash = get_mbpp_plus_hash()
            expected_output = get_groundtruth(
                dataset,
                dataset_hash,
                MBPP_OUTPUT_NOT_NONE_TASKS,
            )
        elif args.dataset == "lcb":
            # load pickle file
            with open("other_data/selected_lcb.pkl", "rb") as f:
                dataset = pickle.load(f)
            dataset_hash = "lcb"
            with open("other_data/refined_lcb_inputs.pkl", "rb") as f:
                refined_inputs = pickle.load(f)
            for task_id in dataset:
                dataset[task_id]["base_input"] = refined_inputs[task_id]["base_input"]
            with open("other_data/lcb_trial_expected_output.pkl", "rb") as f:
                expected_output = pickle.load(f)
            unit_tests = load_jsonl("other_data/lcb_trial_tests.jsonl")
            unit_tests = {x["task_id"]: x["given_tests"] for x in unit_tests}
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        if args.debugging_turn == 1:
            eval_results = load_json(
                os.path.join(workdir, 
                         "eval_results.json")
                         #"eval_results_base_plus.json")
            )
            errors = pickle.load(open(
                os.path.join(workdir, 
                             "errors.pkl"), 
                "rb")
            )
        elif args.change_positive:
            starting_turn = args.debugging_turn - 1
            eval_results = load_json(
                os.path.join(workdir+f"_debug{starting_turn}"+"_sd-ut", 
                         "eval_results.json")
                         #"eval_results_base_plus.json")
            )
            errors = pickle.load(open(
                os.path.join(workdir+f"_debug{starting_turn}"+"_sd-ut", 
                             "errors.pkl"),
                "rb")
            )
            if args.debugging_turn > 1:
                continue_debug = load_json(
                    os.path.join(workdir+f"_debug{starting_turn}"+"_sd-ut", 
                            "continue_debug.json")
                )
        else:
            starting_turn = args.debugging_turn - 1
            eval_results = load_json(
                os.path.join(workdir+f"_debug{starting_turn}"+"_not_change_positive"+"_sd-ut", 
                         "eval_results.json")
                         #"eval_results_base_plus.json")
            )
            errors = pickle.load(open(
                os.path.join(workdir+f"_debug{starting_turn}"+"_sd-ut", 
                             "errors.pkl"),
                "rb")
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
                    starting_dir = os.path.join(workdir+f"_debug{starting_turn}"+"_sd-ut", p_name)
                else:
                    starting_dir = os.path.join(workdir+f"_debug{starting_turn}"+"_not_change_positive"+"_sd-ut", p_name)
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
                debugged_dir = workdir+f"_debug{args.debugging_turn}"+"_sd-ut" + "_b4process"
            else:
                debugged_dir = workdir+f"_debug{args.debugging_turn}"+"_not_change_positive"+"_sd-ut" + "_b4process"
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
            
            if args.dataset == "mbpp":
                # I added the the replace method, just to make sure that the prompt is in the same format as the samples as the samples also include the prompt.
                prompt = task["prompt"].replace("    ", "\t")
                if not model.conversational:
                    assert all([x.startswith(prompt) for x in samples.values()]), "All samples must start with the prompt!"
                    for idx in samples:
                        samples[idx] = samples[idx].replace(prompt, "").strip() # notice that I stripped the spaces and line breaks
                    
                prompt, unit_test = prompt.replace('"""', '').strip().split('\n')
            else:
                prompt = task["prompt"].replace("    ", "\t").strip()
                unit_test = unit_tests[task_id]
            
            base_details = []
            if args.dataset == "mbpp":
                for i, solution in enumerate(eval_results["eval"][task_id]):
                    assert i == int(solution["solution_id"])
                    if len(solution["base_details"]) == 0 or solution["base_details"][0] == 0:
                        base_details.append(False)
                    else:
                        base_details.append(True)
            elif args.dataset in ["humaneval", "lcb"]:
                for i in errors[task_id]:
                    if errors[task_id][i]["base"]["status"] == "pass":
                        base_details.append(True)
                    else:
                        base_details.append(False)
            else:
                raise NotImplementedError("You should select the dataset between mbpp, humaneval and lcb.")
            
            prompts = {}
            for idx in samples:
                if args.dataset == "mbpp":
                    assert type(unit_test) == str, "The unit test should be a string for mbpp since we only have one unit test."
                    prompts[idx] = mbpp_construct_refinement_prompt(prompt, unit_test, samples[idx], errors[task_id][idx]["base"], expected_output[task_id]["base"][0])
                elif args.dataset in ["humaneval", "lcb"]:
                    assert type(unit_test) == list, "The unit test should be a list for humaneval."
                    assert len(unit_test) == len(expected_output[task_id]["base"]), "The length of the unit test should be equal to the length of the expected output."
                    prompts[idx] = humaneval_construct_refinement_prompt(prompt, unit_test, samples[idx], errors[task_id][idx]["base"], expected_output[task_id]["base"])
                else:
                    raise NotImplementedError("You should select the dataset between mbpp, humaneval and lcb.")
            task_continue_debug = continue_debug[task_id] if args.debugging_turn > 1 else {}

            while sidx < args.n_samples:
                if args.debugging_turn == 1:
                    range_to_use = list(range(sidx, min(sidx + model.batch_size, args.n_samples)))
                else:
                    continue_debug_ids = [int(i) for i in task_continue_debug if task_continue_debug[i] and int(i) >= sidx]
                    continue_debug_ids = sorted(continue_debug_ids)
                    samples_to_use = min(model.batch_size, args.n_samples - sidx)
                    range_to_use = continue_debug_ids[:samples_to_use]
                
                if len(range_to_use) == 0:
                    outputs = []
                else:
                    outputs = refinement(model, [prompts[i] for i in range_to_use])
                    assert outputs, "No outputs from model!"
                if args.debugging_turn == 1:
                    for impl in outputs:
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
                else:
                    if range_to_use == []:
                        max_idx = args.n_samples-1
                    else:
                        max_idx = max(range_to_use)
                    while sidx <= max_idx:
                        if sidx in range_to_use:
                            # get the index of sidx in continue_debug_ids
                            index = range_to_use.index(sidx)
                            try:
                                with open(os.path.join(debugged_dir, p_name, f"{sidx}.py"), "w", encoding="utf-8") as f:
                                    if model.conversational:
                                        if args.change_positive or (not base_details[sidx]):
                                            f.write(outputs[index])
                                        else:
                                            raise NotImplementedError("This part is not implemented.")
                                    else:
                                        f.write(task["prompt"] + outputs[index])
                            except UnicodeEncodeError:
                                continue
                        else:
                            try:
                                with open(os.path.join(debugged_dir, p_name, f"{sidx}.py"), "w", encoding="utf-8") as f:
                                    f.write(" correct.")
                            except UnicodeEncodeError:
                                continue
                        sidx += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument(
        "--dataset", required=True, type=str, choices=["humaneval", "mbpp", "lcb"]
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
    model.eos = [e for e in model.eos if e != "\n```"]
    model.eos += ["\n### task end ###"]
    model.eos += ["```\n"]
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
        debugged_dir = workdir+f"_debug{args.debugging_turn}"+"_sd-ut" + "_b4process"
    else:
        debugged_dir = workdir+f"_debug{args.debugging_turn}"+"_not_change_positive"+"_sd-ut" + "_b4process"
    
    os.makedirs(debugged_dir, exist_ok=True)

    with open(os.path.join(debugged_dir, "args.txt"), "w") as f:
        f.write(str(args))

    code_generate(args, workdir=workdir, model=model, id_range=args.id_range)

if __name__ == "__main__":
    main()
