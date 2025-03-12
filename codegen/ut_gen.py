import argparse
import os
from os import PathLike
from typing import List, Any

import json
import pickle
import time
import shutil

from .utils import *

from model import DecoderBase, make_model
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from vllm import SamplingParams

def ut_gen(model: DecoderBase, prompt: str, do_sample: bool = True, num_samples: int = 50) -> List[str]:
    if do_sample:
        assert model.temperature > 0, "Temperature must be greater than 0!"
    batch_size = min(model.batch_size, num_samples)

    vllm_outputs = model.llm.generate(
        [prompt] * batch_size,
        SamplingParams(temperature=model.temperature, max_tokens=model.max_new_tokens, top_p=0.95 if do_sample else 1.0, stop=model.eos),
        use_tqdm=False,
    )

    gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
    return gen_strs


def construct_ut_gen_prompt(prompt: str, entry_point: str) -> str:
    return f"""# Given a docstring, continue to write the following code with 10 valid assertion statements to check the correctness of the function. Provide diverse test cases. 

{prompt}
# check the correctness of `{entry_point}` with 10 different valid assertion statements in the form of \"assert {entry_point}(...) == ...\"
assert """


def check_test_case_validation(test_case):
    if len(test_case.strip()) < 1:
        return False
    if 'assert' not in test_case:
        return False
    try:
        multi_line_test_case = test_case.replace("\n", "\n    ")
        assert_in_a_block = f'try:\n    {multi_line_test_case}\nexcept:\n    pass\n'
        compile(assert_in_a_block, '', 'exec')
        return True
    except Exception:
        return False


def test_case_extract(content, entry_point):
    def _truncate(content):
        for identifier in ['\nclass', '\ndef', '\n#', '\nif', '\nprint']:
            if identifier in content:
                content = content.split(identifier)[0]
        return content.strip()
    
    split_by_assert = [f'assert {part}'.strip() for part in f'assert {content}'.split('assert ') if (entry_point.strip() in part) and len(part.strip()) > 0 and '==' in part]
    truncated_test_cases = [_truncate(i) for i in split_by_assert]
    checked_assertions = [i for i in truncated_test_cases if check_test_case_validation(i)]
    return checked_assertions


def code_generate(args, workdir: PathLike, model: DecoderBase, id_range=None):
    with Progress(
        TextColumn(f"{args.dataset} •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
        if args.dataset == "humaneval":
            from evalplus.data import get_human_eval_plus

            dataset = get_human_eval_plus()
            prompts = load_jsonl("other_data/HumanEval_for_test_case_generation_processed.jsonl")
            prompts = {prompt["task_id"]: prompt for prompt in prompts}
            assert all(task_id in dataset for task_id in prompts)
            
        elif args.dataset == "mbpp":
            from evalplus.data import get_mbpp_plus

            dataset = get_mbpp_plus()
            prompts = load_jsonl("other_data/mbpp_sanitized_for_test_case_generation_processed.jsonl")
            prompts = {prompt["task_id"].replace("MbppEval", "Mbpp"): prompt for prompt in prompts}
            assert all(task_id in prompts for task_id in dataset)
        elif args.dataset == "lcb":
            # load pickle file
            with open("other_data/selected_lcb.pkl", "rb") as f:
                dataset = pickle.load(f)
            prompts = {task_id: {"prompt": dataset[task_id]["prompt"].rstrip() + "\n    pass", "entry_point": dataset[task_id]["entry_point"]} for task_id in dataset}
            # we dont test task_ids since it's the same dataset without version issues
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        
        save_data = []
        for task_id, task in p.track(dataset.items()):
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            p_name = task_id.replace("/", "_")

            log = f"TestCaseGen: {p_name} @ {model}"
            p.console.print(log)
            
            # I added the the replace method, just to make sure that the prompt is in the same format as the samples as the samples also include the prompt.
            prompt = prompts[task_id]["prompt"].replace("    ", "\t")
            entry_point = prompts[task_id]["entry_point"]
            
            prompt = construct_ut_gen_prompt(prompt, entry_point)

            while True:
                try:
                    all_valid_test_cases = []
                    for _ in range(16):
                        rtn = ut_gen(model, prompt, do_sample=not args.greedy, num_samples=args.bs)
                        assert len(rtn) >= 1
                        for test_input_code in rtn:
                            for single_assertion in test_case_extract(test_input_code, entry_point):
                                if single_assertion not in all_valid_test_cases:
                                    all_valid_test_cases.append(single_assertion)

                        if len(all_valid_test_cases) >= 500:
                            break
                        time.sleep(10)

                except Exception as e:
                    print(e)

                p.console.print(len(all_valid_test_cases))
                ### check the generated test cases ###
                if task_id.split("/")[-1] == "2":
                    print("\n".join(all_valid_test_cases[:20]))
                ### END of checking ###
                save_data.append({"prompt": prompt, "test_cases": all_valid_test_cases, "task_id": task_id})

                try:
                    pickle.dump(save_data, open(f"{workdir}/test_cases_new.pkl", 'wb'))
                    break
                except Exception as e:
                    print(e)
                    save_data = save_data[:-1]
                    continue

            shutil.move(f"{workdir}/test_cases_new.pkl", f"{workdir}/test_cases.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--dataset", required=True, type=str, choices=["humaneval", "mbpp", "lcb"])
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    # id_range is list
    parser.add_argument("--id-range", default=None, nargs="+", type=int)
    args = parser.parse_args()

    if args.greedy and (args.temperature != 0 or args.bs != 1):
        args.temperature = 0
        args.bs = 1
        print("Greedy decoding ON (--greedy): setting bs=1, temperature=0")

    if args.id_range is not None:
        assert len(args.id_range) == 2, "id_range must be a list of length 2"
        assert args.id_range[0] < args.id_range[1], "id_range must be increasing"
        args.id_range = tuple(args.id_range)

    os.makedirs(args.root, exist_ok=True)
    os.makedirs(os.path.join(args.root, args.dataset), exist_ok=True)
    
    # Make dir for codes generated by each model
    args.model = args.model.lower()
    model = make_model(name=args.model, batch_size=args.bs, temperature=args.temperature)
    workdir = os.path.join(args.root, args.dataset + "_unit_tests", args.model + f"_temp_{args.temperature}")
    if args.greedy:
        workdir = workdir.replace("temp_0.0", "temp_0")
    
    os.makedirs(workdir, exist_ok=True)

    with open(os.path.join(workdir, "args.txt"), "w") as f:
        f.write(str(args))

    code_generate(args, workdir=workdir, model=model, id_range=args.id_range)


if __name__ == "__main__":
    main()
