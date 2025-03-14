import itertools
from multiprocessing import Array, Value
from typing import Any, Dict, List, Tuple, Union

import json

import numpy as np

from evalplus.eval._special_oracle import (
    MBPP_OUTPUT_NOT_NONE_TASKS,
    MBPP_OUTPUT_SET_EQ_TASKS,
    _poly,
)
from evalplus.eval.utils import (
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)

import argparse
import json
import multiprocessing
import os
import pickle
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Tuple
from warnings import warn

import numpy as np
from termcolor import cprint
from tqdm import tqdm

from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
    load_solutions,
)
from evalplus.data.mbpp import mbpp_serialize_inputs
from evalplus.data.utils import CACHE_DIR
from evalplus.eval import (
    FAIL,
    PASS,
    compatible_eval_result,
    estimate_pass_at_k,
)

from evalplus.gen.util import trusted_exec

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3


def unsafe_execute_with_outputs(
    dataset: str,
    entry_point: str,
    code: str,
    task_id: str,
    solution_id: str,
    inputs,
    expected: List,
    time_limits,
    stat: str,
    details: List[int],
    outputs: Dict[str, Any],
):
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        # Disable functionalities that can make destructive changes to the test.
        # allow only 4GB memory usage
        maximum_memory_bytes = 4 * 1024 * 1024 * 1024
        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)
        exec_globals = {}
        try:
            with swallow_io():
                exec(code, exec_globals)
                fn = exec_globals[entry_point]
            for i, inp in enumerate(inputs):
                if i < len(details):
                    if details[i] == 1:
                        outputs[i] = expected[i]
                        continue
                elif i >= len(details):
                    continue
                try:
                    with time_limit(time_limits[i]):
                        with swallow_io():
                            out = fn(*inp)
                        
                    if dataset == "mbpp":
                        if entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
                            out = set(out)
                        elif entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
                            if not isinstance(out, bool):
                                out = out is not None
                    outputs[i] = out
                except BaseException as e:
                    outputs[i] = f"failed: {e}"
                    continue
        except BaseException as e:
            pass
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def get_groundtruth(problems, hashcode, tasks_only_output_not_none):
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
    if os.path.exists(cache_file):
        #print(f"Load from ground-truth from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    #print("Computing expected output...")
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
    #print(f"Expected outputs computed in {time.time() - tbegin:.2f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(expected_output, f)

    return expected_output


def untrusted_check( dataset: str, entry_point: str, code: str, task_id: str, solution_id: str, inputs: List[Any], expected: List[Any], ref_time, stat, details):
    min_time_limit=1.0
    gt_time_limit_factor=4.0
    fast_check = False
    time_limits = [max(min_time_limit, gt_time_limit_factor * t) for t in ref_time]
    timeout = min(os.getenv("EVALPLUS_TIMEOUT_PER_TASK", 60), sum(time_limits)) + 1
    if not fast_check:
        timeout += 1  # extra time for data collection
    manager = multiprocessing.Manager()
    outputs = manager.dict()

    p = multiprocessing.Process(
        target=unsafe_execute_with_outputs,
        args=(dataset, entry_point, code, task_id, solution_id, inputs, expected, time_limits, stat, details, outputs),
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    return outputs.copy()


def get_outputs(args):
    
    if args.dataset == "mbpp":
        problems = get_mbpp_plus()
        dataset_hash = get_mbpp_plus_hash()
        expected_output = get_groundtruth(
            problems,
            dataset_hash,
            MBPP_OUTPUT_NOT_NONE_TASKS,
        )
    elif args.dataset == "humaneval":
        problems = get_human_eval_plus()
        dataset_hash = get_human_eval_plus_hash()
        expected_output = get_groundtruth(
            problems,
            dataset_hash,
            []
        )
    elif args.dataset == "lcb":
        # load pickle file
        with open("other_data/selected_lcb.pkl", "rb") as f:
            problems = pickle.load(f)
        dataset_hash = "lcb"
        with open("other_data/refined_lcb_inputs.pkl", "rb") as f:
            refined_inputs = pickle.load(f)
        for task_id in problems:
            problems[task_id]["base_input"] = refined_inputs[task_id]["base_input"]
        with open("other_data/refined_lcb_outputs.pkl", "rb") as f:
            expected_output = pickle.load(f)
    
    # check if f"{args.work_dir}/{args.gen_dir}/exec_outputs_v2.pkl" exists. If it does, we can skip the generation of the outputs
    if args.gen_fast:
        if os.path.exists(f"{args.work_dir}/{args.dataset}/{args.gen_dir}/genut_exec_outputs_v2.pkl"):
            print(f"{args.work_dir}/{args.dataset}/{args.gen_dir}/exec_outputs_v2.pkl exists. Skipping generation of outputs.")
            return "skipped"
    else:
        if os.path.exists(f"{args.work_dir}/{args.dataset}/{args.gen_dir}/genut_exec_outputs.pkl"):
            print(f"{args.work_dir}/{args.dataset}/{args.gen_dir}/exec_outputs.pkl exists. Skipping generation of outputs.")
            return "skipped"
    
    with open(f"{args.work_dir}/{args.dataset}/{args.gen_dir}/eval_results.json", "r") as f:
        eval_results = json.load(f)
    for task_id in eval_results["eval"]:
        eval_results["eval"][task_id] = sorted(eval_results["eval"][task_id], key=lambda x: int(x["solution_id"]))
        
    
    # generated_test_cases
    with open(f"/mnt/scratch-artemis/haausing/code_reranking/code/MPSC/save/{args.dataset}/{args.gen_dir}/test_cases.pkl", "rb") as f:
        generated_test_cases_io = pickle.load(f)
    generated_test_cases_io = {test_case["task_id"]: test_case for test_case in generated_test_cases_io}
        

    from tqdm import tqdm
    lcb_imports = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n"
    exec_outputs = {}
    for task_id in tqdm(eval_results["eval"]):
        exec_outputs[task_id] = {}
        
        # first we get all the generated inputs
        generated_inputs = []
        for elem in generated_test_cases_io[task_id]["tc_input_output"]:
            try:
                generated_inputs.append(eval(elem['input']))
            except Exception as e:
                pass
                # print(elem['input'])
        
        for i, hyp in enumerate(eval_results["eval"][task_id]):
            exec_outputs[task_id][i] = {}
            assert int(hyp["solution_id"]) == i
            hyp_solution = hyp["solution"]
            if args.dataset == "lcb":
                hyp_solution = lcb_imports + hyp_solution
            try:
                # Set a timeout of 10 seconds for each untrusted_check call
                from evalplus.eval.utils import time_limit
                with time_limit(10):
                    out = untrusted_check(
                        dataset=args.dataset,
                        entry_point=problems[task_id]["entry_point"],
                        code=hyp_solution,
                        task_id=task_id,
                        solution_id=hyp["solution_id"],
                        inputs=generated_inputs,
                        expected=expected_output[task_id]["base"],
                        ref_time=expected_output[task_id]["base_time"],
                        stat=hyp["base_status"],
                        details=hyp["base_details"],
                    )
            except Exception as e:
                print(f"Error processing task {task_id}, solution {i}: {str(e)}")
                out = {"status": "timeout", "error": str(e)}
            exec_outputs[task_id][i]["base"] = out
            exec_outputs[task_id][i]["plus"] = {}

    # save exec_outputs to pickle
    if args.gen_fast:
        with open(f"{args.work_dir}/{args.dataset}/{args.gen_dir}/genut_exec_outputs_v2.pkl", "wb") as f:
            pickle.dump(exec_outputs, f)
    else:
        with open(f"{args.work_dir}/{args.dataset}/{args.gen_dir}/genut_exec_outputs.pkl", "wb") as f:
            pickle.dump(exec_outputs, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="/path/to/evalplus_outputs")
    parser.add_argument("--gen_dir", type=str, default="deepseek-coder-33b-instruct_temp_0.8")
    parser.add_argument("--dataset", type=str, choices=["mbpp", "humaneval", "lcb"])
    parser.add_argument("--gen_fast", action="store_true")
    args = parser.parse_args()

    get_outputs(args)


if __name__ == "__main__":
    main()