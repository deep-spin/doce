import argparse
import json
import multiprocessing
import os
import pickle
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
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
    untrusted_check,
)
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec

# 1st item: the status
# 2nd item (optional): the detailed pass/fail boolean for each input
Result = Tuple[str, List[Any], List[Any], List[bool]]


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


def check_correctness(
    dataset: str,
    completion_id: int,
    task_id: str,
    solution_id: str,
    problem: Dict[str, Any],
    solution: str,
    expected_output: Dict[str, List],
    base_only=False,
    fast_check=False,
    identifier=None,
    min_time_limit: float = 0.1,
    gt_time_limit_factor: float = 2.0,
) -> Dict[str, Result]:  # {...}, "base" | "plus" -> (status, details)
    ret = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
    }
    
    ret["base"] = untrusted_check(
        dataset,
        solution,
        task_id,
        solution_id,
        "base",
        problem["base_input"],
        problem["entry_point"],
        expected=expected_output["base"],
        atol=problem["atol"],
        ref_time=expected_output["base_time"],
        fast_check=fast_check,
        min_time_limit=min_time_limit,
        gt_time_limit_factor=gt_time_limit_factor,
    )

    if not base_only:
        ret["plus"] = untrusted_check(
            dataset,
            solution,
            task_id,
            solution_id,
            "plus",
            problem["plus_input"],
            problem["entry_point"],
            expected=expected_output["plus"],
            atol=problem["atol"],
            ref_time=expected_output["plus_time"],
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )
    #print(f"done testing {task_id}, {solution_id}")
    return ret


def evaluate(flags):

    # assign number of CPUs if it's not specified
    if flags.parallel is None:
        n_workers = max(1, multiprocessing.cpu_count() // 2)
    else:
        n_workers = flags.parallel

    # decide the format of the result file
    if os.path.isdir(flags.samples):
        result_path = os.path.join(flags.samples, "eval_results.json")
    else:
        assert flags.samples.endswith(".jsonl")
        result_path = flags.samples.replace(".jsonl", "_eval_results.json")

    # if results already exist, just read it
    if os.path.isfile(result_path) and not flags.i_just_wanna_run:
        print(f"Load from previous results from {result_path}")
        with open(result_path, "r") as f:
            results = json.load(f)

        results = compatible_eval_result(results)
    # otherwise, we start to evaluate
    else:
        if flags.dataset == "humaneval":
            problems = get_human_eval_plus(mini=flags.mini, noextreme=flags.noextreme)
            dataset_hash = get_human_eval_plus_hash(
                mini=flags.mini, noextreme=flags.noextreme
            )
            expected_output = get_groundtruth(problems, dataset_hash, [])
        elif flags.dataset == "mbpp":
            problems = get_mbpp_plus(mini=flags.mini, noextreme=flags.noextreme)
            dataset_hash = get_mbpp_plus_hash(
                mini=flags.mini, noextreme=flags.noextreme
            )
            expected_output = get_groundtruth(
                problems,
                dataset_hash,
                MBPP_OUTPUT_NOT_NONE_TASKS,
            )
        elif flags.dataset == "lcb":
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
            
        else:
            raise ValueError(f"Unknown dataset: {flags.dataset}")

        results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "hash": dataset_hash,
            "eval": {},
        }
        
        lcb_imports = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n"
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            eval_results = defaultdict(list)  # task_id ->
            remainings = set()

            print("Reading samples...")
            for sample in tqdm(load_solutions(flags.samples)):
                task_id = sample["task_id"]
                solution_id = sample["solution_id"]
                solution = (
                    sample["solution"]
                    if "solution" in sample
                    else problems[task_id]["prompt"] + sample["completion"]
                )
                if flags.dataset == "lcb":
                    solution = lcb_imports + solution
                remainings.add(sample["_identifier"])
                args = (
                    flags.dataset,
                    completion_id[task_id],
                    task_id,
                    solution_id,
                    problems[task_id],
                    solution,
                    expected_output[task_id],
                    flags.base_only,
                    not flags.test_details,  # fast_check
                    sample["_identifier"],
                    flags.min_time_limit,
                    flags.gt_time_limit_factor,
                )
                futures.append(executor.submit(check_correctness, *args))
                completion_id[task_id] += 1
                n_samples += 1

            assert n_samples == len(remainings), "Missing problems in unfinished"
            ### okay if we turn this on, there'll be problems w.r.t. number of problems unmatched (as we dont generate on all problems)
            #assert len(completion_id) == len(problems), "Missing problems in samples"

            def stucking_checker():
                while remainings:
                    last_size = len(remainings)
                    time.sleep(20)
                    if last_size != len(remainings) or len(remainings) == 0:
                        continue
                    # Potential stucking
                    warn("No samples had finished testing in the last 20s")
                    warn(f"{len(remainings)} samples to be tested: {remainings}")

            threading.Thread(target=stucking_checker).start()

            for future in tqdm(as_completed(futures), total=n_samples):
                try:
                    result = future.result(timeout=10)
                    remainings.remove(result["_identifier"])
                    eval_results[result["task_id"]].append(result)
                except TimeoutError:
                    print(f"A sample timed out after 10 seconds")
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
        # sort the results for each problem by completion_id
        for task_id, task_results in eval_results.items():
            task_results.sort(key=lambda x: x["completion_id"])
            results["eval"][task_id] = []
            for res in task_results:

                def get_failed_tests(stat, details, inputs) -> List[Any]:
                    if stat == PASS or not details:
                        return []

                    if flags.test_details:
                        return [
                            inputs[i] for i in range(len(details)) if not details[i]
                        ]

                    # else => simply return the only and the last fail test
                    return [inputs[len(details)]]

                base_stat, base_details, solution_id = res["base"]
                if flags.dataset == "lcb":
                    base_fail_tests = []
                else:
                    base_fail_tests = get_failed_tests(
                        base_stat, base_details, problems[task_id]["base_input"]
                    )

                # initialize plus tests
                plus_stat = None
                plus_details = []
                plus_fail_tests = []

                # with plus tests
                if not flags.base_only:
                    plus_stat, plus_details, _ = res["plus"]
                    if flags.dataset == "lcb":
                        plus_fail_tests = []
                    else:
                        plus_fail_tests = get_failed_tests(
                            plus_stat, plus_details, problems[task_id]["plus_input"]
                        )

                if flags.dataset == "mbpp":
                    base_fail_tests = mbpp_serialize_inputs(task_id, base_fail_tests)
                    plus_fail_tests = mbpp_serialize_inputs(task_id, plus_fail_tests)

                appended_result ={
                        "task_id": task_id,
                        "solution_id": solution_id,
                        "solution": res["solution"],
                        "base_status": base_stat,
                        "plus_status": plus_stat,
                        "base_details": base_details,
                        "plus_details": plus_details,
                        "base_fail_tests": base_fail_tests,
                        "plus_fail_tests": plus_fail_tests,
                    }
                if flags.dataset == "lcb":
                    appended_result["solution"] = res["solution"].replace(lcb_imports, "")
                    appended_result["difficulty"] = problems[task_id]["difficulty"]
                    appended_result["contest_date"] = problems[task_id]["contest_date"]

                results["eval"][task_id].append(appended_result)

    if os.path.isfile(result_path) and flags.i_just_wanna_run:
        decision = ""
        while decision.lower() not in ["y", "n"]:
            print(f"{result_path} already exists. Press [Y/N] to overwrite or exit...")
            decision = input()

        if decision.lower() == "y":
            # mv the file to a backup
            new_path = result_path + ".bak"
            while os.path.isfile(new_path):
                new_path += ".bak"
            os.rename(result_path, new_path)
            print(f"Backup {result_path} to {new_path}")

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(results, f)

    # Calculate pass@k.
    total = np.array([len(r) for r in results["eval"].values()])
    
    base_correct = []
    new_correct = []

    for res in results["eval"].values():
        bc = sum([r["base_status"] == PASS for r in res])
        base_correct.append(bc)
        if not flags.base_only:
            new_correct.append(
                sum(
                    [
                        res[i]["base_status"] == res[i]["plus_status"] == PASS
                        for i in range(len(res))
                    ]
                )
            )
    base_correct = np.array(base_correct)

    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean()
        for k in [1, 5, 10, 50, 100, 200]
        if total.min() >= k
    }
    cprint(f"{flags.dataset} (base tests)", "red")
    for k, v in pass_at_k.items():
        cprint(f"{k}:\t{v:.6f}", "red")
        pass

    if new_correct:
        cprint(f"{flags.dataset}+ (base + extra tests)", "green")
        pass
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, np.array(new_correct), k).mean()
            for k in [1, 5, 10, 50, 100, 200]
            if (total >= k).all()
        }
        for k, v in pass_at_k.items():
            cprint(f"{k}:\t{v:.6f}", "green")
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, type=str, choices=["humaneval", "mbpp", "lcb"]
    )
    parser.add_argument("--samples", required=True, type=str)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--parallel", default=None, type=int)
    parser.add_argument("--i-just-wanna-run", action="store_true")
    parser.add_argument("--test-details", action="store_true")
    parser.add_argument("--min-time-limit", default=1, type=float)
    parser.add_argument("--gt-time-limit-factor", default=4.0, type=float)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument(
        "--noextreme", action="store_true", help="Omit extreme test inputs"
    )
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
