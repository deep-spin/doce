import os
from evalplus.sanitize import sanitize
from typing import Any, Dict, List, Optional, Tuple, Union
import itertools
import multiprocessing
import time
from multiprocessing import Array, Value
from typing import Any, Dict, List, Tuple, Union

from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
    load_solutions,
)

import argparse

import numpy as np
import pickle

from evalplus.data.utils import CACHE_DIR
from evalplus.eval import *
from evalplus.gen.util import trusted_exec
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS, _poly
from evalplus.eval.utils import TimeoutException
from evalplus.eval.utils import (
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)
import re
import resource
import traceback

SUCCESS = "pass"
FAILED = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: SUCCESS, _FAILED: FAILED, _TIMEOUT: TIMEOUT, _UNKNOWN: None}

def load_jsonl(dir_path: str):
    with open(dir_path, "r") as f:
        return [json.loads(line) for line in f]


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


class MyCustomException(BaseException):
    def __init__(self, message):
        self.message = message


def unsafe_execute(
    dataset: str,
    entry_point: str,
    code: str,
    inputs,
    expected: List,
    time_limits,
    atol,
    fast_check,
    stat: Value,
    details: Array,
    progress: Value,
    feedback: Value,
    feedback_size: int,
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
                try:
                    fn = exec_globals[entry_point]
                except KeyError as e:
                    raise f"Please rename your function to {entry_point} as there is no function named {entry_point}."
                except BaseException as e:
                    raise MyCustomException("An error occurred.")
            for i, inp in enumerate(inputs):
                # try:
                with time_limit(time_limits[i]):
                    with swallow_io():
                        out = fn(*inp)

                exp = expected[i]
                exact_match = out == exp

                # ================================================ #
                # ============== special oracles ================= #
                if dataset == "mbpp":
                    if ("are_equivalent" == entry_point):  # Mbpp/164 special oracle
                        exact_match = exact_match or True
                    elif "sum_div" == entry_point:  # Mbpp/295 special oracle
                        exact_match = exact_match or out == 0
                    elif entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
                        exact_match = set(out) == set(exp)
                    elif entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
                        # exp is True  if not None
                        #        False if None
                        if isinstance(out, bool):
                            exact_match = out == exp
                        else:
                            exact_match = exp == (out is not None)

                if dataset == "humaneval":
                    if "find_zero" == entry_point:
                        assert _poly(*inp, out) <= atol, f"With the above function, the assertion is `{entry_point}({inp}) == {exp}` but the real execution output is {out}."
                        #f"The results aren't as expected.\nInput: {inp}\nExpected Output: {exp}\nActual Output: {out}"
                # ============== special oracles ================= #
                # ================================================ #
                
                if atol == 0 and is_floats(exp):
                    atol = 1e-6  # enforce atol for float comparison
                if not exact_match and atol != 0:
                    try:
                        # explicitly set rtol=1e-07
                        # to match `np.testing.assert_allclose`'s default values
                        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
                    except BaseException as e:
                        raise AssertionError(f"With the above function, the assertion is `{entry_point}({inp}) == {exp}` but the real execution output is {out}.")
                else:
                    assert exact_match, f"With the above function, the assertion is `{entry_point}({inp}) == {exp}` but the real execution output is {out}."

                details[i] = True
                progress.value += 1
            stat.value = _SUCCESS
            padding = feedback_size - len(SUCCESS)
            feedback.value = (SUCCESS + " " * padding).encode('utf-8')
        except TimeoutException as e:
            stat.value = _FAILED
            error_str="Execution timed out."
            padding = max(0, feedback_size - len(error_str))
            feedback.value = (error_str + " " * padding).encode('utf-8')
        except AssertionError as e:
            stat.value = _FAILED
            error_str=str(e)[:feedback_size]
            padding = max(0, feedback_size - len(error_str))
            feedback.value = (error_str + " " * padding).encode('utf-8')
        except MyCustomException as e:
            stat.value = _FAILED
            max_line_lengths = max([len(line) for line in code.split('\n')])
            if max_line_lengths > 300:
                error_str="The generated code has a line that's too long which does not make sense."
            else:
                error_str=e.message[:feedback_size]
            #error_str=e.message[:feedback_size]
            padding = max(0, feedback_size - len(error_str))
            feedback.value = (error_str + " " * padding).encode('utf-8') 
        except BaseException as e:
            stat.value = _FAILED
            max_line_lengths = max([len(line) for line in code.split('\n')])
            if max_line_lengths > 300:
                error_str="The generated code has a line that's too long which does not make sense."
            else:
                error_traceback = traceback.format_exc()
                match = re.search(r'(File "<string>".*)', error_traceback, re.DOTALL)
                if match:
                    error_traceback = match.group(1)
                elif "assert _poly" in error_traceback:
                    if "TypeError: _poly() argument after *" in error_traceback:
                        error_traceback = "TypeError: Invalid output type, output must be an iterable."
                    else:
                        delimiter = f"With the above function, the assertion is `{entry_point}({inp}) == {exp}` but the real execution output is {out}."
                        error_traceback = error_traceback.split(delimiter)[-1]

                error_str=str(error_traceback)[:feedback_size]
            padding = max(0, feedback_size - len(error_str))
            feedback.value = (error_str + " " * padding).encode('utf-8')
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir

def unsafe_execute_fast(
    dataset: str,
    entry_point: str,
    code: str,
    inputs,
    expected: List,
    time_limits,
    atol,
    fast_check,
    stat: Value,
    details: Array,
    progress: Value,
    feedback: Value,
    feedback_size: int,
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
                try:
                    fn = exec_globals[entry_point]
                except KeyError as e:
                    raise f"Please rename your function to {entry_point} as there is no function named {entry_point}."
                except BaseException as e:
                    raise MyCustomException("An error occurred.")
            for i, inp in enumerate(inputs):
                # try:
                with time_limit(time_limits[i]):
                    with swallow_io():
                        out = fn(*inp)

                exp = expected[i]
                exact_match = out == exp

                # ================================================ #
                # ============== special oracles ================= #
                if dataset == "mbpp":
                    if ("are_equivalent" == entry_point):  # Mbpp/164 special oracle
                        exact_match = exact_match or True
                    elif "sum_div" == entry_point:  # Mbpp/295 special oracle
                        exact_match = exact_match or out == 0
                    elif entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
                        exact_match = set(out) == set(exp)
                    elif entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
                        # exp is True  if not None
                        #        False if None
                        if isinstance(out, bool):
                            exact_match = out == exp
                        else:
                            exact_match = exp == (out is not None)

                if dataset == "humaneval":
                    if "find_zero" == entry_point:
                        assert _poly(*inp, out) <= atol, f"With the above function, the assertion is `{entry_point}({inp}) == {exp}` but the real execution output is {out}."
                # ============== special oracles ================= #
                # ================================================ #
                
                if atol == 0 and is_floats(exp):
                    atol = 1e-6  # enforce atol for float comparison
                if not exact_match and atol != 0:
                    try:
                        # explicitly set rtol=1e-07
                        # to match `np.testing.assert_allclose`'s default values
                        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
                    except BaseException as e:
                        raise AssertionError(f"With the above function, the assertion is `{entry_point}({inp}) == {exp}` but the real execution output is {out}.")
                else:
                    assert exact_match, f"With the above function, the assertion is `{entry_point}({inp}) == {exp}` but the real execution output is {out}."

                details[i] = True
                progress.value += 1
            stat.value = _SUCCESS
            padding = feedback_size - len(SUCCESS)
            feedback.value = (SUCCESS + " " * padding).encode('utf-8')
        except BaseException as e:
            stat.value = _FAILED
            error_str=str("failed")
            padding = max(0, feedback_size - len(error_str))
            feedback.value = (error_str + " " * padding).encode('utf-8')
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir

def untrusted_check(
    dataset: str,
    code: str,
    inputs: List[Any],
    entry_point: str,
    expected,
    atol,
    ref_time: List[float],
    fast_check: bool = False,
    min_time_limit: float = 0.1,
    gt_time_limit_factor: float = 4.0,
) -> Tuple[str, np.ndarray]:

    time_limits = [max(min_time_limit, gt_time_limit_factor * t) for t in ref_time]
    timeout = sum(time_limits) + 1
    if not fast_check:
        timeout += 1  # extra time for data collection

    # shared memory objects
    progress = Value("i", 0)
    stat = Value("i", _UNKNOWN)
    details = Array("b", [False for _ in range(len(inputs))])
    feedback_size = 500
    feedback = Array('c', b'\0' * feedback_size)

    if fast_check:
        p = multiprocessing.Process(
            target=unsafe_execute_fast,
            args=(
                dataset,
                entry_point,
                code,
                inputs,
                expected,
                time_limits,
                atol,
                fast_check,
                stat,
                details,
                progress,
                feedback,
                feedback_size,
            ),
        )
    else:
        p = multiprocessing.Process(
            target=unsafe_execute,
            args=(
                dataset,
                entry_point,
                code,
                inputs,
                expected,
                time_limits,
                atol,
                fast_check,
                stat,
                details,
                progress,
                feedback,
                feedback_size,
            ),
        )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat = _mapping[stat.value]
    details = details[: progress.value]
    feedback = feedback.value.decode("utf-8").strip()
    if entry_point not in code:
        feedback = f"Please rename your function to {entry_point} as there is no function named {entry_point}."

    if not stat:
        stat = TIMEOUT

    if stat == SUCCESS:
        if len(details) != len(inputs) or not all(details):
            stat = FAILED

    return stat, details, feedback

        
def get_errors(args, problems, expected_output, inputs = None):
    
    # check if f"{args.work_dir}/{args.dataset}/{args.gen_dir}/exec_outputs.pkl" exists. If it does, we can skip the generation of the outputs
    if os.path.exists(f"{args.work_dir}/{args.dataset}/{args.gen_dir}/errors.pkl"):
        print(f"{args.work_dir}/{args.dataset}/{args.gen_dir}/errors.pkl exists. Skipping generation of outputs.")
        return "skipped"
    
    with open(f"{args.work_dir}/{args.dataset}/{args.gen_dir}/eval_results.json", "r") as f:
        eval_results = json.load(f)
    for task_id in eval_results["eval"]:
        eval_results["eval"][task_id] = sorted(eval_results["eval"][task_id], key=lambda x: int(x["solution_id"]))
        

    from tqdm import tqdm
    exec_outputs = {}
    for task_id in tqdm(eval_results["eval"]):
        exec_outputs[task_id] = {}
        if args.dataset == "mbpp":
            task_inputs = problems[task_id]["base_input"][:1]
            task_expected_output = expected_output[task_id]["base"][:1]
            task_ref_time = expected_output[task_id]["base_time"][:1]
        else:
            assert inputs is not None
            task_inputs = inputs[task_id]["trial_input"]
            task_expected_output = expected_output[task_id]["base"]
            task_ref_time = expected_output[task_id]["base_time"]
        for i, hyp in enumerate(eval_results["eval"][task_id]):
            exec_outputs[task_id][i] = {}
            assert int(hyp["solution_id"]) == i
            ###### start of special case for mbpp ######
            # special case for mbpp, cuz it uses the first test case for eval as the trial test case
            if args.dataset == "mbpp":
                if hyp["base_status"] == "pass":
                    exec_outputs[task_id][i]["base"] = {"status": "pass", "error": "pass"}
                    continue
            ###### end of special case for mbpp ######
            stat, _, feedback = untrusted_check(
                dataset=args.dataset,
                code=hyp["solution"],
                inputs=task_inputs,
                entry_point=problems[task_id]["entry_point"],
                expected=task_expected_output,
                atol=problems[task_id]["atol"],
                ref_time=task_ref_time,
                fast_check=args.fast_check,
                min_time_limit=1,
                gt_time_limit_factor=4.0,
            )

            exec_outputs[task_id][i]["base"] = {"status": stat, "error": feedback}

    # save exec_outputs to pickle
    with open(f"{args.work_dir}/{args.dataset}/{args.gen_dir}/errors.pkl", "wb") as f:
        pickle.dump(exec_outputs, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="/mnt/scratch-artemis/haausing/code_reranking/evalplus_outputs")
    parser.add_argument("--gen_dir", type=str, default="deepseek-coder-33b-instruct_temp_0.8")
    parser.add_argument("--dataset", required=True, type=str, choices=["humaneval", "mbpp"])
    parser.add_argument("--fast_check", action="store_true")
    args = parser.parse_args()
    
    if args.dataset == "mbpp":
        problems = get_mbpp_plus()
        dataset_hash = get_mbpp_plus_hash()
        expected_output = get_groundtruth(
            problems,
            dataset_hash,
            MBPP_OUTPUT_NOT_NONE_TASKS,
        )
        get_errors(args, problems, expected_output)
    else:
        problems = get_human_eval_plus()
        dataset_hash = get_human_eval_plus_hash()
        with open("/mnt/scratch-artemis/haausing/code_reranking/code/evalplus/other_data/trial_expected_output.pkl", "rb") as f:
            expected_output = pickle.load(f)
        inputs = load_jsonl("/mnt/scratch-artemis/haausing/code_reranking/code/evalplus/other_data/trial_inputs.jsonl")
        inputs = {e["task_id"]:e for e in inputs}
        get_errors(args, problems, expected_output, inputs)


if __name__ == "__main__":
    main()