import json

import os
import pickle
import time
from evalplus.data.utils import CACHE_DIR
from evalplus.gen.util import trusted_exec

def load_json(dir_path: str):
    with open(dir_path, "r") as f:
        return json.load(f)

def load_jsonl(dir_path: str):
    # notice that we use the jsonl file to store the data
    with open(dir_path, "r") as f:
        return [json.loads(line) for line in f]
    
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


# We save the unit test generation prompt here for future use.

#def construct_ut_gen_prompt(prompt: str, entry_point: str) -> str:
#    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
#I have this function stub, please generate 50 test cases for this function. The function stub is as follow:
#```python
#{prompt}
#```
#- Each test case is in the form of assertion statement, for example: assert {entry_point}(...) == ...
#- Each test case is in a single line
#- The length of each test case should be too long, ideally less than or equal to 150 letters
#- The test input should not be too long
#- The inputs of test cases should be diverse and cover corner cases of the function
#- Test cases should not be repeated

### Response:
#Here are 50 test cases for function `{entry_point}`:
#```python
#assert {entry_point}("""