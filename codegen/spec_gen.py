import argparse
import os
from os import PathLike
from typing import List, Any
import re

import json
import pickle
import time
import shutil

from transformers import AutoTokenizer

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

def spec_gen(
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

def construct_spec_gen_prompt(prompt: str, tokenizer: AutoTokenizer):
    """
    Prompt to generate spec code in two functions `preconditions` and `postconditions`
    """
    instruction = """Given the python method above with docstring, you should write its pre-conditions in one test function \"def preconditions(input):\" and post-conditions in another test function \"def postconditions(input, output):\".
You should ensure invalid input or output of the method will raise error in the two test functions."""

    exemplars = []
    for i in range(1, 4):
        with open(f"./codegen/exemplars/{i}.py", 'r') as f:
            exemplar_codes = f.read()
            exemplar = dict()
            query, answer, _ = re.match(r"(.*)(# Pre-conditions\n.*)(# Test inputs\n.*)", exemplar_codes, re.DOTALL).groups()
            exemplar['query'] = "```python\n" + query + "```\n"
            exemplar['answer'] = "```python\n" + answer + "```\n"

            exemplars.append(exemplar)
    
    question = "```python\n" + prompt + "\n" + "```\n"

    messages = [{"role": "system", "content": "You are an expert programmer that helps to write Python code based on the user request. Don't be too verbose."},
        ]
    
    if exemplars is not None:
        for i, exemplar in enumerate(exemplars):
            messages.append({"role": "user", "content": exemplar['query'] + "\n" + instruction})
            messages.append({"role": "assistant", "content": exemplar['answer']})
    
    messages.append({"role": "user", "content": question + "\n" + instruction})
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "```python\n"

    return prompt

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
            prompts = {task_id: dataset[task_id]["prompt"].rstrip() + "\n    pass" for task_id in dataset}
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_oname)
        
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

            log = f"SpecGen: {p_name} @ {model}"
            p.console.print(log)
            
            # I added the the replace method, just to make sure that the prompt is in the same format as the samples as the samples also include the prompt.
            prompt = prompts[task_id]["prompt"].replace("    ", "\t")
            
            prompt = construct_spec_gen_prompt(prompt, tokenizer)
            #print(prompt)
            #raise ValueError("Stop here, I want to debug by printing the prompt")
            
            spec_code_choices = []
            for _ in range(5):
                # inference with evalplus api
                answers = spec_gen(model, prompt, do_sample=not args.greedy, num_samples=20)
                
                ### debug by printing the answers ###
                #for answer in answers:
                #    print(answer)
                #    print("-"*100)
                #raise ValueError("Stop here, I want to debug by printing the answers")
                ### end of debug ###
                # extract spec code from answers (deprecated, as we already force the LLM to write code)
                #answers = [re.search(r"```python\n(.*)```", answer, re.DOTALL).group(1) for answer in answers if re.search(r"```python\n(.*)```", answer, re.DOTALL) is not None]
                spec_code_choices.extend(answers)

                time.sleep(10)              # generation token limit
            
            fd_mode = 'a'
            p.console.print(len(spec_code_choices))
            with open(f"{workdir}/specs.jsonl", fd_mode) as f:
                f.write(json.dumps({"prompt": prompt, "spec_code_choices": spec_code_choices, "task_id": task_id}) + "\n")     

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--model_oname", required=True, type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument(
        "--dataset", required=True, type=str, choices=["humaneval", "mbpp", "lcb"]
    )
    parser.add_argument("--root", type=str, required=True)
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

    if args.greedy and (args.temperature != 0 or args.bs != 1):
        args.temperature = 0
        args.bs = 1
        print("Greedy decoding ON (--greedy): setting bs=1, temperature=0")

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
    # We no longer need ``` bc we already force the LLM to write code
    #model.eos = [e for e in model.eos if e != "\n```"]
    ### finish modifying the eos ###
    
    workdir = os.path.join(
        args.root,
        args.dataset+"_specs",
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