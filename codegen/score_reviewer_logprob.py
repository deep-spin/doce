import argparse
import os
import pickle
from os import PathLike
from typing import List, Any

from model import DecoderBase, make_model
from generate import construct_contract_prompt
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")


def process_humaneval_prompt(prompt: str):
    prompt = ''.join(prompt.split('"""')[1:])
    prompt = [e.strip() for e in prompt.split('\n')]
    prompt = '"""\n'+"\n".join(prompt).strip()+'\n"""'
    return prompt

from pyminifier.minification import remove_blank_lines #, remove_comments_and_docstrings
import io, tokenize, re

def remove_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out

def clean_comment(code: str):
    code = remove_comments_and_docstrings(code)
    code = remove_blank_lines(code)
    return code

def remove_print(code):
    code = re.sub("print(.+)", "print('')", code)
    code = re.sub("Error(.+)", "Error('')", code)
    code = re.sub("Exception(.+)", "Exception('')", code)
    code = re.sub("assert (.+), +['\"].+['\"]", "assert \\1", code)
    return code

from evalplus.prompt import MBPP_REVIEWER, HUMANEVAL_REVIEWER

def construct_reviewer_prompt(code: str, task: str):
    if task in ["humaneval", "lcb"]:
        return HUMANEVAL_REVIEWER + f'''### code
```python
{code}
```

Write the docstring for the above code.

### docstring
```python
'''
    elif task == "mbpp":
        return MBPP_REVIEWER + f'''### code
```python
{code}
```

Write the docstring for the above code.

### docstring
```python
'''
    else:
        raise ValueError(f"Unknown task: {task}")
    
def get_logprobs(args, workdir: PathLike, model: DecoderBase, id_range=None):
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
        elif args.dataset == "lcb":
            # load pickle file
            with open("other_data/selected_lcb.pkl", "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
            
        final_logprobs = {}

        for task_id, task in p.track(dataset.items()):
            # temporary solution for mbpp
            if args.dataset == "mbpp":
                if task_id in ["Mbpp/6", "Mbpp/7", "Mbpp/8", "Mbpp/9"]:
                    continue
            #############################################################
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
            for f in os.listdir(os.path.join(workdir, p_name)):
                if f.endswith(".py"):
                    with open(os.path.join(workdir, p_name, f), "r", encoding="utf-8") as file:
                        # delete the suffix .py from f and convert it to interger:
                        samples[int(f.split(".")[0])] = file.read()
            # sort the samples by their keys in ascending order
            samples = dict(sorted(samples.items()))
            # we want to make sure the number of samples is equal to args.n_samples
            assert len(samples) == args.n_samples, f"Number of samples is not equal to {args.n_samples}"
            
            # now we clean comments and prints of all samples, and construct the reviewer prompt (for reviewer)
            for idx in samples:
                try:
                    samples[idx] = clean_comment(samples[idx])
                except:
                    pass
                samples[idx] = remove_print(samples[idx])
                samples[idx] = construct_reviewer_prompt(
                    samples[idx], 
                    args.dataset
                    )
                
            log = f"Reviewer: {p_name} @ {model}"
            n_existing = 0
            nsamples = args.n_samples - n_existing
            p.console.print(log)
            
            sidx = args.n_samples - nsamples
            
            outputs = {}
            # I added the the replace method, just to make sure that the prompt is in the same format as the samples as the samples also include the prompt.
            prompt = task["prompt"].replace("    ", "\t")
            if args.dataset in ["humaneval", "lcb"]:
                prompt = process_humaneval_prompt(prompt)
            # first we get the logprob for the prompt
            #prompt_logprob = model.score_logprob([prompt], num_samples=1)[0]
            #prompt_indices = [list(token.keys())[0] for token in prompt_logprob if token != None]
            #prompt_logprob = [list(token.values())[0] for token in prompt_logprob if token != None]

            timeout_count = 0
            
            while sidx < args.n_samples:
                # now we start to score logprob for each hypothesis, note that we only have code for vllm decoders
                samples_before = [samples[idx].replace("    ", "\t") for idx in samples if idx >= sidx]
                # Set the signal handler and a 5-second alarm
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                if task_id == "LCB/213" and sidx in list(range(60, 68)) + list(range(80, 84)):
                    p.console.print(f"Skipping sidx = {sidx} on task {task_id} as we know it's OOM")
                    logprobs_before = ["no logprobs"]
                    logprobs_after = ["no logprobs"]
                else:
                    try:
                        logprobs_before = model.score_logprob(
                            samples_before,
                            num_samples=args.n_samples - sidx,
                        )

                        samples_after = [samples[idx].replace("    ", "\t") + prompt.strip() for idx in samples if idx >= sidx]
                        logprobs_after = model.score_logprob(
                            samples_after,
                            num_samples=args.n_samples - sidx,
                        )
                        p.console.print(f"Successfully scored sidx = {sidx} on task {task_id}")
                    except TimeoutException:
                        p.console.print(f"Timeout occurred for sidx = {sidx} on task {task_id}")
                        logprobs_before = ["no logprobs"]
                        logprobs_after = ["no logprobs"]
                        timeout_count += 1
                    except Exception as e:
                        # we only provide temporary samples to the model to avoid OOM error
                        p.console.print(f"We have an OOM issue with sidx = {sidx} on task {task_id}")
                        assert args.bs == 1, "We only support batch size 1 for now"
                        logprobs_before = ["no logprobs"]
                        logprobs_after = ["no logprobs"]
                    finally:
                        signal.alarm(0)  # Ensure the alarm is disabled
                    if timeout_count > 1:
                        p.console.print(f"Timeout count for {task_id} exceeds 1, stopping...")
                        raise ValueError("Timeout count exceeds 1")
                
                assert logprobs_before, "No logprobs from model in before!"
                assert logprobs_after, "No logprobs from model in after!"
                
                # now I want to update the outputs dictionary
                
                for idx in range(len(logprobs_after)):
                    if logprobs_before[idx] == "no logprobs":
                        logprob_before_indices = [20] * 5
                        logprob_after_indices = [20] * 5 + [20000] * 5
                        logprob_after_values = [-1000] * 5
                    else:
                        logprob_before_indices = [list(token.keys())[0] for token in logprobs_before[idx] if token != None]
                        logprob_after_indices = [list(token.keys())[0] for token in logprobs_after[idx] if token != None]
                        logprob_after_values = [list(token.values())[0].logprob for token in logprobs_after[idx] if token != None]
                    # check if the starting token ids are equal to the prompt_indices
                    assert logprob_after_indices[:len(logprob_before_indices)] == logprob_before_indices, "Indices are not equal"
                    # only append the logprob values that are not in the prompt_logprob
                    outputs[sidx + idx] = logprob_after_values[len(logprob_before_indices):]
                sidx += len(logprobs_after)
            
            final_logprobs[p_name] = outputs
            with open(os.path.join(workdir, f"reviewer_logprobs_{p_name}.pkl"), "wb") as f:
                pickle.dump(final_logprobs, f)
            ### end ###
    return final_logprobs


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
    parser.add_argument("--suffix", default="", type=str)
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
        + ("" if args.contract_type == "none" else f"-contract-{args.contract_type}")
        + args.suffix,
    )

    final_logprobs = get_logprobs(args, workdir=workdir, model=model, id_range=args.id_range)
    
    # save the final_logprobs to a pickle file
    with open(os.path.join(workdir, "reviewer_logprobs.pkl"), "wb") as f:
        pickle.dump(final_logprobs, f)


if __name__ == "__main__":
    main()
