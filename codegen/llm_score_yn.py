import argparse
import os
import json
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

from evalplus.prompt import MBPP_LEVER_FEW_SHOT, HUMANEVAL_LEVER_FEW_SHOT

def load_json(dir_path: str):
    with open(dir_path, "r") as f:
        return json.load(f)

def load_jsonl(dir_path: str):
    # notice that we use the jsonl file to store the data
    with open(dir_path, "r") as f:
        return [json.loads(line) for line in f]

def mbpp_construct_lever_prompt(prompt: str, unit_test: str, sample: str, inputs: List[Any], outputs: List[Any]) -> str:
    start_prompt = MBPP_LEVER_FEW_SHOT+f"""### instruction
{prompt}
```python
{unit_test}
```

### code
```python
{sample}
```

### feedback
Here are the execution results on unit tests with code above:
"""
    for i, (input, output) in enumerate(zip(inputs, outputs)):
        sample_input = tuple(input)
        sample_output = output
        if type(sample_output) == str and not sample_output.startswith("failed:"):
            sample_output = f"'{sample_output}'"
        start_prompt += f"""# case {i}
Input: {sample_input}
Output: {sample_output}
"""
    start_prompt = start_prompt.strip()
    positive_prompt = f"""{start_prompt}

### decision
Is the code above correct? yes"""
    negative_prompt = f"""{start_prompt}

### decision
Is the code above correct? no"""
    return positive_prompt, negative_prompt

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
        lever_outputs = pickle.load(open(os.path.join(workdir, "lever_outputs.pkl"), "rb"))
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
            
            log = f"LLM_Score: {p_name} @ {model}"
            n_existing = 0
            nsamples = args.n_samples - n_existing
            p.console.print(log)
            
            sidx = args.n_samples - nsamples
            
            outputs = {}
            # I added the the replace method, just to make sure that the prompt is in the same format as the samples as the samples also include the prompt.
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
            positive_prompts = {}
            negative_prompts = {}
            for idx in samples:
                positive_prompts[idx], negative_prompts[idx] = mbpp_construct_lever_prompt(prompt, 
                                                                                           unit_test, 
                                                                                           samples[idx], 
                                                                                           task["base_input"], 
                                                                                           lever_outputs[task_id][idx]["base"])
            
            while sidx < args.n_samples:
                # now we start to score logprob for each hypothesis, note that we only have code for vllm decoders
                if model.conversational:
                    pos_logprobs = model.score_logprob(
                        [positive_prompts[idx] for idx in samples if idx >= sidx],
                        num_samples=args.n_samples - sidx,
                    )
                    neg_logprobs = model.score_logprob(
                        [negative_prompts[idx] for idx in samples if idx >= sidx],
                        num_samples=args.n_samples - sidx,
                    )
                else:
                    raise NotImplementedError("Non-conversational model is not supported yet.")
                    #logprobs = model.score_logprob(
                    #    [samples[idx].replace("    ", "\t") for idx in samples if idx >= sidx],
                    #    num_samples=args.n_samples - sidx,
                    #)
                assert pos_logprobs, "No pos_logprobs from model!"
                assert neg_logprobs, "No neg_logprobs from model!"
                
                # now I want to update the outputs dictionary
                for idx, (pos_logprob, neg_logprob) in enumerate(zip(pos_logprobs, neg_logprobs)):
                    pos_logprob_indices = [list(token.keys())[0] for token in pos_logprob if token != None]
                    neg_logprob_indices = [list(token.keys())[0] for token in neg_logprob if token != None]
                    pos_logprob_values = [list(token.values())[0] for token in pos_logprob if token != None]
                    neg_logprob_values = [list(token.values())[0] for token in neg_logprob if token != None]
                    # check if the tokens except for the last one are the same
                    # I just use a fast version
                    #assert pos_logprob_indices[:-1] == neg_logprob_indices[:-1], "Indices are not equal"
                    assert pos_logprob_indices[-20:-1] == neg_logprob_indices[-20:-1], "Indices are not equal"
                    assert pos_logprob_indices[-1] != neg_logprob_indices[-1], "Should be diff indices for the last token"
                    outputs[sidx + idx] = {
                        "pos": pos_logprob_values[-1], 
                        "neg": neg_logprob_values[-1]
                    }
                sidx += len(pos_logprobs)
            
            final_logprobs[p_name] = outputs
            ### end ###
    return final_logprobs


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
    parser.add_argument("--suffix", default="", type=str)
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
        args.dataset,
        args.model
        + f"_temp_{args.temperature}"
        + ("" if args.contract_type == "none" else f"-contract-{args.contract_type}")
        + args.suffix,
    )

    final_logprobs = get_logprobs(args, workdir=workdir, model=model, id_range=args.id_range)
    
    # save the final_logprobs to a pickle file
    with open(os.path.join(workdir, "llm_score_yn.pkl"), "wb") as f:
        pickle.dump(final_logprobs, f)


if __name__ == "__main__":
    main()
