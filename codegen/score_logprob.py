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
            
        final_logprobs = {}

        for task_id, task in p.track(dataset.items()):
            # temporary solution for mbpp
            #if args.dataset == "mbpp":
                #if task_id in ["Mbpp/6", "Mbpp/7", "Mbpp/8", "Mbpp/9"]:
                    #continue
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
            
            log = f"Logprob: {p_name} @ {model}"
            n_existing = 0
            nsamples = args.n_samples - n_existing
            p.console.print(log)
            
            sidx = args.n_samples - nsamples
            
            outputs = {}
            # I added the the replace method, just to make sure that the prompt is in the same format as the samples as the samples also include the prompt.
            if model.conversational:
                prompt = model.construct_instruction(task["prompt"]).replace("    ", "\t")
            else:
                prompt = task["prompt"].replace("    ", "\t")
            # first we get the logprob for the prompt
            prompt_logprob = model.score_logprob([prompt], num_samples=1)[0]
            prompt_indices = [list(token.keys())[0] for token in prompt_logprob if token != None]
            prompt_logprob = [list(token.values())[0] for token in prompt_logprob if token != None]
            
            while sidx < args.n_samples:
                # now we start to score logprob for each hypothesis, note that we only have code for vllm decoders
                if model.conversational:
                    logprobs = model.score_logprob(
                        [prompt + samples[idx].replace("    ", "\t") for idx in samples if idx >= sidx],
                        num_samples=args.n_samples - sidx,
                    )
                else:
                    logprobs = model.score_logprob(
                        [samples[idx].replace("    ", "\t") for idx in samples if idx >= sidx],
                        num_samples=args.n_samples - sidx,
                    )
                assert logprobs, "No logprobs from model!"
                
                # now I want to update the outputs dictionary
                for idx, logprob in enumerate(logprobs):
                    logprob_indices = [list(token.keys())[0] for token in logprob if token != None]
                    logprob_values = [list(token.values())[0] for token in logprob if token != None]
                    # check if the starting token ids are equal to the prompt_indices
                    assert logprob_indices[:len(prompt_indices)] == prompt_indices, "Indices are not equal"
                    # only append the logprob values that are not in the prompt_logprob
                    outputs[sidx + idx] = logprob_values[len(prompt_logprob):]
                sidx += len(logprobs)
            
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
    with open(os.path.join(workdir, "logprobs_full.pkl"), "wb") as f:
        pickle.dump(final_logprobs, f)


if __name__ == "__main__":
    main()
