import argparse
import os
from os import PathLike

import pickle

from model import DecoderBase, make_model
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

import torch.multiprocessing as mp
from utils import *


def construct_contract_prompt(prompt: str, contract_type: str, contract: str) -> str:
    if contract_type == "none":
        return prompt
    elif contract_type == "docstring":
        # embed within the docstring
        sep = ""
        if '"""' in prompt:
            sep = '"""'
        elif "'''" in prompt:
            sep = "'''"
        assert sep != ""
        l = prompt.split(sep)
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        l[1] = (
            l[1] + contract + "\n" + " " * (len(contract) - len(contract.lstrip()) - 1)
        )
        return sep.join(l)
    elif contract_type == "code":
        # at the beginning of the function
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        return prompt + contract


def ask_llm_worker(args, gpu_ids, dataset_chunk, workdir, result_queue):

    # set GPU id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    model = make_model(name=args.model, batch_size=args.bs, temperature=args.temperature)

    for task_id, task in dataset_chunk:
        id_range = args.id_range
        if id_range is not None:
            id_num = int(task_id.split("/")[1])
            low, high = id_range
            if id_num < low or id_num >= high:
                result_queue.put(("skip", f"Skipping {task_id} as it is not in {id_range}"))
                result_queue.put(("progress", 1))
                continue

        p_name = task_id.replace("/", "_")
        if args.contract_type != "none" and task["contract"] == "":
            result_queue.put(("progress", 1))
            continue
        os.makedirs(os.path.join(workdir, p_name), exist_ok=True)
        log = f"Codegen: {p_name} @ {model}"
        n_existing = 0
        if args.resume:
            # count existing .py files
            n_existing = len(
                [
                    f
                    for f in os.listdir(os.path.join(workdir, p_name))
                    if f.endswith(".py")
                ]
            )
            if n_existing > 0:
                log += f" (resuming from {n_existing})"

        nsamples = args.n_samples - n_existing
        result_queue.put(("log", log))

        sidx = args.n_samples - nsamples
        while sidx < args.n_samples:
            outputs = model.codegen(
                construct_contract_prompt(
                    task["prompt"], args.contract_type, task["contract"]
                ),
                do_sample=not args.greedy,
                num_samples=args.n_samples - sidx,
            )
            assert outputs, "No outputs from model!"
            for idx, impl in enumerate(outputs):
                try:
                    with open(os.path.join(workdir, p_name, f"{sidx}.py"), "w", encoding="utf-8") as f:
                        if model.conversational:
                            f.write(impl)
                        else:
                            f.write(task["prompt"] + impl)
                except UnicodeEncodeError:
                    continue
                sidx += 1
        
        result_queue.put(("progress", 1))


def code_generate(args, workdir: PathLike, id_range=None, free_gpus=[0], tensor_parallel_size=1):

    result_queue = mp.Queue()

    with Progress(
        TextColumn(
            f"{args.dataset} •" + "[progress.percentage]{task.percentage:>3.0f}%"
        ),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as progress:
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

        dataset_items = list(dataset.items())
        num_processes = len(free_gpus) // tensor_parallel_size
        # divide the messages_list into equal part for each process
        chunk_size = (len(dataset_items) + num_processes - 1) // num_processes
        dataset_chunks = [dataset_items[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes)]

        task = progress.add_task("[cyan]Processing...", total=len(dataset_items))

        processes = []
        for i in range(num_processes):
            gpu_ids = ','.join(str(gpu_id) for gpu_id in free_gpus[i*tensor_parallel_size:(i+1)*tensor_parallel_size])
            p = mp.Process(
                target=ask_llm_worker, 
                args=(args, gpu_ids, dataset_chunks[i], workdir, result_queue))
            processes.append(p)
            p.start()

        # Monitor progress and handle messages
        completed = 0
        while completed < len(dataset_items):
            if not result_queue.empty():
                msg_type, msg_content = result_queue.get()
                if msg_type == "progress":
                    completed += msg_content
                    progress.update(task, advance=msg_content)
                elif msg_type == "log":
                    progress.console.print(msg_content)
                elif msg_type == "skip":
                    progress.console.print(msg_content)
                    completed += 1
                    progress.update(task, advance=1)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # # create process
        # processes = []
        # for i in range(num_processes):
        #     gpu_ids = ','.join(str(gpu_id) for gpu_id in free_gpus[i*tp:i*tp+tp])
        #     p = mp.Process(target=ask_llm_worker, args=(gpu_ids, i, config, messages_chunks[i]))
        #     processes.append(p)
        #     p.start()

        # # wait for all process to finish
        # for p in processes:
        #     p.join()

        # for task_id, task in p.track(dataset.items()):
        #     if id_range is not None:
        #         id_num = int(task_id.split("/")[1])
        #         low, high = id_range
        #         if id_num < low or id_num >= high:
        #             p.console.print(f"Skipping {task_id} as it is not in {id_range}")
        #             continue

        #     p_name = task_id.replace("/", "_")
        #     if args.contract_type != "none" and task["contract"] == "":
        #         continue
        #     os.makedirs(os.path.join(workdir, p_name), exist_ok=True)
        #     log = f"Codegen: {p_name} @ {model}"
        #     n_existing = 0
        #     if args.resume:
        #         # count existing .py files
        #         n_existing = len(
        #             [
        #                 f
        #                 for f in os.listdir(os.path.join(workdir, p_name))
        #                 if f.endswith(".py")
        #             ]
        #         )
        #         if n_existing > 0:
        #             log += f" (resuming from {n_existing})"

        #     nsamples = args.n_samples - n_existing
        #     p.console.print(log)

        #     sidx = args.n_samples - nsamples
        #     while sidx < args.n_samples:
        #         outputs = model.codegen(
        #             construct_contract_prompt(
        #                 task["prompt"], args.contract_type, task["contract"]
        #             ),
        #             do_sample=not args.greedy,
        #             num_samples=args.n_samples - sidx,
        #         )
        #         assert outputs, "No outputs from model!"
        #         for idx, impl in enumerate(outputs):
        #             try:
        #                 with open(
        #                     os.path.join(workdir, p_name, f"{sidx}.py"),
        #                     "w",
        #                     encoding="utf-8",
        #                 ) as f:
        #                     if model.conversational:
        #                         f.write(impl)
        #                     else:
        #                         f.write(task["prompt"] + impl)
        #             except UnicodeEncodeError:
        #                 continue
        #             sidx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--dataset", required=True, type=str, choices=["humaneval", "mbpp", "lcb"])
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--contract-type", default="none", type=str, choices=["none", "code", "docstring"])
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--id-range", default=None, nargs="+", type=int) # id_range is list
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
    free_gpus = get_free_gpus(threshold=70000)
    tensor_parallel_size = int(os.getenv("VLLM_N_GPUS", "1"))
    workdir = os.path.join(args.root, args.dataset, args.model + f"_temp_{args.temperature}" + ("" if args.contract_type == "none" else f"-contract-{args.contract_type}"))
    os.makedirs(workdir, exist_ok=True)

    with open(os.path.join(workdir, "args.txt"), "w") as f:
        f.write(str(args))

    code_generate(args, workdir=workdir, id_range=args.id_range, free_gpus=free_gpus, tensor_parallel_size=tensor_parallel_size)


if __name__ == "__main__":
    main()
