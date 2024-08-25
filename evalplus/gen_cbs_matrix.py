import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import pickle

from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
    load_solutions,
)

def load_jsonl(dir_path: str):
    # notice that we use the jsonl file to store the data
    with open(dir_path, "r") as f:
        return [json.loads(line) for line in f]

def process_elem(elem, min_idx, max_idx):
    task_id = elem["task_id"]
    hyp_ids = elem["generated_code_solution_ids"]
    ref_ids = elem["golden_code_solution_ids"]
    # clean hyp_ids and ref_ids to be in the range of min_idx and max_idx
    hyp_ids = [idx for idx in hyp_ids if idx >= min_idx and idx < max_idx]
    ref_ids = [idx for idx in ref_ids if idx >= min_idx and idx < max_idx]
    local_updates_f1 = np.zeros((200, 200))
    local_updates_f3 = np.zeros((200, 200))
    for hyp_id in hyp_ids:
        for ref_id in ref_ids:
            local_updates_f1[hyp_id][ref_id] = elem['pred_f1_score']
            local_updates_f1[ref_id][hyp_id] = elem["pred_f1_score"]
            local_updates_f3[hyp_id][ref_id] = elem["pred_f3_score"]
            local_updates_f3[ref_id][hyp_id] = elem["pred_f3_score"]
    return task_id, local_updates_f1, local_updates_f3

# Setup argparse
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('--work_dir', default="/mnt/scratch-artemis/haausing/code_reranking/evalplus_outputs", type=str, help='Working directory path')
parser.add_argument('--dataset', type=str, choices=["mbpp", "humaneval"], help='Dataset to use')
parser.add_argument('--gen_dir', type=str, help='Generated directory path')
parser.add_argument('--workers', type=int, default=40, help='Number of workers for ThreadPoolExecutor')
args = parser.parse_args()

if args.dataset == "mbpp":
    problems = get_mbpp_plus()
else:
    problems = get_human_eval_plus()

index_pairs = [(0, 50), (50, 100), (100, 150), (150, 200)]
cbs_f1 = {task_id: np.zeros((200, 200)) for task_id in problems}
cbs_f3 = {task_id: np.zeros((200, 200)) for task_id in problems}

for min_idx, max_idx in index_pairs:
    mbr_code_bert_score = load_jsonl(f"{args.work_dir}/{args.dataset}/{args.gen_dir}/code_bert_scores{min_idx}_{max_idx}.jsonl")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(lambda elem: process_elem(elem, min_idx, max_idx), mbr_code_bert_score), total=len(mbr_code_bert_score)))

    # Aggregate results
    for task_id, local_updates_f1, local_updates_f3 in results:
        cbs_f1[task_id] += local_updates_f1
        cbs_f3[task_id] += local_updates_f3

# save cbs_f1 and cbs_f3 to disk
with open(f"{args.work_dir}/{args.dataset}/{args.gen_dir}/cbs_f1.pkl", "wb") as f:
    pickle.dump(cbs_f1, f)
with open(f"{args.work_dir}/{args.dataset}/{args.gen_dir}/cbs_f3.pkl", "wb") as f:
    pickle.dump(cbs_f3, f)
