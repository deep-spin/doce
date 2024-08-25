# doce
This repo is for code in our arxived paper:

```
DOCE: Finding the Sweet Spot for Execution-Based Code Generation
Haau-Sing Li, Patrick Fernandes, Iryna Gurevych, André F. T. Martins
```

Contact person: [Haau-Sing Li](mailto:hli@ukp.tu-darmstadt.de)


### Usage

1. Installing packages from `requirements*.txt`. 

2. Inference on HumanEval/MBPP task

```bash
python3 codegen/generate.py \
    --model ${model} \
    --bs ${batch_size} \
    --temperature ${temperature} \
    --n_samples ${num_of_samples_for_reranking} \
    --dataset ${humaneval/mbpp} \
    --resume \
    --root ${path_to_store_output}
```

3. Evaluation

```bash
evalplus.evaluate \
    --dataset {humaneval/mbpp} \
    --samples ${path to generated samples} \
    --parallel 30 \
    --test-details
```

4. Get execution outputs of generated samples (for MBR-Exec)

```bash
python3 evalplus/gen_outputs.py \
    --gen_dir {model_name_plus_temperature} \
    --dataset {humaneval/mbpp} \
    --gen_fast
```

5. Self-Debugging
You should get execution feedback first:
```bash
python3 evalplus/error_feedback.py \
    --gen_dir {model_name_plus_temperature} \
    --dataset {humaneval/mbpp} 
```

Then we can do self-debugging:
```bash
python3 codegen/ape_sd_ut.py \
    --model ${model} \
    --bs ${batch_size} \
    --temperature ${temperature} \
    --n_samples ${num_of_samples_for_reranking} \
    --dataset ${humaneval/mbpp} \
    --resume \
    --root ${path_to_store_output}
    --debugging_turn ${ith_debugging_turn}
```

5. For MBR and N-Best-Reranking, please refer to our notebooks for now.

We will release our generated candidates soon if you want to save compute.


Our code is built upon [EvalPlus](https://github.com/evalplus/evalplus).
