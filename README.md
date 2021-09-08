# CONAN - CONtinuous pAtterNs
This repository contains the code to reproduce the results from the EMNLP 2021 short paper "Continuous Entailment Patterns for Lexical Inference in Context".

If this code is useful for you, please consider citing:
```
@inproceedings{schmitt-schutze-2021-continuous,
    title = "Continuous Entailment Patterns for Lexical Inference in Context",
    author = {Schmitt, Martin  and
      Sch{\"u}tze, Hinrich},
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
	note = "To appear"
}
```

# Data
Please see the instructions in the repository [Language Models for Lexical Inference in Context](https://github.com/mnschmit/lm-lexical-inference) about retrieving the data.

# Code
This code base is built upon the code in [Language Models for Lexical Inference in Context](https://github.com/mnschmit/lm-lexical-inference).
You can refer to the documentation there for most scripts.
The train script was renamed to `src/train/train.py` and there is only one. The only new arguments, `--num_patterns` and `--num_tokens_per_pattern`, correspond to the hyperparamters n and k and should be self-explanatory.

The following additional scripts are provided:

- `src/analysis/contokens_nn.py`
- `src/analysis/create_heatmap.py`
- `src/train/n_k_loop.py`

## Nearest Neighbor Analysis
The script `src/analysis/contokens_nn.py` computes the nearest neighbors of continuous tokens in the subword embedding space.
It is called like this:
```
python3 -m src.analysis.contokens_nn PATH_TO_CHECKPOINT NUM_PATTERNS NUM_TOKENS_PER_PATTERN
```
where
- `PATH_TO_CHECKPOINT` is a model checkpoint stored after training
- `NUM_PATTERNS` is the number of patterns (called n in the paper)
- `NUM_TOKENS_PER_PATTERN` is the number of continuous tokens per pattern (called k in the paper)

## Optimize n and k
The script `src/train/n_k_loop.py` performs a grid search over all configurations for n and k considered in the paper.
It takes mostly the same arguments as the train script; the arguments `--start_num_patterns`, `--start_num_tokens_per_pattern`, `--start_version` can be useful for restarting an aborted loop.
The results are stored in a tsv file named after the specified experiment name.

The visualization of these results can be done with `src/analysis/create_heatmap.py`.
