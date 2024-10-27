# GradSafe
Official Code for ACL 2024 paper "GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis"
https://arxiv.org/abs/2402.13494

## Overview
Large Language Models (LLMs) face threats from unsafe prompts.
Existing methods for detecting unsafe prompts for LLMs are primarily online moderation APIs or finetuned LLMs. These strategies, however, often require extensive and resource-intensive data collection and training processes.
In this study, we propose GradSafe, which effectively detects unsafe prompts by scrutinizing the gradients of safety-critical parameters in LLMs. 
Our methodology is grounded in a pivotal observation: when LLMs are trained on unsafe prompts paired with compliance responses, the resulting gradients on certain safety-critical parameters exhibit consistent patterns. In contrast, safe prompts lead to markedly different gradient patterns.
Building on this observation, GradSafe analyzes the gradients from prompts (paired with compliance responses) to accurately detect unsafe prompts. 
We show that GradSafe, applied to Llama-2 without further training, outperforms Llama Guard—despite its extensive finetuning with a large collected dataset—in detecting unsafe content. 
This superior performance is consistent across both zero-shot and adaptation scenarios, as evidenced by our evaluations on the ToxicChat and XSTest.


## Dataset

The ToxicChat dataset is available at https://huggingface.co/datasets/lmsys/toxic-chat, we use toxicchat1123 in our evaluation.

The XSTest dataset is available at https://huggingface.co/datasets/natolambert/xstest-v2-copy

Please download the dataset and save at the ./data

## Base model

Please download Llama-2 7b from https://huggingface.co/meta-llama/Llama-2-7b-chat-hf, and save at ./model

## Demo

Evaluate the performance of GradSafe on two dataset:

> python ./code/test_xstest.py

> python ./code/test_toxicchat.py

## Citation

If you find this work useful for your research, please cite:
```
@inproceedings{xie2024gradsafe,
  title={GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis},
  author={Xie, Yueqi and Fang, Minghong and Pi, Renjie and Gong, Neil},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={507--518},
  year={2024}
}
```


