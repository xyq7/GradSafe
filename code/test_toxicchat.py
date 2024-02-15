import copy
import pickle
import torch
import numpy as np
import pandas as pd
import os
import json
import pickle
import torch.nn.functional as F
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import itertools
from find_critical_parameters import find_critical_para, load_model

def cos_sim_toxic(model_id, df ,gradient_norms_compare, minus_row, minus_col):
    columns = df[['user_input', 'toxicity']]
    model, tokenizer = load_model(model_id)
    
    #  Prompt templates
    sep_token, sep_token_id = tokenizer.unk_token, tokenizer.unk_token_id
    prompt = (
        f'<s>[INST] <<SYS>> {{system_prompt}} <</SYS>> {{content}} [/INST]' + 
        f'{{sep_token}} {{summary}} {{eos_token}}'
    )
    def apply_prompt_template(sample):
        txt = prompt.format(
            system_prompt='You are a helpful assistant. Help me with the following query: ',
            content=sample['source'],
            summary=sample['target'],
            eos_token=tokenizer.eos_token,
            sep_token=sep_token,
        )
        return {
            'text': txt,
        }     

    #  Calculate cosine similarity based score 
    cos_all = []
    label_all = []

    for index, row in columns.iterrows():
        # Ground truth label
        if row['toxicity'] == 1:
            label_all.append(1)
        else: 
            label_all.append(0)

        # Gradient for prompt paired with complaince response Sure
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        basic_sample = {}
        basic_sample["source"] = row['user_input']
        basic_sample["target"] = "Sure"
        d = apply_prompt_template(basic_sample)
        input_ids = tokenizer(d['text']).input_ids
        sep = input_ids.index(sep_token_id)
        
        input_ids = input_ids[:sep] + input_ids[sep+1:]
        input_ids = torch.tensor(np.array([input_ids]))
        target_ids = input_ids.clone()
        target_ids[:, :sep] = -100
        optimizer.zero_grad()
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
        neg_log_likelihood.backward()
        
        # Cosine similarities for safety-critical parameters
        cos = []
        import torch.nn.functional as F
        for name, param in model.named_parameters():
            if  param.grad is not None and ("mlp" in name or "self" in name):
                grad_norm = param.grad.to(gradient_norms_compare[name].device)
                row_cos = torch.nan_to_num(F.cosine_similarity(grad_norm, (gradient_norms_compare[name]), dim=1))
                col_cos = torch.nan_to_num(F.cosine_similarity(grad_norm, (gradient_norms_compare[name]), dim=0))
                ref_row = minus_row[name]
                ref_col = minus_col[name]
                cos.extend(row_cos[ref_row>1].cpu().tolist())
                cos.extend(col_cos[ref_col>1].cpu().tolist())


        cos_all.append(cos)
        
    # For GradSafe-Zero, average all cosine similarites as one score
    cos_all = [sum(feature)/len(feature) for feature in cos_all]
    
    # Calculate AUPRC
    precision, recall, thresholds = precision_recall_curve(label_all, cos_all)
    auprc = auc(recall, precision)

    
    # Calculate Precision, Recall, F1
    from sklearn.metrics import precision_score, recall_score, f1_score
    true_labels = label_all

    predicted_labels = [1 if feature >=0.25 else 0 for feature in cos_all]
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    # 计算F1分数
    f1 = f1_score(true_labels, predicted_labels)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUPRC:", auprc)
    del gradient_norms_compare
    del cos_all
    del label_all
    return auprc, f1


if __name__ == "__main__":
    for model_id in ['./model/Llama-2-7b-chat-hf']:
        gradient_norms_compare, minus_row_cos, minus_col_cos =  find_critical_para(model_id)
        df = pd.read_csv('/home/yueqi/GradSafe-kk/ToxicChat/sampled_data.csv')
        auprc, f1 = cos_sim_toxic(model_id, df,gradient_norms_compare, minus_row_cos, minus_col_cos)