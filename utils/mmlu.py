import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import os
import gc
from torch.nn.functional import pad
from datasets import load_from_disk
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from transformers import AutoConfig
import random
from promptsource.templates import DatasetTemplates
from functools import partial
#from custom.modeling_llama import LlamaForCausalLM
from datasets.utils.logging import disable_progress_bar
from transformers.utils import logging
from accelerate import Accelerator, load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map
import datasets
from typing import List, Mapping, NewType, Optional, Tuple, Union
import pickle
from transformers import pipeline
import pandas as pd
import importlib
import copy
disable_progress_bar()


subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

c1toc2 = subcategories

c1toc3 = {'abstract_algebra': 'STEM',
     'anatomy': 'other (business, health, misc.)',
     'astronomy': 'STEM',
     'business_ethics': 'other (business, health, misc.)',
     'clinical_knowledge': 'other (business, health, misc.)',
     'college_biology': 'STEM',
     'college_chemistry': 'STEM',
     'college_computer_science': 'STEM',
     'college_mathematics': 'STEM',
     'college_medicine': 'other (business, health, misc.)',
     'college_physics': 'STEM',
     'computer_security': 'STEM',
     'conceptual_physics': 'STEM',
     'econometrics': 'social sciences',
     'electrical_engineering': 'STEM',
     'elementary_mathematics': 'STEM',
     'formal_logic': 'humanities',
     'global_facts': 'other (business, health, misc.)',
     'high_school_biology': 'STEM',
     'high_school_chemistry': 'STEM',
     'high_school_computer_science': 'STEM',
     'high_school_european_history': 'humanities',
     'high_school_geography': 'social sciences',
     'high_school_government_and_politics': 'social sciences',
     'high_school_macroeconomics': 'social sciences',
     'high_school_mathematics': 'STEM',
     'high_school_microeconomics': 'social sciences',
     'high_school_physics': 'STEM',
     'high_school_psychology': 'social sciences',
     'high_school_statistics': 'STEM',
     'high_school_us_history': 'humanities',
     'high_school_world_history': 'humanities',
     'human_aging': 'other (business, health, misc.)',
     'human_sexuality': 'social sciences',
     'international_law': 'humanities',
     'jurisprudence': 'humanities',
     'logical_fallacies': 'humanities',
     'machine_learning': 'STEM',
     'management': 'other (business, health, misc.)',
     'marketing': 'other (business, health, misc.)',
     'medical_genetics': 'other (business, health, misc.)',
     'miscellaneous': 'other (business, health, misc.)',
     'moral_disputes': 'humanities',
     'moral_scenarios': 'humanities',
     'nutrition': 'other (business, health, misc.)',
     'philosophy': 'humanities',
     'prehistory': 'humanities',
     'professional_accounting': 'other (business, health, misc.)',
     'professional_law': 'humanities',
     'professional_medicine': 'other (business, health, misc.)',
     'professional_psychology': 'social sciences',
     'public_relations': 'social sciences',
     'security_studies': 'social sciences',
     'sociology': 'social sciences',
     'us_foreign_policy': 'social sciences',
     'virology': 'other (business, health, misc.)',
     'world_religions': 'humanities'}
    
c2toc1 = {'physics': ['astronomy', 'college_physics', 'conceptual_physics', 'high_school_physics'], 'chemistry': ['college_chemistry', 'high_school_chemistry'], 'biology': ['college_biology', 'high_school_biology'], 'computer science': ['college_computer_science', 'computer_security', 'high_school_computer_science', 'machine_learning'], 'math': ['abstract_algebra', 'college_mathematics', 'elementary_mathematics', 'high_school_mathematics', 'high_school_statistics'], 'engineering': ['electrical_engineering'], 'history': ['high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'prehistory'], 'philosophy': ['formal_logic', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'world_religions'], 'law': ['international_law', 'jurisprudence', 'professional_law'], 'politics': ['high_school_government_and_politics', 'public_relations', 'security_studies', 'us_foreign_policy'], 'culture': ['human_sexuality', 'sociology'], 'economics': ['econometrics', 'high_school_macroeconomics', 'high_school_microeconomics'], 'geography': ['high_school_geography'], 'psychology': ['high_school_psychology', 'professional_psychology'], 'other': ['global_facts', 'miscellaneous', 'professional_accounting'], 'business': ['business_ethics', 'management', 'marketing'], 'health': ['anatomy', 'clinical_knowledge', 'college_medicine', 'human_aging', 'medical_genetics', 'nutrition', 'professional_medicine', 'virology']}

c2toc3 = {
    'physics': 'STEM',
    'chemistry': 'STEM',
    'biology': 'STEM',
    'computer science': 'STEM',
    'math': 'STEM',
    'engineering': 'STEM',
    'history': 'humanities',
    'philosophy': 'humanities',
    'law': 'humanities',
    'politics': 'social sciences',
    'culture': 'social sciences',
    'economics': 'social sciences',
    'geography': 'social sciences',
    'psychology': 'social sciences',
    'other': 'other (business, health, misc.)',
    'business': 'other (business, health, misc.)',
    'health': 'other (business, health, misc.)'
}

c3toc1 = {'STEM': ['astronomy', 'college_physics', 'conceptual_physics', 'high_school_physics', 'college_chemistry', 'high_school_chemistry', 'college_biology', 'high_school_biology', 'college_computer_science', 'computer_security', 'high_school_computer_science', 'machine_learning', 'abstract_algebra', 'college_mathematics', 'elementary_mathematics', 'high_school_mathematics', 'high_school_statistics', 'electrical_engineering'], 'humanities': ['high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'prehistory', 'formal_logic', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'world_religions', 'international_law', 'jurisprudence', 'professional_law'], 'social sciences': ['high_school_government_and_politics', 'public_relations', 'security_studies', 'us_foreign_policy', 'human_sexuality', 'sociology', 'econometrics', 'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_geography', 'high_school_psychology', 'professional_psychology'], 'other (business, health, misc.)': ['global_facts', 'miscellaneous', 'professional_accounting', 'business_ethics', 'management', 'marketing', 'anatomy', 'clinical_knowledge', 'college_medicine', 'human_aging', 'medical_genetics', 'nutrition', 'professional_medicine', 'virology']}

c3toc2 = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

data_dir = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujiahao12/Datasets/mmlu/data/'

def mmlu_result(results):
    mmlu_score = {'avg':[], 'humanities':[], 'STEM':[], 'social sciences':[], 'other (business, health, misc.)':[]}
    for k, v in results.items():
        mmlu_score[cate[subcategories[k][0]]].append(v)
    for i in mmlu_score:
        mmlu_score['avg'] += mmlu_score[i]
    for i in mmlu_score:
        mmlu_score[i] = np.array(mmlu_score[i]).mean()
    return " & ".join("{:.2f}".format(100*i) for i in mmlu_score.values())

def load_data(subject):   
    dev_df = pd.read_csv(os.path.join('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujiahao12/Datasets/mmlu/data/', "dev", subject + "_dev.csv"), header=None)#[:4]
    test_df = pd.read_csv(os.path.join('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujiahao12/Datasets/mmlu/data/', "test", subject + "_test.csv"), header=None)
    return dev_df, test_df


class Evaluator:
    def __init__(self, dev_df, test_df, n_shots, tokenizer, bs=10, length=10000, method=''):
        choices = ["A", "B", "C", "D"]
        label2idx = {'A':0,'B':1,'C':2,'D':3}

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'right'
        self.batch_size = bs
        self.pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.example_separator = '\n###\n'
        self.max_length = length
        self.pad_idx = 2
        self.answer_index = [self.tokenizer.encode(i)[1] for i in choices]
        
        self.prefix_idx = [1]
        self.prefix_lengths = []
        self.prefix_examples = []
        for i in range(n_shots):
            example = dev_df.iloc[i, 0]
            k = dev_df.shape[1] - 2
            for j in range(k):
                example += "\n{}. {}".format(choices[j], dev_df.iloc[i, j + 1])
            example += "\nAnswer:"
            example += " {}\n\n".format(dev_df.iloc[i, k + 1])
            example + self.example_separator
            idx = self.tokenizer.encode(example)[1:]
            self.prefix_idx += idx
            self.prefix_lengths.append(len(idx))
            self.prefix_examples.append(example)

        
        def add_prompt(x):
            x['input_ids'] = copy.deepcopy(self.prefix_idx)
            x['lengths'] = copy.deepcopy(self.prefix_lengths)
            x['prompt'] = ''.join(self.prefix_examples)
            
            example = x['0']
            k = dev_df.shape[1] - 2
            for j in range(k):
                example += "\n{}. {}".format(choices[j], x[str(j+1)])
            example += "\nAnswer:"
            # example += " {}\n\n".format(x[str(k+1)])
            x['prompt'] += example
            idx = self.tokenizer.encode(example)[1:]
            x['input_ids'] += idx
            x['lengths'].append(len(idx))
            x['input_ids'] += [self.pad_idx for _ in range(self.max_length - len(x['input_ids']))] # padding
            x['prompt_label'] = x[str(k+1)]
            x['label'] = label2idx[x['prompt_label']]
            return x
            
            
        def repeat_example(x):
            total_length = 1
            for i in x['lengths']:
                total_length += i
            ids = x['input_ids'][1: total_length]
            x['input_ids'] = [1] + ids[x['lengths'][0]: -x['lengths'][-1]] + ids
            x['lengths'] = x['lengths'][1:-1] + x['lengths']
            x['input_ids'] += [self.pad_idx for _ in range(self.max_length - len(x['input_ids']))]
            return x
        
        def get_attention_mask(x):
            ls = x['lengths']
            c = len(ls)//2 
            total_length = self.max_length - 1
            pre = 0
            mask = torch.zeros(total_length,total_length).to(torch.bool)
            for i in range(c):
                s = 0
                for j in range(c):
                    s += ls[i+j]
                if i == c - 1:
                    s = total_length - pre
                t = torch.cat([torch.ones(pre), torch.zeros(s), torch.ones(total_length-pre-s)])
                t = t[None, :].expand(total_length, total_length)
                t = ~(t + t.T).to(torch.bool)
                pre += ls[i]
                mask = mask | t
            res = torch.zeros(total_length+1, total_length+1)
            res[0, :] = 1
            res[:, 0] = 1
            res[1:, 1:] = mask
            x['attention_mask'] = res.to(torch.bool).view(-1)
            return x

        self.ds = datasets.Dataset.from_pandas(test_df)
        self.ds = self.ds.map(add_prompt, load_from_cache_file=False)
        if method == 'repeat':
            self.ds = self.ds.map(repeat_example, load_from_cache_file=False)
            self.ds = self.ds.filter(lambda x: len(x['input_ids']) <= self.max_length, load_from_cache_file=False)
            self.ds = self.ds.map(get_attention_mask, load_from_cache_file=False)
            self.ds.set_format(type='torch', columns=['input_ids', 'label', 'attention_mask'])
           # self.ds.cast_column('attention_mask', datasets.features.features.Array2D(shape=(self.max_length, self.max_length), dtype="bool"))
        elif method == 'vanilla-repeat':
            self.ds = self.ds.map(repeat_example, load_from_cache_file=False)
            self.ds = self.ds.filter(lambda x: len(x['input_ids']) <= self.max_length, load_from_cache_file=False)
            self.ds.set_format(type='torch', columns=['input_ids', 'label'])
            # self.ds = self.ds.map(get_attention_mask, load_from_cache_file=False)
        else:
            self.ds = self.ds.filter(lambda x: len(x['input_ids']) <= self.max_length, load_from_cache_file=False)
            self.ds.set_format(type='torch', columns=['input_ids', 'label'])
    
    
    @torch.no_grad()
    def train(self, model, batch):
        bs = batch['label'].size(0)
        input_ids = batch['input_ids'].cuda()
        if 'attention_mask' in batch.keys():
                attention_mask = batch['attention_mask'].view(bs, 1, self.max_length, self.max_length).cuda()
        else:
                attention_mask = None
        outputs = model(input_ids, attention_mask=attention_mask)
        pad_len = (input_ids==self.pad_idx).sum(-1)
        last_token_logits = outputs.logits[range(bs), self.max_length-1-pad_len-0, :]#[range(bs), range(bs), :]
        return (last_token_logits[:, self.answer_index].max(-1).indices == batch['label'].cuda()).sum().item() 
        
    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
    
        dataloader = torch.utils.data.DataLoader(self.ds, batch_size=self.batch_size)
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        cnt = 0
        
        
        for _, batch in enumerate(iter(dataloader)):
            cnt += 1
            # if cnt > self.max_instance / self.batch_size:
            #     break                
            hit += self.train(model, batch)            
            total += batch['label'].size(0)        
            
        acc = hit / total
        return acc
    
    
def determine_max_len(length):
    if length <= 600:
        return 10
    if length <= 1200:
        return 6
    if length <= 2000:
        return 4
    if length <= 3000:
        return 2

def load_evaluate(name, tokenizer, k):
    dev_df, test_df = load_data(name)
    evaluate = Evaluator(dev_df, test_df[:200], k, tokenizer, method='')
    max_len = torch.tensor(evaluate.ds['lengths']).sum(-1).max().item()
    max_len = min(max_len, 3000)
    bs = determine_max_len(max_len)
    evaluate = Evaluator(dev_df, test_df[:200], k, tokenizer, bs=bs, length=max_len, method='')
    return evaluate
