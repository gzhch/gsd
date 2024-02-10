import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import os
import gc
from torch.nn.functional import pad
from datasets import load_from_disk
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.nn as nn
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
from custom.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
disable_progress_bar()
#logging.disable_progress_bar()

# class speculative_classifier(torch.nn.Module):
#     def __init__(self, hidden_size):
#         super(speculative_classifier, self).__init__()
#         self.linear = torch.nn.Linear(hidden_size, 2)
#     def forward(self, x):
#         return self.linear(x)
    
class speculative_classifier(torch.nn.Module):
    def __init__(self, hidden_size, config=None):
        super(speculative_classifier, self).__init__()
        self.down_proj = nn.Linear(hidden_size, 768)
        self.linear_final = nn.Linear(768, 2)
        self.layers = None
        if config is not None:
            self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(1)])
     #   self.linears = nn.ModuleList([nn.Linear(1000, 1000) for i in range(0)])
    def forward(self, inputs):
        attention_mask = inputs['attention_mask']
        position_ids = inputs['position_ids']
        x = inputs['hidden_states']
        
        x = self.down_proj(x)
        if self.layers is not None:
            for decoder_layer in self.layers:
                layer_outputs = decoder_layer(
                            x,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        )

            x = layer_outputs[0]
        return self.linear_final(x)
    
# def logits_compare(t1, t2, k):
#     assert t1.shape == t2.shape
#     vs = t1.shape[-1]
#     t1 = t1.view(-1, vs).float().topk(k).indices.tolist()
#     t2 = t2.view(-1, vs).float().topk(k).indices.tolist()
    
#     r = [len(set(t1[i][:k]).intersection(set(t2[i][:k])))/k for i in range(len(t1))]
#     return r
    
def logits_compare(t1, t2, k):
    assert t1.shape == t2.shape
    vs = t1.shape[-1]
    t1 = t1.float().topk(k).indices.tolist()
    t2 = t2.float().topk(k).indices.tolist()
    r = []
    for i in range(len(t1)):
        r.append([len(set(t1[i][j]).intersection(set(t2[i][j])))/k for j in range(len(t1[i]))])
    return torch.tensor(r)


### 比较大小模型的logits
def logits_diff(t1, t2, prompt, tokenizer, ks=[]):
    assert t1.shape == t2.shape
    vs = t1.shape[-1]
    
    res = {}
    
    KLDivLoss = torch.nn.KLDivLoss(reduction='none')
    p_output = F.softmax(t1.float(), dim=-1)
    q_output = F.softmax(t2.float(), dim=-1)
    log_mean_output = ((p_output + q_output)/2).log()
    res['js'] = (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2
    res['js'] = res['js'].sum(dim=-1)
    
    t1 = t1.view(-1, vs)
    t2 = t2.view(-1, vs)
    
    pred_t1 = (t1.max(axis=-1).indices[:-1] == tokenizer([prompt], return_tensors='pt')['input_ids'][0, 1:]).view(-1).tolist()
    pred_t2 = (t2.max(axis=-1).indices[:-1] == tokenizer([prompt], return_tensors='pt')['input_ids'][0, 1:]).view(-1).tolist()
    pred_t1.append(False)
    pred_t2.append(False)
    
    
    res['l2_dist'] = (t1-t2).norm(dim=-1).tolist()
    t1 = t1.float().topk(ks[-1]).indices.tolist()
    t2 = t2.float().topk(ks[-1]).indices.tolist()
    for k in ks:
        res[str(k)+'_top_sim'] = [len(set(t1[i][:k]).intersection(set(t2[i][:k])))/k for i in range(len(t1))]
        
    
    res['pred_acc_1'] = pred_t1
    res['pred_acc_2'] = pred_t2
    
    
    return res

def get_result_scale(input_text, t1, t2):
    datas = {}
    keys = ['index', 'token', 'l2_dist', '1_top_sim', '3_top_sim', '5_top_sim', '10_top_sim', 'pred_acc_1', 'pred_acc_2']
    for i in keys:
        datas[i] = []
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))
    res = logits_diff(t1, t2, input_text, [1,3,5,10])
    for k, v in res.items():
        datas[k] = v
    datas['index'] += list(range(len(tokens)))
    datas['token'] += tokens
    return datas, t1, t2

def visualize_diff(df):

    tokens = df.token.tolist()
    print(''.join(i for i in tokens).replace('▁', ' '))
    res = []
    for i in range(len(tokens)):
        if i-1 not in df[df['1_top_sim']<0.4].index.tolist():
            t = ''
            for _ in range(len(tokens[i])):
                t += '_'
            res.append(t)
        else:
            res.append(tokens[i])
    print(''.join(i for i in res).replace('▁', ' '))
    
    
    
def forward_speculative_switcher(input_text, args=None):
    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors='pt', max_length=1000, padding='max_length', truncation=True)
        hidden_states_small = llama_small.model(**inputs)[0]
        logits_small = llama_small.lm_head(hidden_states_small)
        hidden_states_large = llama_large.model(**inputs)[0]
        logits_large = llama_large.lm_head(hidden_states_large)
    
    if args['label'] == 'js':
        switch_label = js_div(logits_small.cpu(), logits_large.cpu()).mean(dim=-1)
    elif args['label'] == 'topk':
        switch_label = sd.logits_compare(logits_large, logits_small, args['k']) 
    
    sd_clf_large.to(hidden_states_large.device)
    sd_clf_small.to(hidden_states_small.device)
    
    if args['arch'] == 'pre_lm':
        y_large = sd_clf_large(hidden_states_large.float())#.view(-1, 2)
        y_small = sd_clf_small(hidden_states_small.float())#.view(-1, 2)
    elif args['arch'] == 'post_lm':
        y_large = sd_clf_large(logits_large.float())#.view(-1, 2)
        y_small = sd_clf_small(logits_small.float())#.view(-1, 2)
        
        
    res = {}
    res['inputs'] = inputs
    res['y_large'] = y_large
    res['y_small'] = y_small
    res['label'] = switch_label
    return res

@torch.no_grad()
def evaluate(eval_idx, args):
    encode_length = args['encode_length']
    batch_size = args['batch_size']
    result = {'tpl':0,'tnl':0,'fpl':0,'fnl':0,'tps':0,'tns':0,'fps':0,'fns':0}
    tpl,tnl,fpl,fnl,tps,tns,fps,fns = 0,0,0,0,0,0,0,0
    
    N = args['eval_num'] // batch_size
    for b in range(N):
        input_text = []
        for j in range(b * batch_size, (b + 1) * batch_size):
            input_text.append(process_lima(lima['train'][eval_idx[j]]))
        res = forward_speculative_switcher(input_text, args)
        
        if args['label'] == 'topk':
            label = res['label'] <= args['eval_k_th']
        elif args['label'] == 'js':
            label = res['label'] > args['eval_js_th']
            
        mask = res['inputs'].attention_mask
        
        y_large = res['y_large']
        pred_large = y_large[:, :, 0] < y_large[:, :, 1]
        y_small = res['y_small']
        pred_small = y_small[:, :, 0] < y_small[:, :, 1]
        
        target = label.to(pred_large.device)
        pred_small = pred_small.to(pred_large.device)
        mask = mask.to(pred_large.device)
        
        target = target[:, encode_length:]
        pred_large = pred_large[:, encode_length:]
        pred_small = pred_small[:, encode_length:]
        mask = mask[:, encode_length:]
        
        result['tpl'] += ((pred_large==1) & (target==1) & mask).sum().item()
        result['tnl'] += ((pred_large==0) & (target==0) & mask).sum().item()
        result['fpl'] += ((pred_large==1) & (target==0) & mask).sum().item()
        result['fnl'] += ((pred_large==0) & (target==1) & mask).sum().item()
        # fpr, tpr, threshold = metrics.roc_curve(target.cpu(), pred.cpu(), pos_label=0)
        # res['aucl'] = metrics.auc(fpr, tpr)
        result['tps'] += ((pred_small==1) & (target==1) & mask).sum().item()
        result['tns'] += ((pred_small==0) & (target==0) & mask).sum().item()
        result['fps'] += ((pred_small==1) & (target==0) & mask).sum().item()
        result['fns'] += ((pred_small==0) & (target==1) & mask).sum().item()
        # fpr, tpr, threshold = metrics.roc_curve(target.cpu(), pred.cpu(), pos_label=0)
        # res['aucs'] = metrics.auc(fpr, tpr)
        
    result['precision_l'] = result['tpl'] / (result['tpl'] + result['fpl']+0.01)
    result['precision_s'] = result['tps'] / (result['tps'] + result['fps']+0.01)
    result['recall_l'] = result['tpl'] / (result['tpl'] + result['fnl'])
    result['recall_s'] = result['tps'] / (result['tps'] + result['fns'])
    result['f1_l'] = 2 * result['precision_l'] * result['recall_l'] / (result['precision_l'] + result['recall_l'])
    result['f1_s'] = 2 * result['precision_s'] * result['recall_s'] / (result['precision_s'] + result['recall_s'])
    return result

def loss(criterion, res, args):
    # y shape: batch x length x n_class
    # label shape: batch x length x n_class
    batch_size = args['batch_size']
    encode_length = args['encode_length']
    
    if args['label'] == 'topk':
        label = res['label'] <= args['k_th']
    elif args['label'] == 'js':
        label = res['label'] > args['js_th']
        
    y_large = res['y_large'].view(-1,2)
    y_small = res['y_small'].to(y_large.device).view(-1,2)
    label = label.long().to(y_large.device).view(-1)
    
    loss_large = criterion(y_large, label).view(batch_size, -1)
    loss_small = criterion(y_small, label).view(batch_size, -1)
    
    label = label.view(batch_size, -1).to(y_large.device)
    mask = res['inputs'].attention_mask.to(y_large.device)
    
    if args['down_sampling_p'] is not None:
        neg_mask = (1 - label) & mask & (torch.rand(mask.shape).to(y_large.device) < down_sampling_p)
        pos_mask = (label) & mask
        mask = neg_mask | pos_mask
        
    loss_large = (loss_large * mask)[:, encode_length:]
    loss_small = (loss_small * mask)[:, encode_length:]
    loss_large = loss_large.sum() / mask[:, encode_length:].sum()
    loss_small = loss_small.sum() / mask[:, encode_length:].sum()
    
    return loss_large, loss_small
