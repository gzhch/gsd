import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import datasets
import random
import self_speculative_decoding.modeling_llama2 as ssd_llama

import utils.decoding as dec
import utils.graph_decoding as gdec

from itertools import product

torch.set_grad_enabled(False)


def load_data(task_name, n_shot=1, seed=42):
    data_dirs = {
        'xsum' : '/ossfs/workspace/nas/gzhch/data/datasets/xsum',
        # 'cnndm' : '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujiahao12/Datasets/huggingface/cnn_dailymail-3.0.0',
        # 'lima' : '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujiahao12/Datasets/huggingface/lima',
        'gsm8k' : '/ossfs/workspace/nas/gzhch/data/datasets/gsm8k',
        'alpaca' : '/ossfs/workspace/nas/gzhch/data/datasets/alpaca',
        'wmt' : '/ossfs/workspace/nas/gzhch/data/datasets/wmt14_de-en_test',
    }

    if task_name == 'gsm8k':
        dataset = datasets.load_dataset(data_dirs[task_name])
    else:
        dataset = datasets.load_from_disk(data_dirs[task_name])

    
    if task_name == 'xsum':
        data = dataset['test'].shuffle(seed=seed).select(range(1000))
        shots = dataset['train'].shuffle(seed=seed).select(range(n_shot))
        prompt_shots = ''
        prompt_keys=['document','summary']

        for i in range(n_shot):
            prompt = 'Article: ' + shots[i][prompt_keys[0]] + '\nSummary: ' + shots[i][prompt_keys[1]].replace('\n', '') + '\n'
            prompt_shots += prompt
        
        def process_input(x):
            x['input'] = prompt_shots +'Article: ' + x[prompt_keys[0]] + '\nSummary:'
            return x
        dataset = data.map(process_input, load_from_cache_file=False)

    elif task_name == 'alpaca':
        data = dataset['train'].shuffle(seed=seed).select(range(1000))
        shots = dataset['train'].shuffle(seed=seed).select(range(1000, 1000 + n_shot))
        prompt_shots = ''
        template = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive{instruction}\n\n### Response:\n{output}'
        
        for i in range(n_shot):
            prompt = template.format(instruction=shots[i]['instruction'], output=shots[i]['output']) + '\n\n'
            prompt_shots += prompt
        
        def process_input(x):
            x['input'] = prompt_shots + template.format(instruction=x['instruction'], output='')
            x['ground_truth'] = x['output']
            return x
        dataset = data.map(process_input, load_from_cache_file=False)

    elif task_name == 'wmt':
        data = dataset.shuffle(seed=42).select(range(1000))
        shots = dataset.shuffle(seed=42).select(range(1000, 1000+n_shot))
        prompt_shots = ''
        template = 'Translate Germany to English:\nGermany: {de}\nEnglish: {en}'

        for i in range(n_shot):
            prompt = template.format(de=shots[i]['translation']['de'], en=shots[i]['translation']['en']) + '\n\n'
            prompt_shots += prompt

        def process_input(x):
            x['input'] = prompt_shots + template.format(de=x['translation']['de'], en='')
            x['ground_truth'] = x['translation']['en']
            return x
        dataset = data.map(process_input, load_from_cache_file=False)

        

    elif task_name == 'cnndm':
        prompt_keys=['article','highlights']
        
    elif task_name == 'gsm8k':
        data = dataset['test'].shuffle(seed=seed).select(range(1000))
        shots = dataset['train'].shuffle(seed=seed).select(range(n_shot))
        prompt_shots = ''
        prompt_keys=['question','answer']
        
        for i in range(n_shot):
            prompt = 'Question: ' + shots[i][prompt_keys[0]] + '\nAnswer: ' + shots[i][prompt_keys[1]].replace('\n', '') + '\n'
            prompt_shots += prompt
        
        def process_input(x):
            x['input'] = prompt_shots +'Question: ' + x[prompt_keys[0]] + '\nAnswer:'
            return x
        dataset = data.map(process_input, load_from_cache_file=False)
        
    return dataset



def load_model_and_tokenizer(model_name):
    model_dirs = {
        'tinyllama' : "/ossfs/workspace/nas/gzhch/data/models/tinyllama",
        'llama-2-7b' : "/ossfs/workspace/nas/gzhch/data/models/Llama-2-7b-hf",
        'llama-160m' : "/ossfs/workspace/nas/gzhch/data/models/llama-160m",
    }

    model = ssd_llama.LlamaForCausalLM.from_pretrained(
        model_dirs[model_name], 
        device_map='auto',
        torch_dtype=torch.float16,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dirs[model_name])
    return model, tokenizer


def evaluate(dataset, base_config, draft_config, generate_fn='graph'):
    results = []
    for i, x in enumerate(dataset):
        if i >= base_config['max_instance']:
            break
        
        prompt = x['input']
        
        res = gdec.infer((base_config['model_large'], base_config['model_small']), 
                         base_config['tokenizer'], prompt,
                         generate_fn=generate_fn, 
                         max_new_tokens=base_config['output_length'], 
                         draft_config=draft_config,
                         early_stop=True,
                        )
        results.append(res)
        
        # print(res)
    
    # print(res)

    matchness, drafted_token_num, graph_success, graph_sum, time, acc = 0,0,0,0,0,0
    for r in results:
        matchness += r['matchness']
        drafted_token_num += r['drafted_token_num']
        graph_success += r['graph_success'][0]
        graph_sum += r['graph_success'][1]
        time += r['time']
        acc += r['n_matched']/r['n_draft_step']
        
    matchness /= base_config['max_instance']
    drafted_token_num /= base_config['max_instance']
    graph_success = graph_success / graph_sum
    time  /= base_config['max_instance']
    acc /= base_config['max_instance']
    return {'matchness':matchness, 
            'drafted_token_num':drafted_token_num,
            'graph_success':graph_success,
            'time':time,
            'acc': acc,
           }

def evaluate_generate(dataset, base_config, draft_config, generate_fn='graph'):
    results = []
    for i, x in enumerate(dataset):
        if i >= base_config['max_instance']:
            break
        
        prompt = x['input']
        
        res = gdec.infer((base_config['model_large'], base_config['model_small']), 
                         base_config['tokenizer'], prompt,
                         generate_fn=generate_fn, 
                         max_new_tokens=base_config['output_length'], 
                         draft_config=draft_config,
                         early_stop=True,
                        )
        results.append(res)
        
        # print(res)
    
    # print(res)
    # return results
    matchness, drafted_token_num, graph_success, graph_sum, time, acc = 0,0,0,0,0,0
    for r in results:
        matchness += r['matchness']
        drafted_token_num += r['drafted_token_num']
        graph_success += r['graph_success'][0]
        graph_sum += r['graph_success'][1]
        time += r['time']
        acc += r['n_matched']/r['n_draft_step']
        
    matchness /= base_config['max_instance']
    drafted_token_num /= base_config['max_instance']
    graph_success = graph_success / graph_sum
    time  /= base_config['max_instance']
    acc /= base_config['max_instance']
    return {'matchness':matchness, 
            'drafted_token_num':drafted_token_num,
            'graph_success':graph_success,
            'time':time,
            'acc': acc,
           }

def evaluate_base(dataset, base_config, generate_fn='base'):
    results = []
    for i, x in enumerate(dataset):
        if i >= base_config['max_instance']:
            break
            
        prompt = x['input']
        
        if generate_fn == 'base':
            res = dec.infer(base_config['model_large'], 
                            base_config['tokenizer'], prompt, 
                            generate_fn=generate_fn, 
                            max_new_tokens=base_config['output_length'], 
                            )
        elif generate_fn == 'essg':
            res = dec.infer(base_config['model_large'], 
                            base_config['tokenizer'], prompt, 
                            generate_fn=generate_fn, 
                            max_new_tokens=base_config['output_length'], 
                            max_step_draft=12, 
                            #th_stop_draft=0.8,
                            auto_th_stop_draft=True
                            )
         
        results.append(res)
        
        # print(res)
    time = 0
    for r in results:
        time += r['time']
    time  /= base_config['max_instance']
    return {
            'time':time,
           }



verify_model, verify_tokenizer = load_model_and_tokenizer('llama-2-7b')
draft_model, draft_tokenizer = load_model_and_tokenizer('tinyllama')


data = load_data('xsum', 1) # xsum gsm8k alpaca


## set hyperparameters
base_config = {}
base_config['model_large'] = verify_model
base_config['tokenizer'] = verify_tokenizer
base_config['input_length'] = 512
base_config['output_length'] = 100.   # output_length = 100 or 512
base_config['max_instance'] = 10

res = evaluate_base(data, base_config)
print(res)




base_config['model_small'] = draft_model
tree_decoding = [4]
repeat_threshold = [0, 1, -1]
prob_threshold = [0.3,0.4]
sibling_threshold = [0.1,0.2,0.3,0.4]
hyperparameters = list(product(tree_decoding, repeat_threshold, prob_threshold, sibling_threshold))
hyperparameters = [(1, -1, 0.4, 0), (4, -1, 0.4, 0)] + hyperparameters

draft_config['exact_match'] = True
draft_config['sample'] = False
for p in hyperparameters:
    draft_config['tree_decoding'] = p[0]
    draft_config['repeat_threshold'] = p[1]
    draft_config['prob_threshold'] = p[2]
    draft_config['sibling_threshold'] = p[3]
    res = evaluate(data, base_config, draft_config, generate_fn='dev')
    print('deterministic', p, res)

draft_config['exact_match'] = False
draft_config['sample'] = False
for p in hyperparameters:
    draft_config['tree_decoding'] = p[0]
    draft_config['repeat_threshold'] = p[1]
    draft_config['prob_threshold'] = p[2]
    draft_config['sibling_threshold'] = p[3]
    res = evaluate(data, base_config, draft_config, generate_fn='dev')
    print('non-deterministic', p, res)


