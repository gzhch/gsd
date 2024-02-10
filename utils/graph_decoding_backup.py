import torch
import torch.nn as nn
import time
import copy
from transformers import top_k_top_p_filtering
import utils.activation as ana


## visualization
def token_tree_analysis(tree):
    draft_tree, position_ids, ground_truth = tree
    child = [[] for _ in range(len(draft_tree))]
    #child_node[0].append(1)
    for i, f in enumerate(position_ids):
        child[f+1].append(i+1)
        
    for i, c in enumerate(child):
        print(i, draft_tree[i], c)
    print(ground_truth)
    return child
def print_token_tree(tree):
    
    def _print_token_tree(draft_tree, child, c):
        list_of_tokens = []
        if child[c] == []:
            return [[draft_tree[c]]]
        for cc in child[c]:
            _list_of_tokens = _print_token_tree(draft_tree, child, cc)
            for i in _list_of_tokens:
                list_of_tokens.append([draft_tree[c]] + i)
        return list_of_tokens

    draft_tree, position_ids, ground_truth = tree
    child = [[] for _ in range(len(draft_tree))]
    for i, f in enumerate(position_ids):
        child[f+1].append(i+1)
    # print(draft_tree)
    # print(ground_truth)
    return _print_token_tree(draft_tree, child, 0)


class TokenGraphNode:
    def __init__(self, token_id, position_id, father_index, prob, leaf):
        self.token_id = token_id
        self.pos_id = position_id
        self.prob = prob
        self.father = father_index
        self.pseudo_child = None
        self.child = []
        self.leaf = leaf
        
class TokenGraph:
    def __init__(self, repeat_threshold = 0):
        # each node have one father node and may have more than one child 
        self.nodes = []
        self.repeat_threshold = repeat_threshold
         
    def add_node(self, token_id, father_index, current_index, prob=-1, leaf=False):
        assert current_index == len(self.nodes)
        
        if father_index == -1:
            position_id = 0
            current_node = TokenGraphNode(token_id, position_id, father_index, prob, leaf)
            self.nodes.append(current_node)
            return
        
        else:
            position_id = self.nodes[father_index].pos_id + 1
            
        current_node = TokenGraphNode(token_id, position_id, father_index, prob,leaf)
        self.nodes.append(current_node)
        self.nodes[father_index].child.append(current_index)
        
    def get_path(self, index):
        path = []
        while index >= 0:
            path.append(index)
            index = self.nodes[index].father
        return path
    def check_repetition(self, token_id, father_index):
        ## return True if repetition
        
        if self.repeat_threshold == -1:
            return False
        
        if father_index == -1:
            return False
        
        path = self.get_path(father_index)
        
        for idx, node in enumerate(self.nodes):
            if idx in path:
                continue
            if node.token_id == token_id:
                fid0 = father_index
                fid1 = node.father
                flag = True
                for k in range(self.repeat_threshold):
                    if fid1 == -1 or fid0 == -1:
                        flag = False
                        break
                    if self.nodes[fid0].token_id != self.nodes[fid1].token_id:
                        flag = False
                        break
                    fid0 = self.nodes[fid0].father
                    fid1 = self.nodes[fid1].father
                if flag == True: 
                    self.nodes[father_index].pseudo_child = idx
                    return True
        return False
    
    def check_prob(self, token_id, father_index, threshold):
        ## return True if under the threshold
        if self.nodes[father_index].prob < threshold:
            return True
        return False
    
    def check_leaf(self, token_id, father_index):
        ## return True if father is leaf
        if self.nodes[father_index].leaf:
            return True
        return False

def sample(logits, return_probs: bool=False, do_sample: bool=False, top_k: int=50, top_p: float=0.7, temperature: float=0.7):

    if return_probs:

        all_probs = logits.softmax(-1)
        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
            probs = torch.gather(all_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
        else:
            probs, output_ids = torch.max(all_probs, dim=-1)
            
        return output_ids, probs

    else:

        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
        else:
            output_ids = torch.argmax(logits, dim=-1)
            
        return output_ids
    
    
    
def sample_n(logits, return_probs: bool=False, do_sample: bool=False, top_k: int=50, top_p: float=0.7, temperature: float=0.7, n=1):

    if return_probs:
        all_probs = logits.softmax(-1)
            
        t = torch.topk(all_probs, k=n, dim=-1)
        probs, output_ids = t.values, t.indices
            
        return output_ids, probs

    else:
        output_ids = torch.topk(logits, k=n, dim=-1).indices
            
        return output_ids


## verify the drafted decoding graph
def verify(draft_tree, position_ids, ground_truth):
    child = [[] for _ in range(len(draft_tree))]
    #child_node[0].append(1)
    for i, f in enumerate(position_ids):
        child[f+1].append(i+1)
    
    index = [0]
    current_depth = 0
    current_node = 0
    end = False
    is_leaf = False
    while not end:
        end = True
        if child[current_node] == []:
            is_leaf = True
            break
        for child_node in child[current_node]:
            if ground_truth[current_node] == draft_tree[child_node]:
                current_depth += 1
                current_node = child_node 
                index.append(current_node)
                end = False
                break
                        
    return index, is_leaf

    
def graph_speculative_generation_backup(models, tokenizer, input_ids, draft_config, max_new_tokens=100):
    # tic = time.time()
    
    max_step_draft = draft_config['max_step_draft']
    repeat_threshold = draft_config['repeat_threshold']

    token_trees = []
    
    #input_ids = tokenizer([prompt], return_tensors='pt')['input_ids']

    assert len(models) == 2
    
    model_large = models[0]
    model_small = models[1]
    
    early_stop=False
    max_step_draft=max_step_draft
    th_stop_draft=0.8
    do_sample=False
    do_sample_draft=False
    top_k=0
    top_p=0.85
    temperature=0.2


    step = 0
    step_draft = 0
    step_verify = 0

    n_tree = draft_config['tree_decoding']

    current_input_ids = input_ids
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens+max_step_draft], dtype=torch.long, device=model_large.device)
    if n_tree > 1:
        generate_ids_small = torch.empty([input_ids.size(0), n_tree**(max_step_draft+1)], dtype=torch.long, device=model_large.device)
    else:
        generate_ids_small = torch.empty([input_ids.size(0), max_step_draft+1], dtype=torch.long, device=model_large.device)
    

    n_matched = 0
    n_drafted = 0
    n_node = 0
    n_graph_success = 0
    tmp_n_matched = 0
    tmp_n_drafted = 0
    tmp_matchness = 0

    past_key_values_large = None
    past_key_values_small = None

    with torch.no_grad():
        
        ## initialize the kv-cache for both large and small models
        current_input_ids_large = current_input_ids
        output_large = model_large(input_ids=current_input_ids_large,
                        past_key_values=past_key_values_large,
                        return_dict=True,
                        use_cache=True)
        output_ids = sample_n(output_large['logits'].float(), n=1)   
        generate_ids[:, 0] = output_ids[:, -1, 0]
        past_key_values_large = output_large['past_key_values']
        
        # current_input_ids_small = torch.cat([current_input_ids, output_ids[:, -1:, 0]], dim=-1)
        current_input_ids_small = current_input_ids
        output_small = model_small(input_ids=current_input_ids_small,
                        past_key_values=past_key_values_small,
                        return_dict=True,
                        use_cache=True)
        past_key_values_small = output_small['past_key_values']
        current_input_ids_small = output_ids[:, -1:, 0]
        
        step += 1
        is_leaf = False
        
        ## list of token graph 
        tgs = []
        tgs_debug = []

        while True:
            if step >= max_new_tokens:
                break
                
            tg = TokenGraph(repeat_threshold)
            tg.add_node(current_input_ids_small.squeeze().item(), -1, 0)
            token_node_num = 1
            token_node_num_debug = 1
            
            
            num_of_generated_token_small = 1
            generate_ids_small[:, 0] = current_input_ids_small[:, -1]
            previous_position = -1
            token_dependency = []
            current_position_ids_large = []
            current_position_ids_large.append(past_key_values_large[0][0].shape[2])
            current_position_ids_small = [past_key_values_small[0][0].shape[2]]
            previous_past_key_values_length_small = past_key_values_small[0][0].shape[2]
                    
            current_step_draft = max_step_draft
            for step_draft in range(max_step_draft):
                ## tree-like decoding时需要注意位置编码
                draft_position_id = previous_past_key_values_length_small + step_draft + 1
                
                # print('small pos', current_position_ids_small)
                output_small = ana.custom_forward(model_small, input_ids=current_input_ids_small,
                                                    position_ids=current_position_ids_small,
                                                    past_key_values=past_key_values_small,
                                                    draft_config=draft_config,
                                                    use_cache=True)
                # if is_leaf:
                #     is_leaf = False
                #     output_small['logits'] = output_small['logits'][:, 1:]
                output_ids_small, output_probs_small = sample_n(output_small['logits'], return_probs=True, n=n_tree)
                    
                past_key_values_small = output_small['past_key_values']
                    
                
                ## 把n个candidate添加到generate_ids_small中
                ## 需要追踪token dependency状态
                # draft_output_ids : batch_size x decode_node_num x n_candidate
                token_tree_node_num = output_ids_small.shape[1]
                current_input_ids_small = []
                current_position_ids_small = []
                for i in range(token_tree_node_num):
                    prob = output_probs_small[0, i, 0]
                    for j in range(n_tree):

                        ## 生成draft_output_ids后需要判断生成的candidate是否可用，把生成概率太低的candidate过滤掉
                        #if True:
                        if output_probs_small[0, i, j] >= 0.7 * prob:

                            if not tg.check_repetition(output_ids_small[:, i, j].squeeze().item(), previous_position + 1):
                                tg.add_node(output_ids_small[:, i, j].squeeze().item(), previous_position + 1, token_node_num)
                                token_node_num += 1

                                token_dependency.append(previous_position)
                                current_input_ids_small.append(output_ids_small[:, i, j])
                                current_position_ids_small.append(draft_position_id)
                    previous_position += 1
                if current_input_ids_small != []:
                    current_input_ids_small = torch.stack(current_input_ids_small, dim = -1)
                else:
                    current_step_draft = step_draft
                    break
                num_generated_tokens = current_input_ids_small.shape[1]
                generate_ids_small[:, num_of_generated_token_small:num_of_generated_token_small + num_generated_tokens] = current_input_ids_small
                num_of_generated_token_small += num_generated_tokens

                current_position_ids_large += current_position_ids_small

                if n_tree > 1:
                    draft_config['token_dependency'] = token_dependency

            

            
            ## verify
            verify_config = copy.deepcopy(draft_config)
            input_ids_large = generate_ids_small[:, :num_of_generated_token_small]

            previous_past_key_values_length = past_key_values_large[0][0].shape[2]

            ## convert token graph to tree
            start, end = 0, 1
            max_token = 100
            queue = [0]
            input_ids_large = [tg.nodes[0].token_id]
            position_ids_large = [0]
            dependency = [-2]
            

            while start < end and end < max_token:
                cur_node = tg.nodes[queue[start]]
                for c in cur_node.child:
                    input_ids_large.append(tg.nodes[c].token_id)
                    dependency.append(start-1)
                    position_ids_large.append(position_ids_large[start] + 1)
                    queue.append(c)
                    end += 1
                if cur_node.pseudo_child is not None:
                    input_ids_large.append(tg.nodes[cur_node.pseudo_child].token_id)
                    dependency.append(start-1)
                    position_ids_large.append(position_ids_large[start] + 1)
                    queue.append(cur_node.pseudo_child)
                    end += 1  
                start += 1
                
            
            index_convert = queue ## 
            verify_config['token_dependency'] = dependency[1:]
            input_ids_large = torch.tensor(input_ids_large).view(1, -1)
            current_position_ids_large = [i + past_key_values_large[0][0].shape[2] for i in position_ids_large]
            
            # print('father', [i.father for i in tg.nodes])
            # print('child', [i.child for i in tg.nodes])
            # print('pseudo_child', [i.pseudo_child for i in tg.nodes])
            # print('id', [i.token_id for i in tg.nodes])

#             print(input_ids_large.shape[-1], input_ids_large)
#             print(len(current_position_ids_large), current_position_ids_large)
#             print(len(dependency), dependency)
 
            #return

            output_large = ana.custom_forward(model_large, input_ids=input_ids_large,
                                        position_ids=current_position_ids_large,
                                        #past_key_values=None,
                                        past_key_values=past_key_values_large,
                                        draft_config=verify_config,
                                        use_cache=True, debug=False)

            ## 删除decoding tree中用不到的节点，并整理kv_cache
            output_ids_large = sample_n(output_large['logits'], n=1)   
            matched_token_index, is_leaf = verify(input_ids_large.squeeze().tolist(), verify_config['token_dependency'], output_ids_large.squeeze().tolist())
    
            token_trees.append([input_ids_large.squeeze().tolist(), token_dependency, output_ids_large.squeeze().tolist()])
            tgs.append(tg)
        
            # print(input_ids_large.squeeze().tolist(), token_dependency, output_ids_large.squeeze().tolist())
            # print(matched_token_index)
            kv_ids_large = list(range(previous_past_key_values_length)) + [previous_past_key_values_length + i for i in matched_token_index]
            past_key_values_large = tuple((i[0][:, :, kv_ids_large, :], i[1][:, :, kv_ids_large, :]) for i in output_large['past_key_values'])
            
            ## reset draft_config
            draft_config['token_dependency'] = None
            
            
            
            output_ids_large = output_ids_large[:, matched_token_index, 0]
            
            # print(input_ids_large)
            # print(output_ids_large)
            # print(matched_token_index)
            
            # matched_token_index = [index_convert[i] for i in matched_token_index]
            
            
        
            new_matched_token_index = []
            flag = 0
            for i in matched_token_index:
                if i != index_convert[i]:
                    flag = 1
                new_matched_token_index.append(index_convert[i])
            n_graph_success += flag
            matched_token_index = new_matched_token_index
            
            # print(index_convert)
            # print(matched_token_index)
            # print(flag)
            # print('\n')
            # if flag:
            
            if is_leaf:
                kv_ids_small = list(range(previous_past_key_values_length_small)) + [previous_past_key_values_length + i for i in matched_token_index[:-1]]
                past_key_values_small = tuple((i[0][:, :, kv_ids_small, :], i[1][:, :, kv_ids_small, :]) for i in past_key_values_small)
                
                output_small = ana.custom_forward(model_small, input_ids=output_ids_large[:, -2:-1],
                                                    position_ids=[past_key_values_small[0][0].shape[2]],
                                                    past_key_values=past_key_values_small,
                                                    draft_config=draft_config,
                                                    use_cache=True)
                past_key_values_small = output_small['past_key_values']
                
                current_input_ids_small = output_ids_large[:, -1:]
                
            else:
                kv_ids_small = list(range(previous_past_key_values_length_small)) + [previous_past_key_values_length + i for i in matched_token_index]
                past_key_values_small = tuple((i[0][:, :, kv_ids_small, :], i[1][:, :, kv_ids_small, :]) for i in past_key_values_small)
                current_input_ids_small = output_ids_large[:, -1:]
            
            
            generate_ids[:, step:step+len(matched_token_index)] = output_ids_large
            
            #current_input_ids_small = output_ids_large[:, -1:]

            step += len(matched_token_index)

            n_matched += len(matched_token_index) - 1
            n_drafted += max_step_draft
            n_node += num_of_generated_token_small
    
    generate_ids = generate_ids[:, :step]
    
    # toc = time.time()
    # print(toc - tic)
    return {
        'generate_ids': generate_ids,
        # 'token_tree': token_trees,
        'matchness': n_matched/n_drafted,
        'drafted_token_num': n_node,
        'graph_success': [n_graph_success, len(tgs), n_graph_success/len(tgs)],
        # 'tg': tgs,
        #'tg_debug': tgs_debug,
        # 'num_drafted_tokens': n_drafted,
        # 'th_stop_draft': th_stop_draft,
    }

def graph_speculative_generation(models, tokenizer, input_ids, draft_config, max_new_tokens=100):
    # tic = time.time()
    
    max_step_draft = draft_config['max_step_draft']
    repeat_threshold = draft_config['repeat_threshold']

    token_trees = []
    
    assert len(models) == 2
    
    model_large = models[0]
    model_small = models[1]
    
    early_stop=False
    max_step_draft=max_step_draft
    th_stop_draft=0.8
    do_sample=False
    do_sample_draft=False
    top_k=0
    top_p=0.85
    temperature=0.2


    step = 0
    step_draft = 0
    step_verify = 0

    n_tree = draft_config['tree_decoding']

    current_input_ids = input_ids
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens+max_step_draft], dtype=torch.long, device=model_large.device)
    if n_tree > 1:
        generate_ids_small = torch.empty([input_ids.size(0), n_tree**(max_step_draft+1)], dtype=torch.long, device=model_large.device)
    else:
        generate_ids_small = torch.empty([input_ids.size(0), max_step_draft+1], dtype=torch.long, device=model_large.device)
    

    n_matched = 0
    n_drafted = 0
    n_node = 0
    n_graph_success = 0
    tmp_n_matched = 0
    tmp_n_drafted = 0
    tmp_matchness = 0

    past_key_values_large = None
    past_key_values_small = None

    with torch.no_grad():
        
        ## initialize the kv-cache for both large and small models
        current_input_ids_large = current_input_ids
        output_large = model_large(input_ids=current_input_ids_large,
                        past_key_values=past_key_values_large,
                        return_dict=True,
                        use_cache=True)
        output_ids = sample_n(output_large['logits'].float(), n=1)   
        generate_ids[:, 0] = output_ids[:, -1, 0]
        past_key_values_large = output_large['past_key_values']
        
        # current_input_ids_small = torch.cat([current_input_ids, output_ids[:, -1:, 0]], dim=-1)
        current_input_ids_small = current_input_ids
        output_small = model_small(input_ids=current_input_ids_small,
                        past_key_values=past_key_values_small,
                        return_dict=True,
                        use_cache=True)
        past_key_values_small = output_small['past_key_values']
        current_input_ids_small = output_ids[:, -1:, 0]
        
        step += 1
        is_leaf = False
        
        ## list of token graph 
        tgs = []
        tgs_debug = []

        while True:
            if step >= max_new_tokens:
                break
                
            tg = TokenGraph(repeat_threshold)
            tg.add_node(current_input_ids_small.squeeze().item(), -1, 0, prob=1)
            token_node_num = 1
            token_node_num_debug = 1
            
            
            num_of_generated_token_small = 1
            generate_ids_small[:, 0] = current_input_ids_small[:, -1]
            previous_position = -1
            token_dependency = []
            current_position_ids_large = []
            current_position_ids_large.append(past_key_values_large[0][0].shape[2])
            current_position_ids_small = [past_key_values_small[0][0].shape[2]]
            previous_past_key_values_length_small = past_key_values_small[0][0].shape[2]
                    
            current_step_draft = max_step_draft
            for step_draft in range(max_step_draft):
                ## tree-like decoding时需要注意位置编码
                draft_position_id = previous_past_key_values_length_small + step_draft + 1
                
                # print('small pos', current_position_ids_small)

                
                output_small = ana.custom_forward_dev(model_small, input_ids=current_input_ids_small,
                                                    position_ids=current_position_ids_small,
                                                    past_key_values=past_key_values_small,
                                                    draft_config=draft_config,
                                                    use_cache=True)
                # if is_leaf:
                #     is_leaf = False
                #     output_small['logits'] = output_small['logits'][:, 1:]
                output_ids_small, output_probs_small = sample_n(output_small['logits'], return_probs=True, n=n_tree)
                    
                    
                past_key_values_small = output_small['past_key_values']
                    
                
                ## 把n个candidate添加到generate_ids_small中
                ## 需要追踪token dependency状态
                # draft_output_ids : batch_size x decode_node_num x n_candidate
                token_tree_node_num = output_ids_small.shape[1]
                current_input_ids_small = []
                current_position_ids_small = []
                for i in range(token_tree_node_num):
                    prob = output_probs_small[0, i, 0]
                    for j in range(n_tree):

                        ## 生成draft_output_ids后需要判断生成的candidate是否可用，把生成概率太低的candidate过滤掉
                        #if True:
                        if output_probs_small[0, i, j] >= 0.7 * prob:

                            if not tg.check_repetition(output_ids_small[:, i, j].squeeze().item(), previous_position + 1) and not tg.check_prob(output_ids_small[:, i, j].squeeze().item(), previous_position + 1, draft_config['prob_threshold']):
                                tg.add_node(output_ids_small[:, i, j].squeeze().item(), previous_position + 1, token_node_num, output_probs_small[0, i, j].item())
                                token_node_num += 1

                                token_dependency.append(previous_position)
                                current_input_ids_small.append(output_ids_small[:, i, j])
                                current_position_ids_small.append(draft_position_id)
                    previous_position += 1
                if current_input_ids_small != []:
                    current_input_ids_small = torch.stack(current_input_ids_small, dim = -1)
                else:
                    current_step_draft = step_draft
                    break
                num_generated_tokens = current_input_ids_small.shape[1]
                generate_ids_small[:, num_of_generated_token_small:num_of_generated_token_small + num_generated_tokens] = current_input_ids_small
                num_of_generated_token_small += num_generated_tokens

                current_position_ids_large += current_position_ids_small

                if n_tree > 1:
                    draft_config['token_dependency'] = token_dependency

            #print(current_step_draft)

            
            ## verify
            verify_config = copy.deepcopy(draft_config)
            input_ids_large = generate_ids_small[:, :num_of_generated_token_small]

            previous_past_key_values_length = past_key_values_large[0][0].shape[2]

            ## convert token graph to tree
            start, end = 0, 1
            max_token = 100
            queue = [0]
            input_ids_large = [tg.nodes[0].token_id]
            position_ids_large = [0]
            dependency = [-2]
            

            while start < end and end < max_token:
                cur_node = tg.nodes[queue[start]]
                for c in cur_node.child:
                    input_ids_large.append(tg.nodes[c].token_id)
                    dependency.append(start-1)
                    position_ids_large.append(position_ids_large[start] + 1)
                    queue.append(c)
                    end += 1
                if cur_node.pseudo_child is not None:
                    input_ids_large.append(tg.nodes[cur_node.pseudo_child].token_id)
                    dependency.append(start-1)
                    position_ids_large.append(position_ids_large[start] + 1)
                    queue.append(cur_node.pseudo_child)
                    end += 1  
                start += 1
                
            
            index_convert = queue ## 
            verify_config['token_dependency'] = dependency[1:]
            input_ids_large = torch.tensor(input_ids_large).view(1, -1)
            current_position_ids_large = [i + past_key_values_large[0][0].shape[2] for i in position_ids_large]

            output_large = ana.custom_forward_dev(model_large, input_ids=input_ids_large,
                                        position_ids=current_position_ids_large,
                                        #past_key_values=None,
                                        past_key_values=past_key_values_large,
                                        draft_config=verify_config,
                                        use_cache=True, debug=False)

            ## 删除decoding tree中用不到的节点，并整理kv_cache
            output_ids_large = sample_n(output_large['logits'], n=1)   
            matched_token_index, is_leaf = verify(input_ids_large.squeeze().tolist(), verify_config['token_dependency'], output_ids_large.squeeze().tolist())
    
            token_trees.append([input_ids_large.squeeze().tolist(), token_dependency, output_ids_large.squeeze().tolist()])
            tgs.append(tg)
        
            # print(input_ids_large.squeeze().tolist(), token_dependency, output_ids_large.squeeze().tolist())
            # print(matched_token_index)
            kv_ids_large = list(range(previous_past_key_values_length)) + [previous_past_key_values_length + i for i in matched_token_index]
            past_key_values_large = tuple((i[0][:, :, kv_ids_large, :], i[1][:, :, kv_ids_large, :]) for i in output_large['past_key_values'])
            
            ## reset draft_config
            draft_config['token_dependency'] = None
            
            
            
            output_ids_large = output_ids_large[:, matched_token_index, 0]
            

            
        
            new_matched_token_index = []
            flag = 0
            for i in matched_token_index:
                if i != index_convert[i]:
                    flag = 1
                new_matched_token_index.append(index_convert[i])
            n_graph_success += flag
            matched_token_index = new_matched_token_index
            
            
            if is_leaf:
                kv_ids_small = list(range(previous_past_key_values_length_small)) + [previous_past_key_values_length + i for i in matched_token_index[:-1]]
                past_key_values_small = tuple((i[0][:, :, kv_ids_small, :], i[1][:, :, kv_ids_small, :]) for i in past_key_values_small)
                
                output_small = ana.custom_forward_dev(model_small, input_ids=output_ids_large[:, -2:-1],
                                                    position_ids=[past_key_values_small[0][0].shape[2]],
                                                    past_key_values=past_key_values_small,
                                                    draft_config=draft_config,
                                                    use_cache=True)
                past_key_values_small = output_small['past_key_values']
                
                current_input_ids_small = output_ids_large[:, -1:]
                
            else:
                kv_ids_small = list(range(previous_past_key_values_length_small)) + [previous_past_key_values_length + i for i in matched_token_index]
                past_key_values_small = tuple((i[0][:, :, kv_ids_small, :], i[1][:, :, kv_ids_small, :]) for i in past_key_values_small)
                current_input_ids_small = output_ids_large[:, -1:]
            
            
            generate_ids[:, step:step+len(matched_token_index)] = output_ids_large
            
            #current_input_ids_small = output_ids_large[:, -1:]

            step += len(matched_token_index)

            n_matched += len(matched_token_index) - 1
            n_drafted += current_step_draft
            n_node += num_of_generated_token_small
    
    generate_ids = generate_ids[:, :step]
    
    # toc = time.time()
    # print(toc - tic)
    return {
        'generate_ids': generate_ids,
        # 'token_tree': token_trees,
        'matchness': n_matched/n_drafted,
        'drafted_token_num': n_node,
        'graph_success': [n_graph_success, len(tgs), n_graph_success/len(tgs)],
        # 'tg': tgs,
        #'tg_debug': tgs_debug,
        # 'num_drafted_tokens': n_drafted,
        # 'th_stop_draft': th_stop_draft,
    }

def graph_speculative_generation_dev(models, tokenizer, input_ids, draft_config, max_new_tokens=100):
    # tic = time.time()
    
    max_step_draft = draft_config['max_step_draft']
    repeat_threshold = draft_config['repeat_threshold']

    token_trees = []
    
    assert len(models) == 2
    
    model_large = models[0]
    model_small = models[1]
    
    early_stop=False
    max_step_draft=max_step_draft
    th_stop_draft=0.8
    do_sample=False
    do_sample_draft=False
    top_k=0
    top_p=0.85
    temperature=0.2


    step = 0
    step_draft = 0
    step_verify = 0

    n_tree = draft_config['tree_decoding']

    current_input_ids = input_ids
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens+max_step_draft], dtype=torch.long, device=model_large.device)
    if n_tree > 1:
        generate_ids_small = torch.empty([input_ids.size(0), n_tree**(max_step_draft+1)], dtype=torch.long, device=model_large.device)
    else:
        generate_ids_small = torch.empty([input_ids.size(0), max_step_draft+1], dtype=torch.long, device=model_large.device)
    

    n_matched = 0
    n_drafted = 0
    n_node = 0
    n_graph_success = 0
    tmp_n_matched = 0
    tmp_n_drafted = 0
    tmp_matchness = 0

    past_key_values_large = None
    past_key_values_small = None

    with torch.no_grad():
        
        ## initialize the kv-cache for both large and small models
        current_input_ids_large = current_input_ids
        output_large = model_large(input_ids=current_input_ids_large,
                        past_key_values=past_key_values_large,
                        return_dict=True,
                        use_cache=True)
        output_ids = sample_n(output_large['logits'].float(), n=1)   
        generate_ids[:, 0] = output_ids[:, -1, 0]
        past_key_values_large = output_large['past_key_values']
        
        # current_input_ids_small = torch.cat([current_input_ids, output_ids[:, -1:, 0]], dim=-1)
        current_input_ids_small = current_input_ids
        output_small = model_small(input_ids=current_input_ids_small,
                        past_key_values=past_key_values_small,
                        return_dict=True,
                        use_cache=True)
        past_key_values_small = output_small['past_key_values']
        current_input_ids_small = output_ids[:, -1:, 0]
        
        step += 1
        is_leaf = False
        
        ## list of token graph 
        tgs = []
        tgs_debug = []

        while True:
            if step >= max_new_tokens:
                break
                
            tg = TokenGraph(repeat_threshold)
            tg.add_node(current_input_ids_small.squeeze().item(), -1, 0, prob=1)
            token_node_num = 1
            token_node_num_debug = 1
            
            
            num_of_generated_token_small = 1
            generate_ids_small[:, 0] = current_input_ids_small[:, -1]
            previous_position = -1
            token_dependency = []
            current_position_ids_large = []
            current_position_ids_large.append(past_key_values_large[0][0].shape[2])
            current_position_ids_small = [past_key_values_small[0][0].shape[2]]
            previous_past_key_values_length_small = past_key_values_small[0][0].shape[2]
                    
            current_step_draft = max_step_draft
            for step_draft in range(max_step_draft):
                ## tree-like decoding时需要注意位置编码
                draft_position_id = previous_past_key_values_length_small + step_draft + 1
                
                # print('small pos', current_position_ids_small)

                output_small = model_small(
                    input_ids=current_input_ids_small,
                    position_ids=current_position_ids_small,
                    past_key_values=past_key_values_small,                                      
                    draft_config=draft_config,
                    return_dict=True,
                    use_cache=True,
                )

                output_ids_small, output_probs_small = sample_n(output_small['logits'], return_probs=True, n=n_tree)
                    
                past_key_values_small = output_small['past_key_values']
                    
                
                ## 把n个candidate添加到generate_ids_small中
                ## 需要追踪token dependency状态
                # draft_output_ids : batch_size x decode_node_num x n_candidate
                token_tree_node_num = output_ids_small.shape[1]
                current_input_ids_small = []
                current_position_ids_small = []
                terminate_drafting = True
                for i in range(token_tree_node_num):
                    prob = output_probs_small[0, i, 0]
                    for j in range(n_tree):

                        ## 生成draft_output_ids后需要判断生成的candidate是否可用，把生成概率太低的candidate过滤掉
                        #if True:
                        if output_probs_small[0, i, j] >= 0.7 * prob:

                            if not tg.check_repetition(output_ids_small[:, i, j].squeeze().item(), previous_position + 1) and not tg.check_prob(output_ids_small[:, i, j].squeeze().item(), previous_position + 1, draft_config['prob_threshold']):
                                tg.add_node(output_ids_small[:, i, j].squeeze().item(), previous_position + 1, token_node_num, output_probs_small[0, i, j].item())
                                token_node_num += 1

                                token_dependency.append(previous_position)
                                current_input_ids_small.append(output_ids_small[:, i, j])
                                current_position_ids_small.append(draft_position_id)
                    previous_position += 1
                if current_input_ids_small != []:
                    current_input_ids_small = torch.stack(current_input_ids_small, dim = -1)
                else:
                    current_step_draft = step_draft
                    break
                num_generated_tokens = current_input_ids_small.shape[1]
                generate_ids_small[:, num_of_generated_token_small:num_of_generated_token_small + num_generated_tokens] = current_input_ids_small
                num_of_generated_token_small += num_generated_tokens

                current_position_ids_large += current_position_ids_small

                if n_tree > 1:
                    draft_config['token_dependency'] = token_dependency

            #print(current_step_draft)

            
            ## verify
            verify_config = copy.deepcopy(draft_config)
            input_ids_large = generate_ids_small[:, :num_of_generated_token_small]

            previous_past_key_values_length = past_key_values_large[0][0].shape[2]

            ## convert token graph to tree
            start, end = 0, 1
            max_token = 100
            queue = [0]
            input_ids_large = [tg.nodes[0].token_id]
            position_ids_large = [0]
            dependency = [-2]
            

            while start < end and end < max_token:
                cur_node = tg.nodes[queue[start]]
                for c in cur_node.child:
                    input_ids_large.append(tg.nodes[c].token_id)
                    dependency.append(start-1)
                    position_ids_large.append(position_ids_large[start] + 1)
                    queue.append(c)
                    end += 1
                if cur_node.pseudo_child is not None:
                    input_ids_large.append(tg.nodes[cur_node.pseudo_child].token_id)
                    dependency.append(start-1)
                    position_ids_large.append(position_ids_large[start] + 1)
                    queue.append(cur_node.pseudo_child)
                    end += 1  
                start += 1
                
            
            index_convert = queue ## 
            verify_config['token_dependency'] = dependency[1:]
            input_ids_large = torch.tensor(input_ids_large).view(1, -1)
            current_position_ids_large = [i + past_key_values_large[0][0].shape[2] for i in position_ids_large]

            output_large = model_large(input_ids=input_ids_large,
                position_ids=current_position_ids_large,
                past_key_values=past_key_values_large,
                draft_config=verify_config,
                return_dict=True,
                use_cache=True, 
            )


            ## 删除decoding tree中用不到的节点，并整理kv_cache
            output_ids_large = sample_n(output_large['logits'], n=1)   
            matched_token_index, is_leaf = verify(input_ids_large.squeeze().tolist(), verify_config['token_dependency'], output_ids_large.squeeze().tolist())
    
            token_trees.append([input_ids_large.squeeze().tolist(), token_dependency, output_ids_large.squeeze().tolist()])
            tgs.append(tg)
        
            # print(input_ids_large.squeeze().tolist(), token_dependency, output_ids_large.squeeze().tolist())
            # print(matched_token_index)
            kv_ids_large = list(range(previous_past_key_values_length)) + [previous_past_key_values_length + i for i in matched_token_index]
            past_key_values_large = tuple((i[0][:, :, kv_ids_large, :], i[1][:, :, kv_ids_large, :]) for i in output_large['past_key_values'])
            
            ## reset draft_config
            draft_config['token_dependency'] = None
            
            
            
            output_ids_large = output_ids_large[:, matched_token_index, 0]
            

            
        
            new_matched_token_index = []
            flag = 0
            for i in matched_token_index:
                if i != index_convert[i]:
                    flag = 1
                new_matched_token_index.append(index_convert[i])
            n_graph_success += flag
            matched_token_index = new_matched_token_index
            
            
            if is_leaf:
                kv_ids_small = list(range(previous_past_key_values_length_small)) + [previous_past_key_values_length + i for i in matched_token_index[:-1]]
                past_key_values_small = tuple((i[0][:, :, kv_ids_small, :], i[1][:, :, kv_ids_small, :]) for i in past_key_values_small)
                
                output_small = model_small(
                    input_ids=output_ids_large[:, -2:-1],
                    position_ids=[past_key_values_small[0][0].shape[2]],
                    past_key_values=past_key_values_small,                                      
                    draft_config=draft_config,
                    return_dict=True,
                    use_cache=True,
                )
                
                past_key_values_small = output_small['past_key_values']
                
                current_input_ids_small = output_ids_large[:, -1:]
                
            else:
                kv_ids_small = list(range(previous_past_key_values_length_small)) + [previous_past_key_values_length + i for i in matched_token_index]
                past_key_values_small = tuple((i[0][:, :, kv_ids_small, :], i[1][:, :, kv_ids_small, :]) for i in past_key_values_small)
                current_input_ids_small = output_ids_large[:, -1:]
            
            
            generate_ids[:, step:step+len(matched_token_index)] = output_ids_large
            
            #current_input_ids_small = output_ids_large[:, -1:]

            step += len(matched_token_index)

            n_matched += len(matched_token_index) - 1
            n_drafted += current_step_draft
            n_node += num_of_generated_token_small
    
    generate_ids = generate_ids[:, :step]
    
    # toc = time.time()
    # print(toc - tic)
    return {
        'generate_ids': generate_ids,
        # 'token_tree': token_trees,
        'matchness': n_matched/n_drafted,
        'drafted_token_num': n_node,
        'graph_success': [n_graph_success, len(tgs), n_graph_success/len(tgs)],
        # 'tg': tgs,
        #'tg_debug': tgs_debug,
        # 'num_drafted_tokens': n_drafted,
        # 'th_stop_draft': th_stop_draft,
    }


def graph_speculative_generation_ssd(models, tokenizer, input_ids, draft_config, max_new_tokens=100):
    # tic = time.time()
    
    max_step_draft = draft_config['max_step_draft']
    repeat_threshold = draft_config['repeat_threshold']

    token_trees = []
    
    #input_ids = tokenizer([prompt], return_tensors='pt')['input_ids']

    assert len(models) == 2
    
    model_large = models[0]
    # model_small = models[1]
    
    early_stop=False
    max_step_draft=max_step_draft
    th_stop_draft=0.8
    do_sample=False
    do_sample_draft=False
    top_k=0
    top_p=0.85
    temperature=0.2


    step = 0
    step_draft = 0
    step_verify = 0

    n_tree = draft_config['tree_decoding']

    current_input_ids = input_ids
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens+max_step_draft], dtype=torch.long, device=model_large.device)
    if n_tree > 1:
        generate_ids_small = torch.empty([input_ids.size(0), n_tree**(max_step_draft+1)], dtype=torch.long, device=model_large.device)
    else:
        generate_ids_small = torch.empty([input_ids.size(0), max_step_draft+1], dtype=torch.long, device=model_large.device)
    

    n_matched = 0
    n_drafted = 0
    n_node = 0
    n_graph_success = 0
    tmp_n_matched = 0
    tmp_n_drafted = 0
    tmp_matchness = 0

    past_key_values_large = None
    past_key_values_small = None

    with torch.no_grad():
        
        ## initialize the kv-cache for both large and small models
        current_input_ids_large = current_input_ids
        output_large = model_large(input_ids=current_input_ids_large,
                        past_key_values=past_key_values_large,
                        return_dict=True,
                        use_cache=True)
        output_ids = sample_n(output_large['logits'].float(), n=1)   
        generate_ids[:, 0] = output_ids[:, -1, 0]
        past_key_values_large = output_large['past_key_values']
        
        # current_input_ids_small = torch.cat([current_input_ids, output_ids[:, -1:, 0]], dim=-1)
        current_input_ids_small = current_input_ids

         
        
        past_key_values_small = past_key_values_large
        current_input_ids_small = output_ids[:, -1:, 0]
        
        step += 1
        is_leaf = False
        
        ## list of token graph 
        tgs = []
        tgs_debug = []

        while True:
            if step >= max_new_tokens:
                break
            
            tg = TokenGraph(repeat_threshold)
            tg.add_node(current_input_ids_small.squeeze().item(), -1, 0, prob=1)
            token_node_num = 1
            token_node_num_debug = 1
            
            
            num_of_generated_token_small = 1
            generate_ids_small[:, 0] = current_input_ids_small[:, -1]
            previous_position = -1
            token_dependency = []
            current_position_ids_large = []
            current_position_ids_large.append(past_key_values_large[0][0].shape[2])
            current_position_ids_small = [past_key_values_small[0][0].shape[2]]
            previous_past_key_values_length_small = past_key_values_small[0][0].shape[2]
                    
            current_step_draft = max_step_draft
            for step_draft in range(max_step_draft):
                ## tree-like decoding时需要注意位置编码
                draft_position_id = previous_past_key_values_length_small + step_draft + 1
                
                # print('small pos', current_position_ids_small)
                
                
                with model_large.self_draft():
                    output_small = model_large(input_ids=current_input_ids_small,
                                                position_ids=current_position_ids_small,
                                                past_key_values=past_key_values_small,
                                                self_speculative=True,                                      
                                                draft_config=draft_config,
                                                return_dict=True,
                                                use_cache=True)
                
                
                output_ids_small, output_probs_small = sample_n(output_small['logits'], return_probs=True, n=n_tree)
                    
                    
                past_key_values_small = output_small['past_key_values']
                    
                
                ## 把n个candidate添加到generate_ids_small中
                ## 需要追踪token dependency状态
                # draft_output_ids : batch_size x decode_node_num x n_candidate
                token_tree_node_num = output_ids_small.shape[1]
                current_input_ids_small = []
                current_position_ids_small = []
                terminate_drafting = True
                for i in range(token_tree_node_num):
                    prob = output_probs_small[0, i, 0]
                    for j in range(n_tree):

                        ## 生成draft_output_ids后需要判断生成的candidate是否可用，把生成概率太低的candidate过滤掉
                        #if True:
                        if output_probs_small[0, i, j] >= 0.7 * prob:

                            # if not tg.check_repetition(output_ids_small[:, i, j].squeeze().item(), previous_position + 1) and not tg.check_prob(output_ids_small[:, i, j].squeeze().item(), previous_position + 1, draft_config['prob_threshold']):
                            if not tg.check_repetition(output_ids_small[:, i, j].squeeze().item(), previous_position + 1):
                                
                                ## 如果父节点为leaf，那么把当前节点也标记为leaf
                                if tg.check_leaf(output_ids_small[:, i, j].squeeze().item(), previous_position + 1):
                                    tg.add_node(output_ids_small[:, i, j].squeeze().item(), previous_position + 1, token_node_num, output_probs_small[0, i, j].item(), leaf=True)
                                ## 如果当前节点的生成概率小于阈值，则把当前节点也标记为leaf
                                elif output_probs_small[0, i, j] < draft_config['prob_threshold']:
                                    tg.add_node(output_ids_small[:, i, j].squeeze().item(), previous_position + 1, token_node_num, output_probs_small[0, i, j].item(), leaf=True)
                                else:
                                    tg.add_node(output_ids_small[:, i, j].squeeze().item(), previous_position + 1, token_node_num, output_probs_small[0, i, j].item(), leaf=False)
                                    terminate_drafting = False
                                    
                                token_node_num += 1
                                    
                                token_dependency.append(previous_position)
                                current_input_ids_small.append(output_ids_small[:, i, j])
                                current_position_ids_small.append(draft_position_id)
                    previous_position += 1
                    
               
                if current_input_ids_small != []:
                    current_input_ids_small = torch.stack(current_input_ids_small, dim = -1)
                else:
                    current_step_draft = step_draft + 1
                    break
                    
                
                num_generated_tokens = current_input_ids_small.shape[1]
                generate_ids_small[:, num_of_generated_token_small:num_of_generated_token_small + num_generated_tokens] = current_input_ids_small
                num_of_generated_token_small += num_generated_tokens

                current_position_ids_large += current_position_ids_small

                if n_tree > 1:
                    draft_config['token_dependency'] = token_dependency
                    
                if terminate_drafting or step + step_draft + 2 >= max_new_tokens:
                    current_step_draft = step_draft + 1
                    break
                    
            #print(current_step_draft)
            
            
            
            ## verify
            verify_config = copy.deepcopy(draft_config)
            input_ids_large = generate_ids_small[:, :num_of_generated_token_small]

            previous_past_key_values_length = past_key_values_large[0][0].shape[2]

            ## convert token graph to tree
            start, end = 0, 1
            max_token = 100
            queue = [0]
            input_ids_large = [tg.nodes[0].token_id]
            position_ids_large = [0]
            dependency = [-2]
            

            while start < end and end < max_token:
                cur_node = tg.nodes[queue[start]]
                for c in cur_node.child:
                    input_ids_large.append(tg.nodes[c].token_id)
                    dependency.append(start-1)
                    position_ids_large.append(position_ids_large[start] + 1)
                    queue.append(c)
                    end += 1
                if cur_node.pseudo_child is not None:
                    input_ids_large.append(tg.nodes[cur_node.pseudo_child].token_id)
                    dependency.append(start-1)
                    position_ids_large.append(position_ids_large[start] + 1)
                    queue.append(cur_node.pseudo_child)
                    end += 1  
                start += 1
                
            
            
            index_convert = queue ## 
            verify_config['token_dependency'] = dependency[1:]
            input_ids_large = torch.tensor(input_ids_large).view(1, -1)
            current_position_ids_large = [i + past_key_values_large[0][0].shape[2] for i in position_ids_large]
            
            output_large = model_large(input_ids=input_ids_large,
                                        position_ids=current_position_ids_large,
                                        past_key_values=past_key_values_large,
                                        draft_config=verify_config,
                                        self_speculative=True,
                                       return_dict=True,
                                        use_cache=True, 
                                       )
            
            
                               
            
                
            ## 删除decoding tree中用不到的节点，并整理kv_cache
            output_ids_large = sample_n(output_large['logits'], n=1)   
            matched_token_index, is_leaf = verify(input_ids_large.squeeze().tolist(), verify_config['token_dependency'], output_ids_large.squeeze().tolist())
    
            token_trees.append([input_ids_large.squeeze().tolist(), token_dependency, output_ids_large.squeeze().tolist()])
            tgs.append(tg)
            
            past_key_values_large = output_large['past_key_values']
            
            
            past_key_values_large = efficient_indexing_kv_cache(past_key_values_large, previous_past_key_values_length, matched_token_index)
            
            
            ## reset draft_config
            draft_config['token_dependency'] = None
            
            output_ids_large = output_ids_large[:, matched_token_index, 0]
            
            
        
            new_matched_token_index = []
            flag = 0
            for i in matched_token_index:
                if i != index_convert[i]:
                    flag = 1
                new_matched_token_index.append(index_convert[i])
            n_graph_success += flag
            matched_token_index = new_matched_token_index
            
            # print(input_ids_large)
            # print(index_convert)
            # print(matched_token_index)
            # print(num_of_generated_token_small)
            # print('\n')
            
            
            past_key_values_small = past_key_values_large
            current_input_ids_small = output_ids_large[:, -1:]
   
            generate_ids[:, step:step+len(matched_token_index)] = output_ids_large
            
            #current_input_ids_small = output_ids_large[:, -1:]

            step += len(matched_token_index)

            n_matched += len(matched_token_index) - 1
            n_drafted += current_step_draft
            n_node += num_of_generated_token_small - 1
    
             
            
            
    generate_ids = generate_ids[:, :step]
    
    # toc = time.time()
    # print(toc - tic)
    return {
        'generate_ids': generate_ids,
        # 'token_tree': token_trees,
        'matchness': n_matched/n_drafted,
        'drafted_token_num': n_node,
        'graph_success': [n_graph_success, len(tgs), n_graph_success/len(tgs)],
        # 'tg': tgs,
        #'tg_debug': tgs_debug,
        # 'num_drafted_tokens': n_drafted,
        # 'th_stop_draft': th_stop_draft,
    }

generate_fn_mapping = {
    'graph': graph_speculative_generation,
    'dev': graph_speculative_generation_dev,
    'self_graph': graph_speculative_generation_ssd,
}

def infer(models, tokenizer, prompt, generate_fn, 
          decode_timing=True, seed=42, *args, **kargs):

    if isinstance(generate_fn, str):
        generate_fn = generate_fn_mapping[generate_fn]

    if seed is not None:
        torch.manual_seed(seed)
              
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(models[0].device)
    if decode_timing:
        tic = time.time()
    generate_dict = generate_fn(models, tokenizer, input_ids, *args, **kargs)
    generate_ids = generate_dict['generate_ids']
    if decode_timing:
        toc = time.time()
        decode_time = toc - tic
    else:
        decode_time = None
    completion = tokenizer.decode(generate_ids[0])
    generate_dict['completion'] = completion
    generate_dict['time'] = decode_time
    return generate_dict


def efficient_indexing_kv_cache(past_key_values_large, previous_past_key_values_length, matched_token_index):
    index = matched_token_index
    kv_length = past_key_values_large[0][0].shape[2]  
    kv_ids_large_1 = list(range(previous_past_key_values_length)) 
    kv_ids_large_2 = [previous_past_key_values_length + i for i in matched_token_index]
    kv_ids_large = kv_ids_large_1 + kv_ids_large_2
    
    past_key_1 = [
                (k[:, :, :-(kv_length-previous_past_key_values_length)], v[:, :, :-(kv_length-previous_past_key_values_length)]) for k, v in past_key_values_large
    ]
    
    past_key_2 = [
                (k[:, :, kv_ids_large_2], v[:, :, kv_ids_large_2]) for k, v in past_key_values_large
    ]
            
    past_key_values_large = [(torch.cat([past_key_1[i][0], past_key_2[i][0]], dim=2), torch.cat([past_key_1[i][1], past_key_2[i][1]], dim=2)) for i in range(len(past_key_1))]
#     devices = [k.device for k, _ in past_key_values_large]
#     # print(devices)
    
#     device = past_key_values_large[0][0].device
#     # while True:
    
#     k_cache = torch.stack([k.to(device) for k, v in past_key_values_large])
#     v_cache = torch.stack([v.to(device) for k, v in past_key_values_large])
    
#     t1 = time.time()
    
#     k_cache = k_cache[:, :, :, kv_ids_large]
#     v_cache = v_cache[:, :, :, kv_ids_large]
    
    
#     t2 = time.time()
#     print('kv', t2-t1)
    
#     k_cache = k_cache.split(1, dim=0)
#     v_cache = v_cache.split(1, dim=0)
    
    
#     past_key_values_large = [(k_cache[i].to(devices[i]).squeeze(0) ,v_cache[i].to(devices[i]).squeeze(0)) for i in range(len(k_cache))]
    return past_key_values_large