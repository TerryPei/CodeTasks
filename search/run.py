import argparse
import logging
import os
import pickle
import random
# from pyrsistent import T
import torch
import json
import numpy as np
import math
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  AutoTokenizer, AutoModel, AutoConfig)
from tqdm import tqdm
import itertools
import multiprocessing
import re
import sys
# sys.path.append("..")
from codeparser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from codeparser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('./codeparser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser

logger = logging.getLogger(__name__)




def extract_dataflow(code, parser, lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
    
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]

    control_keywords = {'for', 'while', 'if', 'else', 'elif', 'try', 'except', 'raise'}
    code_tokens = [re.sub(r"[-()\"#/@;:<>{}`\[\]|_.!=,]", "", file) for file in code_tokens]
    code_tokens = [token for token in code_tokens if token!=""]
    
    # petorch.sin(control_position)
    return code_tokens, dfg, control_keywords
    # return code_tokens,dfg

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def save_model(model, tokenizer, path):
    model_path = path + 'model'
    tokenizer_path = path + 'tokenizer'

    assert os.path.exists(model_path) == 1
    model.save_pretrained(model_path)

    assert os.path.exists(tokenizer_path) == 1
    tokenizer.save_pretrained(tokenizer_path)

def load_model(path):
    model_path = path + 'model'
    tokenizer_path = path + 'tokenizer'

    assert os.path.exists(model_path) == 1
    model =  AutoModel.from_pretrained(model_path)
    # model =  RobertaModel.from_pretrained(model_path)

    assert os.path.exists(tokenizer_path) == 1
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, config=AutoConfig.from_pretrained(model_path))
    # tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

class Pair(object):
    """
    A Pre-training Pairs for a Code and Description.
    """
    def __init__(self, code_token_ids, code_mask_ids, code_pos_ids, \
        graph_token_ids, graph_mask_ids, graph_pos_ids, \
            comment_token_ids, comment_mask_ids, url=None):
        
        self.code_token_ids = code_token_ids
        self.code_mask_ids = code_mask_ids
        self.code_pos_ids = code_pos_ids

        self.graph_token_ids = graph_token_ids
        self.graph_mask_ids = graph_mask_ids
        self.graph_pos_ids = graph_pos_ids

        self.comment_token_ids = comment_token_ids
        self.comment_mask_ids = comment_mask_ids

        self.url=url

# args.lang = 'ruby'
# file_type = 'valid'
# file_path = './dataset/'+args.lang+'/'+file_type+'.jsonl'
# prefix=file_path.split('/')[-1][:-6]
# cache_file=args.output_dir+'/'+prefix+'.pkl'
def convert_codes_to_pairs(item):
    js, tokenizer, args = item
    #code
    parser = parsers[args.lang]
    #extract code context and graph information
    code_tokens, dfg, control_keywords = extract_dataflow(js['original_string'], parser, args.lang)
    # print(len(code_tokens)) # 31
    # code_tokens = code_tokens[:args.max_code_length-1]
    # code_tokens=[tokenizer.tokenize('@'+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
    # print(len(code_tokens)) # 31
    # code_tokens = [tokenizer.tokenize(x) for _ ,x in enumerate(code_tokens)]
    # code_tokens=[y for x in code_tokens for y in x]
    # print(len(code_tokens)) # 33
    #truncating
    code_tokens = code_tokens[:args.max_code_length-1]
    code_tokens = [tokenizer.cls_token] + code_tokens
    # Transfer token to token_ids
    code_token_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    # print(code_token_ids)
    # mask_ids
    code_mask_ids = [1] * (len(code_tokens))
    # print(code_mask_ids)
    # 基于control_keywords给每个变量加上相对于关键词的逻辑位置
    control_flow_pos = [0] * len(code_tokens)
    pos_count = 1
    for i, token in enumerate(code_tokens):
        if i == 0: # CLS不计入内
            continue
        if token.lower() in control_keywords:
            pos_count += 1
        control_flow_pos[i] = pos_count
    code_pos_ids = control_flow_pos
        # control_flow_pos[i] = float(pos_count)/10
    # code_pos_ids = list(np.sin(control_flow_pos))
    # print(len(code_pos_ids), len(code_token_ids))
    # print(code_token_ids, code_pos_ids)
    # print(type(code_token_ids), type(code_pos_ids)) # list, numpy
    #padding
    code_padding_length = args.max_code_length - len(code_tokens)
    code_token_ids += [tokenizer.pad_token_id] * code_padding_length
    code_mask_ids += [0] * code_padding_length 
    code_pos_ids += [0] * code_padding_length

    # graph
    graph_tokens = [tokenizer.cls_token] + [x[0] for x in dfg]
    graph_tokens = graph_tokens[:args.max_graph_length - 1]
    graph_token_ids = tokenizer.convert_tokens_to_ids(graph_tokens)
    graph_mask_ids = [1] * (len(graph_tokens))

    # data_flow_tokens = [x[0] for x in dfg]
    data_flow_index = [0] + [x[1] for x in dfg]
    graph_pos_ids = data_flow_index
    # graph_pos_ids = list(np.sin(data_flow_index))
    
    
    #padding
    graph_padding_length = args.max_graph_length - len(graph_token_ids)
    graph_token_ids += [tokenizer.pad_token_id] * graph_padding_length
    graph_mask_ids += [0] * graph_padding_length
    graph_pos_ids += [0] * graph_padding_length
    # print(len(graph_pos_ids), len(graph_token_ids))
    # print(graph_tokens)
    # print(graph_token_ids)
    #extract code description information labeled bu human beings.
    #comment
    comment = js['docstring_tokens']
    comment_tokens = comment[:args.max_comment_length - 1]
    comment_tokens = [tokenizer.cls_token] + comment_tokens
    comment_token_ids =  tokenizer.convert_tokens_to_ids(comment_tokens)
    comment_mask_ids = [1] * (len(comment_tokens))
    comment_padding_length = args.max_comment_length - len(comment_token_ids)

    comment_token_ids += [tokenizer.pad_token_id] * comment_padding_length
    comment_mask_ids += [0] * comment_padding_length # mask中1是实际需要的，0是pad掉的

    return Pair(code_token_ids, code_mask_ids, code_pos_ids, \
                graph_token_ids, graph_mask_ids, graph_pos_ids, comment_token_ids, comment_mask_ids)

def preprocess(file_path):
    pairs = []
    data = []
    cpu_count = 8
    pool = multiprocessing.Pool(processes=cpu_count)
    with open(file_path) as f:
        for i, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            data.append((js, tokenizer, args))

    pairs=pool.map(convert_codes_to_pairs, tqdm(data,total=len(data)))
            # # print(comment_token_ids)
            # pairs.append(Pair(code_token_ids, code_mask_ids, code_pos_ids, \
            #     graph_token_ids, graph_mask_ids, graph_pos_ids, comment_token_ids, comment_mask_ids))
    pool.close()
    return pairs
 
class Dataset(torch.utils.data.Dataset):
      
    def __init__(self,  data_path=None, transforms=None):
        """
        If there are multiple structures for each code pure context, the code_filenames must have repetitive
        file names 
        """
        self.pairs = preprocess(data_path)

        self.transforms = transforms
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx): 
        # Pair(code_token_ids, code_mask_ids, code_pos_ids, comment_token_ids, comment_mask_ids, comment_pos_ids)
        item = {
            'code_token_ids': torch.tensor(self.pairs[idx].code_token_ids),
            'code_mask_ids': torch.tensor(self.pairs[idx].code_mask_ids),
            'code_pos_ids': torch.tensor(self.pairs[idx].code_pos_ids),
            'graph_token_ids': torch.tensor(self.pairs[idx].code_token_ids),
            'graph_mask_ids': torch.tensor(self.pairs[idx].code_mask_ids),
            'graph_pos_ids': torch.tensor(self.pairs[idx].code_pos_ids),
            'comment_token_ids': torch.tensor(self.pairs[idx].comment_token_ids),
            'comment_mask_ids': torch.tensor(self.pairs[idx].comment_mask_ids),
        }
        return item

def build_loaders(data_path, batch_size=8, num_workers=4):
    # transforms = get_transforms(mode=mode)
    dataset = Dataset(
        data_path=data_path
    )
    # if args:
    #     sampler = RandomSampler(dataset)
    #     dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=num_workers)
    # else:
    #     dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
    return dataloader

def get_mrr(logits):   

    sort_inds=np.argsort(logits, axis=-1, kind='quicksort', order=None)[:,::-1]  

    ranks = []
    find = False

    row_ids, col_ids = np.arange(sort_inds.shape[0]), np.arange(sort_inds.shape[1])

    for query, sort_id in zip(row_ids, sort_inds):
        rank=0
        find=False
        for idx in sort_id[:1000]:
            if find is False:
                rank+=1
            if col_ids[idx]==query:
                find=True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr": np.mean(ranks)
    }

    return result["eval_mrr"]

def evaluate(args, tcp_code, tcp_comment, dataloader, device, query='nl'):

    code_features, comment_features = [], []

    try:

        tcp_code.eval()
        tcp_comment.eval()

        with torch.no_grad():
        
            for batch in dataloader:
                code_emb = tcp_code(input_ids=batch['code_token_ids'].to(device), \
                                    attention_mask=batch['code_mask_ids'].to(device), \
                                    position_ids=batch['code_pos_ids'].to(device))[1]
                # graph_emb = tcp_graph(input_ids=batch['graph_token_ids'].to(device), \
                #                     attention_mask=batch['graph_mask_ids'].to(device), \
                #                     position_ids=batch['graph_pos_ids'].to(device))[1]

                comment_emb = tcp_comment(input_ids=batch['comment_token_ids'].to(device), \
                                        attention_mask=batch['comment_mask_ids'].to(device))[1]

                code_feature = code_emb / code_emb.norm(dim=-1, keepdim=True)
                comment_feature = comment_emb / comment_emb.norm(dim=-1, keepdim=True)
                # graph_feature = graph_emb / graph_emb.norm(dim=-1, keepdim=True)

                code_features.append(code_feature.cpu().numpy())
                comment_features.append(comment_feature.cpu().numpy())

        tcp_code.train()
        tcp_comment.train()

        code_features = np.concatenate(code_features, 0)
        comment_features = np.concatenate(comment_features, 0)

        if query=='nl':
            similarity = comment_features @ code_features.T
        else:
            similarity = code_features @ comment_features.T
    except:
        logging.exception('Got exception on main handler')
        raise
    mrr = get_mrr(similarity)
    return mrr

def get_cos_sim(tcp_code, tcp_graph, tcp_comment, dataloader, device):

    cos_xy, cos_yz, cos_xz = 0, 0, 0
    max_s, triple_sim = 0, 0

    try:
        tcp_code.eval()
        tcp_graph.eval()
        tcp_comment.eval()

        with torch.no_grad():
        
            for batch in dataloader:

                # batch_size = len(batch)

                code_emb = tcp_code(input_ids=batch['code_token_ids'].to(device), \
                                    attention_mask=batch['code_mask_ids'].to(device), \
                                    position_ids=batch['code_pos_ids'].to(device))[1]

                graph_emb = tcp_graph(input_ids=batch['graph_token_ids'].to(device), \
                                    attention_mask=batch['graph_mask_ids'].to(device), \
                                    position_ids=batch['graph_pos_ids'].to(device))[1]

                comment_emb = tcp_comment(input_ids=batch['comment_token_ids'].to(device), \
                                        attention_mask=batch['comment_mask_ids'].to(device))[1]

                code_feature = code_emb / code_emb.norm(dim=-1, keepdim=True)
                graph_feature = graph_emb / graph_emb.norm(dim=-1, keepdim=True)
                comment_feature = comment_emb / comment_emb.norm(dim=-1, keepdim=True)

                emb = torch.stack((code_feature, graph_feature, comment_feature), dim=1) # [n, 3, d]

                # emb = emb / emb.norm(dim=-1, keepdim=True)
                u, s, v = torch.linalg.svd(emb, full_matrices=True)

                max_s += s[:, 0].mean().data.item()
                # triple_sim += (max_s ** 2 - 1) / (3 - 1)

                cos_xy += np.mean([(code_feature[i] @ graph_feature[i].T).data.item() for i in range(code_feature.shape[0])])
                cos_yz += np.mean([(graph_feature[i] @ comment_feature[i].T).data.item() for i in range(graph_feature.shape[0])])
                cos_xz += np.mean([(code_feature[i] @ comment_feature[i].T).data.item() for i in range(comment_feature.shape[0])])
            
            cos_xy = cos_xy / len(dataloader)
            cos_yz = cos_yz / len(dataloader)
            cos_xz = cos_xz / len(dataloader)
            max_s = max_s / len(dataloader)
            triple_sim = max_s ** 2 / 3

        tcp_code.train()
        tcp_graph.train()
        tcp_comment.train()

    except:
        logging.exception('Got exception on cos_sim handler')
        raise

    return cos_xy, cos_yz, cos_xz, max_s, triple_sim

def train(args, tcp_code, tcp_graph, tcp_comment, device, resume=False):
    
    """ Train the model """

    train_dataloader = build_loaders(args.train_data_file, batch_size=args.train_batch_size, num_workers=4)

    params = list(tcp_code.parameters()) + list(tcp_graph.parameters()) + list(tcp_comment.parameters())
    loss_f = nn.MSELoss()
    # loss_cos = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, eps=args.eps, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.num_train_epochs*0.1, num_training_steps=len(train_dataloader)*args.num_train_epochs)

    train_loss = []
    best_mrr = 0

    for epoch in range(args.num_train_epochs):
        b_loss = 0
        # b_loss_cos = 0
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        # logger.info("check1")
        # logger.info("tcp_code.device: {} | tcp_graph.device: {} | tcp_comment.device: {} ".format(str(tcp_code.module.device), str(tcp_graph.module.device), str(tcp_comment.module.device)))
        # tcp_code.device: cuda:0 | tcp_graph.device: cuda:0 | tcp_comment.device: cuda:0 
        # logger.info("tcp_code.device: {} | tcp_graph.device: {} | tcp_comment.device: {} ".format(str(tcp_code.device_ids), str(tcp_graph.device_ids), str(tcp_comment.device_ids)))
        # tcp_code.device: [0, 1] | tcp_graph.device: [0, 1] | tcp_comment.device: [0, 1]
        # logger.info("device:", device)
        for batch in bar:
            # logger.info("check start.")
            optimizer.zero_grad()
            # batch['code_token_ids'] = batch['code_token_ids'].to(device)
            try:
                code_emb = tcp_code(input_ids=batch['code_token_ids'].to(device), \
                        attention_mask=batch['code_mask_ids'].to(device), \
                        position_ids=batch['code_pos_ids'].to(device))[1]

                graph_emb = tcp_graph(input_ids=batch['graph_token_ids'].to(device), \
                                    attention_mask=batch['graph_mask_ids'].to(device), \
                                    position_ids=batch['graph_pos_ids'].to(device))[1]

                comment_emb = tcp_comment(input_ids=batch['comment_token_ids'].to(device), \
                                        attention_mask=batch['comment_mask_ids'].to(device))[1]
            except:
                logging.exception('Got exception on main handler')
                raise
            # [n, d]
            # logger.info("code_emb.device: {} | graph_emb.device: {} | comment_emb.device: {} ".format(code_emb.device, graph_emb.device, comment_emb.device))
            # cuda:0
            code_emb = code_emb / code_emb.norm(dim=-1, keepdim=True)
            graph_emb = graph_emb / graph_emb.norm(dim=-1, keepdim=True)
            comment_emb = comment_emb / comment_emb.norm(dim=-1, keepdim=True) # [n, d]

            emb = torch.stack((code_emb, graph_emb, comment_emb), dim=1) # [n, 3, d]

            # emb = torch.stack((code_emb[:], graph_emb, comment_emb), dim=1) # [n, 3, d]
            # emb = torch.stack((code_emb, graph_emb, comment_emb), dim=1) # [n, 3, d]
            # emb = torch.stack((code_emb, graph_emb, comment_emb), dim=1) # [n, 3, d]
            
            # logger.info("emb.device: ", emb.device) # cuda:0
            # logger.info("check2")

            # emb = emb / emb.norm(dim=-1, keepdim=True)
            u, s, v = torch.linalg.svd(emb, full_matrices=True)
            pred = (s[:, 0] ** 2 - 1) / (3 - 1)

            t = torch.ones(s[:, 0].shape).to(device)
            # t_cos = torch.arange(code_emb.size(0), device = code_emb.device)
            
            loss = loss_f(pred, t)
            # loss_cos = loss_f(comment_emb @ code_emb.T, t_cos)

            if args.n_gpu > 1:
                loss = loss.mean()
            
            loss.backward()

            optimizer.step()
            scheduler.step()

            b_loss += loss.cpu().data.item()
            # b_loss_cos += loss_cos.cpu().data.item()
        
        train_loss.append(b_loss/len(train_dataloader))
        
        if (epoch+1) % args.log_interval == 0:
            # bar.set_description("epoch {} loss {}".format(idx,avg_loss))
            eval_dataloader = build_loaders(args.eval_data_file, args.eval_batch_size)
            mrr = evaluate(args, tcp_code, tcp_comment, eval_dataloader, args.device, query='nl')
            cos_xy, cos_yz, cos_xz, max_s, triple_sim = get_cos_sim(tcp_code, tcp_graph, tcp_comment, eval_dataloader, device)
            logger.info("epoch {} | loss: {:.6f} | max singular: {:.6f} | triple sim: {:.6f} | cos_xy: {:.6f} | cos_yz: {:.6f} | cos_xz: {:.6f}".format(epoch+1, b_loss, max_s, triple_sim, cos_xy, cos_yz, cos_xz)) # right!

            if mrr > best_mrr:
                best_mrr = mrr
                if not os.path.isdir(args.cpkt_model_path):
                    os.mkdir(args.cpkt_model_path)
                save_dir = args.cpkt_model_path+str(epoch)+'_ckpt_best_{}.pth'.format(str(args.lang))
                tcp_code_to_save = tcp_code.module if hasattr(tcp_code,'module') else tcp_code
                tcp_graph_to_save = tcp_graph.module if hasattr(tcp_graph,'module') else tcp_graph
                tcp_comment_to_save = tcp_comment.module if hasattr(tcp_comment,'module') else tcp_comment

                cpkt = {
                    'epoch': epoch,
                    'tcp_code': tcp_code_to_save.state_dict(),
                    'tcp_graph': tcp_graph_to_save.state_dict(),
                    'tcp_comment': tcp_comment_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }

                torch.save(cpkt, save_dir)
                logger.info("Update the checkpoints.")
    
    return train_loss
    # # multi-gpu 
    # if args.n_gpu > 1 and device != 'gpu':
    #     model = torch.nn.DataParallel(model)


# import argparse
# import sys
# import torch
# The following code is a Python program that takes a list of integers and produces either the sum or the max:
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process Parameters (For Ablation Study).')
    # logger.info("check1")
    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")
    
    parser.add_argument("--model_path", default="./pretrained/TCP/", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--max_code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--max_comment_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--max_graph_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  

    parser.add_argument("--log_interval", default=1, type=int,
                        help="Log interval steps for epoch and loss.")
    
    parser.add_argument("--eval_interval", default=1, type=int,
                        help="Evaluation interval steps for MRR.")

    parser.add_argument("--cpkt_model_path", default="./results/cpkts/", type=str,
                        help="Checkpoint path for saving model and optimizer.")
                        
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--eps", default=1e-8, type=float,
                        help="The eps for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help="Weight decay")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")
    # parser.add_argument('--sum', dest='accmulate', action='store_const', const=sum, default=None)
    args = parser.parse_args()
    # logger.info("check2")

    #set log
    logging.basicConfig(filename='results/logs/main.log', format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.DEBUG)
    #set seed
    set_seed(seed=args.seed)
    #set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    # gpus = [0, 1]
    # torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    logger.info("device: %s, n_gpu: %s", args.device, args.n_gpu)

    # load model and tokenizer
    tcp_code, tokenizer = load_model(path=os.path.join(args.model_path, 'CodeEncoder/'))
    tcp_graph, _ = load_model(path=os.path.join(args.model_path, 'GraphEncoder/'))
    tcp_comment, _ = load_model(path=os.path.join(args.model_path, 'NLEncoder/'))
    logger.info("Have loaded encoders.")

    tcp_code.to(args.device)
    tcp_graph.to(args.device)
    tcp_comment.to(args.device)
    # multi-gpu 
    if args.n_gpu > 1 and args.device != 'cpu':
        # tcp_code = torch.nn.DataParallel(tcp_code, device_ids=gpus, output_device=gpus[0])
        tcp_code = torch.nn.DataParallel(tcp_code)
        tcp_graph = torch.nn.DataParallel(tcp_graph)
        tcp_comment = torch.nn.DataParallel(tcp_comment)

    eval_dataloader = build_loaders(args.eval_data_file, args.eval_batch_size)
    mrr = evaluate(args, tcp_code, tcp_comment, eval_dataloader, args.device, query='nl')
    logger.info("The initial MRR on valid dataset is: {:.6f}".format(mrr)) # right.

    if args.do_train:
        logger.info("do train")
        train_loss = train(args, tcp_code, tcp_graph, tcp_comment, args.device)
        logger.info("Have pre-trained encoders.")

    if args.do_test:
        logger.info("do test")
        eval_dataloader = build_loaders(args.test_data_file, args.eval_batch_size)
        mrr = evaluate(args, tcp_code, tcp_comment, eval_dataloader, args.device, query='nl')
        logger.info("MRR on test dataset WithOut fine-tuning is: {:.6f}".format(mrr)) 
