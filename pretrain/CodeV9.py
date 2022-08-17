import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import logging
import pickle
import random
# from pyrsistent import T
import torch
import json
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  AutoTokenizer, AutoModel, AutoConfig)
from tqdm import tqdm
import itertools
import multiprocessing
cpu_count = 2
import re
from codeparser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from codeparser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import warnings
from utils import our_loss

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


def extract_ast(code, parser, lang):
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
            tree_df,_=parser[1](root_node,index_to_code,{}) 
        except:
            tree_df=[]
    
        tree_df=sorted(tree_df,key=lambda x:x[1])


        indexs=set()
        for d in tree_df:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_tree_df=[]
        for d in tree_df:
            if d[1] in indexs:
                new_tree_df.append(d)
        tree_df=new_tree_df
    except:
        tree_df=[]

    # code_tokens = [re.sub(r"[-()\"#/@;:<>{}`\[\]_.!,]", "", file) for file in code_tokens]
    code_tokens = [token for token in code_tokens if token!=""]

    ast_tokens = [x[0] for x in tree_df]
    
    return code_tokens, ast_tokens, tree_df

    # return code_tokens,dfg

def set_seed(seed=123456):
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

    assert os.path.exists(tokenizer_path) == 1
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, config=AutoConfig.from_pretrained(model_path))
    return model, tokenizer

class Pair(object):
    """
    A Pre-training Pairs for a Code and Description.
    """
    def __init__(self, code_token_ids, code_mask_ids, code_pos_ids, \
        ast_token_ids, ast_mask_ids, ast_pos_ids, \
            comment_token_ids, comment_mask_ids, url=None):
        
        self.code_token_ids = code_token_ids
        self.code_mask_ids = code_mask_ids
        self.code_pos_ids = code_pos_ids

        self.ast_token_ids = ast_token_ids
        self.ast_mask_ids = ast_mask_ids
        self.ast_pos_ids = ast_pos_ids

        self.comment_token_ids = comment_token_ids
        self.comment_mask_ids = comment_mask_ids

        self.url=url


def preprocess(file_path):
    # data_type = file_path.split('/')[-1][:-6] #train/valid/test
    lang = file_path.split('/')[-2]
    lang = 'ruby'
    assert lang in {'ruby', 'java', 'python', 'php', 'javascript', 'go'}

    pairs = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)

            #code
            parser = parsers[lang]

            #extract code context and ast information
            code_tokens, ast_tokens, dfg = extract_ast(js['original_string'], parser, lang)
            # print(len(code_tokens)) # 31
            # code_tokens = code_tokens[:args.max_code_length-1]
            # print(len(code_tokens)) # 33
            #truncating
            code_tokens = code_tokens[:args.max_code_length-2]
            code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
            # Transfer token to token_ids
            code_token_ids = tokenizer.convert_tokens_to_ids(code_tokens)
            # mask_ids
            code_mask_ids = [1] * (len(code_tokens))
            # print(code_mask_ids)
            # control_flow_pos = [0] * len(code_tokens)
            # pos_count = 1
            # for i, token in enumerate(code_tokens):
            #     if i == 0: 
            #         continue
            #     if token.lower() in control_keywords:
            #         pos_count += 1
            #     control_flow_pos[i] = 
            control_flow_pos = [0] * len(code_tokens)
            code_pos_ids = control_flow_pos
                
            #padding
            code_padding_length = args.max_code_length - len(code_tokens)
            code_token_ids += [tokenizer.pad_token_id] * code_padding_length
            code_mask_ids += [0] * code_padding_length 
            code_pos_ids += [0] * code_padding_length

            # ast
            ast_tokens = [tokenizer.cls_token] + [x[0] for x in dfg][:args.max_ast_length - 2] + [tokenizer.sep_token]

            ast_token_ids = tokenizer.convert_tokens_to_ids(ast_tokens)
            ast_mask_ids = [0] + [1] * (len(ast_tokens) - 2) + [0]
            ast_pos_ids = [0] + [x[1] for x in dfg][:args.max_ast_length - 2] + [0]

            # ast_pos_ids = list(np.sin(data_flow_index))
            
            #padding
            ast_padding_length = args.max_ast_length - len(ast_token_ids)
            ast_token_ids += [tokenizer.pad_token_id] * ast_padding_length
            ast_mask_ids += [0] * ast_padding_length
            ast_pos_ids += [0] * ast_padding_length


            #extract code description information labeled bu human beings.
            #comment
            comment = js['docstring_tokens']
            comment_tokens = comment[:args.max_comment_length - 1]
            comment_tokens = [tokenizer.cls_token] + comment_tokens
            comment_token_ids =  tokenizer.convert_tokens_to_ids(comment_tokens)
            comment_mask_ids = [1] * (len(comment_tokens))
            comment_padding_length = args.max_comment_length - len(comment_token_ids)

            comment_token_ids += [tokenizer.pad_token_id] * comment_padding_length
            comment_mask_ids += [0] * comment_padding_length
            # print(comment_token_ids)
            pairs.append(Pair(code_token_ids, code_mask_ids, code_pos_ids, \
                ast_token_ids, ast_mask_ids, ast_pos_ids, comment_token_ids, comment_mask_ids))
            
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
            'ast_token_ids': torch.tensor(self.pairs[idx].ast_token_ids),
            'ast_mask_ids': torch.tensor(self.pairs[idx].ast_mask_ids),
            'ast_pos_ids': torch.tensor(self.pairs[idx].ast_pos_ids),
            'comment_token_ids': torch.tensor(self.pairs[idx].comment_token_ids),
            'comment_mask_ids': torch.tensor(self.pairs[idx].comment_mask_ids),
        }
        return item

def build_loaders(data_path, batch_size=8, num_workers=4):
    # transforms = get_transforms(mode=mode)
    dataset = Dataset(
        data_path=data_path
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers,
                                               pin_memory=True, sampler=sampler)
    # if args:
    #     sampler = RandomSampler(dataset)
    #     dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=num_workers)
    # else:
    #     dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers)
    # sampler = RandomSampler(dataset)
    # dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
    return dataloader


def split_pos_neg(code, ast, comment):
    
    assert code.size(0) == ast.size(0) == comment.size(0)
    assert code.size(1) == ast.size(1) == comment.size(1)

    batch_size = code.size(0)

    # samples = torch.zeros((batch_size, batch_size, 3, dim))

    r_pos, r_neg = [], []

    #------------------------------------------------------#
    # Here negative sample can be replaced with any one of the negative samples in 19-22. 
    # There are multiple negs in this step in the paper, here we sample n negs for the running speed.
    #------------------------------------------------------#

    pos_idx = list(range(batch_size))

    for i in pos_idx:
        pos, neg = [], []
        # pos
        j = k = i
        # print(pos)
        pos.append(torch.stack((code[i], ast[j], comment[k]), dim=0))

        # neg: sample strategy
        count = batch_size

        while (count):

            neg_idx = list(range(batch_size))
            neg_idx.remove(i)
            j = random.choice(neg_idx)
            neg_idx.remove(j)
            k = random.choice(neg_idx)

            neg.append(torch.stack((code[i], ast[j], comment[k]), dim=0))

            count = count - 1

        pos = torch.stack(pos)
        neg = torch.stack(neg)

        r_pos.append(pos)
        r_neg.append(neg)

    r_pos = torch.stack(r_pos)
    r_neg = torch.stack(r_neg)

    return r_pos, r_neg 


def train(args, our_model, local_rank, nprocs, resume=False):
    
    #-----------------------------------------------------#
    # pre-train model with ****
    #-----------------------------------------------------#
    # torch.cuda.set_device(local_rank)
    # our_model.cuda(local_rank)
    # our_model = torch.nn.parallel.DistributedDataParallel(our_model,
    #                                                   device_ids=[local_rank])

    train_dataloader = build_loaders(args.train_data_file, batch_size=args.train_batch_size, num_workers=4)

    params = our_model.parameters()
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, eps=args.eps, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.num_train_epochs*0.1, num_training_steps=len(train_dataloader)*args.num_train_epochs)

    train_loss = []
    best_mrr = 0

    for epoch in range(args.num_train_epochs):
        b_loss = 0
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        # logger.info("check1")
        # logger.info("device:", device)
        for batch in bar:
            
            batch_size =  len(batch['code_token_ids'])
            # logger.info(batch['code_token_ids'].shape)
            optimizer.zero_grad()
            # batch['code_token_ids'] = batch['code_token_ids'].to(device)
            try:
                code_emb = our_model(input_ids=batch['code_token_ids'].cuda(non_blocking=True), \
                        attention_mask=batch['code_mask_ids'].cuda(non_blocking=True), \
                        position_ids=batch['code_pos_ids'].cuda(non_blocking=True))[0][:, 0, :] # [cls]

                ast_emb = our_model(input_ids=batch['ast_token_ids'].cuda(non_blocking=True), \
                                    attention_mask=batch['ast_mask_ids'].cuda(non_blocking=True), \
                                    position_ids=batch['ast_pos_ids'].cuda(non_blocking=True))[0][:, 0, :]

                comment_emb = our_model(input_ids=batch['comment_token_ids'].cuda(non_blocking=True), \
                                        attention_mask=batch['comment_mask_ids'].cuda(non_blocking=True))[0][:, 0, :]
            except:
                logging.exception('Got exception on main handler')
                raise
            # cuda:0
            code_emb = code_emb / code_emb.norm(dim=-1, keepdim=True)
            ast_emb = ast_emb / ast_emb.norm(dim=-1, keepdim=True)
            comment_emb = comment_emb / comment_emb.norm(dim=-1, keepdim=True)

            pos_samples, neg_samples = split_pos_neg(code_emb, ast_emb, comment_emb) 

            loss = our_loss(pos_samples, neg_samples, batch_size, k=3)


            if args.n_gpu > 1:
                loss = loss.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(our_model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            b_loss += loss.cpu().data.item()
        
        train_loss.append(b_loss/len(train_dataloader))
        # logger.info("check3")
        if (epoch+1) % args.log_interval == 0:
        # bar.set_description("epoch {} loss {}".format(idx,avg_loss))
            eval_dataloader = build_loaders(args.eval_data_file, args.eval_batch_size)
            mrr = evaluate(args, our_model, eval_dataloader, args.device, query='nl')
            logger.info("epoch {} | loss: {:.6f} | mrr: {:.6f}".format(epoch+1, b_loss, mrr)) # right!

        #     if mrr > best_mrr:
        #         best_mrr = mrr
        #         if not os.path.isdir(args.cpkt_model_path):
        #             os.mkdir(args.cpkt_model_path)
        #         save_dir = args.cpkt_model_path+str(epoch)+'_ckpt_best_{}.pth'.format(str(args.lang))
        #         our_model_to_save = our_model.module if hasattr(our_model,'module') else our_model

        #         cpkt = {
        #             'epoch': epoch,
        #             'our_model': our_model_to_save.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'scheduler': scheduler.state_dict()
        #         }

        #         torch.save(cpkt, save_dir)
        #         logger.info("Update the checkpoints.")
    cleanup()
    return train_loss

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    dist.init_process_group(backend='nccl',
                        init_method='tcp://127.0.0.1:23456',
                        world_size=args.nprocs,
                        rank=local_rank)
    
    # create model
    # load model and tokenizer
    our_model, tokenizer = load_model(path=os.path.join(args.model_path, 'CodeEncoder/'))
    torch.cuda.set_device(local_rank)
    our_model.cuda(local_rank)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have

    args.batch_size = int(args.batch_size / args.nprocs)
    our_model = torch.nn.parallel.DistributedDataParallel(our_model,
                                                      device_ids=[local_rank])
    

    loss_func = our_loss().cuda(local_rank)

    optimizer = torch.optim.AdamW(our_model.parameters(), lr=args.learning_rate, eps=args.eps, weight_decay=args.weight_decay)
    
    cudnn.benchmark = True

    train_dataloader = build_loaders(args.train_data_file, batch_size=args.train_batch_size)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.num_train_epochs*0.1, num_training_steps=len(train_dataloader)*args.num_train_epochs)
   
    if args.do_eval:
        logger.info("do eval")
        eval_dataloader, eval_sampler = build_loaders(args.eval_data_file, args.eval_batch_size)
        mrr = evaluate(args, model, eval_dataloader, args.device, query='nl')
        logger.info("Initial Mrr is: {:.6f}".format(mrr)) 
        return

    if args.do_train:
        logger.info("do train")
        train_dataloader, train_sampler = build_loaders(args.train_data_file, args.train_batch_size)
        for epoch in range(args.start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            eval_sampler.set_epoch(epoch)
            
            loss = train(train_dataloader, model, loss_func, optimizer, epoch, local_rank,
              args, scheduler)
            
            if (epoch+1) % args.log_interval == 0:
                # bar.set_description("epoch {} loss {}".format(idx,avg_loss))
                eval_dataloader, _ = build_loaders(args.eval_data_file, args.eval_batch_size)
                mrr = evaluate(args, model, eval_dataloader, args.device, query='nl')
                logger.info("epoch {} | valid loss: {:.6f} | valid mrr: {:.6f}".format(epoch+1, loss/len(train_dataloader), mrr)) # right!

    if args.do_test:
        logger.info("do test")
        test_dataloader, _ = build_loaders(args.test_data_file, args.eval_batch_size)
        mrr = evaluate(args, model, test_dataloader, args.device, query='nl')
        logger.info("MRR on test dataset WithOut fine-tuning is: {:.6f}".format(mrr)) 


def main(args):
    # args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process Parameters (For Ablation Study).')
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
    
    parser.add_argument("--model_path", default="./pretrain/CodeV9/", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    # If use single encoder and 

    parser.add_argument("--max_code_length", default=128, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--max_comment_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--max_ast_length", default=128, type=int,
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
                        
    parser.add_argument("--train_batch_size", default=1024, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=1024, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--eps", default=1e-8, type=float,
                        help="The eps for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=200, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help="Weight decay")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")


    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

    parser.add_argument('--world_size', default=-1, type=int,
                help='node rank for distributed training')
    

    args = parser.parse_args()

        #set log
    logging.basicConfig(filename='results/logs/pre_train_v9.log', format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.DEBUG)
 
    main(args)

