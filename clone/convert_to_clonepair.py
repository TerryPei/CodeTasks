import argparse
import logging
import os
import pickle
import torch
import json

from tqdm import tqdm

import sys
sys.path.append("..")
from codeparser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from codeparser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)

from transformers import RobertaTokenizer, RobertaModel
import multiprocessing
from tree_sitter import Language, Parser

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
    LANGUAGE = Language('../codeparser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser



class Args():
    def __init__(self, lang, max_context_length=256, max_graph_length=64, max_comment_length=128):
        self.lang = lang
        
        self.max_context_length = max_context_length
        self.max_graph_length = max_graph_length
        self.max_comment_length = max_comment_length
        # self.train_batch_size = 32

        # self.train_data_file = './dataset/python/train.jsonl'
        self.output_dir = './dataset/'
        self.pretrain_path = '../pretrained/RobertaBERT/'


#remove comments, tokenize code and extract dataflow                                        
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
    return code_tokens,dfg


def save_model(model, tokenizer, path):
    model_path = path + 'bert'
    tokenizer_path = path + 'tokenizer'

    assert os.path.exists(model_path) == 1
    model.save_pretrained(model_path)

    assert os.path.exists(tokenizer_path) == 1
    tokenizer.save_pretrained(tokenizer_path)

def load_model(path):
    model_path = path + 'bert'
    tokenizer_path = path + 'tokenizer'

    assert os.path.exists(model_path) == 1
    # model =  AutoModel.from_pretrained(model_path)
    model =  RobertaModel.from_pretrained(model_path)

    assert os.path.exists(tokenizer_path) == 1
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, config=AutoConfig.from_pretrained(model_path))
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

class ClonePair(object):
    """
    A Pre-training Pairs for a Code and Description.
    """
    def __init__(self, url1, code_token_ids1, code_mask_ids1, code_pos_ids1, 
                       url2, code_token_ids2, code_mask_ids2, code_pos_ids2, label
                ):
        
        self.url1 = url1
        self.url2 = url2

        self.code_token_ids1  = code_token_ids1
        self.code_mask_ids1 =  code_mask_ids1
        self.code_pos_ids1  = code_pos_ids1

        self.code_token_ids2 = code_token_ids2
        self.code_mask_ids2 = code_mask_ids2
        self.code_pos_ids2 = code_pos_ids2

        self.label = label


def convert_codes_to_pairs(item):
    #source
    url1, url2, label, cache, args, url_to_code, tokenizer = item
    parser=parsers['java']

    for url in [url1,url2]:
        
        if url not in cache:

            func = url_to_code[url]

            #extract code context and graph information
            code_tokens, dfg=extract_dataflow(func, parser, args.lang)
            code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
            code_tokens=[y for x in code_tokens for y in x]

            #truncating
            code_tokens = code_tokens[:args.max_context_length+args.max_graph_length-2-min(len(dfg), args.max_graph_length)] # 实际上后面减掉了graph的部分
            code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]# context_token

            # Transfer token to token_ids
            code_token_ids = tokenizer.convert_tokens_to_ids(code_tokens)

            #pos_ids
            code_pos_ids = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]

            # pad with dfg
            dfg_pad = dfg[:args.max_context_length + args.max_graph_length - len(code_tokens)]
            code_tokens += [x[0] for x in dfg_pad]
            code_pos_ids +=[0 for x in dfg_pad]
            code_token_ids += [tokenizer.unk_token_id for x in dfg_pad]

            # mask_ids
            code_mask_ids = [1] * (len(code_tokens))

            # code context and graph information padding
            code_padding_length = args.max_context_length + args.max_graph_length - len(code_tokens)

            code_token_ids += [tokenizer.pad_token_id] * code_padding_length
            code_pos_ids += [tokenizer.pad_token_id] * code_padding_length
            code_mask_ids += [0] * code_padding_length 

            # print(code_token_ids, code_pos_ids, code_mask_ids) # right, have been checked!

            cache[url] = code_token_ids, code_mask_ids, code_pos_ids
        
    code_token_ids1, code_mask_ids1, code_pos_ids1 = cache[url1]   
    code_token_ids2, code_mask_ids2, code_pos_ids2 = cache[url2]  
    

    return ClonePair(url1, code_token_ids1, code_mask_ids1, code_pos_ids1, 
                    url2, code_token_ids2, code_mask_ids2, code_pos_ids2, label)

    

def double_func(a):
    return a * 2 # Just for test on mutipoolprocessing


if __name__ == '__main__':
    
    args = Args(lang='java', max_context_length=256, max_graph_length=64, max_comment_length=128)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    _, tokenizer = load_model(path=args.pretrain_path)

    cache_file=args.output_dir+'test.pkl'
    file_path = './dataset/test.txt'
    data_path = './dataset/data.jsonl'
    # examples = []
    # data=[]
    pairs = []
    data=[]
    cpu_count = 2
    
    # with open(file_path) as f:
    #     for i, line in enumerate(f):
    #         line=line.strip()
    #         js=json.loads(line)
    #         data.append((js, tokenizer, args))
    # pairs=pool.map(convert_codes_to_pairs, tqdm(data,total=len(data)))


    examples = []

    url_to_code={}
    with open(data_path) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            url_to_code[js['idx']]=js['func']

    #load code function according to index
    #
    data=[]
    cache={}
    f=open(file_path)
    with open(file_path) as f:
        for line in f:
            line=line.strip()
            url1,url2,label=line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                # print('false')
                continue
            if label=='0':
                label=0
            else:
                label=1
            # print(url1,url2,label)

            data.append((url1, url2, label, cache, args, url_to_code, tokenizer))

    pool = multiprocessing.Pool(processes=cpu_count)
    clonepairs=pool.map(convert_codes_to_pairs, tqdm(data,total=len(data)))
    pickle.dump(clonepairs, open(cache_file,'wb'))
    pool.close()
    pool.join()