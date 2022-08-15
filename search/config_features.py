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
    def __init__(self, lang, context_length=256, graph_length=64, comment_length=128, file_type='train'):
        self.lang = lang
        # self.temperature = 1.0
        # # self.graph_embedding = nn.Embedding(n, num_features)
        # # self.graph_embedding = nn.Embedding(6, 10).weight
        # # self.context_embedding = nn.Embedding(6, 11).weight
        # self.projection_dim = 128
        # self.pretrained = False
        # self.trainable = False
        # self.dropout = 0.1

        # self.vocab_size = 50000
        # self.max_context_length = 128
        # self.max_graph_length = 128
        # # self.max_ques_length = 32
        # # self.max_ans_length  = 32
        # self.context_embedding_dim = 256
        # self.graph_embedding_dim = 256

        # self.context_encode_features = 64
        # self.graph_encode_features = 64

        # self.batch_size = 32
        # self.test_batch_size = 1
        # self.epochs = 20
        # self.lr = 1e-3
        # self.weight_decay = 1e-3
        # self.momentum = 0.5
        # self.no_cuda = False
        # self.seed = 1
        # self.log_interval = 10
        # self.valid_freq = 1
        # self.save_model = False
        # self.num_workers = 1
        
        self.context_length = context_length
        self.graph_length = graph_length
        self.comment_length = comment_length
        # self.train_batch_size = 32

        # self.train_data_file = './dataset/python/train.jsonl'
        self.config_features_data_file = '../dataset/'+self.lang+'/'+file_type+'.jsonl'
        self.output_dir = '../dataset/'+self.lang
        self.pretrain_path = '../pretrained/RobertaBERT/'
        # self.graph_encoder_path = '../results/bert'
        # self.context_encoder_path = '../pretrained/RobertaBERT/'
        # self.graph_encoder_path = '../pretrained/RobertaBERT/'

        # self.save_model_path = '../results/best_model/checkpoint1.pt'


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


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,                 
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx=position_idx
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg        
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url



def convert_examples_to_features(item):

    js,tokenizer,args=item
    #code
    parser = parsers[args.lang]
    #extract data flow
    code_tokens,dfg=extract_dataflow(js['original_string'],parser,args.lang)
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
    code_tokens=[y for x in code_tokens for y in x]
    #truncating
    code_tokens=code_tokens[:args.context_length+args.graph_length-2-min(len(dfg),args.graph_length)]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg=dfg[:args.context_length+args.graph_length-len(code_tokens)]
    code_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    code_ids+=[tokenizer.unk_token_id for x in dfg]
    code_padding_length=args.context_length+args.graph_length-len(code_ids)
    position_idx+=[tokenizer.pad_token_id]*code_padding_length
    code_ids+=[tokenizer.pad_token_id]*code_padding_length  

    code_mask_ids = [1] * (len(code_tokens))
    code_mask_ids += [0] * code_padding_length

    #reindex
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
    #nl
    comment=' '.join(js['docstring_tokens'])
    comment_tokens=tokenizer.tokenize(comment)[:args.comment_length-2]
    comment_tokens =[tokenizer.cls_token]+comment_tokens+[tokenizer.sep_token]
    comment_ids =  tokenizer.convert_tokens_to_ids(comment_tokens)
    padding_length = args.comment_length - len(comment_ids)
    comment_ids+=[tokenizer.pad_token_id]*padding_length


    # attention normal?



    # attention by CodeBERT?


    
    return InputFeatures(code_tokens,code_ids,position_idx,dfg_to_code,dfg_to_dfg,comment_tokens,comment_ids,js['url'])




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
    

def double_func(a):
    return a * 2


if __name__ == '__main__':
    
    args = Args(lang='ruby', context_length=256, graph_length=64, comment_length=128, file_type='test')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    _, tokenizer = load_model(path=args.pretrain_path)

    file_path = args.config_features_data_file
    prefix=file_path.split('/')[-1][:-6]
    cache_file=args.output_dir+'/'+prefix+'.pkl'
    output_test_examples = args.output_dir+'/'+prefix+'_example.pkl'

    # examples = []
    # data=[]
    examples = []
    data=[]

    cpu_count = 2
    pool = multiprocessing.Pool(processes=cpu_count)
    
    with open(file_path) as f:
        for i, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            data.append((js, tokenizer, args))
    
    examples=pool.map(convert_examples_to_features, tqdm(data,total=len(data)))
    # self.examples=pool.map(convert_examples_to_features, tqdm(data,total=10))
    # pool map error: cannot pickle 'tree_sitter.Parser' object
    pickle.dump(examples, open(cache_file,'wb'))
    pool.close()
    pool.join()
    # pool = multiprocessing.Pool(processes=2)
    # data = [1, 2, 3]
    # examples = []
    # examples = pool.map(double_func, tqdm(data,total=len(data)))
    # print(examples)
    # pickle.dump(examples, open('./test2.pkl', 'wb'))
    # pool.close()
    # pool.join()