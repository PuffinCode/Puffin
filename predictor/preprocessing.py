from gensim.models import word2vec
from collections import defaultdict
from random import choice
import w2v
import pandas as pd
import numpy as np
import json
#import resnet_model
#import ggnn_model
#import mlp_model
import cfg
import loaddata
import argparse
import torch
import torch.nn as nn
import random
from torch.autograd import Variable
import os
import math


#data_path="../o3_data/o3_result/"
#data_path2="../o3_data/o3_memory/"
data_path=""
data_path2=""
data_path3=""
model_name = ""


def drop_part_value(func_dic):
    print("drop part value ...")
    random_seed=42
    np.random.seed(random_seed)

    for key in func_dic.keys():
        func_addr_list = list(func_dic[key])
        if len(func_addr_list)>5000:
            print("drop values in key:",key,len(func_addr_list),500+int(math.log(len(func_addr_list),10))*100)
            slice_list = random.sample(func_addr_list,500+int(math.log(len(func_addr_list),10))*100) 
            func_dic[key]=set(slice_list)
            print("after drop:",len(func_dic[key]))


def create_dic(func_dic,f,func_bb):
    for line in f:
        items = line.split()
        #print("line:",line)
        #print("key:",items[0],int(items[0],16))
        key_it = int(items[0],16)
        #print("key_it:",key_it)
        #debug
        #value_it = str(items[1])+"-8" #+"-"+str(items[2])
        value_it = str(items[1])+"-"+str(items[2])
        #print("value_it:",value_it)
        #value_it = str(int(value_it) // 4096)
        #if (len(func_dic[key_it]))<(10 * len(func_bb[key_it])):
        func_dic[key_it].add(value_it)

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError



def load_cfg_data(file_name,config,redundant_t):
    func_name=[]
    func_bb=defaultdict(list)
    labels=[]
    node_size=[]
    matrix=[]
    sentencess=[]
    data_amount=0
    
    if redundant_t=="dw":
        redundant_t=""
    else:
        redundant_t="_"+redundant_t
    bb_f = open(data_path+"bb_"+file_name)
    adj_f = open(data_path+"adj_"+file_name)
    arg_f = open(data_path+"label_"+file_name+redundant_t)
    file_data_amount=0

    arg_line = arg_f.readline()
    all_bb_amount=0
    while arg_line:
        cfg_line_cnt=0
        addr,bb_cnt,cfg_label = map(int,arg_line.split())

        cfg_inst_amount=0
        arg_line = arg_f.readline()
        ins_cnt = list(map(int,arg_line.split()))
        sentences=[]
        func_bb_list=[]
        datamat=np.empty([bb_cnt,bb_cnt]) 
        for i in range(bb_cnt):
            adj_line = adj_f.readline()
            line=adj_line.strip().split()
            datamat[i,:]=line[:]
            cfg_line_cnt += ins_cnt[i]

            words=[]
            for j in range(ins_cnt[i]):
                bb_line = bb_f.readline()
                tmp_s=bb_line.split()
                func_bb_list.append(int(tmp_s[0].rstrip(':'),16))
                del tmp_s[0]
                words += tmp_s  #w2v
                #tmp_s_str = "_".join(str(token) for token in tmp_s)
                #words.append(tmp_s_str) #bert
                cfg_inst_amount = cfg_inst_amount+1
            if len(words)==0:
                print("!!!! null words")
            sentences.append(words)
            all_bb_amount += 1
        func_bb[addr]=func_bb_list
        
        if cfg_inst_amount>1 and bb_cnt<config.n_node_type:
            func_name.append(addr)
            node_size.append(bb_cnt)
            sentencess.append(sentences)
            matrix.append(datamat)
            labels.append(cfg_label)
            data_amount=data_amount+1
            file_data_amount = file_data_amount+1
        arg_line = arg_f.readline()
        arg_line = arg_f.readline()
    print("file "+file_name+":",file_data_amount)
    return func_name,labels,node_size,sentencess,matrix,func_bb

#def load_cg_data(file_name,func_name,func_bb,config):
def load_cg_data(file_name,func_name,func_bb,config):
    call_matrix=[]
    data_list=[]
    func_cg=defaultdict(int)

    func_memory_dic = defaultdict(set)
    print("load dic from:",data_path2+file_name+"_dic.txt")
    if not os.path.isfile(data_path2+file_name+"_dic.txt"):
        print("create dic:",data_path2+"memory_"+file_name)
        data_f = open(data_path2+"memory_"+file_name)
        create_dic(func_memory_dic,data_f,func_bb)
        drop_part_value(func_memory_dic)
        fw = open(data_path2+file_name+"_dic.txt",'w+')
        #fw.write(str(func_memory_dic)) #把字典转化为str
        js = json.dumps(func_memory_dic, default=set_default)
        fw.write(js)
        fw.close()
        print("no dic : func_meory_dic length:",len(func_memory_dic))

        #note: func_memory_dic is different (defaultdict cannot use, reload a new dict)
        print("load dic:",file_name)
        fr = open(data_path2+file_name+"_dic.txt",'r+')
        js = fr.read()
        func_memory_dic = json.loads(js)
        fr.close()
        print("has dic : func_meory_dic length:",len(func_memory_dic))


    else:
        print("load dic:",file_name)
        fr = open(data_path2+file_name+"_dic.txt",'r+')
        js = fr.read()
        func_memory_dic = json.loads(js)
        fr.close()
        print("has dic : func_meory_dic length:",len(func_memory_dic))
    #print("start drop data")
    #drop_part_value(func_memory_dic)
    
    #for key in func_memory_dic.keys():
    #    print("key:",key,len(func_memory_dic[key]),type(key))
    print("get func_memory_dic")
       
    call_adj_f = open(data_path3+"adj_"+file_name)
    call_arg_f = open(data_path3+"node_"+file_name)
    arg_line = call_arg_f.readline()
    f_i=0
    no_memory_node_cnt=0
    get_memory_node_cnt=0
    print("use dic : func_meory_dic length:",len(func_memory_dic))
    while arg_line:
        addr,node_cnt = map(int,arg_line.split())
        arg_line = call_arg_f.readline()
        nodes = list(map(int,arg_line.split()))
        call_datamat=np.empty([node_cnt,node_cnt]) 
        for i in range(node_cnt):
            adj_line = call_adj_f.readline()
            line=adj_line.strip().split()
            call_datamat[i,:]=line[:]

        if addr in func_name:
            func_cg[addr]=f_i
            f_i += 1
            call_matrix.append(call_datamat)
            n_list=[]
            for n in nodes:
                n_addr_list=[]
                #if str(n) in func_memory_dic:
                #    n_addr_list = n_addr_list+(func_memory_dic[str(n)])
                #print("check memory1:",n,func_bb[0])
                if n in func_bb:
                    for n_inst in func_bb[n]:
                        #print("check memory2:",n_inst,func_bb[n][0])
                        if str(n_inst) in func_memory_dic:
                            #print("check memory3:",str(n_inst),func_memory_dic[str(n_inst)])
                            n_addr_list = n_addr_list+(func_memory_dic[str(n_inst)])
                elif str(n) in func_memory_dic:
                    n_addr_list = n_addr_list+(func_memory_dic[str(n)])

                if len(n_addr_list)==0:
                    n_addr_list=["0"]
                    no_memory_node_cnt += 1
                else:
                    get_memory_node_cnt += 1
                n_addr_list=list(set(n_addr_list))
                #print("n_addr_list:",n_addr_list)
                n_list.append(n_addr_list)
            data_list.append(n_list)    
    
        arg_line = call_arg_f.readline()
    print("memory state sta:",no_memory_node_cnt,get_memory_node_cnt)
    return call_matrix,data_list,func_cg 


def delete_func(t_func_name,t_labels,t_node_size,t_sentencess,t_matrix,t_call_matrix,t_data_list,func_cg):
    new_call_matrix=[]
    new_data_list=[]
    i=0
    while True:
        if i>=len(t_labels):
            break
        if t_func_name[i] not in func_cg:
            print("notin cg:",t_func_name[i])
            del t_labels[i]
            del t_node_size[i]
            del t_sentencess[i]
            del t_matrix[i]
            del t_func_name[i]
        else:
            new_call_matrix.append(t_call_matrix[func_cg[t_func_name[i]]])
            new_data_list.append(t_data_list[func_cg[t_func_name[i]]])
            i += 1
    return t_func_name,t_labels,t_node_size,t_sentencess,t_matrix,new_call_matrix,new_data_list 


def drop_ill_data(labels,node_size,sentencess,matrix,call_matrix,func_name,data_list,start_i,end_i,config,target_start):
    i=start_i
    while True:
        if i>=end_i:
            break
        if len(data_list[i])>config.call_m_size:
            del labels[i]
            del node_size[i]
            del sentencess[i]
            del matrix[i]
            del call_matrix[i]
            del func_name[i]
            del data_list[i]
            end_i -= 1
            if i<target_start:
                target_start -= 1
        else:
            i += 1

    return labels,node_size,sentencess,matrix,call_matrix,func_name,data_list,target_start



def drop_nodead_data(labels,node_size,sentencess,matrix,call_matrix,func_name,data_list,start_i,end_i,target_start):
    i=start_i
    while True:
        if i>=end_i:
            break
        if labels[i]==0:
            drop_t = random.randint(0,99)
            if  drop_t < 88:
                del labels[i]
                del node_size[i]
                del sentencess[i]
                del matrix[i]
                del call_matrix[i]
                del func_name[i]
                del data_list[i]
                end_i -= 1
                if i<target_start:
                    target_start -= 1
            else:
                i += 1
        else:
            i += 1

    return labels,node_size,sentencess,matrix,call_matrix,func_name,data_list,target_start,end_i

