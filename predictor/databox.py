'''
change callgraph memory embedding method
(no w2v)
'''
from collections import Counter
from random import choice
import datetime
import w2v
import pandas as pd
import numpy as np
import cfg
import random
import os
import math

def build_memory_vector(data_list,dim):
    addr_list=[]
    page_list=[]
    for addr_s in data_list:
        addr = int(addr_s.split("-")[0],16)
        if addr_s == "0":
            cnt=0
        else:
            cnt = int(addr_s.split("-")[1])
        #for j in range(cnt):
            #addr_list.append(addr+j)
        addr_list.append(addr)
        page_list.append(addr%4096)
    data_eb = []
    data_eb.append(min(addr_list))
    data_eb.append(max(addr_list))
    data_eb.append(len(set(addr_list)))
    data_eb.append(len(addr_list))
    data_eb.append(len(set(page_list)))
    data_eb.append(len(page_list))
    dim -= 6
    addr_c = Counter(addr_list)
    page_c = Counter(page_list)
    index=0
    while((dim > 4) and (len(page_c.most_common())>index)):
        data_eb.append(page_c.most_common()[index][0])
        data_eb.append(page_c.most_common()[index][1])
        data_eb.append(addr_c.most_common()[index][0])
        data_eb.append(addr_c.most_common()[index][0])
        index += 1
        dim -= 4
    while(dim > 0  and  len(addr_c.most_common())>index):
        data_eb.append(addr_c.most_common()[index][0])
        data_eb.append(addr_c.most_common()[index][1])
        dim -= 2
    while(dim > 0):
        data_eb.append(0)
        dim -= 1

    return data_eb
    


class dataBox():
    def __init__(self,labels,benchmark_name,node_size,sentencess,matrix,call_matrix,func_name,data_list):
        self.labels = labels
        self.benchmark_name = benchmark_name
        self.node_size = node_size
        self.sentencess = sentencess
        self.matrix = matrix
        self.call_matrix = call_matrix
        self.func_name = func_name
        self.data_list = data_list
        self.bbs=[]
        self.data_eb=[]
        self.adj_matrix=[]
        self.adj_matrix2=[]
        self.adj_call_matrix=[]

    def random_data(self):
        new_labels = []
        new_benchmark_name = []
        new_node_size = []
        new_sentencess = []
        new_matrix = []
        new_call_matrix = []
        new_func_name = []
        new_data_list = []
       
        index_list=list(range(len(self.labels)))
        np.random.seed()
        np.random.shuffle(index_list)
        for i in range(len(self.labels)):
            new_labels.append(self.labels[index_list[i]])
            new_benchmark_name.append(self.benchmark_name[index_list[i]])
            new_node_size.append(self.node_size[index_list[i]])
            new_sentencess.append(self.sentencess[index_list[i]])
            new_matrix.append(self.matrix[index_list[i]])
            new_call_matrix.append(self.call_matrix[index_list[i]])
            new_func_name.append(self.func_name[index_list[i]])
            new_data_list.append(self.data_list[index_list[i]])
        self.labels = new_labels
        self.benchmark_name = new_benchmark_name
        self.node_size = new_node_size
        self.sentencess = new_sentencess
        self.matrix = new_matrix
        self.call_matrix = new_call_matrix
        self.func_name = new_func_name
        self.data_list = new_data_list

    def combine(self,data_b):
        self.labels += data_b.labels
        self.benchmark_name += data_b.benchmark_name
        self.node_size += data_b.node_size
        self.sentencess += data_b.sentencess
        self.matrix += data_b.matrix
        self.call_matrix += data_b.call_matrix
        self.func_name += data_b.func_name
        self.data_list += data_b.data_list

    def getsub(self,start,length):
        length += start
        return self.labels[start:length:1],self.benchmark_name[start:length:1],self.node_size[start:length:1],self.sentencess[start:length:1],self.matrix[start:length:1],self.call_matrix[start:length:1],self.func_name[start:length:1],self.data_list[start:length:1]


    def drop_ill_data(self,config):
        i=0
        end_i = len(self.labels)
        while True:
            if i>=end_i:
                break
            if len(self.data_list[i])>config.call_m_size:
                del self.labels[i]
                del self.benchmark_name[i]
                del self.node_size[i]
                del self.sentencess[i]
                del self.matrix[i]
                del self.call_matrix[i]
                del self.func_name[i]
                del self.data_list[i]
                end_i -= 1
            else:
                i += 1


        
    def drop_nodead_data(self,config):
        i=0
        end_i = len(self.labels)
        label_dead=0
        label_nodead=0
        label_drop=0
        random.seed()

        while True:
            if i>=end_i:
                break
            if self.labels[i]==0:
                label_nodead += 1
                drop_t = random.randint(0,99)
                if  drop_t < config.drop_rate:
                    label_drop += 1
                    del self.labels[i]
                    del self.benchmark_name[i]
                    del self.node_size[i]
                    del self.sentencess[i]
                    del self.matrix[i]
                    del self.call_matrix[i]
                    del self.func_name[i]
                    del self.data_list[i]
                    end_i -= 1
                else:
                    i += 1
            else:
                label_dead += 1
                i += 1
        print("before drop: all,label0,label1, drop:",label_nodead+label_dead,label_nodead,label_dead,label_drop)
        
        print("before copy label 1:",len(self.labels))
        if config.copy_rate>0:
            length = len(self.labels)
            for i in range(length):
                if self.labels[i]==1:
                    resample_t = random.randint(0,99)
                    if resample_t < config.copy_rate:
                        self.labels.append(self.labels[i])
                        self.benchmark_name.append(self.benchmark_name[i])
                        self.node_size.append(self.node_size[i])
                        self.sentencess.append(self.sentencess[i])
                        self.matrix.append(self.matrix[i])
                        self.call_matrix.append(self.call_matrix[i])
                        self.func_name.append(self.func_name[i])
                        self.data_list.append(self.data_list[i])
            if config.copy_rate>100:
                config.copy_rate -= 100
                for i in range(length):
                    if self.labels[i]==1:

                        resample_t = random.randint(0,99)
                        if resample_t < config.copy_rate:
                            self.labels.append(self.labels[i])
                            self.benchmark_name.append(self.benchmark_name[i])
                            self.node_size.append(self.node_size[i])
                            self.sentencess.append(self.sentencess[i])
                            self.matrix.append(self.matrix[i])
                            self.call_matrix.append(self.call_matrix[i])
                            self.func_name.append(self.func_name[i])
                            self.data_list.append(self.data_list[i])
                
        print("after copy label 1:",len(self.labels))

    def drop_notest(self):
        i=0
        end_i = len(self.labels)
        #keys = ["505.mcf_r","508.namd_r","510.parest_r","520.omnetpp_r","523.xalancbmk_r","544.nab_r","557.xz_r","526.blender_r"]
        keys = ["nab_r_base.mytest-64","omnetpp_r_base.mytest-64", "xalancbmk_r_base.mytest-64","mcf_r_base.mytest-64", "namd_r_base.mytest-64","parest_r_base.mytest-64","xz_r_base.mytest-64","blender_r_base.mytest-64"]
        values = [0,0,0,0,0,0,0,0]
        cnt_dict = dict(zip(keys,values))
        print("before drop_test drop:",end_i)
        while True:
            if i>=end_i:
                break
            if cnt_dict[self.benchmark_name[i]]<500:
                drop_f = False
                cnt_dict[self.benchmark_name[i]] += 1
            else:
                drop_f = True

            if  drop_f:
                del self.labels[i]
                del self.benchmark_name[i]
                del self.node_size[i]
                del self.sentencess[i]
                del self.matrix[i]
                del self.call_matrix[i]
                del self.func_name[i]
                del self.data_list[i]
                end_i -= 1
            else:
                i += 1
        print("after drop_notest ",len(self.labels))


    def random_drop(self,config):
        i=0
        end_i = len(self.labels)
        print("before random drop:",end_i)
        random.seed()

        while True:
            if i>=end_i:
                break
            drop_t = random.randint(0,99)
            if  drop_t < 25:
                del self.labels[i]
                del self.benchmark_name[i]
                del self.node_size[i]
                del self.sentencess[i]
                del self.matrix[i]
                del self.call_matrix[i]
                del self.func_name[i]
                del self.data_list[i]
                end_i -= 1
            else:
                i += 1
        print("after random drop",len(self.labels))

    def cnt_label_0_1(self):
        label_0_cnt=0
        label_1_cnt=1
        for label in self.labels:
            if label == 0:
                label_0_cnt += 1
            else:
                label_1_cnt += 1
        print("label 0 & 1:",label_0_cnt,label_1_cnt)


    def train_w2v(self,target,path,n_dim,save_model):
        sentences=[]

        if target == 'instructions':
            for i in range(len(self.sentencess)):
                sentences += self.sentencess[i]
        else:
            for i in range(len(self.data_list)):
                sentences += self.data_list[i]

        # 数据集获取
        #sentences = w2v.word2vec.LineSentence('./data/ass_test_text.txt') 
        print("start train w2v...")
        # word2vec词向量训练,并保存模型(w2v.py中实现)
        w2v_model = w2v.get_dataset_vec(sentences,n_dim,path,save_model,target)
        if target == "instructions":
            cfg.w2v_model = w2v_model
        else:
            cfg.data_model = w2v_model
        print("train w2v finish")




    def data_padding(self,config,is_train_load):
        if config.load_test and is_train_load:
            w2v_model = w2v.load_model(cfg.w2v_model_path,"instructions")
            data_model = w2v.load_model(cfg.data_w2v_model_path,"data")
            cfg.w2v_model = w2v_model
            cfg.data_model = data_model
        else:
            w2v_model=cfg.w2v_model  # w2v.load_model(cfg.w2v_model_path,"instructions")
            data_model=cfg.data_model # w2v.load_model(cfg.data_w2v_model_path,"data")

        before_sent = "null_sent"
        for i in range(len(self.labels)):
            bb_cnt = self.node_size[i]
            sentences = self.sentencess[i]
            datamat = self.matrix[i]
            bb=np.zeros([bb_cnt,config.annotation_dim])
            for sent_i in range(len(sentences)):
                tt = (w2v.build_sentence_vector(sentences[sent_i],config.annotation_dim,w2v_model))
                bb[sent_i]=tt
       
            sentences = self.data_list[i]
            func_n=np.zeros([len(sentences),config.call_g_annotation])
            for node_i in range(len(sentences)):
                #tt = w2v.build_sentence_vector(sentences[node_i],config.call_g_annotation,data_model)
                tt = (build_memory_vector(sentences[node_i],config.call_g_annotation))
                func_n[node_i]=tt

            #padding
            padding_matrix_1 = np.zeros((np.size(datamat,1),config.n_node_type - np.size(datamat,1)))
            datamat = np.concatenate((datamat,padding_matrix_1),1)
            padding_matrix_0 = np.zeros((config.n_node_type - np.size(datamat,0),config.n_node_type))
            datamat = np.concatenate((datamat,padding_matrix_0),0)

            call_datamat = self.call_matrix[i]
            call_padding_matrix_1 = np.zeros((np.size(call_datamat,1),config.call_m_size - np.size(call_datamat,1)))
            call_datamat = np.concatenate((call_datamat,call_padding_matrix_1),1)
            call_padding_matrix_0 = np.zeros((config.call_m_size - np.size(call_datamat,0),config.call_m_size))
            call_datamat = np.concatenate((call_datamat,call_padding_matrix_0),0)


            padding0 = np.zeros((config.n_node_type-np.size(bb,0),config.annotation_dim))
            bb = np.concatenate((bb,padding0),0)

            padding0 = np.zeros((config.call_m_size-np.size(func_n,0),config.call_g_annotation))
            func_n = np.concatenate((func_n,padding0),0)

            self.data_eb.append(func_n)
            self.bbs.append(bb)
            self.adj_matrix.append(datamat)
            self.adj_matrix2.append(datamat)
            self.adj_call_matrix.append(call_datamat)
