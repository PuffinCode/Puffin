#change w2v to memory feature
from gensim.models import word2vec
from collections import defaultdict
from random import choice
import w2v
import pandas as pd
import numpy as np
import json

#import mymodel_data3 as mymodel 
import mymodel_data as mymodel 
import cfg
import loaddata
import preprocessing as prep
import databox as box
import argparse
import torch
import torch.nn as nn
import random
from torch.autograd import Variable
import os
import math
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--no_memory', type=bool, default=False)
parser.add_argument('--load_test', type=int, default=0)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--model_type', type=str, default="all")
parser.add_argument('--lossf', type=str, default="bce")
parser.add_argument('--drop_rate', type=int, default=86)
parser.add_argument('--copy_rate', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--target', type=str, default="o2_test")
parser.add_argument('--train', type=str, default="o2_test")
parser.add_argument('--compl', type=str, default="")
parser.add_argument('--redundant_t', type=str, default="dw")
parser.add_argument('--model_tag', type=str, default="test")
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--hidden_dim', type=int, default=60, help='GGNN hidden state size')
parser.add_argument('--annotation', type=int, default=60, help='GGNN annotation_dim')
parser.add_argument('--ggnn_out', type=int, default=70, help='GGNN output_dim')
parser.add_argument('--resnet_out', type=int, default=40, help='ResNet output_dim')
parser.add_argument('--node_cnt', type=int, default=128, help='GGNN node cnt')
parser.add_argument('--call_hidden_dim', type=int, default=30, help='GGNN2 hidden state size')
parser.add_argument('--call_g_annotation', type=int, default=30, help='GGNN2 output_dim')
parser.add_argument('--call_steps', type=int, default=10, help='GGNN2 steps')
parser.add_argument('--call_out', type=int, default=50, help='GGNN2 out_size')
parser.add_argument('--call_m_size', type=int, default=256, help='GGNN2 node_size')


data_file_list1=["505.mcf_r","508.namd_r","510.parest_r","520.omnetpp_r","523.xalancbmk_r","544.nab_r","557.xz_r","526.blender_r","502.gcc_r","511.povray_r","538.imagick_r","541.leela_r"]
#data_file_list1=["505.mcf_r","508.namd_r","510.parest_r","520.omnetpp_r","523.xalancbmk_r","544.nab_r","557.xz_r","526.blender_r"]
data_file_list2=["nab_r_base.mytest-64","omnetpp_r_base.mytest-64", "xalancbmk_r_base.mytest-64","mcf_r_base.mytest-64", "namd_r_base.mytest-64","parest_r_base.mytest-64","xz_r_base.mytest-64","blender_r_base.mytest-64","502.gcc_r","511.povray_r","538.imagick_r","541.leela_r"]
#data_file_list2=["nab_r_base.mytest-64","omnetpp_r_base.mytest-64", "xalancbmk_r_base.mytest-64","mcf_r_base.mytest-64", "namd_r_base.mytest-64","parest_r_base.mytest-64","xz_r_base.mytest-64","blender_r_base.mytest-64"]

#data_file_list3=["505.mcf_r"]
#data_file_list3=["526.blender_r"]
data_file_list3=["502.gcc_r"]
data_file_list4=["541.leela_r"]

data_file_list=[]
test_result_label=[]
test_result_real_label=[]
test_result_func_name=[]
test_result_benchmark_name=[]

data_path=prep.data_path
data_path2=prep.data_path2

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



def random_memory_part(length):
    call_matrix=[]
    data_list=[]
    func_cg=[]
    for i in range(length):
        call_datamat=np.empty([1,1]) 
        call_matrix.append(call_datamat)
        n_addr_list=["0"]
        n_list=[]
        n_list.append(n_addr_list)
        data_list.append(n_list)    

    return call_matrix,data_list


def create_dic(func_dic,f):
    for line in f:
        items = line.split()
        #print("key:",items[0],int(items[0],16))
        key_it = int(items[0],16)
        value_it = str(items[1])
        #value_it = str(int(value_it) // 4096)
        func_dic[key_it].add(value_it)

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError



def train_test(train,test,data_loader,spy_model,epoch):
    lr = config.lr * (0.1 ** (epoch // 30))
    optimizer = torch.optim.Adam(spy_model.parameters(), lr) 
    if config.lossf == "focal":
        print("lossf:focal")
        criterion = mymodel.BCEFocalLoss().cuda()
    else:
        criterion = torch.nn.BCELoss().cuda()

    #criterion = nn.BCEWithLogitsLoss().cuda()
    #criterion = nn.CrossEntropyLoss().cuda()
    

    if not train:
        FP=0
        TP=0
        FN=0
        TN=0
        epoch = 1
    for i in range(epoch):
        total_loss=0
        total_loss_cnt=0
        for adj_matrix,adj_matrix2,bbs,labels,adj_call_matrix,data_eb,func_name,benchmark_name in data_loader: 

            padding_2 = torch.zeros(len(adj_matrix2),config.n_node_type,config.n_node_type).double()
            padding_2 = padding_2.permute(0,2,1)
            adj_matrix2 = torch.cat((adj_matrix2,padding_2),2)# batch_size,node_size,node_size*2
            adj_m = adj_matrix.unsqueeze(1)
            adj_matrix1 = adj_m
            #adj_matrix1=torch.cat((adj_matrix1,adj_m),1)
            #adj_matrix1=torch.cat((adj_matrix1,adj_m),1) #input of CNN need 3channel
    
            padding_2 = torch.zeros(len(adj_call_matrix),config.call_m_size,config.call_m_size).double()
            padding_2 = padding_2.permute(0,2,1)
            adj_call_matrix = torch.cat((adj_call_matrix,padding_2),2)# batch_size,node_size,node_size*2

            call_x = data_eb
            padding = torch.zeros(len(call_x), config.call_m_size, config.call_hidden_dim - config.call_g_annotation).double()
            call_x = torch.cat((call_x, padding), 2)

            x = bbs
            padding = torch.zeros(len(bbs), config.n_node_type, config.hidden_dim - config.annotation_dim).double()
            x = torch.cat((x, padding), 2)
          
            #change labels type

            x = Variable(x.cuda()).double()
            m = Variable(adj_matrix1.cuda()).double()
            m2 = Variable(adj_matrix2.cuda()).double()
            a = Variable(bbs.cuda()).double()
            t = Variable(labels.cuda()).double()
            
            data_m = Variable(adj_call_matrix.cuda()).double()
            data_a = Variable(data_eb.cuda()).double()
            data_x = Variable(call_x.cuda()).double()

            output = spy_model(x,m,m2,a,data_x,data_a,data_m)
            t = t.view(-1)
            #print("output:",output)
            #print("     t:",t)
            loss = criterion(output, t)
            total_loss += loss.item()
            total_loss_cnt += 1
            print("loss:",loss.item())
            if train:
                spy_model.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                prediction = output
                #prediction = torch.max(output, 1)[1]  # troch.max(out,1)取每行的最大值，troch.max()[1]， 只返回最大值的每个索引
                prediction = prediction.detach().cpu().numpy()
                t = t.cpu().numpy()
                #print("pred,t:",prediction,t)
               
                for i in range(len(t)):
                    train_correct01 = ((prediction[i]<=0.5)&(t[i]==1))
                    train_correct10 = ((prediction[i]>0.5)&(t[i]==0))
                    train_correct11 = ((prediction[i]>0.5)&(t[i]==1))
                    train_correct00 = ((prediction[i]<=0.5)&(t[i]==0))

                    FN += train_correct01.item()       
                    FP += train_correct10.item()
                    TP += train_correct11.item()
                    TN += train_correct00.item()

                    #output predict result
                    if test:
                        if prediction[i]<=0.5:
                            this_label=0
                        else:
                            this_label=1
                        test_result_label.append(this_label)
                        test_result_real_label.append(t[i])
                        test_result_func_name.append(func_name[i].numpy())
                        test_result_benchmark_name.append(benchmark_name[i])
                
        print("epoch_loss:",total_loss/total_loss_cnt)


    stop=False
    if not train:
        print("validation----------------------------")
        print("total_loss,min:",total_loss/total_loss_cnt,config.validation_loss_min)
        print("TP,TN,FP,FN: ",TP,TN,FP,FN)
        if TP==0:
            ALL_P = (TP+TN) / (TP+FP+TN+FN)
            print("precise,recall,all_precise:",0,0,ALL_P)
            R=0
            P=0
        else:
            P = TP/ (TP+FP)
            R = TP/ (TP+FN)
            ALL_P = (TP+TN) / (TP+FP+TN+FN)
            print("precise,recall,all_precise:",P,R,ALL_P)
            #if R>0.76 and ALL_P>0.76:
            #    stop=True
        print("---------------------------------------")      
        if ALL_P > config.accuracy_max:
            config.accuracy_max = ALL_P
            config.accuracy_stop_raise=0
        else:
            config.accuracy_stop_raise += 1
            print("max_accuracy, stop raise cnt:",config.accuracy_max,config.accuracy_stop_raise)
            if config.accuracy_stop_raise > 10:
                stop=True
                print("stop: accuracy raise stop.")
        '''
        if total_loss/total_loss_cnt < config.validation_loss_min:
            config.validation_loss_min=total_loss/total_loss_cnt
            config.valid_stop_down=0
        else:
            config.valid_stop_down += 1
            if config.valid_stop_down>10:
                stop=True
                print("stop: validation loss stop drop.")
        '''
    return stop




def train_model(train,opt,config):
    func_name=[]
    func_bb=defaultdict(list)
    bbs=[]
    adj_matrix=[]
    adj_matrix2=[]
    labels=[]
    node_size=[]
    matrix=[]
    sentencess=[]
    call_matrix=[]
    func_addr=[]
    adj_call_matrix=[]
    data_list=[]
    data_eb=[]
    data_amount=0
    benchmark_name=[]

    
    for file_name in data_file_list:
        func_bb=defaultdict(list)
        t_func_name,t_labels,t_node_size,t_sentencess,t_matrix,func_bb = prep.load_cfg_data(file_name,config,opt.redundant_t)
        
        #mode dont't have memory part
        if config.no_memory:
            t_call_matrix,t_data_list = random_memory_part(len(t_labels)) 
        else:
            t_call_matrix,t_data_list,func_cg = prep.load_cg_data(file_name,t_func_name,func_bb,config)
            t_func_name,t_labels,t_node_size,t_sentencess,t_matrix,t_call_matrix,t_data_list = prep.delete_func(t_func_name,t_labels,t_node_size,t_sentencess,t_matrix,t_call_matrix,t_data_list,func_cg)
        t_benchmark_name = [ str(file_name) for i in range(len(t_labels))]

        benchmark_name += t_benchmark_name
        func_name += t_func_name
        labels += t_labels
        node_size += t_node_size
        sentencess += t_sentencess
        matrix += t_matrix
        print(file_name,len(labels),len(call_matrix),len(data_list))
   
    #for file_name in data_file_list:
        #t_call_matrix,t_data_list = prep.load_cg_data(file_name,t_func_name,func_bb,config)
        call_matrix += t_call_matrix
        data_list += t_data_list
        print(file_name,len(labels),len(call_matrix),len(data_list))

    data1 = box.dataBox(labels,benchmark_name,node_size,sentencess,matrix,call_matrix,func_name,data_list) 
    func_name=[]
    labels=[]
    node_size=[]
    matrix=[]
    sentencess=[]
    call_matrix=[]
    data_list=[]
 
    data1.random_data()
    data1.drop_ill_data(config)
    data_amount1 = len(data1.labels)
    train_size1 = math.floor(0.9*data_amount1)
    validation_size1 = math.floor(0.5*(data_amount1-train_size1))
    test_size1 = data_amount1 - train_size1 - validation_size1


    labels,benchmark_name,node_size,sentencess,matrix,call_matrix,func_name,data_list = data1.getsub(0,train_size1)
    data1_sub = box.dataBox(labels,benchmark_name,node_size,sentencess,matrix,call_matrix,func_name,data_list)
    data_train = data1_sub
    labels,benchmark_name,node_size,sentencess,matrix,call_matrix,func_name,data_list = data1.getsub(train_size1,validation_size1)
    data1_sub = box.dataBox(labels,benchmark_name,node_size,sentencess,matrix,call_matrix,func_name,data_list)
    data_validation = data1_sub
    labels,benchmark_name,node_size,sentencess,matrix,call_matrix,func_name,data_list = data1.getsub(train_size1+validation_size1,test_size1)
    data1_sub = box.dataBox(labels,benchmark_name,node_size,sentencess,matrix,call_matrix,func_name,data_list)
    data_test = data1_sub


    data_train.drop_nodead_data(config)
    data_train.random_data()
    print("train,validation,test size:",len(data_train.labels),len(data_validation.labels),len(data_test.labels))
    if config.load_test == 0:
        data_train.train_w2v('instructions',cfg.w2v_model_path,config.annotation_dim,config.save_model)
        #data_train.train_w2v('memory',cfg.data_w2v_model_path,config.call_g_annotation,config.save_model)

    
    print("start padding ...")
    data_train.data_padding(config,True) #only load w2v model once,here set true
    data_validation.data_padding(config,False)
    data_test.data_padding(config,False)
    

    print("padding finish")
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if config.cuda else {}
    train_data_loader = torch.utils.data.DataLoader(
                        loaddata.MyDataset(data_train.adj_matrix,data_train.adj_matrix2,data_train.bbs,data_train.labels,data_train.adj_call_matrix,data_train.data_eb,data_train.func_name,data_train.benchmark_name), batch_size=config.batch_size, **kwargs)
    test_data_loader = torch.utils.data.DataLoader(
                        loaddata.MyDataset(data_test.adj_matrix,data_test.adj_matrix2,data_test.bbs,data_test.labels,data_test.adj_call_matrix,data_test.data_eb,data_test.func_name,data_test.benchmark_name), batch_size=config.batch_size, **kwargs)
    validation_data_loader = torch.utils.data.DataLoader(
                        loaddata.MyDataset(data_validation.adj_matrix,data_validation.adj_matrix2,data_validation.bbs,data_validation.labels,data_validation.adj_call_matrix,data_validation.data_eb,data_validation.func_name,data_validation.benchmark_name), batch_size=config.batch_size, **kwargs)
    print("create data loader finish")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config.load_test == 0:
        if config.model_type=="w2v+ggnn1+resnet":
            spy_model = mymodel.in_func_model(opt,config)
        elif config.model_type == "w2v":
            spy_model = mymodel.word2vec(opt,config)
        elif config.model_type == "ggnn2":
            spy_model = mymodel.GGNN_memory(opt,config)
        elif config.model_type == "w2v+ggnn1":
            spy_model = mymodel.semantic(opt,config)
        elif config.model_type == "resnet7":
            spy_model = mymodel.Resnet7(opt,config)
        elif config.model_type == "resnet11":
            spy_model = mymodel.Resnet11(opt,config)
        elif config.model_type == "CNN3":
            spy_model = mymodel.CNN3(opt,config)
        elif config.model_type == "no_resnet":
            spy_model = mymodel.SPY_noresnet(opt,config)
        else:
            spy_model = mymodel.SPY(opt,config)
        spy_model = spy_model.double()
        #if torch.cuda.device_count() > 1:
        #    spy_model = nn.DataParallel(spy_model,device_ids=[0,1])
            #spy_model = nn.DataParallel(spy_model)
        spy_model.to(device)
        print("model to device finish")

        for i in range(config.epoch):
            if i==0:
                continue
            epoch_i = i*1
            print("train epoch:",epoch_i)
            spy_model.train() 
            stop = train_test(True,False,train_data_loader,spy_model,1)
            print("start validation epoch:",epoch_i)
            spy_model.eval()
            stop = train_test(False,False,validation_data_loader,spy_model,1)
            if i <3 :
                stop=False
            if stop:
                break
    else:
        spy_model = torch.load('data-model/spy/'+prep.model_name+'_spy_model.pkl')
        spy_model.to(device)
    
    print("start test:")
    now_time = datetime.datetime.now()
    print("time",now_time)

   
    stop = train_test(False,True,test_data_loader,spy_model,1)

    now_time = datetime.datetime.now()
    print("time",now_time)

    if config.save_model:
        torch.save(spy_model, 'data-model/spy/'+prep.model_name+'_spy_model.pkl')
 
def debug():
    print("in degub.")
    
if __name__ == '__main__':

    opt = parser.parse_args()
    config=cfg.CONFIG(opt)
    config.print_c()
    print(opt.train,opt.target)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    if opt.no_memory:
        print("no memory")
    if opt.compl=='gcc':
        compiler=''
    elif opt.compl=='llvm':
        compiler='llvm_'
    else:
        print("ERROR: wrong compiler")

    prep.data_path="../../angr/"+compiler+opt.train+"_test_data/"+compiler+opt.train+"_test_result/"
    prep.data_path2="../../angr/"+compiler+opt.train+"_test_data/"+compiler+opt.train+"_test_memory/"
    prep.data_path3="../../angr/"+compiler+opt.train+"_test_data/"+compiler+opt.train+"_test_cg/"
    prep.model_name = opt.compl+"-"+opt.train+"-"+opt.model_tag

    if opt.load_test:
        cfg.w2v_model_path = 'data-model/w2v/'+prep.model_name+'_w2v_model.pkl'
        cfg.data_w2v_model_path = 'data-model/w2v/'+prep.model_name+'_data_w2v_model.pkl'
    else:
        cfg.w2v_model_path = 'data-model/w2v/'+prep.model_name+'_w2v_model.pkl'
        cfg.data_w2v_model_path = 'data-model/w2v/'+prep.model_name+'_data_w2v_model.pkl'
 
    if opt.train=="arm_o2" or opt.train == "arm_o3":
        data_file_list = data_file_list2
    else:
        data_file_list = data_file_list1

    if opt.compl == "llvm":
        data_file_list = data_file_list1

    if opt.debug:
        if opt.train=="arm_o2" or opt.train == "arm_o3":
            data_file_list = data_file_list4
        else:
            data_file_list = data_file_list3

    #debug()
    now_time = datetime.datetime.now()
    print("time:",now_time)
    train_model(True,opt,config)
    now_time = datetime.datetime.now()
    print("time",now_time)

    for i in range(len(test_result_label)):
        print(test_result_label[i],end=" ")
        print(test_result_real_label[i],end=" ")
        print(test_result_func_name[i],end=" ")
        print(test_result_benchmark_name[i])
