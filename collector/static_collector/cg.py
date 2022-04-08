#! /usr/bin/env python
import sys  
import networkx as nx
import numpy as np
import angr
import argparse
import os
from collections import defaultdict
#from angrutils import plot_cfg, hook0, set_plot_style
#import bingraphvis

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='dead', help='name of program')
parser.add_argument('--ptype', type=str, default='o3_test', help='type of program')

child_dic = defaultdict(int)
father_dic = defaultdict(int)
func_list=[]
def create_subgraph(G,sub_G,start_node,hop):
    if hop == 0:
        return
    for n in G.successors(start_node):
        if (n not in sub_G) and (n in func_list): # and (sub_G.number_of_nodes()<256):
            #if n in func_memory_dic:
            #    if (func_memory_dic[n]&node_set):
            sub_G.add_node(n)
            sub_G.add_edge(start_node,n)
            #child_list.append(n)
            if (father_dic[n]+child_dic[n])<20:  #gcc:100:
                create_subgraph(G,sub_G,n,hop-1)
    for n in G.predecessors(start_node):
        if (n not in sub_G) and (n in func_list): # and (sub_G.number_of_nodes()<256):
            #if n in func_memory_dic:
            #    if (func_memory_dic[n]&node_set):
            sub_G.add_node(n)
            sub_G.add_edge(n,start_node)
            #father_list.append(n)
            if (father_dic[n]+child_dic[n])<20:  #gcc:100:     
                create_subgraph(G,sub_G,n,hop-1)
    '''
    for n in child_list:
        create_subgraph(G,sub_G,n,hop-1)
    for n in father_list:
        create_subgraph(G,sub_G,n,hop-1)
    '''


def analyze(b, addr, name=None):
    cfg = b.analyses.CFGFast()
    #cfg = b.analyses.CFG(show_progressbar=True) 
    cg = cfg.functions.callgraph
    print(cg.number_of_nodes(),len(func_list))
    A=np.array(nx.adjacency_matrix(cg).todense())

    for node in cg.nodes():
        child_n=0
        father_n=0
        for n in cg.successors(node):
            child_n += 1
        for n in cg.predecessors(node):
            father_n += 1
        child_dic[node]=child_n
        father_dic[node]=father_n


    #np.savetxt(f_adj,A)
    for node in cg.nodes():
        sub_G = nx.DiGraph()
        sub_G.add_node(node)
        if (node in func_list):
            #node_set={}
            #if node in func_memory_dic:
            #    node_set = func_memory_dic[node]
            create_subgraph(cg, sub_G,node,5)
            sub_A=np.array(nx.adjacency_matrix(sub_G).todense())
            #print(sub_A)
            #print(node,sub_G.number_of_nodes())
            np.savetxt(f_adj,sub_A)
            print(node,sub_G.number_of_nodes(),file=f_node)
            for n in sub_G.nodes():
                print(n,end=" ",file=f_node)
            print("\n",end="",file=f_node)

def create_dic(func_dic,f):
    for line in f:
        items = line.split()
        func_dic[items[0]].add(str(items[1]))


if __name__ == "__main__":
    arg = parser.parse_args()
    name = arg.name
    ptype = arg.ptype
      
    f_node_name = ptype+"_cg/node_"+name
    f_node = open(f_node_name,'w+')
    f_adj_name = ptype+"_cg/adj_"+name
    f_adj = open(f_adj_name,'w+')
    proj = angr.Project(ptype+"_benchmark/"+name, load_options={'auto_load_libs':False})

    
    #f_label_name = "cfg_label/"+name+"_label"
    f_label_name = ptype+"_result/label_"+name
    arg_f = open(f_label_name,'r')
    arg_line = arg_f.readline()
    while arg_line:
        addr,bb_cnt,cfg_label = map(int,arg_line.split())
        if addr not in func_list:
            func_list.append(addr)
        arg_line = arg_f.readline()
        arg_line = arg_f.readline()
        arg_line = arg_f.readline()
    
    '''
    data_path="func_memory_dic/"
    f_memory_name = "memory_access/memory_"+name
    func_memory_dic = defaultdict(set)
    if not os.path.isfile(data_path+name+"_dic.txt"):
        print("create dic:",name)
        data_f = open(f_memory_name)
        create_dic(func_memory_dic,data_f)
        fw = open(data_path+name+"_dic.txt",'w+')
        #fw.write(str(func_memory_dic)) #把字典转化为str
        js = json.dumps(func_memory_dic)
        fw.write(js)
        fw.close()
    else:
        print("load dic:",name)
        fr = open(data_path+name+"_dic.txt",'r+')
        #func_memory_dic = eval(fr.read()) #读取的str转换为字典
        js = fr.read()
        func_memory_dic = json.loads(js)
        fr.close()
    '''
    #proj = angr.Project(name, load_options={'auto_load_libs':False})
    main = proj.loader.main_object.get_symbol("main")
    analyze(proj, main.rebased_addr)
