#! /usr/bin/env python
import sys  
import networkx as nx
import numpy as np
import angr
import argparse
#from angrutils import plot_cfg, hook0, set_plot_style
#import bingraphvis

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='dead', help='name of program')
parser.add_argument('--ptype', type=str, default='o3_test', help='type of program')

def analyze(b, addr, name=None):
    start_state = b.factory.blank_state(addr=addr)
    start_state.stack_push(0x0)
    cfg = b.analyses.CFGFast()
    for addr,func in cfg.kb.functions.items():
        graph_f = func.transition_graph
        #print(func.nodes())
        i=0
        block_cnt=0;
        node_list=[]
        for node in func.nodes():
            try:
                if node.is_blocknode:
                    node_list.append(i)
                    block_cnt = block_cnt+1
            except:
                pass
            i=i+1
        #print(node_list)
        block_cnt_r=0
        for block in func.blocks:
            block_cnt_r = block_cnt_r+1
        if block_cnt_r != block_cnt:
            continue
        if block_cnt==0:
            continue
        A=np.array(nx.adjacency_matrix(graph_f).todense())
        A_block = A[:,node_list]
        A_block = A_block[node_list,:]
        #print(A_block)

        print(addr,block_cnt,file=f_arg)
        np.savetxt(f_adj,A_block.astype(int))
        #print("new cfg:")
        for block in func.blocks:
            #print(block)
            if block.size>0:
                print(block.instructions,end=" ",file=f_arg)
                block.pp()
            else:
                print(0,end=" ",file=f_arg)
                
        print("\n",file=f_arg)
           
if __name__ == "__main__":
    arg = parser.parse_args()
    name = arg.name
    ptype = arg.ptype
    save_stdout=sys.stdout
    f_bb_name = ptype+"_result/bb_"+name
    f_adj_name = ptype+"_result/adj_"+name
    f_arg_name = ptype+"_result/arg_"+name
    f_bb = open(f_bb_name,'w+')
    f_adj = open(f_adj_name,'w+')
    f_arg = open(f_arg_name,'w+')
    sys.stdout=f_bb
    proj = angr.Project(ptype+"_benchmark/"+name, load_options={'auto_load_libs':False})
    #debug
    #cfg0 = proj.analyses.CFGFast()
    #print("debug: cfg0 finish\n")

    #proj = angr.Project("test", load_options={'auto_load_libs':False})
    main = proj.loader.main_object.get_symbol("main")
    analyze(proj, main.rebased_addr)
    sys.stdout=save_stdout
