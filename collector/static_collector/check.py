import sys
import argparse
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='dead', help='name of program')
parser.add_argument('--ptype', type=str, default='o3_test', help='type of program')

if __name__ == "__main__":
    arg = parser.parse_args()
    name = arg.name
    ptype = arg.ptype
    f_in = open(ptype+"_result/arg_"+name,'r')
    f_dead = open(ptype+"_dead/"+name+".dead.rw",'r')
    f_bb = open(ptype+"_result/bb_"+name,'r')
    print(ptype+"_result/bb_"+name)
    deadstore_amount=0
    nodead_amount=0
    f_out = open(ptype+"_result/label_"+name,"w")
    deadaddr_dic = defaultdict(int)
    dead_addr = np.loadtxt(f_dead,skiprows=2,usecols=(0),unpack=True)
    dead_addr_list = dead_addr.tolist()
    dead_addr_list = list(map(int,dead_addr_list))
    
    for addr in dead_addr_list:
        deadaddr_dic[addr]=1

    line = f_in.readline()
    while line:
        addr,cnt = map(int,line.split())
        label_str=[]
        inst_cnt=0
        dead=False
        #addr = hex(addr)
        line_cnt = f_in.readline()
        bb_cnt = list(map(int,line_cnt.split()))
        for bb_cnt_i in bb_cnt:
            for bb_i in range(bb_cnt_i):
                inst_cnt = inst_cnt+1
                label_str.append(line_cnt)
                ints = f_bb.readline().split() 
                ints_addr = ints[0]
                ints_addr = ints_addr[:-1]
                ints_addr_i = int(ints_addr,16)
                if not dead:
                    if (ints_addr_i in deadaddr_dic):
                        dead=True
        #for i in range(inst_cnt):
        if dead:
            deadstore_amount=deadstore_amount+1
            print(addr,end=" ",file=f_out)
            print(cnt,end=" ",file=f_out)
            print("1",file=f_out)
            print(line_cnt,file=f_out)
        else:
            nodead_amount = nodead_amount+1
            print(addr,end=" ",file=f_out)
            print(cnt,end=" ",file=f_out)
            print("0",file=f_out)
            print(line_cnt,file=f_out)

        line = f_in.readline()
        line = f_in.readline()
    print("deadstore:",name,deadstore_amount)
    print("nodead:",name,nodead_amount)
    print("all:",name,nodead_amount+deadstore_amount)
  
