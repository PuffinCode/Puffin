# Hardware
All the hardware used in this paper is the corporate resources of TikTok Inc. According to the corporate policy, there is no way to vpn-access the hardware resources from outside at this moment. But, we would like to give the detailed machine architectures and configurations as below.
We have used two types of machines in our experiments and their detailed descriptions are shown below:
+ Client machine: Intel Xeon CPU 8163 2.50GHz with 512GB RAM, running Linux kernel 5.0. 
+ Central server: Accelerator is 8 Nvidia Tesla V100 GPU cards of the Volta architecture, each of which is with 5120 streaming cores, 640 tensor cores and 32GB memory capacity. CUDA version is 11.0 and pytorch version is 1.6.0. Host is Intel Xeon CPU 8163 2.50GHz with 512GB RAM, running Linux kernel 5.0. 
+ The central server is connected with the client machines through 100Gb NIC.

# Software and Code
The code is divided into four pieces, three of which are the predictor, collector and monitor. The fourth is our CIDetector implementation that provides the ground truth labels when training the model. All these codes is partially running in the production at TikTok Inc. According to the corporate security policy, there is no way to open-source the code at this moment. But, we would like to describe some basics of the code structures and the instructions to run.


# Build
+ Static collector: download and install angr on client machines. Copy the code in Puffin's collector/static_collector/ to the directory angr-dev/angr-utils/examples/plot_cfg
+ Dynamic collector: install intel xed tool on client machines. Compile the monitor with command: g++ -o collector collector.cpp elf_parser.cpp -I path_of_xed/xed-tool/xed-kit/kits/xed-install-base-2020-12-19-lin-x86-64/include -L path_of_xed/xed-tool/xed-kit/kits/xed-install-base-2020-12-19-lin-x86-64/lib -l xed -g -pthread
+ Predictor: installing pytorch on the training server and copy the code in Puffin's predictor to this machine. Install genism with pip. Start the training of the model by executing python train.py
+ Monitor: download and install DynamoRIO on client machines. Copy Puffin's monitor to DynamoRIO/clients and build it with cmake.

# Collector
We need to collect both static data and dynamic data for predictor. The static data of each sample contains 3 parts: CG, CFG, raw data of instructions. The dynamic data of each sample contains memory states. In order to perform model training and accuracy evaluation, we also need to collect labels for each sample. The following describes the scripts we use and the process of generating data.

## scripts
+ get_cfg.py: to obtain the CFG of the function and the instructions of each BB through angr
+ cg.py: to obtain the CG for each function with the help of angr
+ check.py: add ground truth labels to all training data
+ collector: to obtain the memory states for each function 
+ CIDetector: tool to get ground truth labels, implemented based on DynamoRIO. Since this code is not publicly available, it is not provided here


## run steps
+ compile the target program according to the specified option (can be gcc-O2, gcc-O3, llvm-O2, and llvm-O3), and store all the executable program in the option_benchmark directory
+ create a new output folder named option_result to store the results
+ run the following instruction to obtain the CFG of the function and the instructions of each BB
   + python get_cfg.py --name=program --ptype=option
   + parameter:
      + --name: the program we try to run
      + --ptype: can be gcc-O2, gcc-O3, llvm-O2, and llvm-O3
   + output:
      + ***arg_program***: record the CFG and instruction parameters of the function. Each function has two lines in this file. The first line has two numbers, which are the address of the first instruction of the function (addr), and the number of BBs contained in the function (bb_cnt). The second line has bb_cnt numbers, which are the number of instructions of each BB
      + ***adj_program***: records the CFG of all functions, expressed in the form of adjacency matrix. For example, if the function has bb_cnt BBs, then an adjacency matrix of bb_cnt*bb_cnt is recorded in the file ***adj_program***
      + ***bb_program***: records the instructions of each BB in the function. If the function has 3 BBs and the number of instructions is 4, 7, and 12 respectively, there will be 23 consecutive instructions in the file ***bb_program***
+ execute the cg.py to obtain CG 
   + python cg.py --name=program --ptype=option
   + parameter:
      + --name: the program we try to run
      + --ptype: can be gcc-O2, gcc-O3, llvm-O2, and llvm-O3
   + output:
      + ***cg_program***: contains the adjacency matrix of the CG of each function. In the adjacency matrix, a node represents a function, an edge represents a call relationship, and the direction of the edge is from the caller to the callee
      + ***node_program***: contains the node information of the CG. Each function has two lines. The first line has two parameters, which are the address of the first instruction of the target function (addr) and the number of nodes contained in its CG (node_cnt). The second line is the address of the first instruction of each function in its CG
+ run CIDetector to get the basic blocks that have 3 kinds of unnecessary memory operations. 
   + output:
      + ***target_program_deadstore***: contains all the basic blocks with dead store
      + ***target_program_silentstore***: contains all the basic blocks with silent store
      + ***target_program_silentload***: contains all the basic blocks with silent load
+ run collector to get memory states: 
   + ./collector pid binary sample_freq detect_t phase_get_memory memory_out
   + parameter:
      + pid: the pid of the target program running on the server
      + binary: the binary of the target program
      + sample_freq: sample freq of Puffin, the way to choose this parameter is discussed in the paper
      + detect_t: duration of memory state sampling execution, the way to choose this parameter is discussed in the paper
      + memory_out: path to the output file
   + output:
      + ***memory_program*** contains two columns, the first column is the BB that initiated the memory access, and the second column is the target address of the access
+ run check.py to relabel all the training data with the help of file ***target_program*** (dead store, silent store and silent load):
   + python check.py --name=program --ptype=option
   + parameter:
      + --name: the program we try to run
      + --ptype: can be gcc-O2, gcc-O3, llvm-O2, and llvm-O3
   + output:
      + ***label_program***: add a label parameter to the first line of each function in ***arg_program*** to indicate whether the function has unnecessary memory operations. 3 ***label_program*** file will be created, corresponding to dead store, silent store and silent load

# Predictor 
Once we have obtained ***label_program***, ***bb_program***, ***adj_program***, ***cg_program***, ***node_program***, and ***memory_program*** via collector, we can start training. 
## scripts
+ train.py: script for training model
+ cfg.py: configuration file
+ preprocessing.py: script for reading all data
+ databox.py: script for data preprocessing
+ loaddata.py: script for batching training data
+ w2v.py: Word2vec model
+ mymodel_data.py: Puffin model
## run steps
+ create 2 folders to store the trained model: data-model/spy and data-model/w2v
+ train and test the model by running train.py
   + python train.py --target=option --data_path=path
   + parameter:
      + --target: can be gcc-O2, gcc-O3, llvm-O2, and llvm-O3
      + --data_path: path to store previously mentioned ***label_program***, ***bb_program***, ***adj_program***, ***cct_program***, ***node_program***, and ***memory_program***
   + output:
      + three parameters: precise, recall, accuracy, which is the prediction result of the model on the test set
      + ***predicted_label*** file: the prediction results for all functions in the test set, including two columns, the first column is the function name and the second column is the predicted label
+ for different training and prediction targets, we will choose the corresponding label files. For example, when we test the prediction accuracy of the model on dead store, we set the path of ***label_program*** to the path of the label file of dead store. At this time, the model uses the label of dead store when training and inferencing.

# Monitor
After get the ***predicted_label*** file for the target program with the help of predictor, we can start performing online monitoring. Since this code is not publicly available, it is not provided here. In this section, we will describe how to run the monitor.
## run steps
+ Move ***predicted_label*** file to the installation directory of Puffin's monitor
+ Run monitor with the following command: ./bin64/drrun -t puffin_monitor -- target_program
+ The monitor will output the PC pairs of the detected dead stores, silent stores and silent loads. The programmer can optimize the program based on the output results

# Benchmarks and tool links
+ Benchmarks: 
+ SPEC CPU 2017: http://www.spec.org/cpu2017
+ Tools: We use four tools to help implement the collector, predictor and monitor :
+ Angr: https://github.com/angr/angr
+ DynamoRIO: https://dynamorio.org
+ Genism: http://radimrehurek.com/gensim
+ Intel xed: https://intelxed.github.io/build-manual/


