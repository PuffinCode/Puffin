# Introduction of RedSparrow

# Description
Redsparrow is divided into three parts. The first part is the static data collector, the second part is the dynamic data collector and the online monitor, the third part is the predictor. Among them, the dynamic data collector and online monitor are switched by modifying the parameters. We will describe in detail how to use these tools to detect inefficiencies in the program.

# Build
Static collector: download and install angr on client machines, copy the code in Redsparrow's monitor_collector/static_collector/ to the directory angr-dev/angr-utils/examples/plot_cfg
Dynamic collector and monitor: install intel xed tool on client machines. Compile the monitor with command: g++ -o redsparrow redsparrow.cpp elf_parser.cpp -I path_of_xed/xed-tool/xed-kit/kits/xed-install-base-2020-12-19-lin-x86-64/include -L path_of_xed/xed-tool/xed-kit/kits/xed-install-base-2020-12-19-lin-x86-64/lib -l xed -g -pthread
Predictor: installing pytorch on the training server and copy the code in Redsparrow's predictor to this machine. Install genism with pip. Start the training of the model by executing python train.py

# Collector
We need to collect both static data and dynamic data for predictor. The static data of each sample contains 3 parts: CG, CFG, raw data of instructions. The dynamic data of each sample contains memory states. In order to perform model training and accuracy evaluation, we also need to collect labels for each sample. The following describes the scripts we use and the process of generating data.

## scripts
+ get_cfg.py: to obtain the CFG of the function and the instructions of each BB through angr
+ cg.py: to obtain the CG for each function with the help of angr
+ check.py: add ground truth labels to all training data
+ redsparrow: to obtain the memory states for each function 
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
+ run redsparrow to get memory states: 
   + ./redsparrow pid binary sample_freq detect_t phase_get_memory memory_out
   + parameter:
      + pid: the pid of the target program running on the server
      + binary: the binary of the target program
      + sample_freq: sample freq of redsparrow, the way to choose this parameter is discussed in the paper
      + detect_t: duration of memory state sampling execution, the way to choose this parameter is discussed in the paper
      + phase_get_memory: this parameter should be set to 1 in order to switch it to collector
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
+ mymodel_data.py: RedSparrow model
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
After get the ***predicted_label*** file for the target program with the help of predictor, we can start performing online monitoring. In this section, we will describe how to run the monitor.
## run steps
+ Obtain the assembly file ***ass_program*** of the target program by objdump
+ Run Puffin with the following command:
./puffin xx  ***predicted_label*** 
   + parameter:
      + xx
      + xx

+ The monitor will output the PC pairs of the detected deadstore. The programmer can optimize the program based on the output results

# Benchmarks and tool links
+ Benchmarks: 
+ SPEC CPU 2017: http://www.spec.org/cpu2017
+ Tools: We use four tools to help implement the collector, predictor and monitor :
+ Angr: https://github.com/angr/angr
+ DynamoRIO: https://dynamorio.org
+ Genism: http://radimrehurek.com/gensim
+ Intel xed: https://intelxed.github.io/build-manual/


