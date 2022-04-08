# config.py maintains those hyperparameters and information that don't change 
import sys

w2v_model_path = 'data/w2v/w2v_model.pkl'
data_w2v_model_path = 'data/w2v/data_w2v_model.pkl'

w2v_model=None
data_model=None

class CONFIG():
    def __init__(self,opt):
        self.cuda = True
        self.gpu = opt.gpu
        self.no_memory = opt.no_memory
        self.load_test = opt.load_test
        self.epoch = opt.epoch
        self.batch_size = opt.batch_size
        self.lr = opt.lr
        self.target_start = 0
        self.max_recall = 0
        self.max_pre = 0
        self.stop_epoch = 0
        self.maxRP = 0
        self.lossf = opt.lossf
        self.drop_rate = opt.drop_rate
        self.copy_rate = opt.copy_rate
        self.validation_loss_min=1000
        self.valid_stop_down=0
        self.model_type = opt.model_type
        self.save_model = opt.save_model
        self.accuracy_max = 0
        self.accuracy_stop_raise = 0

        #arg for data GGNN
        self.call_m_size=opt.call_m_size
        self.call_g_annotation=opt.call_g_annotation
        self.call_hidden_dim=opt.call_hidden_dim
        self.call_out=opt.call_out
        self.call_n_steps=opt.call_steps
        
        #arg for GGNN
        self.annotation_dim=opt.annotation
        self.n_node_type=opt.node_cnt
        self.n_edge_type=1
        self.n_label_type=opt.ggnn_out #for task 18
        self.task_id=18
        self.hidden_dim=opt.hidden_dim
        self.n_steps=5
        #arg for resnet
        self.resnet_out_size= 64 #opt.resnet_out #num of classes

        #arg for MLP
        if self.model_type=="resnet" or self.model_type=="resnet7" or self.model_type=="resnet11" or self.model_type=="CNN3":
            self.mlp_input_size=self.resnet_out_size
        elif self.model_type=="w2v":
            self.mlp_input_size=self.annotation_dim*self.n_node_type
        elif self.model_type=="w2v+ggnn1":
            self.mlp_input_size=self.n_label_type
        elif self.model_type=="ggnn2":
            self.mlp_input_size=self.call_out
        elif self.model_type=="w2v+ggnn1+resnet":
            self.mlp_input_size=self.n_label_type + self.resnet_out_size
        elif self.model_type=="no_resnet":
            self.mlp_input_size=self.n_label_type + self.call_out
        elif self.model_type=="all":
            self.mlp_input_size=self.resnet_out_size + self.n_label_type + self.call_out
        
        self.common_size=1


        if self.annotation_dim > self.hidden_dim:
            print("ERROR: annotation > hidden_dim")
            sys.exit()

    def reset_model(self):
        self.target_start = 0
        self.max_recall = 0
        self.max_pre = 0
        self.stop_epoch = 0
        self.maxRP = 0
        self.validation_loss_min=1000
        self.valid_stop_down=0
        self.accuracy_max = 0
        self.accuracy_stop_raise = 0
 
        #arg for MLP
        if self.model_type=="resnet" or self.model_type=="resnet7" or self.model_type=="resnet11" or self.model_type=="CNN3":
            self.mlp_input_size=self.resnet_out_size
        elif self.model_type=="w2v":
            self.mlp_input_size=self.annotation_dim*self.n_node_type
        elif self.model_type=="w2v+ggnn1":
            self.mlp_input_size=self.n_label_type
        elif self.model_type=="ggnn2":
            self.mlp_input_size=self.call_out
        elif self.model_type=="w2v+ggnn1+resnet":
            self.mlp_input_size=self.n_label_type + self.resnet_out_size
        elif self.model_type=="no_resnet":
            self.mlp_input_size=self.n_label_type + self.call_out
        elif self.model_type=="all":
            self.mlp_input_size=self.resnet_out_size + self.n_label_type + self.call_out
        

    def print_c(self):
        print("gpu:",self.gpu)
        print("load_test:",self.load_test)
        print("lossf:",self.lossf)
        print("drop_rate:",self.drop_rate)
        print("copy_rate:",self.copy_rate)

        print("batch_size:",self.batch_size)
        print("lr:",self.lr)
        print("call_matrix_size:",self.call_m_size)
        print("call_g_annotation:",self.call_g_annotation)
        print("call_hidden_dim:",self.call_hidden_dim)
        print("call_out:",self.call_out)
        print("call_n_steps:",self.call_n_steps)
        print("semantic ggnn annotation:",self.annotation_dim)
        print("semantic ggnn node_cnt:",self.n_node_type)
        print("semantic ggnn out_size:",self.n_label_type)
        print("semantic ggnn hidden_dim:",self.hidden_dim)
        print("semantic ggnn steps:",self.n_steps)
        print("resnet_out_size:",self.resnet_out_size)
        print("common_size(model output size):",self.common_size)
        print("opt:Adam")
