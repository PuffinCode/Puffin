import torch
import torch.nn as nn
import cfg
import numpy as np
import torch.nn.functional as F

class GGNN2(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt,config):
        super(GGNN2, self).__init__()
        self.task_id = 4
        self.hidden_dim = config.call_hidden_dim
        self.annotation_dim = config.call_g_annotation
        self.n_node = config.call_m_size
        self.n_edge = config.n_edge_type
        self.n_output = config.call_out
        self.n_steps = config.call_n_steps

        self.fc_in = nn.Linear(self.hidden_dim, self.hidden_dim * self.n_edge)
        self.fc_out = nn.Linear(self.hidden_dim, self.hidden_dim * self.n_edge)

        self.gated_update = GatedPropagation(self.hidden_dim, self.n_node, self.n_edge)

        self.attention = Attention(self.hidden_dim,self.n_output)

        if self.task_id == 18 or self.task_id == 19:
            self.graph_aggregate =  GraphFeature(self.hidden_dim, self.n_node, self.n_edge, self.annotation_dim)
            self.fc_output = nn.Linear(self.hidden_dim, self.n_output)
        else:
            self.fc1 = nn.Linear(self.hidden_dim+self.annotation_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, 1)
            self.fc3 = nn.Linear(self.hidden_dim, self.n_output)
            self.tanh = nn.Tanh()

    def forward(self, x, a, m):
        '''
        init state x: [batch_size, num_node, hidden_size] , pad zero from annoatation
        annoatation x: [batch_size, num_node, 1] 
        adj matrix m: [batch_size, num_node, num_node * n_edge_types * 2]
        output out: [batch_size, n_label_types], for task 4, 15, 16, n_label_types == num_nodes
        '''
        x, a, m = x.double(), a.double(), m.double()
        all_x = [] # used for task 19, to track 
        for i in range(self.n_steps):
            in_states = self.fc_in(x)
            out_states = self.fc_out(x)
            in_states = in_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge, self.hidden_dim)
            out_states = out_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge, self.hidden_dim)
            x = self.gated_update(in_states, out_states, x, m)
            all_x.append(x)

        if self.task_id == 18:
            output = self.graph_aggregate(torch.cat((x, a), 2))
            output = self.fc_output(output)
        elif self.task_id == 19:
            step1 = self.graph_aggregate(torch.cat((all_x[0], a), 2))
            step1 = self.fc_output(step1).view(-1,1,self.n_output)
            step2 = self.graph_aggregate(torch.cat((all_x[1], a), 2))
            step2 = self.fc_output(step2).view(-1,1,self.n_output)
            output = torch.cat((step1,step2), 1)
        else:
            output = self.fc1(torch.cat((x, a), 2))
            #output = self.attention(torch.cat((x, a), 2))
            output = self.tanh(output)
            print(output.size()) # batch_size,node_size,hidden_size
            #output = self.fc3(output) #.sum(2)
            output= self.attention(output)
            print(output.size()) # batch_size,node_size,output_size
        return output


class Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Attention, self).__init__()
        self.d_k = out_dim
        self.d_v = out_dim 
        self.w_qs = nn.Linear(in_dim, out_dim, bias=False)
        self.w_ks = nn.Linear(in_dim, out_dim, bias=False)
        self.w_vs = nn.Linear(in_dim, out_dim, bias=False)
        
        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)

    def forward(self, x):
        d_k, d_v = self.d_k, self.d_v
        n_head = 1
        sz_b, len_q, len_k, len_v = x.size(0), x.size(1), x.size(1), x.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(x).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        return q

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt,config):
        super(GGNN, self).__init__()
        self.task_id = config.task_id
        self.hidden_dim = config.hidden_dim
        self.annotation_dim = config.annotation_dim
        self.n_node = config.n_node_type
        self.n_edge = config.n_edge_type
        self.n_output = config.n_label_type
        self.n_steps = config.n_steps

        self.fc_in = nn.Linear(self.hidden_dim, self.hidden_dim * self.n_edge)
        self.fc_out = nn.Linear(self.hidden_dim, self.hidden_dim * self.n_edge)

        self.gated_update = GatedPropagation(self.hidden_dim, self.n_node, self.n_edge)

        if self.task_id == 18 or self.task_id == 19:
            self.graph_aggregate =  GraphFeature(self.hidden_dim, self.n_node, self.n_edge, self.annotation_dim)
            self.fc_output = nn.Linear(self.hidden_dim, self.n_output)
        else:
            self.fc1 = nn.Linear(self.hidden_dim+self.annotation_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, 1)
            self.tanh = nn.Tanh()

    def forward(self, x, a, m):
        '''
        init state x: [batch_size, num_node, hidden_size] , pad zero from annoatation
        annoatation x: [batch_size, num_node, 1] 
        adj matrix m: [batch_size, num_node, num_node * n_edge_types * 2]
        output out: [batch_size, n_label_types], for task 4, 15, 16, n_label_types == num_nodes
        '''
        x, a, m = x.double(), a.double(), m.double()
        all_x = [] # used for task 19, to track 
        for i in range(self.n_steps):
            in_states = self.fc_in(x)
            out_states = self.fc_out(x)
            in_states = in_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge, self.hidden_dim)
            out_states = out_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge, self.hidden_dim)
            x = self.gated_update(in_states, out_states, x, m)
            all_x.append(x)

        if self.task_id == 18:
            output = self.graph_aggregate(torch.cat((x, a), 2))
            output = self.fc_output(output)
        elif self.task_id == 19:
            step1 = self.graph_aggregate(torch.cat((all_x[0], a), 2))
            step1 = self.fc_output(step1).view(-1,1,self.n_output)
            step2 = self.graph_aggregate(torch.cat((all_x[1], a), 2))
            step2 = self.fc_output(step2).view(-1,1,self.n_output)
            output = torch.cat((step1,step2), 1)
        else:
            output = self.fc1(torch.cat((x, a), 2))
            output = self.tanh(output)
            output = self.fc2(output).sum(2)
        return output


class GraphFeature(nn.Module):
    '''
    Output a Graph-Level Feature
    '''
    def __init__(self, hidden_dim, n_node, n_edge, n_anno):
        super(GraphFeature, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_node = n_node
        self.n_edge = n_edge
        self.n_anno = n_anno

        self.fc_i = nn.Linear(self.hidden_dim + self.n_anno, self.hidden_dim)
        self.fc_j = nn.Linear(self.hidden_dim + self.n_anno, self.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        '''
        input x: [batch_size, num_node, hidden_size + annotation]
        output x: [batch_size, hidden_size]
        '''
        x_sigm = self.sigmoid(self.fc_i(x))
        x_tanh = self.tanh(self.fc_j(x))
        x_new = (x_sigm * x_tanh).sum(1)

        return self.tanh(x_new)


class GatedPropagation(nn.Module):
    '''
    Gated Recurrent Propagation
    '''
    def __init__(self, hidden_dim, n_node, n_edge):
        super(GatedPropagation, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_node = n_node
        self.n_edge = n_edge

        self.gate_r = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.gate_z = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.trans  = nn.Linear(self.hidden_dim*3, self.hidden_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x_in, x_out, x_curt, matrix):
        matrix_in  = matrix[:, :, :self.n_node*self.n_edge]
        matrix_out = matrix[:, :, self.n_node*self.n_edge:]

        a_in  = torch.bmm(matrix_in, x_in)
        a_out = torch.bmm(matrix_out, x_out)
        a = torch.cat((a_in, a_out, x_curt), 2)

        z = self.sigmoid(self.gate_z(a))
        r = self.sigmoid(self.gate_r(a))

        joint_input = torch.cat((a_in, a_out, r * x_curt), 2)
        h_hat = self.tanh(self.trans(joint_input))
        output = (1 - z) * x_curt + z * h_hat

        return output
        


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None): #64->3
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        #if groups != 1 or base_width != 64:
        #    raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, config, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, #64->3
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.num_classes=config.resnet_out_size
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  #64->3
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=2, padding=0,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0]) #64->3
        #self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
        #                               dilate=replace_stride_with_dilation[0])
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
        #                               dilate=replace_stride_with_dilation[1])
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                               dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.adpmaxpool = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(64 * block.expansion, self.num_classes) #512->3

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)

        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)

        #x = self.avgpool(x)
        #print("before avgpool:",x.size())
        x = self.adpmaxpool(x)
        #print("after avgpool:",x.size())
        x = torch.flatten(x, 1)
        #print("after flatten:",x.size())
        #x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, common_size)
        )
                                            
    def forward(self, x):
        out = self.linear(x)
        return out  


class SPY(nn.Module):
    def __init__(self,opt,config):
        super(SPY, self).__init__()
        self.ggnn = GGNN(opt,config)
        self.ggnn2 = GGNN2(opt,config)
        self.resnet = ResNet(BasicBlock, [3, 0, 0, 0],config)
        self.mlp = MLP(config.mlp_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.mlp_input_size])
        self.ln_data_x = nn.LayerNorm([config.call_m_size,config.call_hidden_dim])
        self.ln_data_a = nn.LayerNorm([config.call_m_size,config.call_g_annotation])
        self.ln_x = nn.LayerNorm([config.n_node_type,config.hidden_dim])
        self.ln_a = nn.LayerNorm([config.n_node_type,config.annotation_dim])

    def forward(self,x,m,m2,a,data_x,data_a,data_m):
        res_out = self.resnet(m) 
        x=self.ln_x(x)
        a=self.ln_a(a)
        ggnn_out = self.ggnn(x,a,m2)

        data_x = self.ln_data_x(data_x)
        data_a = self.ln_data_a(data_a)
        ggnn_data_out_all = self.ggnn2(data_x,data_a,data_m)
        ggnn_data_out = ggnn_data_out_all[:,0]
        out = torch.cat((res_out,ggnn_out),1)
        out = torch.cat((out,ggnn_data_out),1)
        
        out = self.ln(out)
        out = self.mlp(out)
        out = torch.sigmoid(out)
        out = out.view(-1)
        return out
        


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, _input, target):
        #pt = torch.sigmoid(_input)
        pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
            (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



class word2vec(nn.Module):
    def __init__(self,opt,config):
        super(word2vec, self).__init__()
        self.mlp = MLP(config.mlp_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.mlp_input_size])
        #self.ln = nn.GroupNorm(config.mlp_input_size,config.mlp_input_size,eps=1e-20)

    def forward(self,x,m,m2,a,data_x,data_a,data_m):
        a = a.view(a.size(0),-1)

        out = a
        out = self.ln(out)
        out = self.mlp(out)
        out = torch.sigmoid(out)
        out = out.view(-1)
        return out
 

class GGNN_memory(nn.Module):
    def __init__(self,opt,config):
        super(GGNN_memory, self).__init__()
        self.ggnn2 = GGNN2(opt,config)
        self.mlp = MLP(config.mlp_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.mlp_input_size])
        self.ln_data_x = nn.LayerNorm([config.call_m_size,config.call_hidden_dim])
        self.ln_data_a = nn.LayerNorm([config.call_m_size,config.call_g_annotation])

    def forward(self,x,m,m2,a,data_x,data_a,data_m):
        data_x = self.ln_data_x(data_x)
        data_a = self.ln_data_a(data_a)
        ggnn_data_out_all = self.ggnn2(data_x,data_a,data_m)
        ggnn_data_out = ggnn_data_out_all[:,0]

        out = ggnn_data_out
        
        out = self.ln(out)
        out = self.mlp(out)
        out = torch.sigmoid(out)
        out = out.view(-1)
        return out
        

class in_func_model(nn.Module):
    def __init__(self,opt,config):
        super(in_func_model, self).__init__()
        self.ggnn = GGNN(opt,config)
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2],config)
        self.mlp = MLP(config.mlp_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.mlp_input_size])
        self.ln_x = nn.LayerNorm([config.n_node_type,config.hidden_dim])
        self.ln_a = nn.LayerNorm([config.n_node_type,config.annotation_dim])

    def forward(self,x,m,m2,a,data_x,data_a,data_m):
        res_out = self.resnet(m) 
        x=self.ln_x(x)
        a=self.ln_a(a)
        ggnn_out = self.ggnn(x,a,m2)
        out = torch.cat((res_out,ggnn_out),1)

        out = self.ln(out)
        out = self.mlp(out)
        out = torch.sigmoid(out)
        out = out.view(-1)
        return out
        

class semantic(nn.Module):
    def __init__(self,opt,config):
        super(semantic, self).__init__()
        self.ggnn = GGNN(opt,config)
        self.mlp = MLP(config.mlp_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.mlp_input_size])
        self.ln_x = nn.LayerNorm([config.n_node_type,config.hidden_dim])
        self.ln_a = nn.LayerNorm([config.n_node_type,config.annotation_dim])

    def forward(self,x,m,m2,a,data_x,data_a,data_m):
        x=self.ln_x(x)
        a=self.ln_a(a)
        ggnn_out = self.ggnn(x,a,m2)

        out = ggnn_out
        
        out = self.ln(out)
        out = self.mlp(out)
        out = torch.sigmoid(out)
        out = out.view(-1)
        return out
        



class Resnet11(nn.Module):
    def __init__(self,opt,config):
        super(Resnet11, self).__init__()
        self.resnet = ResNet(BasicBlock, [3, 0, 0, 0],config)
        self.mlp = MLP(config.mlp_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.mlp_input_size])
        self.ln_x = nn.LayerNorm([config.n_node_type,config.hidden_dim])
        self.ln_a = nn.LayerNorm([config.n_node_type,config.annotation_dim])

    def forward(self,x,m,m2,a,data_x,data_a,data_m):
        res_out = self.resnet(m) 
        out = res_out
        
        out = self.ln(out)
        out = self.mlp(out)
        out = torch.sigmoid(out)
        out = out.view(-1)
        return out
        
class Resnet7(nn.Module):
    def __init__(self,opt,config):
        super(Resnet7, self).__init__()
        self.resnet = ResNet(BasicBlock, [1, 0, 0, 0],config)
        self.mlp = MLP(config.mlp_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.mlp_input_size])
        self.ln_x = nn.LayerNorm([config.n_node_type,config.hidden_dim])
        self.ln_a = nn.LayerNorm([config.n_node_type,config.annotation_dim])

    def forward(self,x,m,m2,a,data_x,data_a,data_m):
        res_out = self.resnet(m) 
        out = res_out
        
        out = self.ln(out)
        out = self.mlp(out)
        out = torch.sigmoid(out)
        out = out.view(-1)
        return out
 


class CNN3(nn.Module):
    def __init__(self,opt,config):
        super(CNN3,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,stride=2,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(4096,config.resnet_out_size)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = MLP(config.mlp_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.mlp_input_size])


    def forward(self,x,m,m2,a,data_x,data_a,data_m):
        out = self.relu(self.conv1(m))
        out = self.pool(out)        
        out = self.relu(self.conv2(out))
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        
        out = self.ln(out)
        out = self.mlp(out)
        out = torch.sigmoid(out)
        out = out.view(-1)
        return out
 
class SPY_noresnet(nn.Module):
    def __init__(self,opt,config):
        super(SPY_noresnet, self).__init__()
        self.ggnn = GGNN(opt,config)
        self.ggnn2 = GGNN2(opt,config)
        self.mlp = MLP(config.mlp_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.mlp_input_size])
        self.ln_data_x = nn.LayerNorm([config.call_m_size,config.call_hidden_dim])
        self.ln_data_a = nn.LayerNorm([config.call_m_size,config.call_g_annotation])
        self.ln_x = nn.LayerNorm([config.n_node_type,config.hidden_dim])
        self.ln_a = nn.LayerNorm([config.n_node_type,config.annotation_dim])

    def forward(self,x,m,m2,a,data_x,data_a,data_m):
        x=self.ln_x(x)
        a=self.ln_a(a)
        ggnn_out = self.ggnn(x,a,m2)

        data_x = self.ln_data_x(data_x)
        data_a = self.ln_data_a(data_a)
        ggnn_data_out_all = self.ggnn2(data_x,data_a,data_m)
        ggnn_data_out = ggnn_data_out_all[:,0]
        out = torch.cat((ggnn_out,ggnn_data_out),1)
        
        out = self.ln(out)
        out = self.mlp(out)
        out = torch.sigmoid(out)
        out = out.view(-1)
        return out
        

