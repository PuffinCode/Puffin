import torch.utils.data as data
import torch

class MyDataset(data.Dataset):
    #def __init__(self, adj_matrix,adj_matrix2, annotation, labels, real_node_size):
    def __init__(self, adj_matrix,adj_matrix2, annotation, labels, adj_call_matrix, data_eb,func_name,benchmark_name):
        self.adj_matrix = adj_matrix
        self.adj_matrix2 = adj_matrix2
        self.annotation = annotation
        self.labels = labels
        self.adj_call_matrix = adj_call_matrix
        self.data_eb = data_eb
        self.func_name = func_name
        self.benchmark_name = benchmark_name

    def __getitem__(self, index):#返回的是tensor
        #adj_matrix, adj_matrix2, annotation, labels, real_node_size = self.adj_matrix[index], self.adj_matrix2[index], self.annotation[index], self.labels[index], self.real_node_size[index]
        adj_matrix, adj_matrix2, annotation, labels, adj_call_matrix, data_eb, func_name,benchmark_name = self.adj_matrix[index], self.adj_matrix2[index], self.annotation[index], self.labels[index],self.adj_call_matrix[index],self.data_eb[index],self.func_name[index],self.benchmark_name[index]
        #return adj_matrix, adj_matrix2, annotation, labels, real_node_size
        return adj_matrix, adj_matrix2, annotation, labels, adj_call_matrix, data_eb,func_name,benchmark_name


    def __len__(self):
        return len(self.labels)

