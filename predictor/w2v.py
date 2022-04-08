from gensim.models import word2vec
import pandas as pd
import numpy as np
import cfg as config

this_model=None
data_model=None

def get_dataset_vec(dataset,n_dim,path,save_model,tag):
    w2v_model = word2vec.Word2Vec(dataset, sg=1, size=n_dim, min_count=1, hs=0)  # 初始化模型并训练
    # 在测试集上训练
    # w2v_model.train(x_test,total_examples=w2v_model.corpus_count,epochs=w2v_model.iter) #追加训练模型
    # 将imdb_w2v模型保存，训练集向量，测试集向量保存到文件
    # print(w2v_model['会议'])
    
    print("train:",tag)
    if tag=="instructions":
        this_model = w2v_model
    else:
        data_model = w2v_model
    if save_model:
       w2v_model.save(path)  # 保存训练结果
    #w2v_model.wv.save_word2vec_format('data/w2v/w2v_model_18.pkl')
    #print("word:",w2v_model.wv.index2word) #输出单词表
    return w2v_model

def print_vocab(model):
    print("word:",model.wv.index2word)

def load_model(path,tag):
    #w2v_model = word2vec.Word2Vec.load(path)
    #print("word:",w2v_model.wv.index2word)
    print("w2v:load ",tag)
    if tag=="instructions":
        #w2v_model=this_model
                w2v_model = word2vec.Word2Vec.load(path)
    else:
        #w2v_model = data_model
                w2v_model = word2vec.Word2Vec.load(path)
    return w2v_model

#对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(sentence,size,w2v_model):
     sen_vec=np.zeros(size,dtype=float).reshape((1,size))
     count=0
     for word in sentence:
        try:
            sen_vec+=w2v_model[word].reshape((1,size))
            count+=1
        except KeyError:
            #print("KeyError:",word)
            continue
     if count!=0:
        sen_vec/=count
     return sen_vec                                


def print_vocab(w2v_model):
    vocab = w2v_model.vocab
    print(vocab)
