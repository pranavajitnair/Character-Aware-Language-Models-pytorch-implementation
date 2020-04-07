import torch.nn as nn
import torch.optim as optim
import numpy as np

import os

from model import ConvolutionModel
from data_loader import read_data,DataLoader,get_dicts,get_training_data

def train(epochs,dataloader,model):
#        learning_rate=1.0
        lossFunction=nn.CrossEntropyLoss()
        
#        perplexity_best=1000.0
        
        for epoch in range(epochs):
            
#                x_batch,y_batch=dataloader.load_train_batch()
#                output=model(x_batch)
#                
#                loss=lossFunction(output,y_batch)
#                perplexity=np.exp(loss.item())
#                
#                if perplexity_best-perplexity<=1.0:
#                        learning_rate/=2
#                        
#                perplexity_best=min(perplexity_best,perplexity)
            
                optimizer=optim.SGD(model.parameters(),lr=0.3)
                optimizer.zero_grad()
                
                x_batch,y_batch=dataloader.load_train_batch()
                output=model(x_batch)
                
                loss=lossFunction(output,y_batch)
                print(epoch,loss.item())
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)
                optimizer.step()

sentences=read_data(os.getcwd()+'/train.txt')

batch_size=100
seq_size=50
char_dim=15
hidden_size=300

word_to_int,int_to_word,char_to_int,int_to_char,vocab_size,n_chars,ma=get_dicts(sentences)
data_x,data_y=get_training_data(sentences,char_to_int,word_to_int,ma,batch_size,seq_size)

dataLoader=DataLoader(data_x,data_y,batch_size,seq_size)
epochs=400

model=ConvolutionModel(char_dim,n_chars,ma+2,vocab_size,batch_size,seq_size,hidden_size)

train(epochs,dataLoader,model)