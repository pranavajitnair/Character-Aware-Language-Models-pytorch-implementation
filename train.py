import torch.nn as nn
import torch.optim as optim
import numpy as np

import os

from model import ConvolutionModel
from data_loader import read_data,DataLoader,get_dicts,get_training_data

def train(epochs,model,dataloader_train,dataloader_valid,iterations_valid,iterations_train):
        learning_rate=1.0
        lossFunction=nn.CrossEntropyLoss()
        
        best_perplexity=1000
        old_perplexity=10000
         
        for epoch in range(epochs):
                model.eval()
                
                valid_loss=[]
                perplexities=[]
               
                for _ in range(iterations_valid):
                        x,y=dataloader_valid.load_train_batch()
                        output=model(x)
                        
                        loss=lossFunction(output,y)
                        valid_loss.append(loss.item())
                        perplexities.append(np.exp(loss.item()))
                
                perplexity=np.mean(perplexities)
                validation_loss=np.mean(valid_loss)
                
                if old_perplexity-perplexity<=1.0:
                        learning_rate/=2
                        
                if best_perplexity>perplexity:
                        best_perplexity=perplexity
                        
                old_perplexity=perplexity
                
                model.train()
                
                training_losses=[]
                
                for _ in range(iterations_train):
                        optimizer=optim.SGD(model.parameters(),lr=learning_rate)
                        optimizer.zero_grad()
                        
                        x_batch,y_batch=dataloader_train.load_train_batch()
                        output=model(x_batch)
                        
                        loss=lossFunction(output,y_batch)
                        training_losses.append(loss.item())
                        
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)
                        optimizer.step()
                        
                training_loss=np.mean(training_losses)
                print('epoch=',epoch+1,'training loss=',training_loss,'validation loss=',validation_loss,'learning rate=',learning_rate)
                        

sentences_train=read_data(os.getcwd()+'/train.txt')
sentences_test=read_data(os.getcwd()+'/test.txt')
sentences_validation=read_data(os.getcwd()+'/valid.txt')

batch_size=20
seq_size=35

char_dim=15
hidden_size=300

epochs=400
iterations_valid=13
iterations_train=23

word_to_int,int_to_word,char_to_int,int_to_char,vocab_size,n_chars,ma=get_dicts(sentences_train+sentences_test+sentences_validation)

data_x,data_y=get_training_data(sentences_train,char_to_int,word_to_int,ma,batch_size,seq_size)
dataLoader_train=DataLoader(data_x,data_y,batch_size,seq_size)

#test_x,test_y=get_training_data(sentences_test,char_to_int,word_to_int,ma,batch_size,seq_size)
#dataLoader_test=DataLoader(test_x,test_y,batch_size,seq_size)

valid_x,valid_y=get_training_data(sentences_validation,char_to_int,word_to_int,ma,batch_size,seq_size)
dataLoader_valid=DataLoader(valid_x,valid_y,batch_size,seq_size)

model=ConvolutionModel(char_dim,n_chars,ma+2,vocab_size,batch_size,seq_size,hidden_size)

train(epochs,model,dataLoader_train,dataLoader_valid,iterations_valid,iterations_train)