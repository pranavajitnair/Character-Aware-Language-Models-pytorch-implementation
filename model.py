import torch
import torch.nn as nn
import torch.nn.functional as F


class Classfier(nn.Module):
        def __init__(self,filter_size,vocab_size,hidden_size,batch_size,seq_len):
                super(Classfier,self).__init__()
                self.seq_len=seq_len
                self.batch_size=batch_size
                
                self.lstm=nn.LSTM(filter_size,hidden_size,num_layers=2,bidirectional=True,batch_first=True,dropout=0.5,bias=True)
                
                self.Dense=nn.Linear(hidden_size*2,vocab_size)
                
                self.dropout=nn.Dropout()
                
        def forward(self,input):
                output,hidden=self.lstm(input,None)
                
                output=self.dropout(output)
                output=self.Dense(output).contiguous().view(self.batch_size*self.seq_len,-1)
                
                return output

            
class HighwayNetwork(nn.Module):
        def __init__(self,size):
                super(HighwayNetwork,self).__init__()
                
                self.Dense=nn.Linear(size,size)
                self.Transform=nn.Linear(size,size)
                
        def forward(self,input):
                output=F.relu(self.Dense(input))
                transform_gate_output=torch.sigmoid(self.Transform(input))
                
                return output*transform_gate_output+input*(1-transform_gate_output)

            
class ConvolutionModel(nn.Module):
        def __init__(self,embedding_size,char_size,max_len,vocab_size,batch_size,seq_len,hidden_size):
                super(ConvolutionModel,self).__init__()
                self.max_word_len=max_len
                self.embedding_size=embedding_size
                
                self.seq_len=seq_len
                self.batch_size=batch_size
                
                self.embeddings=nn.Embedding(char_size,embedding_size)
                
                self.conv1=nn.Conv2d(1,25,(1,embedding_size))
                self.conv2=nn.Conv2d(1,50,(2,embedding_size))
                self.conv3=nn.Conv2d(1,75,(3,embedding_size))
                self.conv4=nn.Conv2d(1,100,(4,embedding_size))
                self.conv5=nn.Conv2d(1,125,(5,embedding_size))
                self.conv6=nn.Conv2d(1,150,(6,embedding_size))
                
                self.maxpool1=nn.MaxPool2d((max_len-1+1,1))
                self.maxpool2=nn.MaxPool2d((max_len-2+1,1))
                self.maxpool3=nn.MaxPool2d((max_len-3+1,1))
                self.maxpool4=nn.MaxPool2d((max_len-4+1,1))
                self.maxpool5=nn.MaxPool2d((max_len-5+1,1))
                self.maxpool6=nn.MaxPool2d((max_len-6+1,1))
                
                self.highway=HighwayNetwork(525)
                
                self.classifier=Classfier(525,vocab_size,hidden_size,batch_size,seq_len)
                
                self.batchnorm=nn.BatchNorm1d(525,affine=False)
                
        def forward(self,input):
                input=self.embeddings(input).view(self.batch_size*self.seq_len,1,self.max_word_len,self.embedding_size)
                
                output1=torch.tanh(self.conv1(input))
                output1=self.maxpool1(output1).squeeze()
                
                output2=torch.tanh(self.conv2(input))
                output2=self.maxpool2(output2).squeeze()
                
                output3=torch.tanh(self.conv3(input))
                output3=self.maxpool3(output3).squeeze()
                
                output4=torch.tanh(self.conv4(input))
                output4=self.maxpool4(output4).squeeze()
                
                output5=torch.tanh(self.conv5(input))
                output5=self.maxpool5(output5).squeeze()
                
                output6=torch.tanh(self.conv6(input))
                output6=self.maxpool6(output6).squeeze()
                
                output=torch.cat((output1,output2,output3,output4,output5,output6),dim=1)
                
                output=self.batchnorm(output).contiguous().view(self.batch_size,-1,525)
                
                output=self.highway(output)
                output=self.classifier(output)
                
                return output