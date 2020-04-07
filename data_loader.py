import torch
import re
from math import floor as floor
import numpy as np

def read_data(file_name):
        with open(file_name, 'r') as f:
            corpus = f.read().lower()
    
        corpus = re.sub(r"<unk>", "unk", corpus)
        
        return corpus.split()

def get_dicts(sentences):
        s=list(set(sentences))
        
        word_to_int={}
        int_to_word={}
        char_to_int={}
        int_to_char={}
        
        for i in range(len(s)):
                word_to_int[s[i]]=i
                int_to_word[i]=s[i]
        
        k=set()
        ma=0
        
        for word in s:
                ma=max(ma,len(word))
                for character in word:
                        k.add(character)
        k=list(k)

        for i in range(len(k)):
                int_to_char[i+1]=k[i]
                char_to_int[k[i]]=i+1
                
        char_to_int['SOW']=len(k)+1
        int_to_char[len(k)+1]='SOW'
        
        char_to_int['EOW']=len(k)+2
        int_to_char[len(k)+2]='EOW'
        
        char_to_int['PAD']=0
        int_to_char[0]='PAD'
                
        return word_to_int,int_to_word,char_to_int,int_to_char,len(s),len(k)+3,ma
   
def get_training_data(sentences,char_to_int,word_to_int,max_word_len,batch_size,seq_size):
        k=0
        final=[]
        final_gold=[]
        
        for _ in range(floor(len(sentences)/(batch_size*seq_size))):
                l=[]
                l_gold=[]
                
                for _ in range(batch_size):
                        s=[]
                        s_gold=[]
                        
                        for _ in range(seq_size):
                                m=[]
                                s_gold.append(word_to_int[sentences[k+1]])
                                
                                m.append(char_to_int['SOW'])
                                
                                for i in range(len(sentences[k])):
                                        m.append(char_to_int[sentences[k][i]])
                                        
                                m.append(char_to_int['EOW'])
                                
                                o=len(m)
                                
                                for  i in range(max_word_len+2-o):
                                        m.append(char_to_int['PAD'])
                                        
                                s.append(m)
                                
                                k+=1
                                
                        l.append(s)
                        l_gold.append(s_gold)
                        
                final.append(l)
                final_gold.append(l_gold)
                
        return np.array(final),np.array(final_gold)


class DataLoader(object):
        def __init__(self,data_x,data_y,batch_size,seq_len):
                self.x_len=data_x.shape[0]
                self.counter=0
                
                self.batch_size=batch_size
                self.seq_len=seq_len
                 
                self.data_x=data_x
                self.data_y=data_y
                
        def load_train_batch(self):
                
                x,y=torch.from_numpy(self.data_x[self.counter]),torch.from_numpy(self.data_y[self.counter])
                y=y.view(self.batch_size*self.seq_len)
                self.counter=(self.counter+1)%self.x_len
                
                return x,y