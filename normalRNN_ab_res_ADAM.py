# Architecture for network without bayesian sgld
# Hyper Parameters
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
nb_tokens = 20
num_classes = 15
batch_size = 64
hidden_size = 64
num_layers = 3
d_a = 100
r = 30

# BiRNN Model (Many-to-One)
class SelfAttentionRNNnormal(nn.Module):
    def __init__(self, hidden_size, num_layers, nb_tokens, embedding_dim, num_classes):
        super(SelfAttentionRNNnormal, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(nb_tokens, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                            batch_first=True, dropout = 0.4, bidirectional=True)#dr=0.3before
        self.linear_first = nn.Linear(hidden_size * 2, d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a, r)
        self.linear_second.bias.data.fill_(0)
        self.batch_size = batch_size
        self.hidden_state = self.init_hidden()
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection 
    
    def init_hidden(self):
        h0 = Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size)).cuda() # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size)).cuda()
        return (h0, c0)
        
    def softmax(self,input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d, dim = 1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)

    def forward(self, x):
        emb = self.embed(x)
        outputs, self.hidden_state = self.lstm(emb, self.hidden_state)
        
        x = F.tanh(self.linear_first(outputs))
        dr = nn.Dropout(p = 0.4)
        x = dr(x)
        x = self.linear_second(x)
        x = self.softmax(x,1)       
        attention = x.transpose(1,2)       
        sentence_embeddings = attention @ outputs       
        avg_sentence_embeddings = torch.sum(sentence_embeddings,1) / r
        
        #print (avg_sentence_embeddings.size())

        # Log-softmax is for training
        outputs = F.log_softmax(self.fc(avg_sentence_embeddings), dim = 1)


        #outputs = F.softmax(self.fc(avg_sentence_embeddings), dim = 1)
        #print (attention.size())
        return outputs, attention