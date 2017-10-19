# -*- coding: utf-8 -*-
"""
Other Layers.
Reference: Bilateral Multi-Perspective Matching for Natural Language Sentences.

Sheng Liu
All rights reserved
Report bugs to ShengLiu shengliu@nyu.edu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .MultiPerspective import MultiPerspective

class Charrepresentation(nn.Module):
    '''
    First Layer. Feeding each character (represented as a character embedding) within a word into a LSTM.
    Input:
         character batch, size: batch_size * character vector size of a sentence
         (character vector size = nb_chars * sequence_length)
    Output size: batch_size * sequence_length * rnn_dim
    
    '''
    def __init__(self, sequence_length, nb_chars, nb_per_word, 
                    embedding_dim, rnn_dim, rnn_layers, rnn_unit = 'gru', dropout = 0.1):
        super(Charrepresentation,self).__init__()
        self.sequence_length = sequence_length
        self.nb_chars = nb_chars
        self.nb_per_word = nb_per_word
        #print("nb_per_word", nb_per_word)
        self.embedding_dim = embedding_dim
        self.rnn_dim = rnn_dim
        if rnn_unit == 'gru':
            self.rnn = nn.GRU(embedding_dim, rnn_dim, rnn_layers,
                            bias = False,
                            batch_first = True,
                            dropout = dropout,
                            bidirectional = False)
        if rnn_unit == 'lstm':
            self.rnn = nn.lstm(embedding_dim, rnn_dim, rnn_layers,
                            bias = False,
                            batch_first = True,
                            dropout = dropout,
                            bidirectional = False)
        

        self.embedding = nn.Embedding(nb_chars, embedding_dim)
        #self.rnn = rnn_unit(embedding_dim, rnn_dim, rnn_layers)

    def _collapse_input(self, x, nb_per_word=0):
        x = x.view(-1, nb_per_word)
        return x

    def _unroll_input(self, x, sequence_length=0, rnn_dim=0):
        #print(rnn_dim* self.nb_per_word)
        #print(sequence_length)
        #print(x.size())
        x = x.view(-1 , sequence_length, rnn_dim)
        return x

    def forward(self, x):
        #print(self.nb_per_word)
        #print(x.size())
        out = self._collapse_input(x, self.nb_per_word)
        #print(out.size())
        out = self.embedding(out)
        #print(out.size())
        out, _ = self.rnn(out)
        out = out[:,-1,:]
        #print(out.size())
        #ans = self.rnn_dim * self.nb_per_word
        return self._unroll_input(out.contiguous(), self.sequence_length, self.rnn_dim)

class ContextRepresentation(nn.Module):
    '''
    Second Layer, bidirectional rnn.
    Input size: batch_size * sequence_length * (word_embedding size + rnn_dim)
    output size: batch_size * sequence_length * (2 * hidden_size)
    '''
    def __init__(self, input_size, hidden_size = 100, rnn_unit = 'gru', dropout = 0.1):
        super(ContextRepresentation, self).__init__()
        if rnn_unit == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, 
                            bias = False,
                            batch_first = True,
                            dropout = dropout,
                            bidirectional = True)
        if rnn_unit == 'lstm':
            self.rnn = nn.lstm(input_size, hidden_size,
                            bias = False,
                            batch_first = True,
                            dropout = dropout,
                            bidirectional = True)
    def forward(self, x):
        out = self.rnn(x)

        return out


class PredictionLayer(nn.Module):
    def __init__(self, pre_in, hidden_size, pre_out, dropout = 0.1):
        super(PredictionLayer, self).__init__()
        self.linear1 = nn.Linear(pre_in,hidden_size)
        self.linear2 = nn.Linear(hidden_size,pre_out)
        self.dropout = nn.Dropout(p = dropout)
        self.softmax = nn.Softmax()

    def forward(self,x):
        out = self.dropout(self.linear2(self.linear1(x)))

        return self.softmax(out)




class BiMPM(nn.Module):
    def __init__(self, embedding_dim, sequence_length, nb_chars, nb_per_word, char_embedding_dim, \
                 rnn_dim, rnn_layers, perspective, hidden_size = 100, epsilon = 1e-6, num_classes = 2, rnn_unit = 'gru'):
        super(BiMPM,self).__init__()
        self.hidden_size = hidden_size
        self.rnn_unit = rnn_unit
        self.char_representation = Charrepresentation(sequence_length, nb_chars, \
            nb_per_word, char_embedding_dim, rnn_dim, rnn_layers)
        self.contex_rep = ContextRepresentation(embedding_dim, hidden_size)
        self.multiperspective = MultiPerspective(hidden_size, epsilon, perspective)
        self.aggregation = ContextRepresentation(4 * perspective, hidden_size)
        self.pre = PredictionLayer(4 * hidden_size, hidden_size, num_classes)





    def forward(self, x1, x2 ,y1, y2, hidden_size):
        out1 = self.char_representation(y1)
        out2 = self.char_representation(y2)
        out1 = torch.cat([x1,out1], 2)
        out2 = torch.cat([x2,out2], 2)
        #print(out1.size())
        out1,_ = self.contex_rep(out1)
        out2,_ = self.contex_rep(out2)
        out3 = self.multiperspective(out1, out2)
        out4 = self.multiperspective(out2, out1)
        out3,_ = self.aggregation(out3)
        out4,_ = self.aggregation(out4)
        #timestep x batch x (2*hidden_size)
        pre_list = []
        pre_list.append(out3[:,-1,:hidden_size])
        pre_list.append(out3[:,0,hidden_size:])
        pre_list.append(out4[:,-1,:hidden_size])
        pre_list.append(out4[:,0,:hidden_size])
        pre1 = torch.cat(pre_list,-1)
        # batch x (4*hidden_size)
        out = self.pre(pre1)

        return out
        
