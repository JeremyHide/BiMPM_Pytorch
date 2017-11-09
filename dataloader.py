
#from torchvision import transforms
#import torchvision
#from multi_perspective import MultiPerspective
'''
# (some of the) hyper parameters
learning_rate = 0.001
num_epochs = 5 # number epoch to train
batch_size = 32
max_char = 10

char_embedding_dim = 20
word_embedding_dim = 300
char_rnn_dim = 50
rnn_layers = 1
'''
# I/O Param
#data_dir = "Quora_question_pair_partition/"
#train_dir = os.path.join(data_dir, "train.tsv")
#dev_dir = os.path.join(data_dir, "dev.tsv")
#test_dir = os.path.join(data_dir, "test.tsv")
#vec_path = "Quora_question_pair_partition/wordvec.txt"
#PADDING_IDX = 0

'''
Get data_loader for train, dev, test set
arg:
    vec_path - file path of wordvec
return:
    train_loader - 
    validation_loader - 
    test_loader - 
'''
import numpy as np
import os
import torch
from collections import Counter
from sklearn.feature_extraction import stop_words
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
def data_loader(train_dir, dev_dir, test_dir, vec_path, batch_size, max_char, max_sequence_length):
    voc = get_word_lists(vec_path)
    wordvecs, word2id = embedding(vec_path , voc)
    train_set, chars = construct_dataset(train_dir , [])
    validation_set, chars  = construct_dataset(dev_dir, chars)
    test_set, chars = construct_dataset(test_dir, chars)
    len_chars = len(chars)
    train_data = process_text_dataset(train_set, word2id, wordvecs, chars,max_char)
    validation_data = process_text_dataset(validation_set, word2id, wordvecs, chars, max_char)
    test_data = process_text_dataset(test_set, word2id, wordvecs, chars, max_char)
    # consturct datasets
    #quora_train = QuoraDataset(train_data)
    #quora_validation = QuoraDataset(validation_data)
    #quora_test = QuoraDataset(test_data)    
    quora_train = QuoraDataset(train_data, max_sequence_length, max_char)
    quora_validation = QuoraDataset(validation_data, max_sequence_length, max_char)
    quora_test = QuoraDataset(test_data, max_sequence_length, max_char)
        
    # construct data loader
    train_loader = torch.utils.data.DataLoader(dataset=quora_train, 
                                               batch_size=batch_size,
                                               collate_fn=quora_collate_func,
                                               shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=quora_validation, 
                                               batch_size=batch_size, 
                                               collate_fn=quora_collate_func,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=quora_test,  
                                               batch_size=batch_size,
                                               collate_fn=quora_collate_func,
                                               shuffle=False)
    return (train_loader, validation_loader, test_loader, len_chars,len(quora_train),len(quora_validation))

'''
Get embeddingof 300-dimensional GloVe word vectors pretrained from the 840B Common Crawl corpus [Pennington
et al., 2014].
args:
    vec_path - file path of wordvec
return:
    300-dimensional word embedding
'''
def embedding(vec_path, voc):
    ## word2id : word to its corresponding index
    ## id2word : index to its corresponding word
    # don't need to run for every code change
    word2id = {}
    id2word = {}
    vec_file = open(vec_path, 'rt')
    word_vecs = {}
    for line in vec_file:
        line = line.strip()
        parts = line.split(' ')
        word = parts[0]
        word_dim = len(parts[1:])
        if (word_dim > 300):
            parts = parts[-301:]
        vector = np.array(parts[1:], dtype='float32')
        cur_index = len(word2id)
        word2id[word] = cur_index 
        id2word[cur_index] = word
        word_vecs[cur_index] = vector
    vec_file.close()
    
    vocab_size = len(word2id)
    wordvecs = np.zeros((vocab_size+1, word_dim), dtype=np.float32) # the last dimension is all zero
    for cur_index in range(vocab_size):
        wordvecs[cur_index] = word_vecs[cur_index]
    
    return wordvecs, word2id

# vocabulary in wordvec.txt, len(voc) = 106685
# don't need to run for every code change
'''
Get List of vocabulary from 300-dimensional GloVe word vectors
args:
    vec_path: file path of wordvec
return:
    vocabulary list
'''
def get_word_lists(vec_path):
    voc = []
    with open(vec_path) as f:
        for i, line in enumerate(f):
            text = line.strip("\n").split(" ")
            voc.append(text[0])
    return voc
        
class QuoraDatum():
    """
    Class that represents a train/validation/test datum
    - self.raw_text
    - self.label: 0 neg, 1 pos
    - self.file_name: dir for this datum
    - self.tokens: list of tokens
    - self.token_idx: index of each token in the text
    """
    def __init__(self, raw_text1, raw_text2, char_text1, char_text2, label, question_id):
        self.raw_text1 = raw_text1
        self.raw_text2 = raw_text2
        self.char_text1 = list(raw_text1)
        self.char_text2 = list(raw_text2)
        self.label = label
        self.question_id = question_id
        #self.file_name = file_name
        
    def set_embedding(self, word_embedding1, word_embedding2):
        self.word_embedding1 = word_embedding1
        self.word_embedding2 = word_embedding2

    def set_chartokens(self, tokens1, tokens2):
        self.chartokens1 = tokens1
        self.chartokens2 = tokens2



def construct_dataset(dataset_dir , chars):
    """
    Function that loads a dataset
    @param offset: skip first offset items in this dir
    """
    output = []
    with open(dataset_dir) as f:
        for i, line in enumerate(f):
            text = line.split("\t")
            raw_text1 = text[1].replace("<br />", "")
            raw_text2 = text[2].replace("<br />", "")
            char_text1 = list(text[1])
            char_text2 = list(text[2])
            for char in char_text1:
                if char not in chars:
                    chars.append(char)
            label = int(line[0])
            question_id = text[3].strip("\n")
            output.append(QuoraDatum(raw_text1 = raw_text1 , raw_text2 = raw_text2, char_text1 = char_text1, \
                                     char_text2 = char_text2, label = label, question_id = question_id))
    return output, chars

def char_tokenize(string, chars):
    return [chars.index(x) if x in chars else 0 for x in string]

def process_text_dataset(dataset, word2id, wordvecs, chars, max_char):
    len_chars = len(chars)
    for i in range(len(dataset)):
        text1_datum = dataset[i].raw_text1
        text2_datum = dataset[i].raw_text2
        #print("text1_datum" , list(text1_datum.split(" ")))
        word_embedding1 = []
        word_embedding2 = []
        char_token1 = []
        char_token2 = []
        for word in list(text1_datum.split(" ")):
            if word.lower() in word2id:
                word_embedding1.append(wordvecs[word2id[word.lower()]])
            else:
                word_embedding1.append(np.random.randn(300))
            char_inword = [chars.index(x) if x in chars else 0 for x in word]
            if (len(word) > 10):
                char_token1 += char_inword[0:10]
            else:
                char_token1 += char_inword + [len_chars] * (max_char - len(word))
        for word in list(text2_datum.split(" ")):
            #print(word)
            #print(wordvecs[word2id[word.lower()]])
            if word.lower() in word2id:
                word_embedding2.append(wordvecs[word2id[word.lower()]])
            else:
                word_embedding2.append(np.random.randn(300))
            char_inword = [chars.index(x) if x in chars else 0 for x in word]
            if (len(word) > 10):
                char_token2 += char_inword[0:10]
            else:
                char_token2 += char_inword + [len_chars] * (max_char - len(word))
        dataset[i].set_embedding(word_embedding1, word_embedding2)
        char_datum1 = dataset[i].char_text1
        char_datum2 = dataset[i].char_text2
        #char_token1 = char_tokenize(char_datum1)
        #char_token2 = char_tokenize(char_datum2)
        dataset[i].set_chartokens(char_token1, char_token2)
    return dataset

class QuoraDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list, max_sequence_length, max_char):
        """
        @param data_list: list of QuoraDatum
        """
        self.data_list = data_list
        self.max_sequence_length = max_sequence_length
        self.max_char = max_char
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        emb1, emb2, token1, token2, label = self.data_list[key].word_embedding1,self.data_list[key].word_embedding2, \
        self.data_list[key].chartokens1, self.data_list[key].chartokens2, self.data_list[key].label
        #token_idx1, token_idx2, label = self.data_list[key].token_idx1, self.data_list[key].token_idx2, self.data_list[key].label
        #return (token_idx1, token_idx2, len(token_idx1)), label
        return (emb1, emb2, token1, token2, label, self.max_sequence_length, self.max_char)

def quora_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list1 = []
    data_list2 = []
    char_list1 = []
    char_list2 = []
    label_list = []
    word_length_list1 = []
    word_length_list2 = []
    char_length_list1 = []
    char_length_list2 = []
    for datum in batch:
        #print(datum)
        #print("datum[1]", datum[1])
        #print("datum[0][0]", datum[0][0])
        #print("datum[0][1]", datum[0][1])
        label_list.append(datum[4])
        word_length_list1.append(len(datum[0]))
        word_length_list2.append(len(datum[1]))
        char_length_list1.append(len(datum[2]))
        char_length_list2.append(len(datum[3]))
        #length_list1.append(len(datum[0][0]))
        #length_list2.append(len(datum[0][1]))
    #print(length_list1)
    #print(length_list2)
    max_word_length = max(np.max(word_length_list1), np.max(word_length_list2))
    max_char_length = max(np.max(char_length_list1), np.max(char_length_list2))
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]), 
                                pad_width=((0,datum[5]-len(datum[0])),(0, 0)), 
                                mode="constant", constant_values=0)
        data_list1.append(padded_vec)
        padded_vec = np.pad(np.array(datum[1]),
                                pad_width=((0,datum[5]-len(datum[1])),(0, 0)), 
                                mode="constant", constant_values=0)
        data_list2.append(padded_vec)
        padded_vec = np.pad(np.array(datum[2]), 
                                pad_width=((0,datum[6]*datum[5]-len(datum[2]))), 
                                mode="constant", constant_values=0)
        char_list1.append(padded_vec)
        padded_vec = np.pad(np.array(datum[3]),
                                pad_width=((0,datum[6]*datum[5]-len(datum[3]))), 
                                mode="constant", constant_values=0)
        char_list2.append(padded_vec)
    #print(data_list1)
    #print(data_list2)
    return [torch.from_numpy(np.array(data_list1)), torch.from_numpy(np.array(data_list2)) , torch.LongTensor(word_length_list1), torch.LongTensor(word_length_list2), \
            torch.from_numpy(np.array(char_list1)), torch.from_numpy(np.array(char_list2)) , torch.LongTensor(char_length_list1), torch.LongTensor(char_length_list2), torch.LongTensor(label_list)]