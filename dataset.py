"""
Creating desired final form of train, validation and test data of loded text 
collection from specified path. It includs 2 major Classes
 - Class Dataset(object) : For dictionary setting, and also block and batch 
   generation  
 - class Config(object): For initializing paths (both to load and save),some 
   important items such as batch size, ... . This class is used as object for
   Dataset class in main. 
"""
import numpy as np
import os
import nltk
import pickle
import pprint
#import copy
#import sys

nltk.download('punkt')

class Dataset(object):
    def __init__(self, config):
        self.config = config

## dictionary settings
        # Initialize dictionary (call initialize_dictionary)
        self.initialize_dictionary()
        # Build train, valid., and test corpus
        self.build_corpus(self.config.train_path, self.train_corpus)
        self.build_corpus(self.config.valid_path, self.valid_corpus)
        self.build_corpus(self.config.test_path, self.test_corpus)
        print()
         
        # Process data 
        self.train_data = self.process_data(
                self.train_corpus, 
                update_dict=True)
        self.valid_data = self.process_data(
                self.valid_corpus,
                update_dict=True)
        self.test_data = self.process_data(
                self.test_corpus,
                update_dict=True)
        print()
        # Padding data     
        self.pad_data(self.train_data)
        self.pad_data(self.valid_data)
        self.pad_data(self.test_data)
        # Reshape data to (batch_size, seq_len=data size//batch_size, max word length)
        self.train_data = self.reshape_data(self.train_data)
        self.valid_data = self.reshape_data(self.valid_data)
        self.test_data = self.reshape_data(self.test_data)
        print()
        
        # Initialize dc for each mode at first step as zeros
        self.train_dc = 0
        self.valid_dc = 0
        self.test_dc = 0
        
        # Print No. of words and No. of characters in dictionary
        print('char_dict', len(self.char2idx))
        print('word_dict', len(self.word2idx), end='\n\n')

    def initialize_dictionary(self):
        self.train_corpus = []
        self.valid_corpus = []
        self.test_corpus = []
        self.char2idx = {}
        self.idx2char = {}
        self.word2idx = {}
        self.idx2word = {}
        self.UNK = '<unk>'
        self.PAD = 'PAD'
        self.CONJ = '+'
        self.START = '{'
        self.END = '}'
        self.char2idx[self.UNK] = 0
        self.char2idx[self.PAD] = 1
        self.char2idx[self.CONJ] = 2
        self.char2idx[self.START] = 3
        self.char2idx[self.END] = 4
        self.idx2char[0] = self.UNK
        self.idx2char[1] = self.PAD
        self.idx2char[2] = self.CONJ
        self.idx2char[3] = self.START
        self.idx2char[4] = self.END
        self.word2idx[self.UNK] = 0
        self.word2idx[self.PAD] = 1
        self.word2idx[self.CONJ] = 2
        self.idx2word[0] = self.UNK    # For OOV tokens
        self.idx2word[1] = self.PAD
        self.idx2word[2] = self.CONJ   
    
    def update_dictionary(self, key, mode=None):
        if mode == 'c':     # character mode
            if key not in self.char2idx:
                self.char2idx[key] = len(self.char2idx)
                self.idx2char[len(self.idx2char)] = key
        elif mode == 'w':   # word/token mode
            if key not in self.word2idx:
                self.word2idx[key] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = key
    
    def map_dictionary(self, key_list, dictionary, reverse=False):
        output = []
        # reverse=False : word2idx, char2idx
        # reverse=True : idx2word, idx2char
        for key in key_list:
            if key in dictionary:
                if reverse and key == 1: # PAD
                    continue
                else:
                    output.append(dictionary[key])
            else:
                if not reverse:
                    output.append(dictionary[self.UNK])
                else:
                    output.append(dictionary[0]) # 0 for UNK
        return output
    # split text file to lines (sentenses), and separate sentences with <conj> (+)
    def build_corpus(self, path, corpus):
        print('building corpus %s' % path)
        with open(path) as f:

            lines    = f.readlines()
            
            num_doc   = 0
            doc_start = []  # a list of no. that each of them shows in which line doc start

            minword = 1 #10 #100     # min. word in each sentence 
            minsent = 1 #5  #10      # min. sent. in each document
            
            spad = []
            for i in range(self.config.blk_len):
                spad.append(self.PAD)
            
            ## Count no. of docs in corpus
            doc = []
            
            for i in range(len(lines)):
                if lines[i][1] == '=':
                   num_doc += 1 
                   doc_start.append(i)
                   doc.append(0)
            
            ## Save docs separately
            for k in range(num_doc):
                docl = []
                c    = doc_start[k]   #+1
    
                if k < num_doc-1:
                    c_end = doc_start[k+1]        
                else:
                    c_end = len(lines)        
    
                while c < c_end:
                   docl.append(lines[c])
                   c += 1
     
                doc[k] = ''.join(docl)

            docs = []
            for t in range(len(doc)):
                text = doc[t]
                text = text.replace("\n"," ")
                text = text.replace(".",".<stop>")
                text = text.replace("?","?<stop>")
                text = text.replace("!","!<stop>")
                text = text.replace("<prd>",".")
                sentences = text.split("<stop>")
    
                s1 = []
                for j in range(len(sentences)):
                    a = sentences[j].split()
                    if len(a) >= minword:
                        if len(a) > self.config.blk_len:
                            a = a[:self.config.blk_len]    #sentences[j] = sentences[j][:self.blk_len]
                        else:
                            while len(a) < self.config.blk_len:
                               a.append(self.PAD)
                        sentences[j] = ' '.join(a)
                        s1.append(sentences[j])
                         
                if len(s1) >= minsent:
                    if len(s1) > self.config.num_blk:
                        s1 = s1[:self.config.num_blk]      #s1.truncate(self.num_blk)
                    else:
                        while len(s1) < self.config.num_blk:
                            s1.append(' '.join(spad))
                            
                    docs.append(' '.join(s1))   #doc[t]
               
            for i in range(len(docs)):
                for j in range(len(docs[i].split())):
                    word = docs[i].split()[j]
                    corpus.append(word)


    # Process data by calling map_dic and update_dict            
    def process_data(self, corpus, update_dict=False):

        print('processing corpus %d' % len(corpus))
        total_data = []
        max_wordlen = 0

        for k, word in enumerate(corpus):        
            # dictionary update
            if update_dict:
                self.update_dictionary(word, 'w')
                for char in word:
                    self.update_dictionary(char, 'c')
            
            # user special characters or mapping
            if word == self.UNK or word == self.CONJ or word == self.PAD:
                word_char = word
                charidx = [self.char2idx[word_char]]
            else:
                word_char = self.START + word + self.END
                charidx = self.map_dictionary(word_char, self.char2idx)
            
            # get max word length
            max_wordlen = (len(word_char) 
                    if len(word_char) > max_wordlen else max_wordlen)
            
#            if len(word_char)> 22#                
#               print('******', word_char)
            if max_wordlen > self.config.max_wordlen:
                self.config.max_wordlen = max_wordlen
                
            # word / char
            total_data.append([self.word2idx[word], charidx])
                
        
        if update_dict:
            self.config.char_vocab_size = len(self.char2idx)
            self.config.word_vocab_size = len(self.word2idx)

        print('data size', len(total_data))
        print('max word_len', max_wordlen)

        return total_data
## Word padding NOT sentences or document padding    
    def pad_data(self, dataset):        
        for data in dataset:
            sentword, sentchar = data            
            # pad words which no. of their characters is less than max_wordlen

            while len(sentchar) != self.config.max_wordlen:
                sentchar.append(self.char2idx[self.PAD])

        return dataset

## Block generation
        
    def reshape_data(self, dataset):
        n_doc   = len(dataset)//(self.config.num_blk*self.config.blk_len)
        inputs  = [d[1] for d in dataset]
        targets = [d[0] for d in dataset]
            
        inputs = np.array(inputs)
        targets = np.array(targets)

        inputs = np.reshape(inputs, (n_doc,self.config.num_blk,self.config.blk_len, -1)) # inputs[batch_size, seq_len, batch_num]
        targets = np.reshape(targets, (n_doc,self.config.num_blk, -1))        # targets[batch_size, seq_len*batch_num]  
        
        return inputs, targets
  


    def get_next_doc(self, mode='tr'):
        if mode == 'tr':
            dc = self.train_dc
            data = self.train_data
            x0   = 14
        elif mode == 'va':
            dc = self.valid_dc
            data = self.valid_data
            x0   = 3
        elif mode == 'te':
            dc = self.test_dc
            data = self.test_data
            x0   = 3
           
# type(data) => <class 'tuple'>
# data[0] => inputs  ; data[1] => targets
# In character level LM targets are just inputs shifted over by one character
             
        inputs = data[0][dc,:,:,:]        
        targets = data[1][dc,:,:]  #[dc,:,:+1]
        
        for ii in range(self.config.blk_len):
            if ii<self.config.blk_len-1:
                targets[:,ii] = targets[:,ii+1] #[dc,:,:+1]
            else:
                if dc < len(data[0])-1:
                   targets[:,ii] = data[1][dc+1,:,0]
                else:
                   targets[:,ii] = data[1][0,:,0]
        
        dc += 1       
        if dc >= x0-1: 
            TT0 = 0
        else:
            TT0 = 1

        if mode == 'tr':
            self.train_dc = dc
        elif mode == 'va':
            self.valid_dc = dc
        elif mode == 'te':
            self.test_dc = dc

        return inputs, targets, TT0
    # Get document counter (dc)
    def get_doc_dc(self, mode):
        if mode == 'tr':
            self.train_dc = 0
            return self.train_dc
        elif mode == 'va':
            self.valid_dc = 0
            return self.valid_dc
        elif mode == 'te':
            self.test_dc = 0
            return self.test_dc

## Specified path and batch size and zero initilization of max_wordlen, char_
## and word_vocab size         
class Config(object):
    def __init__(self):
        user_home = os.getcwd() #os.path.expanduser('~')
        self.train_path = os.path.join(user_home, 'datasets/wiki3/train.txt')
        self.valid_path = os.path.join(user_home, 'datasets/wiki3/valid.txt')
        self.test_path = os.path.join(user_home, 'datasets/wiki3/test.txt')
        self.batch_size = 1
        self.num_doc    = 0
        self.num_blk    = 10  #75
        self.blk_len    = 10  #75
        self.max_wordlen = 0
#        self.max_sentlen = 75 
        self.char_vocab_size = 0
        self.word_vocab_size = 0
        self.save_preprocess = True
        self.preprocess_save_path = './data/preprocess(tmp).pkl'
        self.preprocess_load_path = './data/preprocess(tmp).pkl'

## Create and load desired dataset through Dataset and Config classes (defined 
## above) and somehow checking them   
if __name__ == '__main__':
    # create data directory if it is not created before!
    if not os.path.exists('./data'):
        os.makedirs('./data')
    # Call Dataset class by config class as its object and make and load desired dataset    
    config = Config()
    if config.save_preprocess:
        dataset = Dataset(config)
        pickle.dump(dataset, open(config.preprocess_save_path, 'wb'), protocol=4)
    else:
        print('## load preprocess %s' % config.preprocess_load_path)
        dataset = pickle.load(open(config.preprocess_load_path, 'rb'))

## Check some aspect of resulted dataset to see if it is the desired one!
    ## Print dataset config to see it is valid
    pp = lambda x: pprint.PrettyPrinter().pprint(x)
    pp(([(k,v) for k, v in vars(dataset.config).items() if '__' not in k]))
    print()
    
#    input, target = dataset.get_next_doc()
#    print([dataset.map_dictionary(i, dataset.idx2char) for i in input[0,:,:]])
##    print([dataset.idx2word[t] for t in target[0,:]])
#    print()
    
    ## (batch_size, data size, max word length)
    print('train', dataset.train_data[0].shape)   # train (75, 27848, 22)
    print('valid', dataset.valid_data[0].shape)   # valid (75, 2901, 22)
    print('test', dataset.test_data[0].shape)     # test (75, 3274, 22)  
    print()
    
    ## Check dc for test mode and a seq_len of 100
#    while True:
#        i, t = dataset.get_next_doc(mode='te')
#        print(dataset.test_dc, len(i[0]))
#        if dataset.test_dc == 0:
#            print('\niteration test pass!')
#            break

