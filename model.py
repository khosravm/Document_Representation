""""
GOAL: Pytorch Implementation of a hierarchical bi-lstm model for document 
representations from the paper:
   "Language Model Pre-training for Hierarchical Document Representations"

===============================================================================
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
dilation=1, groups=1, bias=True, padding_mode='zeros')
Applies a 2D convolution over an input signal composed of several input planes
===============================================================================
torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
A simple lookup table that stores embeddings of a fixed dictionary and size.
This module is often used to store word embeddings and retrieve them using 
indices. The input to the module is a list of indices, and the output is the 
corresponding word embeddings.
===============================================================================
nn.lstm()
Some Arguments: ---------------------------------------------------------------
input_size – The number of expected features in the input x
batch_first – If True, then the input and output tensors are provided as 
(batch, seq, feature). Default: False
bidirectional – If True, becomes a bidirectional LSTM. Default: False
Inputs and Outputs ------------------------------------------------------------
Inputs: input, (h_0, c_0) - shape (seq_len, batch, input_size)
h_0: hidden state of shape (num_layers * num_directions, batch, hidden_size)
c_0: cell state of shape (num_layers * num_directions, batch, hidden_size)
Outputs: output, (h_n, c_n) - shape (seq_len, batch, num_directions * hidden_size)
h_n: hidden state of shape (num_layers * num_directions, batch, hidden_size)
c_n: cell state of shape (num_layers * num_directions, batch, hidden_size)
LSTM and Sequence modeling ----------------------------------------------------
out, hidden = lstm(inputs, hidden)
1st- Step through the sequence one element at a time. In this way, after each 
step, hidden contains the hidden state.
2nd- Alternatively, we can do the entire sequence all at once. in this case, 
the first value returned by LSTM is all of the hidden states throughout the 
sequence. And, the second is just the most recent hidden state (compare the last 
slice of "out" with "hidden", they are the same)
The reason for this is that: "out" will give you access to all hidden states in 
the sequence "hidden" will allow you to continue the sequence and backpropagate,
by passing it as an argument  to the lstm at a later time
References --------------------------------------------------------------------
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
https://pytorch.org/docs/stable/nn.html
https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
===============================================================================
torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, 
weight_decay=0, nesterov=False)
Implements stochastic gradient descent (optionally with momentum).
Nesterov momentum is based on the formula from On the importance of 
initialization and momentum in deep learning.

"""
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Final_model(nn.Module):
    def __init__(self, config):
        super(Final_model, self).__init__()
        self.config = config
        
        self.char_embed = nn.Embedding(config.char_vocab_size, config.char_embed_dim,
                padding_idx=1)
        #Store any desired number of nn.Module’s (like a python list)
        self.char_conv = nn.ModuleList([nn.Conv2d(
                config.char_embed_dim, config.char_conv_fn[i],
                (config.char_conv_fh[i], config.char_conv_fw[i]),
                stride=1) for i in range(len(config.char_conv_fn))])
        self.input_dim = int(np.sum(config.char_conv_fn))
        self.hidden_dim = config.hidden_dim
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, config.layer_num,
                dropout=config.rnn_dr, batch_first=True) #, bidirectional=True

        self.hw_nonl = nn.Linear(self.input_dim, self.input_dim)
        self.hw_gate = nn.Linear(self.input_dim, self.input_dim)

        self.word_vocab_size = config.word_vocab_size
        self.blk_len = config.blk_len
        self.num_blk = config.num_blk
        self.hidden_dimg = config.hidden_dimg
        self.input_dimg  = config.input_dimg 
        
        # Readout layer
#        self.fc_blk = nn.Linear(2*config.hidden_dim, config.word_vocab_size)   # 2: No. of directions
#        self.fc_glb = nn.Linear(2*config.hidden_dimg, config.word_vocab_size)
        self.fc     = nn.Linear(2*config.hidden_dimg,1*config.hidden_dimg)
        self.fc_out = nn.Linear(1*config.hidden_dimg, config.word_vocab_size)

        self.fc1 = nn.Linear(1*config.hidden_dimg*config.blk_len, config.input_dimg)
        self.fc2 = nn.Linear(config.input_dimg,1)
        # Global LSTM layers    
        self.lstm_global = nn.LSTM(config.input_dimg, config.hidden_dimg, config.layer_numg,
                dropout=config.rnn_dr, batch_first=True) #, batch_first=True , bidirectional=True
               
        self.init_weights()
        self.hidden = self.init_hidden(config.batch_size)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=config.lr)
        self.model_params(debug=False)
    
    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(
                    self.config.layer_num*1, batch_size, self.config.hidden_dim)).to(device),
                autograd.Variable(torch.zeros(
                    self.config.layer_num*1, batch_size, self.config.hidden_dim)).to(device))
    #** cuda() => .to(device)
            
    def init_weights(self):
        # self.char_embed.weight.data.uniform_(-0.05, 0.05)
        # for conv in self.char_conv:
        #     conv.weight.data.uniform_(-0.05, 0.05)
        #     conv.bias.data.fill_(0)
        # self.hw_nonl.weight.data.uniform_(-0.05, 0.05)
        # self.hw_nonl.bias.data.fill_(0)
        # self.hw_gate.weight.data.uniform_(-0.05, 0.05)
        self.hw_gate.bias.data.uniform_(-2.05, 2.05)
        # self.fc1.weight.data.uniform_(-0.05, 0.05)
        # self.fc1.bias.data.fill_(0)

    def model_params(self, debug=True):
#        print('### model parameters')
        params = []
        total_size = 0
        def multiply_iter(p_list):
            out = 1
            for p in p_list:
                out *= p
            return out

        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
                total_size += multiply_iter(p.size())
            if debug:
                print(p.requires_grad, p.size())
#        print('total size: %s\n' % '{:,}'.format(total_size))
        return params

    def char_conv_layer(self, inputs):
        embeds = self.char_embed(inputs.reshape(-1, self.config.max_wordlen))    #** view => reshape
        embeds = torch.transpose(torch.unsqueeze(embeds, 2), 1, 3).contiguous()
        conv_result = []
        for i, conv in enumerate(self.char_conv):
            char_conv = torch.squeeze(conv(embeds))
            char_mp = torch.max(torch.tanh(char_conv), 2)[0]
            char_mp = char_mp.view(-1, inputs.size(0), char_mp.size(1))   ##inputs.size(1)
            conv_result.append(char_mp)
        conv_result = torch.cat(conv_result, 2)
        return conv_result

    def rnn_layer(self, inputs, hidden):
        lstm_out, hidden = self.lstm(inputs, hidden)        
        return lstm_out , hidden

    ## Adaptively carrying some dimensions of the input directly to the output 
    ## (transformer gate and carrying gate)
    def highway_layer(self, inputs):
        nonl = F.relu(self.hw_nonl(inputs))
        gate = torch.sigmoid(self.hw_gate(inputs))
        z = torch.mul(gate, nonl) + torch.mul(1-gate, inputs)
        return z

    def forward(self, inputs):
        outblks = torch.zeros(self.num_blk, self.blk_len, 1*self.hidden_dim)  #self.word_vocab_size)      # outblks torch.Size([75, 35, vocab_size])
        blkrep  = []    
        ## For each block of length (blk_len) implement char_aware_LM
        for k in range(self.num_blk):
            ##=================================================================
            ## Char_aware_LM employs a CNN & a highway network over characters 
            ##=================================================================
            inputs1 = inputs[k]    # inputs1 torch.Size([35, 21])
            char_conv = self.char_conv_layer(inputs1)
            high_out = self.highway_layer(char_conv)    # high_out torch.Size([1, 35, 525])
            ##=================================================================
            ## Char-based word representations of k-th block 
            ## (out => {h_1,h_2,...,h_blk_len} for k-th block)
            ##=================================================================
            out, hidden = self.rnn_layer(high_out, self.hidden)   # type(high_out) => torch.tensor            
            # Fully-connected layer to project out to vocab_size
            outblks[k]  = out[0]    #self.fc_blk(out[0])       #out: torch.Size([1, 35, 2048]) ;outblks[k]: torch.Size([35, vocab_size])
            # Both max and avg.pooling on the word representation followed by a 
            # feed-forward transformation to obtain block-representation [c_i]
            output_loc  = torch.cat((F.max_pool1d(out,2),F.avg_pool1d(out,2)),2) #output_loc: torch.Size([1, 35, 2*1024])
            # Keep representation of each blk in blkrep
            if k ==0:
               blkrep = output_loc.reshape(1,-1)
            else:               
               blkrep = torch.cat([blkrep,output_loc.reshape(1,-1)],0)
        # Linear transformations to feed blk rep. as input to global Bi-LSTM       
        blkrep  = self.fc1(blkrep)  # torch.Size([75, 1024])     
#        target_glb = self.fc2(blkrep).t()   # torch.Size([1, 75])
        ##=====================================================================
        ## Processing local block rep. by the global Bi-LSTM to generate
        ## document-contextualized block rep. p_i : {p_1 ,...,p_num_blk}
        ##=====================================================================
        out_glb , hidden_glb = self.lstm_global(blkrep.reshape(1,self.num_blk,-1))  #type(outputs) => <class list> ,shape:torch.Size([1, 75, 2*1024])
        # Mapping document-contextualized block rep. to vocab_size
#        outglb = self.fc_glb(out_glb[0])   # torch.Size([75, 10001])
        # The previous tokens with respect to a current token can be categorized 
        # into two sets: the preceding tokens in the same text block, and all 
        #tokens in the preceding text blocks.
        # combine the information from ( h_t−1(i) , p_(i-1)):
        outf  = torch.zeros_like(outblks)
        for i in range(self.num_blk):
            for j in range(self.blk_len):
                if i==0:
                   x          = torch.cat([outblks[i][j],torch.rand_like(outblks[i][j])],0)     # or zeros!!
                else:
                   x          = torch.cat([outblks[i][j],out_glb[0][i-1]],0)
                  
                   outf[i][j] = self.fc(x.reshape(1,-1))
        outf   = self.fc_out(outf)    #torch.Size([75, 35, 10001])        
        return outf, outblks    #outglb ,target_glb
          
    def decay_lr(self):
        self.config.lr *= 0.5     #/= 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.lr

    def save_checkpoint(self, state, is_best, filename=None):
        if filename is None:
            filename = self.config.checkpoint_path
        # print('### save checkpoint %s' % filename)
        torch.save(state, filename)

    def load_checkpoint(self, filename=None):
        if filename is None:
            filename = self.config.load_path
        print('### load checkpoint %s' % filename)
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])  


