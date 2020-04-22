import torch
import argparse
import pickle
import pprint
from torch.autograd import Variable
from dataset import Dataset, Config
from model import Final_model
from run import run_epoch
import time
import matplotlib.pyplot as plt


argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', type=str, default='./data/preprocess(tmp).pkl')
argparser.add_argument('--checkpoint_path', type=str, default='./results/test.pth')
argparser.add_argument('--load_path', type=str, default='./results/test.pth')
argparser.add_argument('--epoch', type=int, default= 20)  #25
argparser.add_argument('--train', action='store_true', default=True)
argparser.add_argument('--valid', action='store_true', default=True)
argparser.add_argument('--test', action='store_true', default=True)
argparser.add_argument('--save', action='store_true', default=False)
argparser.add_argument('--resume', action='store_true', default=False)

argparser.add_argument('--num_blk', type=float, default=10)  # no. of blk in each doc.
argparser.add_argument('--blk_len', type=float, default=10)    # Max no. of tokens in each blk
#argparser.add_argument('--seq_len', type=float, default=35)    #75
argparser.add_argument('--lr', type=float, default=1.0)
argparser.add_argument('--hidden_dim', type=int, default=1024)   #300
argparser.add_argument('--layer_num', type=int, default=2)
argparser.add_argument('--input_dimg', type=int, default=1024)   #512
argparser.add_argument('--hidden_dimg', type=int, default=1024)   
argparser.add_argument('--layer_numg', type=int, default=2)
argparser.add_argument('--rnn_dr', type=float, default=0.5)    #dropout
argparser.add_argument('--rnn_max_norm', type=int, default=5)  # constraint norm of gradient 
argparser.add_argument('--char_embed_dim', type=int, default=15) #dimensionality of character embedding
argparser.add_argument('--char_conv_fn', type=list, 
        default=[25, 50, 75, 100, 125, 150]) #filter size    [25, 50, 75, 100, 125, 150]
argparser.add_argument('--char_conv_fh', type=list, default=[1, 1, 1, 1, 1, 1])
argparser.add_argument('--char_conv_fw', type=list, default=[1, 2, 3, 4, 5, 6]) #filters of width
args = argparser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #**

def run_experiment(model, dataset):
    if model.config.resume:
        model.load_checkpoint()

    if model.config.train:
        print('=' * 64)
        print('                            Training                            ')
        print('=' * 64)      
        prev_vloss = 99999.0
        tr_loss = []
        va_loss = []
        for ep in range(model.config.epoch):
            start_time = time.time()
            print('[Epoch %d] => lr = ' % (ep+1), model.config.lr)
            print('-' * 64)
            tloss = run_epoch(model, dataset, 'tr')
            tr_loss.append(tloss)
            print('-' * 64)
            print('Train_loss = ' , tloss)
            if model.config.valid:
                print('=' * 64)
                print('                           Validation                           ')
                print('=' * 64)
                      
                vloss = run_epoch(model, dataset, 'va', is_train=True)
                va_loss.append(vloss)
                print('-' * 64)
                print('Valid_loss = ' , vloss)
                if vloss < prev_vloss - 1.0:   #1.0
                    prev_vloss = vloss
                    print('Prev_vloss = ' , prev_vloss)
                else:
                    model.decay_lr()
                    print('Learning_rate (lr) decays to %.3f' % model.config.lr)
            print('/' * 64)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  .format(ep+1, (time.time() - start_time), vloss)) 
            print('/' * 64)
            print()
        # Plot losses and saving plot
        plt.plot(tr_loss,'r', label="tr_loss")
        plt.plot(va_loss,'b', label="va_loss")
        plt.legend(bbox_to_anchor=(0.72, 0.95), loc='upper left', borderaxespad=0.)  
        plt.savefig('loss.png')
        
        with open('tr_loss.pkl', 'wb') as f1:
            pickle.dump(tr_loss, f1)
        with open('va_loss.pkl', 'wb') as f2:
            pickle.dump(va_loss, f2)
    
    if model.config.test:
        print('=' * 64)
        print('                            Testing                             ')
        print('=' * 64)
        te_loss = run_epoch(model, dataset, 'te', is_train=False)
        print('-' * 64)
        print('Test_loss = ' , te_loss)


def main():
    print('### load dataset')
    dataset = pickle.load(open(args.data_path, 'rb'))
    
    # update args
    dataset.config.__dict__.update(args.__dict__)
    args.__dict__.update(dataset.config.__dict__)
    pp = lambda x: pprint.PrettyPrinter().pprint(x)
    pp(vars(dataset.config))
    print('train', dataset.train_data[0].shape)
    print('valid', dataset.valid_data[0].shape)
    print('test', dataset.test_data[0].shape)
    print()

    model = Final_model(args).to(device)    #** cuda()
    run_experiment(model, dataset)


if __name__ == '__main__':
    main()

