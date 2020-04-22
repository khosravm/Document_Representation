"""
zero_grad clears old gradients from the last step (otherwise you’d just accumulate 
the gradients from all loss.backward() calls).

retain_graph (bool, optional) – If False, the graph used to compute the grad 
will be freed. Note that in nearly all cases setting this option to True is not 
needed and often can be worked around in a much more efficient way. Defaults to 
the value of create_graph.


"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def run_epoch(m, d, mode='tr', is_train=True):
#    loss_loc     = 0.0
#    loss_glb     = 0.0
    
    loss  = 0.0
    running_loss = total_loss = 0.0
#    print_step = 10
    run_step = total_step = 0.0
    d.get_doc_dc(mode=mode)

    while True:
        m.optimizer.zero_grad()
        inputs, targets, TT0 = d.get_next_doc(mode=mode)
        
        inputs, targets = (
                Variable(torch.LongTensor(inputs).to(device)),    
                Variable(torch.LongTensor(targets).to(device)))  #** cuda()
        
#        print(targets[0][1])
#        print(targets.float().mean(1).long().max())

        
        if is_train:
            m.train()
        else:
            m.eval()

        outf, outblks = m(inputs)   #, outglb,target_glb
#        print('~' * 80)
#        print(outblks[0])
#        print(targets[0])
#        print(targets.shape) # torch.Size([75, 35])
#        print(outblks.shape) # torch.Size([75, 35, 1])
#        print(outglb.shape)  # torch.Size([1, 75, 2048])
#        print(outputs.shape)                  # torch.Size([1, 75, 1024])
#        print(outblks.reshape(-1,1).shape)   # torch.Size([2625, 1])
#        print(targets.reshape(-1).shape)     # torch.Size([2625])
#        print(outglb[0].shape)               # torch.Size([75, 2048])
#        print(targets.float().mean(1).shape) # torch.Size([75])
        
         ## Model 1   
#        for kk in range(len(outblks)): 
#            loss_loc = loss_loc+m.criterion(outblks[kk],targets[kk])
               
#        loss_glb = m.criterion(outglb,target_glb[0].long())
#        loss = loss_loc + loss_glb
        ## Model 2
#        loss = m.criterion(outglb, targets.float().mean(1).long())

        ## Model 3
#        n1 = len(outblks)        # No. of blks
#        n2 = len(outblks[0])     # length of each blk
#        n3 = len(outblks[0][0])  # vocabsize
#        P   = torch.cat([torch.zeros(1,n3),outglb[:n1-1]],0)
#        outf = torch.zeros(n1,n2+1,n3)
#        for i in range(n1):
#            outf[i] = torch.cat([P[i].reshape(1,n3),outblks[i]],0)
        
        loss  = m.criterion(outf.reshape(-1,len(outf[0][0])), targets.reshape(-1))
#        for i in range(len(outf)):
#            loss = m.criterion(outf[i], targets[i])
##            loss += torch.log(loss_blk) 
        if is_train:
#            if run_step==0 :
#               loss.backward(retain_graph=True)
#            else:
#               loss.requres_grad = True
#               loss.backward()  #(retain_graph=True)
            loss.backward(retain_graph=True)    
            # clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(m.parameters(), m.config.rnn_max_norm)
            m.optimizer.step()
                              
        running_loss += np.exp(loss.item())    #** data[0] => item()
        run_step += 1.0
        total_loss += loss.item() #np.exp(loss.item())    #running_loss        #** data[0] => item()
        total_step += 1.0
#        print('total_loss',total_loss)
#        run_step += 1    
#        if run_step % 2 == 0:
#           print('Doc. [%d]: loss = ' %(run_step),loss.item())     #running_loss / run_step)


#        if (d.get_doc_dc(mode)) % (m.config.batch_size * print_step) == 0:
#            print('[%d] loss: %.3f seq_len: %d' % (d.get_batch_ptr(mode), 
#                        running_loss / run_step, inputs.size(1)))
#            run_step = 0
#            running_loss = 0
#            if m.config.save:
#                m.save_checkpoint({
#                    'config': m.config,
#                    'state_dict': m.state_dict(),
#                    'optimizer': m.optimizer.state_dict(),
#                }, False)

        
#        if d.get_doc_dc(mode) == 0:
##            print('total loss: %.3f\n' % (total_loss / total_step))
        if TT0 == 0:            
            break
    return total_loss/total_step  # loss.item()

