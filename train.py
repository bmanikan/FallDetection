import numpy as np
from torch import cuda
import torch
from utils import transform_image

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # training loop   #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):

            data,target = transform_image(data,target)
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                
            # refresh optimizer
            optimizer.zero_grad()
            
            output = model_scratch(data)
            
            loss = criterion(output,target)
            
            loss.backward()
            
            optimizer.step()          
            
            
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validation loop    #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['test']):
            data,target = transform_image(data,target)
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model_scratch(data)
            
            val_loss = criterion(output,target)
            
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (val_loss.data - valid_loss))
            
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            valid_loss_min = valid_loss
            
    return model
