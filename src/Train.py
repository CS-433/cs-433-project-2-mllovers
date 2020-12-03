import torch
import numpy as np
import matplotlib.pyplot as plt

def train(model, sample, lr=0.005, num_iter=1000):
    """
    Train function for model learning
    Parameters:
        - model: DMM class
        Trainable model for prediction
        - sample: tensor (number of samples, number of features)
          Input data(pitch scapes)
        - lr: scalar. Default: 0.005
        Learnig rate for gradient descent
        - num_iter: scalar. Default: 1000
        Number of epochs to train
    Return: numpy array (number of epochs)
    Loss value for every epoch
    """
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # learning rate decrease scheduler 
    scheduler =  torch.optim.lr_scheduler.MultiStepLR(optimizer, [num_iter])
    # array for loss values
    loss = np.zeros(num_iter)

    # iterate over epochs
    for it in range(num_iter):
        optimizer.zero_grad()  # Reset the gradients (PyTorch syntax...).
        cost = model.neglog_likelihood(sample)  # Cost to minimize.
        cost.backward()  # Backpropagate to compute the gradient.
        optimizer.step() # do gradient step

        scheduler.step() # decrease gradient step
        loss[it] = cost.data.cpu().numpy() 
    #plot loss values in dependency on epoch
    plt.figure()
    plt.plot(loss)
    plt.tight_layout()
    plt.grid()
    plt.show()
    
    return loss