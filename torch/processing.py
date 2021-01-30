import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import pandas as pd
import torchtext
import numpy as np
from torch.utils.data import TensorDataset


#this takes a list of songs and breaks it up
def input_label_split_songs(data, sequence_length, overlap_inputs=True):
    """
    Breaks data up from a long list to (input, label) pairs where len(input) = sequence_length
    """
    x = []
    y = []
    #output data is (input, output) = (x,y)
    #unlike tf we will do this
    #[1,2,3,4] with seq_len=3 => [1,2,3], [2,3,4]. aka we will overlap input label sequences. maybe this
    if overlap_inputs:
        num_sequences = len(data) - sequence_length #buffer required to make first sequence
        assert num_sequences > 0, "You created zero data entries. Use more data or a smaller sequence_length"
        for ii in range(num_sequences):
            x.append(list(data[0+ii:sequence_length+ii]))
            y.append(data[sequence_length+ii])
    else: #make it like tf where. [1,2,3,4] => [1,2,3] 
        num_sequences = len(data) // (sequence_length + 1)
        assert num_sequences > 0, "You created zero data entries. Use more data or a smaller sequence_length"
        for ii in range(num_sequences):
            shift = sequence_length + 1
            x.append(list(data[shift*ii : sequence_length+(shift*ii)]))
            y.append(data[sequence_length+(ii*shift)])
    #make into a TensorDataset
    from torch.utils.data import TensorDataset
    #dataset = TensorDataset(torch.tensor(x, device=dev), torch.tensor(y, device=dev))
    #dataset = TensorDataset(torch.tensor(x), torch.tensor(y)) #not sure if putting these on GPU is actually better or not
    return x,y


#this takes a tensor list and breaks it up. much faster
def input_label_split(data, sequence_length, overlap_inputs=True):
    """
    Breaks data up from a long list to (input, label) pairs where len(input) = sequence_length
    MIGRATES INPUT TO CUDA. OUTPUTS OBJECTS ON CUDA
    """
    assert type(data) == torch.Tensor, "This function only works for a torch.Tensor([x,y,z]) input"
    data = data.cuda()
    #x = []
    #y = []
    #output data is (input, output) = (x,y)
    #unlike tf we will do this
    #[1,2,3,4] with seq_len=3 => [1,2,3], [2,3,4]. aka we will overlap input label sequences. maybe this
    if overlap_inputs:
        num_sequences = len(data) - sequence_length #buffer required to make first sequence
        assert num_sequences > 0, "You created zero data entries. Use more data or a smaller sequence_length"
        for ii in range(num_sequences):
            if ii == 0:
                x = data[0+ii:sequence_length+ii].cuda()
                y = data[sequence_length+ii].reshape(1).cuda()
            else:    
                #x.append(list(data[0+ii:sequence_length+ii]))
                x = torch.cat((x, data[0+ii:sequence_length+ii]))
                #y.append(data[sequence_length+ii])
                y = torch.cat((y, data[sequence_length+ii].reshape(1)))
    else: #make it like tf where. [1,2,3,4] => [1,2,3] 
        num_sequences = len(data) // (sequence_length + 1)
        assert num_sequences > 0, "You created zero data entries. Use more data or a smaller sequence_length"
        for ii in range(num_sequences):
            shift = sequence_length + 1
            if ii == 0:
                x = data[shift*ii : sequence_length+(shift*ii)].cuda()
                y = data[sequence_length+(ii*shift)].reshape(1).cuda()
            else:
                #x.append(list(data[shift*ii : sequence_length+(shift*ii)]))
                x = torch.cat((x, data[shift*ii : sequence_length+(shift*ii)]))
                #y.append(data[sequence_length+(ii*shift)])
                y = torch.cat((y, data[sequence_length+(ii*shift)].reshape(1)))
    #make into a TensorDataset
    #from torch.utils.data import TensorDataset
    #dataset = TensorDataset(torch.tensor(x, device=dev), torch.tensor(y, device=dev))
    #dataset = TensorDataset(torch.tensor(x), torch.tensor(y)) #not sure if putting these on GPU is actually better or not
    return x.reshape(num_sequences, sequence_length), y


#this takes a tensor list and breaks it up. much faster
def data_split(data, sequence_length):
    """
    Breaks data up from a long list to (input, label) pairs where len(input) = sequence_length
    MIGRATES INPUT TO CUDA. OUTPUTS OBJECTS ON CUDA
    """
    assert type(data) == torch.Tensor, "This function only works for a torch.Tensor([x,y,z]) input"
    data = data.cuda()
    #x = []
    #y = []
    #output data is (input, output) = (x,y)
    #unlike tf we will do this
    #[1,2,3,4] with seq_len=3 => [1,2,3], [2,3,4]. aka we will overlap input label sequences. maybe this
    num_sequences = len(data) // (sequence_length + 1)

    assert num_sequences > 0, "You created zero data entries. Use more data or a smaller sequence_length"
    for ii in range(num_sequences):
        shift = sequence_length + 1
        if ii == 0:
            x = data[0:sequence_length]
            y = data[1:(sequence_length+1)]
        else:
            #x.append(list(data[shift*ii : sequence_length+(shift*ii)]))
            x = torch.vstack((x, data[shift*ii : sequence_length+(shift*ii)]))
            #y.append(data[sequence_length+(ii*shift)])
            y = torch.vstack((y, data[(ii*shift+1) : (shift+(ii*shift))]))

    return x, y