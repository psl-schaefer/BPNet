
import matplotlib.pyplot as plt
#plt.style.use('dark_background')
import torch
import torch.nn as nn
from architectures import *
from utils import * 
from loss import *
from metrics import *
from interpretation import *
from captum.attr import DeepLift
import seaborn as sns
from matplotlib import pyplot as plt
from modisco.visualization import viz_sequence


def region_of_interest_data(seqname , start, end, dataset, device):

    """Get one-hot-encoding, counts and bias tracks for a specific sequence of interest.
    Params:
        seq_name: string
            specifies the chromosome 
        start: int
            specifies start coordinate of sequence of interest on chromosome
        end: int
            specifies end coordinate of sequence of interest on chromsome
        dataset: utils.ChIP_Nexus_Dataset object
        device: cuda or cpu

    Returns:
        one_hot: tensor
            Tensor of one-hot encoded sequence of interest. 
        bias_raw: tensor
        bias_smooth: tensor
        idx: int
            idx of the region of interest in the dataset
        dist_start: int
            distance between the start of the 1kb sequence and the region of interest
    """
    # subset data to contain only sequence/region of interest
    tmp_df = dataset.region_info.copy().reset_index()
    idx = tmp_df.loc[(tmp_df.seqnames==seqname) & (tmp_df.start > start-1000) & (tmp_df.end < end+1000)].index.to_numpy()[0]
    dist_start = start - tmp_df.start[idx]

    # read in data
    one_hot = torch.unsqueeze(torch.tensor(dataset.one_hot_seqs[idx, :, :]).to(device), axis=0)
    bias_raw = torch.unsqueeze(torch.tensor(dataset.ctrl_counts[idx, :, :]).to(device), axis=0)
    bias_smooth = torch.unsqueeze(torch.tensor(dataset.ctrl_counts_smooth[idx, :, :]).to(device), axis=0)

    return one_hot, bias_raw, bias_smooth, idx, dist_start


def get_contr_region(seqname, start, end, dataset,  dl, device, tf_list, plot=True):
    """Compute the DeepLift contribution scores for a 1kb sequence which contains the shorter 
    region of interest specified by the input arguments.
    Params:
        seq_name: string
            specifies the chromosome 
        start: int
            specifies start coordinate of sequence of interest on chromosome
        end: int
            specifies end coordinate of sequence of interest on chromsome
        dataset: utils.ChIP_Nexus_Dataset object
        device: cuda or cpu
        tf_list: 
            Contains names of TFs for which we want to compute the contributions
        plot: bool
            Whether to visualize the DeepLift contribution scores.

    Returns:
        contr: tensor (4x1000)
            Contains the contribution of each bp to the profile shape predictions for the input sequence.
        dist_start: int
            distance between the start of the 1kb sequence and the region of interest
    """
    # select region of interest
    tmp_df = dataset.region_info.copy().reset_index()
    idx = tmp_df.loc[(tmp_df.seqnames==seqname) & (tmp_df.start > start-1000) & (tmp_df.end < end+1000)].index.to_numpy()[0]
    dist_start = start - tmp_df.start[idx]
    # read in data
    one_hot = torch.unsqueeze(torch.tensor(dataset.one_hot_seqs[idx, :, :]).to(device), axis=0)
    baseline = torch.zeros(one_hot.shape).to(device) #baseline for the deep lift function
    bias_raw = torch.unsqueeze(torch.tensor(dataset.ctrl_counts[idx, :, :]).to(device), axis=0)
    bias_smooth = torch.unsqueeze(torch.tensor(dataset.ctrl_counts_smooth[idx, :, :]).to(device), axis=0)
    # compute contribution scores for each tf
    contr_list = []
    for tf_index, tf in enumerate(tf_list):
        contr = dl.attribute(inputs=one_hot, baselines=baseline, target=(tf_index), additional_forward_args=(bias_raw, bias_smooth, True))
        contr_list.append(contr)
        if plot:
            print(f"TF: {tf}")
            # plot entire sequence
            print(f"Coordinates: {tmp_df.seqnames[idx]}:{tmp_df.start[idx]}-{tmp_df.end[idx]} (1-1000")
            viz_sequence.plot_weights(contr.detach().cpu().numpy(), subticks_frequency=10)
            # plot subsequence of interest
            print(f"Coordinates: {seqname}:{start}-{end}, ({dist_start} - {dist_start + end - start})")
            viz_sequence.plot_weights(contr.detach().cpu().numpy()[:, :,dist_start : (dist_start + end - start)], subticks_frequency=5, figsize=(10,2))
    return contr, dist_start



def input_gradient(seqname, start, end, dataset, model, device, tf_list, plot=True):
    """Compute tthe input x gradient for a specific sequence of interest
    Params:
        seq_name: string
            specifies the chromosome 
        start: int
            specifies start coordinate of sequence of interest on chromosome
        end: int
            specifies end coordinate of sequence of interest on chromsome
        dataset: utils.ChIP_Nexus_Dataset object
        strand: int
        plot: bool
            whether to plot the results for the region of interest

    Returns:
        grad_list: list
            Each entry corresponds to an array of shape (4 x 1000) containing the gradients of the output (all 1000bp collapsed for one strand) with respect to all 1000 bp of the input.
        input_gradient_list: list
            Each entry corresponds to an array of shape (4 x 1000) containing the gradients of the output (all 1000bp collapsed for one strand) with respect to all 1000 bp of the input multiplied by the input. 
    """
    one_hot, bias_raw, bias_smooth, idx, dist_start = region_of_interest_data(seqname, start, end, dataset, device)

    # compute the gradient with respect to its input
    grad_list = []
    input_gradient_list = []
    for tf_index, tf in enumerate(tf_list):
        # read in sequence one-hot-encoding
        one_hot = torch.unsqueeze(torch.tensor(dataset.one_hot_seqs[idx, :, :]).to(device), axis=0)
        # in order to compute the gradients with respect to the input we need to set grad=True
        one_hot.requires_grad=True
        # forward pass
        output = model.forward(one_hot, bias_raw, bias_smooth, interpretation=True)
        # backward pass
        output[tf_index].backward()#, strand].backward()
        one_hot.requires_grad=False
        grad_list.append(one_hot.grad.squeeze())
        # weight the gradients with the input
        mult = one_hot.grad.squeeze() * one_hot.squeeze()
        input_gradient_list.append(mult)

        # plot subsequence of interest 
        if plot:
            print(f"TF: {tf}")
            print(f"Coordinates: {seqname}:{start}-{end}, ({dist_start} - {dist_start + end - start})")
            viz_sequence.plot_weights(mult.detach().cpu().numpy()[:, dist_start : (dist_start + end - start)], figsize=(10,2))
            plt.show()

    return grad_list, input_gradient_list