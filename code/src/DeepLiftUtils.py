import matplotlib
import matplotlib.pyplot as plt
#plt.style.use('dark_background')
import torch
import torch.nn as nn
from src.architectures import *
from src.utils import * 
from src.loss import *
from src.metrics import *
from src.DeepLiftUtils import *
from captum.attr import DeepLift
import seaborn as sns
from matplotlib import pyplot as plt
from modisco.visualization import viz_sequence



def get_seq_oi(seqname, start, end, dataset, device):
    tmp_df = dataset.region_info.copy().reset_index()
    idx = tmp_df.loc[(tmp_df.seqnames==seqname) & (tmp_df.start > start-1000) & (tmp_df.end < end+1000)].index.to_numpy()[0]
    dist_start = start - tmp_df.start[idx]
    # read in data
    one_hot = torch.unsqueeze(torch.tensor(dataset.one_hot_seqs[idx, :, :]).to(device), axis=0)
    baseline = torch.zeros(one_hot.shape).to(device)
    bias_raw = torch.unsqueeze(torch.tensor(dataset.ctrl_counts[idx, :, :]).to(device), axis=0)
    bias_smooth = torch.unsqueeze(torch.tensor(dataset.ctrl_counts_smooth[idx, :, :]).to(device), axis=0)
    tf_counts = dataset.tf_counts[idx, :, :]
    return tmp_df, idx, dist_start, one_hot, baseline, bias_raw, bias_smooth, tf_counts 

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
    #baseline = torch.zeros(one_hot.shape).to(device)
    bias_raw = torch.unsqueeze(torch.tensor(dataset.ctrl_counts[idx, :, :]).to(device), axis=0)
    bias_smooth = torch.unsqueeze(torch.tensor(dataset.ctrl_counts_smooth[idx, :, :]).to(device), axis=0)

    return one_hot, bias_raw, bias_smooth, idx, dist_start


def mutate_sequence(region):
    new = np.zeros(region.shape)
    idx = np.random.choice(np.arange(4), size=region.shape[-1])
    new[:, idx, np.arange(region.shape[-1])] = 1
    assert np.all(new == 0) == False
    return new


def get_contr_region(seqname, start, end, dataset, model, dl, device, tf_list, output_dir, plot=True, figsize1=(20,2), figsize2=(10,1.5)):
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
    # select sequence of interest
    tmp_df, idx, dist_start, one_hot, baseline, bias_raw, bias_smooth, tf_counts = get_seq_oi(seqname, start, end, dataset, device)
    width = end - start


    # compute contribution scores for each tf
    contr_list = []
    plot_df = {}
    for tf_index, tf in enumerate(tf_list):
        contr = dl.attribute(inputs=one_hot, baselines=baseline, target=(tf_index), additional_forward_args=(bias_raw, bias_smooth, True)).detach().cpu().numpy()
        contr_list.append(contr)

        pred = model.forward(one_hot, bias_raw, bias_smooth, interpretation=False).detach().cpu().numpy().squeeze()
        # scale prediciton with total counts
        pred = pred * tf_counts.sum(axis=-1, keepdims=True)
        plot_df[f"{tf}"] = pred[:, :,dist_start : (dist_start + width+1)]

        if plot:
            # entire sequence original
            plot_weights(contr,
            fontsizes=[20,15,15],
            title = f"{tf} - 1kbp sequence", 
            xlabel=f"{tmp_df.seqnames[idx]}: {tmp_df.start[idx]}-{tmp_df.end[idx]}", 
            ylabel="DeepLift contribution scores",
            subticks_frequency=20, figsize=figsize1)
            plt.savefig(f"{output_dir}{tf}_{seqname}_{start}_{end}_entireSeq_DeepLift.pdf")

            # zoomed into motif region
            plot_weights(contr[:, :,dist_start : (dist_start + width+1)],
            fontsizes=[20,15,15],
            title = f"{tf} - Motif of interest", 
            xlabel=f"{seqname}: {start}-{end}, ({dist_start} - {dist_start + width+1})", 
            ylabel="DeepLift contribution scores",
            subticks_frequency=10, figsize=figsize2)
            plt.savefig(f"{output_dir}{tf}_{seqname}_{start}_{end}_zoomedSeq_DeepLift.pdf")  


            #print(f"TF: {tf}")
            # plot entire sequence
            #print(f"Coordinates: {tmp_df.seqnames[idx]}:{tmp_df.start[idx]}-{tmp_df.end[idx]} (1-1000")
            #viz_sequence.plot_weights(contr.detach().cpu().numpy(), subticks_frequency=10, figsize=figsize1)
            # plot subsequence of interest
            #print(f"Coordinates: {seqname}:{start}-{end}, ({dist_start} - {dist_start + end - start})")
            #viz_sequence.plot_weights(contr.detach().cpu().numpy()[:, :,dist_start : (dist_start + end - start)], subticks_frequency=5, figsize=figsize2)
            fig, axis = plt.subplots(1,2,figsize=(12,4))
            axis[0].plot(tf_counts[tf_index, 0, :], label="true counts", color="green", linewidth=0.8)
            axis[0].plot(-tf_counts[tf_index, 1, :], color="green", linewidth=0.8)
            axis[0].plot(pred[tf_index, 0, :], label="pred", color="blue", linewidth=0.8)
            axis[0].plot(-pred[tf_index, 1, :], color="blue", linewidth=0.8)   
            axis[0].set_xlabel("bp")
            axis[0].set_ylabel("Read counts")
            axis[1].plot(pred[tf_index, 0, :], label="pred", color="blue", linewidth=0.8)
            axis[1].plot(-pred[tf_index, 1, :], color="blue", linewidth=0.8)
            axis[1].set_xlabel("bp")
            axis[1].set_ylabel("Predicted probabilitiy * total counts")
            axis[0].legend()
            axis[1].legend()
            plt.show()

            
    return contr, dist_start, plot_df



def input_gradient(seqname, start, end, dataset, model, device, output_dir, tf_list, plot=True, figsize1=(20,2), figsize2=(10,1.5)):
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
    # select sequence of interest
    tmp_df, idx, dist_start, one_hot, _, bias_raw, bias_smooth, _ = get_seq_oi(seqname, start, end, dataset, device)
    width = end - start

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
        mult = (one_hot.grad.squeeze() * one_hot.squeeze()).detach().cpu().numpy()
        input_gradient_list.append(mult)
        print(one_hot.grad.shape)

        # plot subsequence of interest 
        if plot:
            # entire sequence original
            plot_weights(mult,
            fontsizes=[20,15,15],
            title = f"{tf} - 1kbp region", 
            xlabel=f"{tmp_df.seqnames[idx]}: {tmp_df.start[idx]}-{tmp_df.end[idx]}", 
            ylabel="Input x Gradient",
            subticks_frequency=20, figsize=figsize1)            
            plt.show()
            plt.savefig(f"{output_dir}{tf}_{seqname}_{start}_{end}_entireSeq_GradientInput.pdf")

            # zoomed into motif region
            plot_weights(one_hot.grad.squeeze()[:, dist_start : (dist_start + width+1)],
            fontsizes=[20,15,15],
            title = f"{tf} - Motif of interest", 
            xlabel=f"{seqname}: {start}-{end}, ({dist_start} - {dist_start + width+1})", 
            ylabel="Input x Gradient",
            subticks_frequency=10, figsize=figsize2)
            plt.show()
            plt.savefig(f"{output_dir}{tf}_{seqname}_{start}_{end}_zoomedSeq_GradientInput.pdf")




            #print(f"TF: {tf}")
            #print(f"Coordinates: {seqname}:{start}-{end}, ({dist_start} - {dist_start + end - start})")
            #viz_sequence.plot_weights(mult[:, dist_start : (dist_start + end - start)], figsize=(10,2))
            #plt.show()

    return grad_list, input_gradient_list


def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
           [0.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.2, 0.0],
        ]),
        np.array([
           [1.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.8, 0.0],
        ]),
        np.array([
           [0.225, 0.45],
           [0.775, 0.45],
           [0.85, 0.3],
           [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                 + np.array([left_edge,base])[None,:]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                            facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                            facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
                  width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
                  width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))


default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}
def plot_weights_given_ax(ax, array,
                 figsize=(20,2),
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=1.0,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={},
                 ylabel=""):
    if len(array.shape)==3:
        array = np.squeeze(array)
    assert len(array.shape)==2, array.shape
    if (array.shape[0]==4 and array.shape[1] != 4):
        array = array.transpose(1,0)
    assert array.shape[1]==4 
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        #sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color=colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]                
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)
            #now highlight any desired positions; the key of
    #the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                    width=end_pos-start_pos,
                    height=max_height-min_depth,
                    edgecolor=color, fill=False))
            
    ax.set_xlim(-length_padding, array.shape[0]+length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))
    height_padding = max(abs(min_neg_height)*(height_padding_factor),
                            abs(max_pos_height)*(height_padding_factor))
    ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)
    ax.set_ylabel(ylabel)
    ax.yaxis.label.set_fontsize(15)


def plot_weights(array,
                fontsizes,
                title="",
                xlabel="",
                ylabel="",
                figsize=(20,2),
                 **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    plot_weights_given_ax(ax=ax, array=array,**kwargs)
    ax.set_title(title, fontsize=fontsizes[0])
    ax.set_xlabel(xlabel, fontsize=fontsizes[1])
    ax.set_ylabel(ylabel, fontsize=fontsizes[2])