
# Copied/Adapted from: https://github.com/kundajelab/bpnet/blob/master/bpnet/metrics.py
# Publication: 1.Avsec, Ž. et al. Base-resolution models of transcription-factor binding reveal soft motif syntax. Nat Genet 53, 354–366 (2021).
  
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import * 
from numba import jit
import sklearn.metrics as skm
import seaborn as sns

# bpnet.stats
def permute_array(arr, axis=0):
    """Permute array along a certain axis
    Args:
      arr: numpy array
      axis: axis along which to permute the array
    """
    if axis == 0:
        return np.random.permutation(arr)
    else:
        return np.random.permutation(arr.swapaxes(0, axis)).swapaxes(0, axis)


def bin_max_values(seq, binsize):
    assert len(seq.shape) == 3
    if binsize <= 1:
        return seq
    else:                
        n_bins = int(seq.shape[-1] / binsize)
        new = np.zeros([seq.shape[0], seq.shape[1], n_bins])
        for bin in range(n_bins):
            new[:, :, bin] = seq[:, :,  bin*binsize:(bin+1)*binsize].max(axis=-1)
        return new


#@jit(nopython=True)
def bin_counts_amb(x, binsize=2):
    """Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[-1] // binsize
    xout = np.zeros((x.shape[0], x.shape[1], outlen)).astype(float)
    for i in range(outlen):
        iterval = x[:,:,(binsize * i):(binsize * (i + 1))]
        has_amb = np.any(iterval == -1, axis=-1)
        has_peak = np.any(iterval == 1, axis=-1)
        # if no peak and has_amb -> -1
        # if no peak and no has_amb -> 0
        # if peak -> 1
        xout[:, :, i] = (has_peak - (1 - has_peak) * has_amb).astype(float)
    return xout



def plot_prc(test_dataset, test_pred, tf_index, tf_name, plot = True):
    true_counts = test_dataset.tf_counts.copy()
    #subset for one tf
    tf_counts = true_counts[:, tf_index, :, :]
    test_pred = test_pred.cpu().numpy().copy()
    assert np.allclose(test_pred.sum(axis=-1), 1)
    # subset for one tf
    tf_pred = test_pred[:, tf_index, :, :]
    binary_labels, pred_subset, random = binary_labels_from_counts(tf_counts, tf_pred, verbose=False)
    precision, recall, thresholds = skm.precision_recall_curve(binary_labels, pred_subset)
    if plot:
        plt.plot(precision, recall,  label=f"{tf_name}")
        plt.title(f"Precision-Recall Curve: {tf_name}")
        plt.xlabel("recall")
        plt.ylabel("precision")
    else:
        return precision, recall, thresholds


def binary_labels_from_counts(counts, pred, pos_threshold=0.015, neg_threshold=0.005, min_reads=2.5, verbose=False):
    """Create ground truth binary labels from observed counts 
    Args:
      counts (numpy array, B x S x N): This contains the observed counts for each strand.
            With B the batch index, S the strand and N the 1000bp sequence
      pred (numpy array, B x S x N): This contains the predicted profile shape (probability distribution) for the same sequences as in counts.
      pos_threshold (float): Percentage of total counts at position i needs to be bigger or equal to this value for the positive class.
      neg_threshold (float): Percentage of total counts at position i needs to be bigger or equal to this value for the negative class.
      min_reads (float): Minimum number of reads required at a position with percentage of total counts bigger/equal to pos_threshold.
      verbose (boolean): Print out information.
    """

    ### GROUND TRUTH LABELS
    # make sure that a positive position has more than the minimum number of required reads.
    keep = counts.sum(axis=-1).mean(axis=-1) > min_reads / pos_threshold
    counts = counts[keep]
    assert (counts.sum(axis=-1).mean(axis=-1) > min_reads / pos_threshold).sum() == counts.shape[0]

    count_fracs = counts / counts.sum(axis=-1,keepdims=True)
    assert np.allclose(count_fracs.sum(axis=-1), 1)

    if verbose:
        print(f"Count fractions per position. min:{count_fracs.min()}, max: {count_fracs.max()}, mean:{count_fracs.mean()}")
    # create binary labels
    labels = (count_fracs >= pos_threshold).astype(int)
    ambiguous = (count_fracs >= neg_threshold) & (count_fracs < pos_threshold)
    labels[ambiguous] = -1

    if verbose: 
        print(f"positive fraction: {(labels==1).sum() / (labels != -1).sum()}")

    # flatten the labels, because we don't need the information about the sequences anymore
    labels = np.ravel(labels)
    # in our ground truth labels we do not want ambiguous positions
    mask = labels != -1
    labels = labels[mask]
    

    ### RANDOM PROFILE SHAPE
    random = permute_array(permute_array(pred[keep], axis=1), axis=0)
    random = np.ravel(random)[mask]

    ### PREDICTED PROFILE SHAPES
    # subset for positions with more than 2.5 reads in the ground truth
    pred = pred[keep]
    if verbose: 
        print(f"Predicted probabilites per position. min: {pred.min()}, max: {pred.max()}, mean: {pred.mean()}")
    # subset for non-ambigious positions
    pred = np.ravel(pred)[mask]
    assert pred.shape == labels.shape
    assert pred.shape == random.shape


    return labels, pred, random




def compute_auprc_bins(counts, pred, patchcap, pos_threshold=0.015, neg_threshold=0.005, 
min_reads=2.5, binsizes=[1, 2, 4, 6, 8, 10], verbose=False):
    """Create ground truth binary labels from observed counts 
    Args:
      counts (numpy array, B x S x N): This contains the observed counts for each strand.
            With B the batch index, S the strand and N the 1000bp sequence
      pred (numpy array, B x S x N): This contains the predicted profile shape (probability distribution) for the same sequences as in counts.
      pos_threshold (float): Percentage of total counts at position i needs to be bigger or equal to this value for the positive class.
      neg_threshold (float): Percentage of total counts at position i needs to be bigger or equal to this value for the negative class.
      min_reads (float): Minimum number of reads required at a position with percentage of total counts bigger/equal to pos_threshold.
      binsizes (list of integers): Different binsizes used to bin the bp of a sequence together and compute auPRC on bins.
      verbose (boolean): Print out information.
    """
    ### GROUND TRUTH LABELS
    keep = counts.sum(axis=-1).mean(axis=-1) > min_reads / pos_threshold
    if verbose:
        print(f"Number of sequences we keep: {keep.sum()}. Number of sequences in total: {counts.shape}")
    counts = counts[keep]
    assert (counts.sum(axis=-1).mean(axis=-1) > min_reads / pos_threshold).sum() == counts.shape[0]

    count_fracs = counts / counts.sum(axis=-1,keepdims=True)
    assert np.allclose(count_fracs.sum(axis=-1), 1)
    if verbose:
        print(f"Count fractions per position. min:{count_fracs.min()}, max: {count_fracs.max()}, mean:{count_fracs.mean()}")
    labels = (count_fracs >= pos_threshold).astype(int)
    ambiguous = (count_fracs >= neg_threshold) & (count_fracs < pos_threshold)
    labels[ambiguous] = -1



    ### RANDOM PROFILE SHAPE
    random = permute_array(permute_array(pred[keep], axis=-1), axis=0)


    ### AVERAGE PROFILE SHAPE FOR THIS TF
    average_profile = counts.mean(axis=0)
    average_profile /= average_profile.sum(axis=-1, keepdims=True)
    average_profile = np.tile(average_profile, (len(counts), 1, 1))

    ### PATCHCAP PROFILE
    patchcap=patchcap[keep]
    # keep only profiles where the sum of total counts is not zero
    patch_keep = ((patchcap.sum(axis=-1) > 0).sum(axis=-1) == 2)
    patchcap = patchcap[patch_keep]
    patchcap /= patchcap.sum(axis=-1, keepdims=True)
    assert np.all(np.isnan(patchcap)) == False

    ### PROFILE SHAPE PREDICTED
    if verbose:
        print(f"Predicted probability per position. min: {pred.min()}, max: {pred.max()}, mean: {pred.mean()}")
    # subset for positions with more than 2.5 reads in the ground truth
    pred = pred[keep]



    out = []
    for bins in binsizes:
        ### GROUND TRUTH LABELS
        # bin the binary class labels
        # flatten the labels, because we don't need the information about the sequences anymore
        bin_labels = np.ravel(bin_counts_amb(labels, binsize=bins))
        bin_labels_patch = labels[patch_keep]
        bin_labels_patch = np.ravel(bin_counts_amb(bin_labels_patch, binsize=bins))
        patch_mask = (bin_labels_patch != -1)
        bin_labels_patch = bin_labels_patch[patch_mask]
        if verbose: 
            print(f"Number of bins : {bins}")
            print(f"Positive labels: {(bin_labels == 1).sum()/(bin_labels != -1).sum()}")
            print(f"Ambiguous labels: {(bin_labels == -1).sum()/np.ravel(bin_labels).shape[0]}")

        # in our ground truth labels we do not want ambiguous positions
        mask = bin_labels != -1
        bin_labels = bin_labels[mask]
    
        ### RANDOM PROFILE SHAPE
        bin_random = np.ravel(bin_max_values(random, binsize=bins))[mask]
        bin_average = np.ravel(bin_max_values(average_profile, binsize=bins))[mask]
        bin_patchcap = np.ravel(bin_max_values(patchcap, binsize=bins))[patch_mask]

        ### PREDICTED PROFILE SHAPES
        bin_pred = np.ravel(bin_max_values(pred, binsize=bins))[mask]
        assert bin_pred.shape == bin_labels.shape
        assert bin_pred.shape == bin_random.shape

        au = skm.average_precision_score(bin_labels, bin_pred)
        au_random = skm.average_precision_score(bin_labels, bin_random)
        au_average = skm.average_precision_score(bin_labels, bin_average)
        au_patchcap = skm.average_precision_score(bin_labels_patch, bin_patchcap)

        out.append({"binsize": bins,
                    "auprc": au,
                    "random_auprc": au_random,
                    "average_auprc": au_average,
                    "patchcap_auprc": au_patchcap
                    })


    return out





# old versions without random profile and patchcap
def compute_auprc_bins_old(counts, pred, pos_threshold=0.015, neg_threshold=0.005, 
min_reads=2.5, binsizes=[1, 2, 4, 6, 8, 10], verbose=False):
    """Create ground truth binary labels from observed counts 
    Args:
      counts (numpy array, B x S x N): This contains the observed counts for each strand.
            With B the batch index, S the strand and N the 1000bp sequence
      pred (numpy array, B x S x N): This contains the predicted profile shape (probability distribution) for the same sequences as in counts.
      pos_threshold (float): Percentage of total counts at position i needs to be bigger or equal to this value for the positive class.
      neg_threshold (float): Percentage of total counts at position i needs to be bigger or equal to this value for the negative class.
      min_reads (float): Minimum number of reads required at a position with percentage of total counts bigger/equal to pos_threshold.
      binsizes (list of integers): Different binsizes used to bin the bp of a sequence together and compute auPRC on bins.
      verbose (boolean): Print out information.
    """
    ### GROUND TRUTH LABELS
    keep = counts.sum(axis=-1).mean(axis=-1) > min_reads / pos_threshold
    counts = counts[keep]
    assert (counts.sum(axis=-1).mean(axis=-1) > min_reads / pos_threshold).sum() == counts.shape[0]

    count_fracs = counts / counts.sum(axis=-1,keepdims=True)
    assert np.allclose(count_fracs.sum(axis=-1), 1)
    if verbose:
        print(f"Count fractions per position. min:{count_fracs.min()}, max: {count_fracs.max()}, mean:{count_fracs.mean()}")
    labels = (count_fracs >= pos_threshold).astype(int)
    ambiguous = (count_fracs >= neg_threshold) & (count_fracs < pos_threshold)
    labels[ambiguous] = -1

    ### RANDOM PROFILE SHAPE
    random = permute_array(permute_array(pred[keep], axis=1), axis=0)

    ### PROFIEL SHAPE PREDICTED
    if verbose:
        print(f"Predicted probability per position. min: {pred.min()}, max: {pred.max()}, mean: {pred.mean()}")
    # subset for positions with more than 2.5 reads in the ground truth
    pred = pred[keep]



    out = []
    for bins in binsizes:
        ### GROUND TRUTH LABELS
        # bin the binary class labels
        # flatten the labels, because we don't need the information about the sequences anymore
        bin_labels = np.ravel(bin_counts_amb(labels, binsize=bins))
        # in our ground truth labels we do not want ambiguous positions
        mask = bin_labels != -1
        bin_labels = bin_labels[mask]
    
        ### RANDOM PROFILE SHAPE
        bin_random = np.ravel(bin_max_values(random, binsize=bins))[mask]

        ### PREDICTED PROFILE SHAPES
        bin_pred = np.ravel(bin_max_values(pred, binsize=bins))[mask]
        assert bin_pred.shape == bin_labels.shape
        assert bin_pred.shape == bin_random.shape

        au = skm.average_precision_score(bin_labels, bin_pred)
        au_random = skm.average_precision_score(bin_labels, bin_random)

        out.append({"binsize": bins,
                    "auprc": au,
                    "random_auprc": au_random
                    })


    return out