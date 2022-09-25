
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List
import pandas as pd

# globals
ALL_TFs = ["Sox2", "Oct4", "Klf4", "Nanog"]


def _smooth_counts(mtx: np.ndarray, window=50):
    step = int(window/2)
    out = np.zeros_like(mtx, dtype=np.float32)
    for strand_idx in range(mtx.shape[-2]):
        for col in range(mtx.shape[-1]):
            start = col - step if col >= step else 0
            end = col + step if col + step < out.shape[-1] else out.shape[-1]-1
            out[:, strand_idx, col] = mtx[:, strand_idx, start:end].sum(axis=-1)  # mean or sum?
    return out


def _get_position_index(query_list, target_list):
    d = {k: v for v, k in enumerate(target_list)}
    index = (d[k] for k in query_list)
    return list(index)


class ChIP_Nexus_Dataset(Dataset):

    def __init__(self, set_name: str, input_dir: str, TF_list: List[str], subset=True, qval_thr=0.0, subsample=None, seed=42):
        self.indir = input_dir 
        self.set_name = set_name
        self.tf_list = TF_list
        self.one_hot_seqs = np.load(f"{self.indir}{self.set_name}_one_hot_seqs.npy") # [batch, bases=4, pwidth=1000]
        self.tf_counts = np.stack([np.load(f"{self.indir}{tf}/{self.set_name}_counts.npy") for tf in self.tf_list], axis=1) # [batch, TF, strand=2, pwidth=1000]
        self.ctrl_counts = np.load(f"{self.indir}patchcap/{self.set_name}_counts.npy") # [batch, TF, strand=2, pwidth=1000]
        self.ctrl_counts_smooth = _smooth_counts(self.ctrl_counts) # [batch, strand=2, pwidth=1000]
        self.region_info = pd.read_csv(f"{self.indir}region_info.tsv", sep="\t") # [batch, strand=2, pwidth=1000]
        self.seqnames  = np.genfromtxt(f"{self.indir}{self.set_name}_seq_names.txt", dtype=str)

        if subset:
            self.region_info = self.region_info.loc[(self.region_info.set==set_name).to_numpy() & 
                                                     np.isin(self.region_info.TF, np.array(self.tf_list)) & 
                                                     (self.region_info.qValue >= qval_thr).to_numpy()]
            if subsample:
                assert subsample > 0 and subsample <= 1
                np.random.seed(seed)
                self.region_info = self.region_info.iloc[np.random.choice(len(self.region_info), int(len(self.region_info) * subsample), replace=False), ]
            idx_keep = _get_position_index(self.region_info.Region.tolist(), self.seqnames.tolist())
            self.one_hot_seqs = self.one_hot_seqs[idx_keep]
            self.tf_counts = self.tf_counts[idx_keep]
            self.ctrl_counts = self.ctrl_counts[idx_keep]
            self.ctrl_counts_smooth = self.ctrl_counts_smooth[idx_keep]


    def __repr__(self):
        return f"{type(self).__name__}\nSet: {self.set_name}\nTFs: {self.tf_list}\nSize: {self.one_hot_seqs.shape[0]}"


    def check_shapes(self):
        print(f"{self.tf_list=}")
        print(f"{self.one_hot_seqs.shape=} [idx, bases, pwidth]")
        print(f"{self.tf_counts.shape=} [idx, TF, strand, pwidth]")
        print(f"{self.ctrl_counts.shape=} [idx, strand, pwidth]")
        print(f"{self.ctrl_counts_smooth.shape=} [idx, strand, pwidth]")


    def __len__(self):
        return self.one_hot_seqs.shape[0]

    
    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx      
        return self.one_hot_seqs[idx], self.tf_counts[idx], self.ctrl_counts[idx], self.ctrl_counts_smooth[idx]


def dummy_shape_predictions(dataset, plot_mean=False):
  from .loss import neg_log_multinomial
  import matplotlib.pyplot as plt
  mean_prediction = dataset.tf_counts.mean(axis=0)
  mean_prediction /= mean_prediction.sum(axis=-1)[:, :, None]
  mean_loss = neg_log_multinomial(k_obs=torch.tensor(dataset.tf_counts), p_pred=torch.tensor(mean_prediction[None, :]).repeat(len(dataset), 1, 1, 1), device="cpu")


  perfect_prediction = dataset.tf_counts.copy()
  perfect_prediction /= (perfect_prediction.sum(axis=-1)[:, :, :, None] + 1e-8)
  perfect_loss = neg_log_multinomial(k_obs=torch.tensor(dataset.tf_counts), p_pred=torch.tensor(perfect_prediction), device="cpu")

  uniform_prediction = np.ones_like(dataset.tf_counts)
  uniform_prediction /= uniform_prediction.sum(axis=-1)[:, :, :, None]
  uniform_loss = neg_log_multinomial(k_obs=torch.tensor(dataset.tf_counts), p_pred=torch.tensor(uniform_prediction), device="cpu")

  print(f"Unfiform Prediction Loss:\t{uniform_loss:.2f}")  
  print(f"Mean Prediction Loss:\t\t{mean_loss:.2f}")
  print(f"Perfect Prediction Loss:\t{perfect_loss:.2f}")

  if plot_mean:
    fig, axis = plt.subplots(4, 1, figsize=(6, 12))
    fig.suptitle("Mean Counts for each TF", fontsize=12)
    for (i, ax), tf in zip(enumerate(axis), dataset.tf_list):
      ax.plot(mean_prediction[i, 0], label="pos")
      ax.plot(-mean_prediction[i, 1], label="neg")
      ax.set_title(tf)
      ax.legend()


def dummy_total_counts_predictions(dataset):
    from torch.nn.functional import mse_loss
    mean_prediction = dataset.tf_counts.sum(axis=-1).mean(axis=0)
    mean_loss = ((torch.log(1 + torch.from_numpy(mean_prediction).repeat(len(dataset), 1, 1)) - torch.log(1 + torch.from_numpy(dataset.tf_counts.sum(axis=-1))))**2).mean()

    perfect_prediction = dataset.tf_counts.sum(axis=-1)
    perfect_loss = ((torch.log(1 + torch.from_numpy(perfect_prediction)) - torch.log(1 + torch.from_numpy(dataset.tf_counts.sum(axis=-1))))**2).mean()

    print(f"Mean Prediction Loss:\t\t{mean_loss:.2f}")
    print(f"Perfect Prediction Loss:\t{perfect_loss:.2f}")


if __name__ == "__main__":

    # test all TFs
    #ChIP_Nexus_Dataset(input_dir="/home/philipp/AML_Final_Project/output_correct/", set_name="train", TF_list=ALL_TFs)

    # test one TF with qValue threshold
    #ChIP_Nexus_Dataset(set_name="train", input_dir="/home/philipp/AML_Final_Project/output_correct/", TF_list=["Sox2"], qval_thr=2**4.5)

    # test subsampling
    ds = ChIP_Nexus_Dataset(set_name="tune", input_dir="/home/philipp/AML_Final_Project/output_correct/", TF_list=["Nanog", "Sox2"], subsample=0.25)

    # test dummy predictions
    dummy_shape_predictions(ds)
    dummy_total_counts_predictions(ds)