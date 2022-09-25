
import torch

def neg_log_multinomial(k_obs: torch.Tensor, p_pred: torch.Tensor, device, eps=1e-8):    
    """Compute the negative log multinomial ("likelikhood")
    Params:
        k_obs: matrix of size B x TF x S x L
            B = batch size. 
            TF = number of TFs
            S = pos and neg strand
            L = length of sequence. 
            k_obs[b, tf A, pos, l] = number of counts for sequence b 
            for tf A on the positive strand at position l.
        p_pred: matrix of size B x TF x S x L
            p_pred[b, tf A, pos, l]  is the predicted probability of observing 
            a read at postion l of the sequence b for tf A on the positive strand.
    Returns:
        vector of size B: each entry is the loss computed for one training instance/sequence. 
    """
    n_obs = k_obs.sum(axis=-1).to(device) #[B x TF x S]
    loss = (k_obs+1).lgamma().sum(axis=-1).to(device) \
    - (n_obs+1).lgamma().to(device) \
    - (k_obs * torch.log(p_pred+eps)).sum(axis=-1).to(device)
    return loss.mean()


if __name__ == "__main__":

    # tune data
    from src.utils import *
    from architectures import BPNet
    from torch.utils.data import DataLoader
    large = ChIP_Nexus_Dataset("tune", input_dir="/home/philipp/AML_Final_Project/output_correct/", TF_list=ALL_TFs)
    large_loader = DataLoader(large, batch_size=32)
    one_hot, tf_counts, ctrl_counts, ctrl_smooth = next(large_loader.__iter__())

    # construct BPNet for all TFs
    large_model = BPNet(9, TF_list=ALL_TFs, pred_total=False, bias_track=True)
    count_pred = large_model.forward(one_hot, ctrl_counts, ctrl_smooth)
    loss = neg_log_multinomial(k_obs=tf_counts, p_pred=count_pred, device="cpu")