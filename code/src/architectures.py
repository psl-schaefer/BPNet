
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayers(nn.Module):
    def __init__(self, n_dil_layers: int, out_channels=64, size_first_kernel=25, size_dil_kernel=3):
        super(ConvLayers, self).__init__()
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(in_channels=4, 
                                          out_channels=out_channels, 
                                          kernel_size=size_first_kernel,
                                          padding = "same", 
                                          padding_mode="zeros"))
        
        for i in range(1, n_dil_layers+1):
            self.conv_layers.append(nn.Conv1d(in_channels=out_channels, 
                                              out_channels=out_channels, 
                                              kernel_size=size_dil_kernel,
                                              padding="same", 
                                              padding_mode="zeros", 
                                              dilation = 2**i)) # dilation rate doubles at every step

    def forward(self, input):
        x = F.relu(self.conv_layers[0].forward(input))
        # dilated layers with skip connections
        for i in range(1, len(self.conv_layers)):
            out = F.relu(self.conv_layers[i].forward(x))  # TODO: test drop out
            x = x + out
        return x


class TotalCountHead(nn.Module):
    def __init__(self, bias_track: bool, in_channels=64):
        super(TotalCountHead, self).__init__()
        self.bias_track = bias_track
        self.fc1 = nn.Linear(in_features=in_channels, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=2)
        if self.bias_track:
            self.bias_weights = nn.Parameter(torch.tensor([0.01, 0.01], dtype=torch.float32))  # TODO: Initialization

    def forward(self, bottleneck, bias_raw):
        bias = self.bias_weights * torch.log(1 + bias_raw.sum(axis=-1))
        x0 = bottleneck.mean(axis=-1)  # global average pooling
        x1 = self.fc1.forward(x0)
        x2 = self.fc2.forward(x1)
        return x2 + bias


class ProfileShapeHead(nn.Module):
    def __init__(self, bias_track: bool, in_channels=64):
        super(ProfileShapeHead, self).__init__()
        self.bias_track = bias_track
        # with kernel size 25, we loose (25-1)/2 = 12 bp on both sides, which is why we need a padding of 12
        self.deconv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=2, kernel_size=(25), padding=12)
        if self.bias_track:
            self.bias_weights = nn.Parameter(torch.tensor([0.01, 0.01], dtype=torch.float32))  # TODO: Initialization
        
    def forward(self, bottleneck, bias_raw, bias_smooth):
        prediction = self.deconv.forward(bottleneck)
        if self.bias_track:
            bias = self.bias_weights[0] * bias_raw + self.bias_weights[1] * bias_smooth
            return prediction + bias
        return prediction

class BPNet(nn.Module):
    def __init__(self, n_dil_layers, TF_list, conv_channels=64, pred_total=True, bias_track=True, size_first_kernel=25, size_dil_kernel=3):
        super(BPNet, self).__init__()
        self.base_model = ConvLayers(n_dil_layers, out_channels=conv_channels, size_first_kernel=size_first_kernel, size_dil_kernel=size_dil_kernel)
        self.head_description = TF_list
        self.pred_total = pred_total
        self.profile_heads = nn.ModuleList([ProfileShapeHead(bias_track, in_channels=conv_channels) for _ in range(len(self.head_description))])
        if self.pred_total:
            self.count_heads = nn.ModuleList([TotalCountHead(bias_track, in_channels=conv_channels) for _ in range(len(self.head_description))])
        
    def forward(self, sequence, bias_raw, bias_smooth, interpretation=False):
        bottleneck = self.base_model.forward(sequence)
        profile_shapes = torch.stack([head.forward(bottleneck, bias_raw, bias_smooth) for head in self.profile_heads], dim=1)
        softmax_profile = F.softmax(profile_shapes, dim=-1)
        if interpretation:
            return (softmax_profile.detach()*profile_shapes).sum(axis=-1).squeeze().mean(axis=-1)
        elif self.pred_total:
            # use softplus here to ensure that the output is always positive
            total_counts = F.softplus(torch.stack([head.forward(bottleneck, bias_raw) for head in self.count_heads], dim=1))
            return (softmax_profile, total_counts)
        else:
            return F.softmax(profile_shapes, dim=-1)


if __name__ == "__main__":

    # tune data
    from utils import *
    from torch.utils.data import DataLoader
    small = ChIP_Nexus_Dataset("tune", input_dir="/home/philipp/AML_Final_Project/output_correct/", TF_list=["Sox2"])
    small_loader = DataLoader(small, batch_size=32)

    # construct BPNet for one TF, no bias correction, only shape prediction
    small_model1 = BPNet(9, TF_list=["Sox2"], pred_total=False, bias_track=False)
    one_hot, tf_counts, ctrl_counts, ctrl_smooth = next(small_loader.__iter__())
    #small_model1.forward(one_hot, ctrl_counts, ctrl_smooth)
    # deconv = nn.ConvTranspose1d(in_channels=64, out_channels=2, kernel_size=(25), padding=12)
    # comp = nn.Conv1d(in_channels=64, out_channels=2, kernel_size=25, padding="same", padding_mode="zeros")
    # [x.shape for x in list(deconv.parameters())] -> [torch.Size([64, 2, 25]), torch.Size([2])]
    # [x.shape for x in list(comp.parameters())] -> [torch.Size([2, 64, 25]), torch.Size([2])]
    # [x.shape for x in list(self.base_model.conv_layers[0].parameters())] -> [torch.Size([64, 4, 25]), torch.Size([64])]

    # construct BPnet for one TF, with bias correction, shape and total count prediction
    small_model2 = BPNet(9, TF_list=["Sox2"], pred_total=True, bias_track=True)
    small_model2.forward(one_hot, ctrl_counts, ctrl_smooth)


    large = ChIP_Nexus_Dataset("tune", input_dir="/home/philipp/AML_Final_Project/output_correct/", TF_list=ALL_TFs)
    large_loader = DataLoader(large, batch_size=32)
    one_hot, tf_counts, ctrl_counts, ctrl_smooth = next(large_loader.__iter__())
    # construct BPNet for all TFs
    large_model1 = BPNet(9, TF_list=ALL_TFs, pred_total=False, bias_track=False)

    # construct BPNet for all TFs
    large_model2 = BPNet(9, TF_list=ALL_TFs, pred_total=True, bias_track=True)
    large_model2.forward(one_hot, ctrl_counts, ctrl_smooth)

    # Appendix
    # bar = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, bias=False, padding=0)
    # bar.state_dict()["weight"][:] = torch.tensor(np.array([2, 4, 2])[None, None, :])
    # input = torch.tensor(np.arange(10)[None, None, :]).to(torch.float)