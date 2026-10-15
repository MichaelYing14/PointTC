import torch.nn as nn
import torch
import torch.nn.functional as F
from . import symbol_split
from .utils import (
    CoordinateUpsamplingModule, NewRateRefineModule3, NewSnrRefineModule,
    PointNetSetAbstraction, ReconstructionLayer, TransformerBlock,
)
from pytorch3d.loss import chamfer_distance
import math
from torch.nn.functional import gumbel_softmax


class get_model(nn.Module):
    def __init__(self, revise = False, bottleneck_size=300, recon_points=2048, M_QAM=64):
        super(get_model, self).__init__()

        self.revise = revise
        self.bottleneck_size_main, self.bottleneck_size_sub = symbol_split(bottleneck_size)
        self.num_symbols = bottleneck_size
        self.M_QAM = M_QAM
        self.constellation_loc = [-7,-5,-3,-1,1,3,5,7]

        self.sa1_main = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 + 3, mlp=[64, 64, 128], group_all=False)
        self.attn1_main = TransformerBlock(d_points=128, d_model=128, k=32)

        self.attn1_sub = TransformerBlock(d_points=self.bottleneck_size_sub, d_model=self.bottleneck_size_sub, k=32)

        self.sa2_main = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, self.bottleneck_size_main], group_all=False)
        self.attn2_main = TransformerBlock(d_points=self.bottleneck_size_main, d_model=self.bottleneck_size_main, k=64)

        self.feature_dim_convert_sub = nn.Sequential(
            nn.Linear(128, self.bottleneck_size_sub),
        )

        self.feature_downsample_sub = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, int(2*math.sqrt(self.M_QAM)))
        )

        self.feature_downsample_main = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, int(2*math.sqrt(self.M_QAM)))
        )
        
        self.constellation_prob_convert_sub = nn.Sequential(
            nn.Linear(self.bottleneck_size_sub * 2 * int(math.sqrt(self.M_QAM)), self.bottleneck_size_sub * 2 * int(math.sqrt(self.M_QAM))),
            nn.ReLU(inplace=True)
        )

        self.constellation_prob_convert_main = nn.Sequential(
            nn.Linear(self.bottleneck_size_main * 2 * int(math.sqrt(self.M_QAM)),
                      self.bottleneck_size_main * 2 * int(math.sqrt(self.M_QAM))),
            nn.ReLU(inplace=True)
        )

        self.snr_refine_sub = NewSnrRefineModule(int(math.sqrt(self.M_QAM)), int(math.sqrt(self.M_QAM)))
        self.snr_refine_main = NewSnrRefineModule(int(math.sqrt(self.M_QAM)), int(math.sqrt(self.M_QAM)))

        self.attn3 = TransformerBlock(256, 256, 64)

        self.decompression_1 = ReconstructionLayer(recon_points // 32, bottleneck_size, 256)
        self.decompression_2 = ReconstructionLayer(recon_points // 32, bottleneck_size, 256)

        self.coor_reconstruction_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

        self.coor_upsampling1 = CoordinateUpsamplingModule(ratio=4, channels=256, radius=0.10)
        self.coor_upsampling2 = CoordinateUpsamplingModule(ratio=4, channels=256, radius=0.10)

        self.coor_reest_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

    def forward(self, xyz, snr, stat=False):
        B, N, C = xyz.shape
        pc_gd = xyz

        l0_feature = xyz
        l0_xyz = xyz

        l1_xyz_main, l1_feature_main = self.sa1_main(l0_xyz.permute(0, 2, 1), l0_feature.permute(0, 2, 1))
        l1_xyz_sub = l1_xyz_main
        l1_feature_sub = self.feature_dim_convert_sub(l1_feature_main.permute(0, 2, 1))
        l1_feature_main, _ = self.attn1_main(l1_xyz_main.permute(0, 2, 1), l1_feature_main.permute(0, 2, 1))

        x_sub, _ = self.attn1_sub(l1_xyz_sub.permute(0, 2, 1), l1_feature_sub)
        x_sub = self.feature_downsample_sub(x_sub.permute(0, 2, 1))
        x_sub = x_sub.view(-1,self.bottleneck_size_sub * 2 * int(math.sqrt(self.M_QAM)))
        x_sub = self.constellation_prob_convert_sub(x_sub)
        x_sub = x_sub.view(-1, self.bottleneck_size_sub * 2, int(math.sqrt(self.M_QAM)))

        l2_xyz_main, l2_feature_main = self.sa2_main(l1_xyz_main, l1_feature_main.permute(0, 2, 1))
        x_main, _ = self.attn2_main(l2_xyz_main.permute(0, 2, 1), l2_feature_main.permute(0, 2, 1))
        x_main = self.feature_downsample_main(x_main.permute(0, 2, 1))
        x_main = x_main.view(-1, self.bottleneck_size_main * 2 * int(math.sqrt(self.M_QAM)))
        x_main = self.constellation_prob_convert_main(x_main)
        x_main = x_main.view(-1, self.bottleneck_size_main * 2, int(math.sqrt(self.M_QAM)))

        x = torch.cat((x_main,x_sub), dim=1)
        x = x.view(-1, (self.bottleneck_size_main+self.bottleneck_size_sub)*2, int(math.sqrt(self.M_QAM)))
        codewords = gumbel_softmax(x, hard=False, tau=1.5)
        constellation_loc = torch.tensor(self.constellation_loc).to(codewords.device)
        constellation_points = torch.sum(codewords*constellation_loc, dim=-1)
        abs_diff = torch.abs(constellation_points.unsqueeze(-1) - constellation_loc)

        # hard operation
        _, nearest_index = torch.min(abs_diff, dim=-1)
        constellation_points_hard = constellation_loc[nearest_index]

        # soft operation
        softmax_weights = torch.softmax(-abs_diff / 1.5, dim=-1)     # 1.5 temperature
        constellation_points_soft = torch.sum(softmax_weights * constellation_loc, dim=-1)

        # trick to make the gradient flow
        # in foward the hard operation is used
        # in backward the soft operation is used
        constellation_points = constellation_points_hard - constellation_points_soft.detach() + constellation_points_soft
        constellation_points = constellation_points.view(-1, (self.bottleneck_size_main + self.bottleneck_size_sub), 2)

        if stat:
            return constellation_points

        '''normalize'''
        #constellation_points_mean = torch.mean(constellation_points, dim=(1,2)).view(-1,1,1)
        #constellation_points_std =torch.std(constellation_points, dim=(1,2)).view(-1,1,1)
        #normalization_factor = constellation_points_std * (math.sqrt(2))
        #constellation_points = (constellation_points - constellation_points_mean) / normalization_factor
        self.constellation_points_mean, self.constellation_points_std = torch.mean(constellation_points), torch.std(constellation_points)
        self.normalization_factor = self.constellation_points_std * (math.sqrt(2))
        constellation_points = (constellation_points - self.constellation_points_mean) / self.normalization_factor

        '''awgn'''
        n = 1 / (10 ** (snr / 10))
        sqrt_n = n ** 0.5
        noise = torch.randn_like(constellation_points) * sqrt_n / math.sqrt(2)
        noise = noise.to(constellation_points.device)
        constellation_points_hat = constellation_points + noise

        constellation_points_hat_1 = constellation_points_hat[:, :, 0].unsqueeze(-1)
        constellation_points_hat_2 = constellation_points_hat[:, :, 1].unsqueeze(-1)

        decoder_local_feature_1 = self.decompression_1(constellation_points_hat_1.permute(0, 2, 1))
        decoder_local_feature_2 = self.decompression_2(constellation_points_hat_2.permute(0, 2, 1))
        decoder_local_feature = torch.cat((decoder_local_feature_1, decoder_local_feature_2), dim=1)
        new_xyz0 = self.coor_reconstruction_layer(decoder_local_feature)
        new_feature0, _ = self.attn3(new_xyz0, decoder_local_feature)
        if self.revise:
            new_xyz0 = self.coor_reest_layer(new_feature0)

        new_xyz1, new_feature1 = self.coor_upsampling1(new_xyz0, new_feature0)
        new_xyz2, new_feature2 = self.coor_upsampling2(new_xyz1, new_feature1)

        coor_recon = new_xyz2
        cd = chamfer_distance(pc_gd, coor_recon)[0]

        return coor_recon, cd
