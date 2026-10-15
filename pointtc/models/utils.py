"""Point-cloud operators used by the released PointTC models.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    #grouped_xyz_norm = grouped_xyz

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: point cloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled point cloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    Note:
        the grouped points do not need to be the nearest K points, but must in the ball
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm?
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


class PointNetSetAbstraction(nn.Module):

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        xyz = xyz.permute(0, 2, 1)

        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)), inplace=True)

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class ReconstructionLayer(nn.Module):
    def __init__(self, ratio, input_channel, output_channel):
        super(ReconstructionLayer, self).__init__()
        self.deconv_features = nn.ConvTranspose1d(input_channel, output_channel, ratio, stride=ratio)

    def forward(self, x):
        feature = self.deconv_features(x.permute(0, 2, 1)).permute(0, 2, 1)
        return feature


class CoordinateUpsamplingModule(nn.Module):
    def __init__(self, ratio, channels=256, radius=0.05):
        super(CoordinateUpsamplingModule, self).__init__()
        self.ratio = ratio
        self.radius = radius
        self.output_channels = channels
        self.transl_F = nn.Sequential(nn.Linear(channels, out_features=channels // 2),
                                      nn.LeakyReLU())

        self.transl_C = nn.Sequential(nn.Linear(3, out_features=channels // 2),
                                      nn.LeakyReLU())

        self.transl_nn = nn.Sequential(nn.Linear(channels, out_features=channels // 2),
                                       nn.LeakyReLU(),
                                       nn.Linear(in_features=channels // 2, out_features=3 * self.ratio),
                                       nn.Tanh())

        self.feature_nn = nn.Sequential(nn.Linear(channels, out_features=channels // 2),
                                        nn.LeakyReLU(),
                                        nn.Linear(in_features=channels // 2, out_features=channels * self.ratio),
                                        nn.LeakyReLU())

    def forward(self, xyz, feature):
        feature = self.transl_F(feature)
        xyz_mlp = self.transl_C(xyz)
        feature = torch.cat((feature, xyz_mlp), dim=2)

        offset = self.transl_nn(feature)
        offset = offset.reshape((offset.shape[0], offset.shape[1], self.ratio, 3)) * self.radius

        new_xyz = (xyz.unsqueeze(2) + offset).reshape((offset.shape[0], offset.shape[1] * self.ratio, 3))

        new_feature = self.feature_nn(feature).reshape(
            (offset.shape[0], offset.shape[1] * self.ratio, self.output_channels))

        return new_xyz, new_feature


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn


class NewSnrRefineModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NewSnrRefineModule, self).__init__()
        self.feature_path = nn.Sequential(
            nn.Linear(2*in_dim, 2*in_dim),
            nn.ReLU(),
            nn.Linear(2*in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid()
            # Todo Sigmoid or ReLU
        )

    def forward(self, feature, snr_list):
        snr_list_scale = (snr_list / 10).unsqueeze(2)
        snr_list_scale = snr_list_scale.repeat(1, feature.shape[1], feature.shape[2])
        feature_out = self.feature_path(torch.cat([feature, snr_list_scale], dim=-1))
        # Residual connection
        feature_out = feature + feature_out
        return feature_out


class NewRateRefineModule3(nn.Module):
    def __init__(self, bottleneck_size_main, bottleneck_size_sub, rate_levels):
        super(NewRateRefineModule3, self).__init__()

        layers = []
        self.bottleneck_size_main = bottleneck_size_main
        self.bottleneck_size_sub = bottleneck_size_sub
        self.rate_levels = rate_levels
        dim = self.bottleneck_size_main + self.bottleneck_size_sub

        # Gradually reduce the dimension by a factor of 4 until it reaches 5 or below
        while dim > rate_levels:
            next_dim = max(5, dim // 4)  # Ensure the last layer dimension is 5
            layers.append(nn.Linear(dim, next_dim))
            if next_dim > 5:  # Add ReLU for all layers except the last
                layers.append(nn.BatchNorm1d(next_dim))
                layers.append(nn.ReLU())
            dim = next_dim

        # Combine all layers into a sequential model
        self.progressive_mlp = nn.Sequential(*layers)

    def one_hot_to_thermal(self, one_hot):
        thermal = torch.zeros_like(one_hot)

        # Convert one-hot to thermal encoding
        for i in range(self.rate_levels):
            thermal[:, i] = one_hot[:, i:].sum(dim=-1)

        return thermal

    def generate_mask(self, thermal):
        mask = torch.zeros(thermal.shape[0], self.bottleneck_size_sub)
        segment_length = self.bottleneck_size_sub // self.rate_levels

        for i in range(self.rate_levels):
            start_index = i * segment_length
            end_index = start_index + segment_length
            mask[:,start_index:end_index] = thermal[:,i].unsqueeze(-1).repeat(1, segment_length)

        return mask

    def forward(self, feature):
        feature = feature.view(feature.shape[0], self.bottleneck_size_main + self.bottleneck_size_sub, -1)
        feature_pooling = torch.max(feature,-1).values
        logits = self.progressive_mlp(feature_pooling)
        prob = gumbel_softmax(logits, hard=True, tau=1.5)
        thermal = self.one_hot_to_thermal(prob)
        mask = self.generate_mask(thermal).unsqueeze(-1).to(feature.device)
        symbol_num_sub = (torch.sum(thermal,-1) / self.rate_levels * self.bottleneck_size_sub)

        return mask, symbol_num_sub
