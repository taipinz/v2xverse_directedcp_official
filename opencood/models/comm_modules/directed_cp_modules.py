import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DirectionControlModule(nn.Module):
    """
    [修复版] 使用 1x1 卷积替代全连接层
    这样可以适应任意尺寸的 H*W，且大幅减少显存占用。
    """

    def __init__(self, in_channels, hidden_dim=64):
        super(DirectionControlModule, self).__init__()
        # 使用 1x1 Conv 替代 Linear
        # input_dim 变成了 in_channels (例如 128)
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        # 输出为 1 个通道的 Mask 权重 (或者是 input_dim 维度用于点乘，通常 mask 是单通道或与C一致)
        # 这里我们输出 in_channels，让每个通道都有独立的权重
        self.conv3 = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, Q0, pose_embedding, dir_mask):
        # Q0: [B, C, H, W]
        # dir_mask: [B, 1, H, W] (需要广播)

        # 融合输入：应用方向掩码 + 叠加位姿信息
        # 注意：dir_mask 需要与 Q0 通道对齐或广播
        masked_features = Q0 * dir_mask + pose_embedding

        # 直接通过卷积处理，不需要 view/flatten
        x = self.relu(self.conv1(masked_features))
        # x = self.dropout(x) # 卷积层通常很少用 dropout，尤其是在 feature map 上，可以注释掉
        x = self.relu(self.conv2(x))
        # x = self.dropout(x)
        x = self.conv3(x)

        return x


class QueryClippingLayer(nn.Module):
    """查询裁剪层"""

    def __init__(self):
        super(QueryClippingLayer, self).__init__()

    def forward(self, QCMs, budget):
        B, C, H, W = QCMs.shape

        # 计算最大查询数量
        if budget <= 1.0:
            max_queries = int(budget * H * W)
        else:
            max_queries = int(budget)

        # 防止 max_queries 为 0
        max_queries = max(1, max_queries)

        # 展平以便排序: [B, C, N]
        QCMs_flat = QCMs.view(B, C, -1)
        sparse_queries = torch.zeros_like(QCMs_flat)

        for b in range(B):
            # 这里简化逻辑：我们基于 Channel 平均值或者最大值来做 Spatial 裁剪？
            # 或者是对每个 Channel 独立裁剪？
            # 通常 Directed-CP 是对 Spatial Domain 裁剪 (传输某些块的全部Channel)
            # 为了保持维度一致，我们对每个 Channel 应用相同的 Mask (基于 Channel 均值)

            # 1. 计算 Spatial 重要性分数 [H*W]
            spatial_score = QCMs_flat[b].mean(dim=0)

            if max_queries >= spatial_score.numel():
                sparse_mask = torch.ones_like(spatial_score)
            else:
                _, top_k_indices = torch.topk(spatial_score, max_queries)
                sparse_mask = torch.zeros_like(spatial_score)
                sparse_mask[top_k_indices] = 1.0

            # 2. 广播回所有 Channel
            sparse_queries[b] = sparse_mask.unsqueeze(0).expand(C, -1)

        return sparse_queries.view(B, C, H, W)


class QCNet(nn.Module):
    """QC-Net 主类"""

    def __init__(self, input_dim, hidden_dim=64):
        super(QCNet, self).__init__()
        # input_dim 这里实际上代表 Channels
        self.direction_control = DirectionControlModule(in_channels=input_dim, hidden_dim=hidden_dim)
        self.query_clipping = QueryClippingLayer()
        self.sigmoid = nn.Sigmoid()

    def forward(self, com_maps, pose_embedding, dir_mask, budget=0.2):
        # 统一转为 float
        com_maps = com_maps.float()
        pose_embedding = pose_embedding.float()
        dir_mask = dir_mask.float()

        # 1. 生成置信度图 QCMs
        QCMs = self.direction_control(com_maps, pose_embedding, dir_mask)
        QCMs = self.sigmoid(QCMs)

        # 2. 根据 Budget 裁剪生成 Mask
        sparse_query_maps = self.query_clipping(QCMs, budget)

        return sparse_query_maps


class RSUDirectionAttentionScore:
    """RSU 辅助方向注意力评分 (代码保持不变)"""

    def __init__(self, num_directions=4, sigma1=1.0, sigma2=1.0):
        self.num_directions = num_directions
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def compute_das_from_feature_map(self, confidence_map, ego_interest_weights=None):
        if isinstance(confidence_map, torch.Tensor):
            confidence_map = confidence_map.detach().cpu().numpy()
        if len(confidence_map.shape) == 3:
            confidence_map = confidence_map.max(axis=0)

        H, W = confidence_map.shape
        h_mid, w_mid = H // 2, W // 2
        quadrants = [
            confidence_map[:h_mid, w_mid:],  # 前右
            confidence_map[:h_mid, :w_mid],  # 前左
            confidence_map[h_mid:, :w_mid],  # 后左
            confidence_map[h_mid:, w_mid:],  # 后右
        ]
        densities = np.array([q.sum() for q in quadrants])
        total_density = densities.sum()
        das_raw = densities / total_density if total_density > 0 else np.ones(4) / 4

        if ego_interest_weights is not None:
            das_combined = das_raw * 0.5 + np.array(ego_interest_weights) * 0.5
        else:
            das_combined = das_raw

        direction_mask = []
        total_das = das_combined.sum()
        for i in range(self.num_directions):
            rel = das_combined[i] / total_das if total_das > 0 else 0
            abs_val = das_combined[i]
            mask_val = 1 if (rel > self.sigma1 or abs_val > self.sigma2) else 0
            direction_mask.append(mask_val)
        return das_combined.tolist(), direction_mask

    def create_spatial_direction_mask(self, H, W, direction_mask):
        spatial_mask = np.zeros((H, W))
        h_mid, w_mid = H // 2, W // 2
        if direction_mask[0]: spatial_mask[:h_mid, w_mid:] = 1
        if direction_mask[1]: spatial_mask[:h_mid, :w_mid] = 1
        if direction_mask[2]: spatial_mask[h_mid:, :w_mid] = 1
        if direction_mask[3]: spatial_mask[h_mid:, w_mid:] = 1
        return spatial_mask