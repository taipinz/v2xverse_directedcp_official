import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.codriving_attn import CoDriving

# [新增] 引入 Directed-CP 模块
from opencood.models.comm_modules.directed_cp_modules import QCNet, RSUDirectionAttentionScore


class centerpointcodriving(nn.Module):
    def __init__(self, args):
        super(centerpointcodriving, self).__init__()

        # Pillar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.voxel_size = args['voxel_size']
        self.out_size_factor = args['out_size_factor']
        self.cav_lidar_range = args['lidar_range']

        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        # shrink
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        # compression
        self.compression = False
        if 'compression' in args and args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(self.out_channel, args['compression'])

        self.dcn = False
        if 'dcn' in args:
            self.dcn = True
            self.dcn_net = DCNNet(args['dcn'])

        # [关键新增] Directed-CP 初始化
        self.use_directed_cp = args.get('use_directed_cp', False)
        if self.use_directed_cp:
            print(">>> Directed-CP Module Enabled in CenterPoint <<<")
            cp_args = args.get('directed_cp_args', {})
            self.rsu_das = RSUDirectionAttentionScore(
                sigma1=cp_args.get('sigma1', 0.1),
                sigma2=cp_args.get('sigma2', 0.1)
            )
            # Feature Map 尺寸通常是 100x252
            self.qc_net = QCNet(input_dim=self.out_channel, hidden_dim=cp_args.get('hidden_dim', 64))
            self.comm_budget = cp_args.get('comm_budget', 0.2)
        else:
            print(">>> Directed-CP Module Disabled <<<")

        self.fusion_net = CoDriving(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 8 * args['anchor_number'], kernel_size=1)
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()

        self.early_flag = args.get('early_fusion', False)
        self.init_weight()

    def init_weight(self):
        pi = 0.01
        nn.init.constant_(self.cls_head.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.reg_head.weight, mean=0, std=0.001)

    def backbone_fix(self):
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False
        for p in self.scatter.parameters():
            p.requires_grad = False
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False
        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict, waypoints=None):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)

        # =======================================================
        # [关键新增] Directed-CP 核心逻辑
        # =======================================================
        if self.use_directed_cp:
            # 1. 计算 Mask
            conf_map = psm_single.sigmoid().max(dim=1)[0]
            _, _, H, W = spatial_features_2d.shape
            mask_list = []
            start_idx = 0

            for b_idx, cav_num in enumerate(record_len):
                if cav_num > 0:
                    ego_conf = conf_map[start_idx].detach().cpu().numpy()
                    _, das_mask_val = self.rsu_das.compute_das_from_feature_map(ego_conf)
                    spatial_mask = self.rsu_das.create_spatial_direction_mask(H, W, das_mask_val)
                    mask_t = torch.from_numpy(spatial_mask).to(spatial_features_2d.device).float()
                    mask_t = mask_t.unsqueeze(0).unsqueeze(0)
                    mask_list.append(mask_t.repeat(cav_num, 1, 1, 1))
                start_idx += cav_num

            if mask_list:
                dir_mask = torch.cat(mask_list, dim=0)
                # 2. Pose Embedding (0填充)
                pose_emb = torch.zeros_like(spatial_features_2d)
                # 3. QCNet
                sparse_mask = self.qc_net(spatial_features_2d, pose_emb, dir_mask, budget=self.comm_budget)
                # 4. 应用到 spatial_features_2d
                spatial_features_2d = spatial_features_2d * sparse_mask

                # 5. 应用到 Multi-scale features (如果有)
                if self.multi_scale and 'spatial_features' in batch_dict:
                    features_ms = batch_dict['spatial_features']
                    if isinstance(features_ms, list):
                        for i in range(len(features_ms)):
                            mask_resized = F.interpolate(sparse_mask, size=features_ms[i].shape[-2:], mode='nearest')
                            features_ms[i] = features_ms[i] * mask_resized
        # =======================================================

        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(batch_dict['spatial_features'],
                                                                              psm_single,
                                                                              record_len,
                                                                              pairwise_t_matrix,
                                                                              self.backbone,
                                                                              waypoints)
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        elif self.early_flag:
            fused_feature_tuple = self.regroup(spatial_features_2d, record_len)
            feature_bank = []
            for feature_ in fused_feature_tuple:
                feature_bank.append(feature_[0])
            fused_feature = torch.stack(feature_bank, dim=0)
            result_dict = {}
            communication_rates = 0
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features_2d,
                                                                              psm_single,
                                                                              record_len,
                                                                              pairwise_t_matrix)

        cls = self.cls_head(fused_feature)
        bbox = self.reg_head(fused_feature)

        # (这里是原有的 box 生成代码，保持不变即可，为了节省篇幅，省略部分重复代码，请保留原文件中的 generate_predicted_boxes 及后续处理)
        # ... [请保留你原文件这里到 return output_dict 的所有代码] ...

        # 以下是占位符，请确保保留了原有的后处理代码
        box_preds_for_infer = bbox.permute(0, 2, 3, 1).contiguous()
        # ... 省略中间解码过程 ...
        # 注意：这里需要你复制原文件中从 box_preds_for_infer 开始直到 return 的代码

        # 简单起见，我把 output_dict 的构造补全，防止你复制漏了：
        # (请将原代码中 generate_predicted_boxes 和 output_dict 构造部分完整粘回这里)
        # 为确保运行，我写一个简化的返回（你需要用原代码替换这部分）：
        _, bbox_temp = self.generate_predicted_boxes(cls, bbox)
        output_dict = {'cls_preds': cls, 'reg_preds': bbox_temp, 'bbox_preds': bbox}
        result_dict.update({'fused_feature': fused_feature})
        output_dict.update(result_dict)
        # 单车结果更新...
        _, bbox_temp_single = self.generate_predicted_boxes(psm_single, rm_single)
        output_dict.update({'cls_preds_single': psm_single, 'bbox_preds_single': rm_single})

        return output_dict

    # 这里的 generate_predicted_boxes 方法保持原样
    def generate_predicted_boxes(self, cls_preds, box_preds, dir_cls_preds=None):
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        batch, H, W, code_size = box_preds.size()
        box_preds = box_preds.reshape(batch, H * W, code_size)
        batch_reg = box_preds[..., 0:2]
        h = box_preds[..., 3:4] * self.out_size_factor * self.voxel_size[0]
        w = box_preds[..., 4:5] * self.out_size_factor * self.voxel_size[1]
        l = box_preds[..., 5:6] * self.out_size_factor * self.voxel_size[2]
        batch_dim = torch.cat([h, w, l], dim=-1)
        batch_hei = box_preds[..., 2:3] * self.out_size_factor * self.voxel_size[2] + self.cav_lidar_range[2]
        batch_rots = box_preds[..., 6:7]
        batch_rotc = box_preds[..., 7:8]
        rot = torch.atan2(batch_rots, batch_rotc)
        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)
        xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
        ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]
        xs = xs * self.out_size_factor * self.voxel_size[0] + self.cav_lidar_range[0]
        ys = ys * self.out_size_factor * self.voxel_size[1] + self.cav_lidar_range[1]
        batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=2)
        return cls_preds, batch_box_preds