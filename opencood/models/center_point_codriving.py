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


from opencood.utils.waypoint2map import waypoints2map_radius


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
            try:
                from opencood.models.sub_modules.dcn_net import DCNNet
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "DCN is enabled in config but mmcv is not installed. "
                    "Either install mmcv-full or remove the 'dcn' section from the config."
                ) from e
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

            # [新增] 从配置中读取半径和高斯方差，如果没有则使用默认值
            self.req_radius = cp_args.get('request_radius', 160)
            self.req_sigma = cp_args.get('sigma_reverse', 2)
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

    def generate_request_map(self, waypoints, H, W):
        """
        根据规划的 waypoints 生成驾驶需求热力图 (Request Map)
        """
        if waypoints is None:
            return None

        # 计算网格分辨率参数: [H, W, res_x, res_y]
        range_x = self.cav_lidar_range[3] - self.cav_lidar_range[0]
        range_y = self.cav_lidar_range[4] - self.cav_lidar_range[1]

        # 计算每个像素代表的物理尺寸倒数
        res_x = W / range_x
        res_y = H / range_y

        # [修改] 使用 self.req_radius 和 self.req_sigma 替代硬编码
        request_map = waypoints2map_radius(
            waypoints.detach().cpu().numpy(),
            radius=self.req_radius,  # 这里改为 self变量
            sigma_reverse=self.req_sigma,  # 这里改为 self变量
            grid_coord=[H, W, res_x, res_y],
            det_range=self.cav_lidar_range
        )
        return request_map

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
        # [修改后的] Directed-CP 核心逻辑
        # =======================================================
        directed_cp_mask = None
        if self.use_directed_cp:
            # 1. 获取基础感知置信度 (Perception Confidence)
            conf_map = psm_single.sigmoid().max(dim=1)[0]
            _, _, H, W = spatial_features_2d.shape

            # 2. [新增] 获取规划需求图 (Planning Request)
            # 这一步将 waypoints 转化为与特征图对齐的热力图
            req_map_np = None
            if waypoints is not None:
                req_map_np = self.generate_request_map(waypoints, H, W)

            mask_list = []
            start_idx = 0

            for b_idx, cav_num in enumerate(record_len.tolist()):
                if cav_num > 0:
                    # A. 提取当前车辆的感知置信度 (CPU numpy)
                    ego_conf = conf_map[start_idx].detach().cpu().numpy()  # Shape: (H, W)

                    # B. [关键] 融合规划信息
                    guidance_map = ego_conf

                    if req_map_np is not None:
                        # 获取当前 batch 对应的 request map
                        ego_req = req_map_np[b_idx]  # Shape: (H, W)

                        # [融合策略]: 加权融合
                        # alpha=0.6: 感知权重 (确保已看到的物体不丢失)
                        # beta=0.4:  规划权重 (确保规划路径上的区域被保留)
                        alpha = 0.6
                        beta = 0.4
                        guidance_map = alpha * ego_conf + beta * ego_req
                        guidance_map = np.clip(guidance_map, 0.0, 1.0)
                        # 可选: 归一化以保持数值稳定性
                        # guidance_map = np.clip(guidance_map, 0, 1)

                    # C. 将融合后的 guidance_map 传给 DAS 计算方向分
                    # 原代码传的是纯感知 ego_conf，现在传的是融合了规划的 map
                    _, das_mask_val = self.rsu_das.compute_das_from_feature_map(guidance_map)

                    # D. 生成空间 Mask
                    spatial_mask = self.rsu_das.create_spatial_direction_mask(H, W, das_mask_val)
                    mask_t = torch.from_numpy(spatial_mask).to(spatial_features_2d.device).float()
                    mask_t = mask_t.unsqueeze(0).unsqueeze(0)
                    mask_list.append(mask_t.repeat(cav_num, 1, 1, 1))

                start_idx += cav_num

            if mask_list:
                dir_mask = torch.cat(mask_list, dim=0)
                # 2. Pose Embedding (0填充)
                pose_emb = torch.zeros_like(spatial_features_2d)
                # 3. QCNet 计算稀疏 Mask
                sparse_mask = self.qc_net(spatial_features_2d, pose_emb, dir_mask, budget=self.comm_budget)
                # 4. Directed-CP 输出转为单通道 mask，交给通信模块统一裁剪
                directed_cp_mask = sparse_mask.mean(dim=1, keepdim=True)
        # =======================================================

        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(batch_dict['spatial_features'],
                                                                              psm_single,
                                                                              record_len,
                                                                              pairwise_t_matrix,
                                                                              self.backbone,
                                                                              waypoints,
                                                                              directed_cp_mask=directed_cp_mask)
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
                                                                              pairwise_t_matrix,
                                                                              directed_cp_mask=directed_cp_mask)

        cls = self.cls_head(fused_feature)
        bbox = self.reg_head(fused_feature)

        box_preds_for_infer = bbox.permute(0, 2, 3, 1).contiguous()
        bbox_temp_list = []
        num_class = int(box_preds_for_infer.shape[3] / 8)
        box_preds_for_infer = box_preds_for_infer.view(box_preds_for_infer.shape[0],
                                                       box_preds_for_infer.shape[1],
                                                       box_preds_for_infer.shape[2],
                                                       num_class,
                                                       8)
        for i in range(num_class):
            box_preds_for_infer_singleclass = box_preds_for_infer[:, :, :, i, :]
            box_preds_for_infer_singleclass = box_preds_for_infer_singleclass.permute(0, 3, 1, 2)
            _, bbox_temp = self.generate_predicted_boxes(cls[:, i, :, :],
                                                         box_preds_for_infer_singleclass)
            bbox_temp_list.append(bbox_temp)
        bbox_temp_list = torch.stack(bbox_temp_list, dim=1)

        _, bbox_temp = self.generate_predicted_boxes(cls, bbox)

        output_dict = {'cls_preds': cls,
                       'reg_preds': bbox_temp,
                       'reg_preds_multiclass': bbox_temp_list,
                       'bbox_preds': bbox}
        result_dict.update({'fused_feature': fused_feature})
        output_dict.update(result_dict)

        _, bbox_temp_single = self.generate_predicted_boxes(psm_single, rm_single)
        output_dict.update({'cls_preds_single': psm_single,
                            'reg_preds_single': bbox_temp_single,
                            'bbox_preds_single': rm_single,
                            'comm_rate': communication_rates})

        psm_single_regroup = self.regroup(psm_single, record_len)
        rm_single_regroup = self.regroup(rm_single, record_len)
        psm_single_ego_list = []
        rm_single_ego_list = []
        for b in range(len(record_len)):
            psm_single_ego_list.append(psm_single_regroup[b][0:1])
            rm_single_ego_list.append(rm_single_regroup[b][0:1])
        psm_single_ego = torch.cat(psm_single_ego_list, dim=0)
        rm_single_ego = torch.cat(rm_single_ego_list, dim=0)

        box_preds_for_infer = rm_single_ego.permute(0, 2, 3, 1).contiguous()
        bbox_temp_list_single = []
        num_class = int(box_preds_for_infer.shape[3] / 8)
        box_preds_for_infer = box_preds_for_infer.view(box_preds_for_infer.shape[0],
                                                       box_preds_for_infer.shape[1],
                                                       box_preds_for_infer.shape[2],
                                                       num_class,
                                                       8)
        for i in range(num_class):
            box_preds_for_infer_singleclass = box_preds_for_infer[:, :, :, i, :]
            box_preds_for_infer_singleclass = box_preds_for_infer_singleclass.permute(0, 3, 1, 2)
            _, bbox_temp = self.generate_predicted_boxes(psm_single_ego[:, i, :, :],
                                                         box_preds_for_infer_singleclass)
            bbox_temp_list_single.append(bbox_temp)
        bbox_temp_list_single = torch.stack(bbox_temp_list_single, dim=1)

        output_dict.update({'cls_preds_single_ego': psm_single_ego,
                            'reg_preds_multiclass_single_ego': bbox_temp_list_single,
                            'bbox_preds_single_ego': rm_single_ego})

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
