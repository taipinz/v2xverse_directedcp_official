from matplotlib import pyplot as plt
import numpy as np
import copy
import torch

from opencood.tools.inference_utils import get_cav_box
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev
from opencood.utils import box_utils


def visualize(infer_result, pcd, pc_range, save_path, method='3d', left_hand=False):
    """
    Visualize the prediction, ground truth with point cloud together.
    """

    if method == "first_person":
        plt.figure()
    else:
        plt.figure(figsize=[(pc_range[3] - pc_range[0]) / 10, (pc_range[4] - pc_range[1]) / 10])

    pc_range = [int(i) for i in pc_range]
    pcd_np = pcd.cpu().numpy()

    attrib_list = ['pred_box_tensor', 'pred_score', 'gt_box_tensor', 'score_tensor']
    for attrib in attrib_list:
        if attrib in infer_result:
            # 1. 处理 List 类型
            if isinstance(infer_result[attrib], list):
                infer_result_tensor = []
                for i in range(len(infer_result[attrib])):
                    if infer_result[attrib][i] is not None:
                        infer_result_tensor.append(infer_result[attrib][i])
                if len(infer_result_tensor) > 0:
                    merged = torch.cat(infer_result_tensor, dim=0)

                    # [修复逻辑] 智能展平
                    # 只有当最后一维是 7 (参数) 时才展平
                    # 如果是 3 (角点 x,y,z)，保持 (N, 8, 3) 结构
                    if len(merged.shape) == 3:
                        if merged.shape[-1] == 7:
                            merged = merged.view(-1, 7)
                        # else: 保持 (N, 8, 3)

                    infer_result[attrib] = merged
                else:
                    infer_result[attrib] = None

            # 2. 处理 Dict 类型
            elif isinstance(infer_result[attrib], dict):
                infer_result_tensor = []
                for k, v in infer_result[attrib].items():
                    if v is not None:
                        infer_result_tensor.append(v)
                if len(infer_result_tensor) > 0:
                    merged = torch.cat(infer_result_tensor, dim=0)

                    # [修复逻辑] 智能展平
                    if len(merged.shape) == 3:
                        if merged.shape[-1] == 7:
                            merged = merged.view(-1, 7)
                        # else: 保持 (N, 8, 3)

                    infer_result[attrib] = merged
                else:
                    infer_result[attrib] = None

    pred_box_tensor = infer_result.get("pred_box_tensor", None)
    gt_box_tensor = infer_result.get("gt_box_tensor", None)

    if pred_box_tensor is not None:
        pred_box_np = pred_box_tensor.cpu().numpy()
        pred_name = ['pred'] * pred_box_np.shape[0]

        score = infer_result.get("score_tensor", None)
        if score is None and 'pred_score' in infer_result:
            score = infer_result['pred_score']

        if score is not None:
            score_np = score.cpu().numpy()
            pred_name = [f'score:{score_np[i]:.3f}' for i in range(score_np.shape[0])]

        uncertainty = infer_result.get("uncertainty_tensor", None)
        if uncertainty is not None:
            uncertainty_np = uncertainty.cpu().numpy()
            uncertainty_np = np.exp(uncertainty_np)
            d_a_square = 1.6 ** 2 + 3.9 ** 2

            if uncertainty_np.shape[1] == 3:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)
                pred_name = [
                    f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:.3f} a_u:{uncertainty_np[i, 2]:.3f}' \
                    for i in range(uncertainty_np.shape[0])]
            elif uncertainty_np.shape[1] == 2:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)
                pred_name = [f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:3f}' \
                             for i in range(uncertainty_np.shape[0])]
            elif uncertainty_np.shape[1] == 7:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)
                pred_name = [
                    f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:3f} a_u:{uncertainty_np[i, 6]:3f}' \
                    for i in range(uncertainty_np.shape[0])]

    if gt_box_tensor is not None:
        gt_box_np = gt_box_tensor.cpu().numpy()
        gt_name = ['gt'] * gt_box_np.shape[0]

    mode_list = ['gt', 'pred', 'gt+pred']
    canvas_list = []

    def ensure_corners(boxes):
        if boxes is None or len(boxes) == 0:
            return boxes
        # 如果是 (N, 7)，说明是参数，需要转换成角点
        if boxes.ndim == 2 and boxes.shape[1] == 7:
            return box_utils.boxes_to_corners_3d(boxes, order='hwl')
        return boxes

    if method == 'bev':
        for mode in mode_list:
            canvas = canvas_bev.Canvas_BEV_heading_right(
                canvas_shape=((pc_range[4] - pc_range[1]) * 10, (pc_range[3] - pc_range[0]) * 10),
                canvas_x_range=(pc_range[0], pc_range[3]),
                canvas_y_range=(pc_range[1], pc_range[4]),
                left_hand=left_hand)

            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
            canvas.draw_canvas_points(canvas_xy[valid_mask])

            if gt_box_tensor is not None and (mode == 'gt' or mode == 'gt+pred'):
                # 过滤 (N, 7) 的 GT
                if gt_box_np.ndim == 2 and gt_box_np.shape[1] == 7:
                    mask = np.sum(gt_box_np, axis=1) != 0
                    valid_gt = gt_box_np[mask]
                else:
                    valid_gt = gt_box_np

                if len(valid_gt) > 0:
                    valid_gt = ensure_corners(valid_gt)
                    canvas.draw_boxes(valid_gt, colors=(0, 255, 0))

            if pred_box_tensor is not None and (mode == 'pred' or mode == 'gt+pred'):
                valid_pred = ensure_corners(pred_box_np)
                canvas.draw_boxes(valid_pred, colors=(255, 0, 0))

            # heterogeneous
            lidar_agent_record = infer_result.get("lidar_agent_record", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if lidar_agent_record is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, islidar in enumerate(lidar_agent_record):
                    text = ['lidar'] if islidar else ['camera']
                    color = (0, 191, 255) if islidar else (255, 185, 15)
                    box_to_draw = ensure_corners(cav_box_np[i:i + 1])
                    canvas.draw_boxes(box_to_draw, colors=color, texts=text)

            canvas_list.append(canvas)

    elif method == '3d':
        for mode in mode_list:
            canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
            canvas.draw_canvas_points(canvas_xy[valid_mask])

            if gt_box_tensor is not None and (mode == 'gt' or mode == 'gt+pred'):
                if gt_box_np.ndim == 2 and gt_box_np.shape[1] == 7:
                    mask = np.sum(gt_box_np, axis=1) != 0
                    valid_gt = gt_box_np[mask]
                else:
                    valid_gt = gt_box_np

                if len(valid_gt) > 0:
                    valid_gt = ensure_corners(valid_gt)
                    canvas.draw_boxes(valid_gt, colors=(0, 255, 0))

            if pred_box_tensor is not None and (mode == 'pred' or mode == 'gt+pred'):
                valid_pred = ensure_corners(pred_box_np)
                canvas.draw_boxes(valid_pred, colors=(255, 0, 0))

            # heterogeneous
            lidar_agent_record = infer_result.get("lidar_agent_record", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if lidar_agent_record is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, islidar in enumerate(lidar_agent_record):
                    text = ['lidar'] if islidar else ['camera']
                    color = (0, 191, 255) if islidar else (255, 185, 15)
                    box_to_draw = ensure_corners(cav_box_np[i:i + 1])
                    canvas.draw_boxes(box_to_draw, colors=color, texts=text)

            canvas_list.append(canvas)

    elif method == 'first_person':
        canvas = canvas_3d.Canvas_3D(camera_center_coords=(0, 0, 2.3),
                                     camera_focus_coords=(0 + 0.9396926, 0, 2.3 + 0.9396926), left_hand=left_hand,
                                     canvas_shape=(500, 1500))
        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
        canvas.draw_canvas_points(canvas_xy[valid_mask])

        if gt_box_tensor is not None:
            if gt_box_np.ndim == 2 and gt_box_np.shape[1] == 7:
                mask = np.sum(gt_box_np, axis=1) != 0
                valid_gt = gt_box_np[mask]
            else:
                valid_gt = gt_box_np

            if len(valid_gt) > 0:
                valid_gt = ensure_corners(valid_gt)
                canvas.draw_boxes(valid_gt, colors=(255, 0, 0))

        if pred_box_tensor is not None:
            valid_pred = ensure_corners(pred_box_np)
            canvas.draw_boxes(valid_pred, colors=(0, 255, 0))

        # heterogeneous
        lidar_agent_record = infer_result.get("lidar_agent_record", None)
        cav_box_np = infer_result.get("cav_box_np", None)
        if lidar_agent_record is not None:
            cav_box_np = copy.deepcopy(cav_box_np)
            for i, islidar in enumerate(lidar_agent_record):
                text = ['lidar'] if islidar else ['camera']
                color = (0, 191, 255) if islidar else (255, 185, 15)
                box_to_draw = ensure_corners(cav_box_np[i:i + 1])
                canvas.draw_boxes(box_to_draw, colors=color, texts=text)

        plt.axis("off")
        plt.imshow(canvas.canvas)
        plt.tight_layout()
        plt.savefig(save_path, transparent=False, dpi=500)
        plt.clf()
        plt.close()
        return

    else:
        raise (f"Not Completed for f{method} visualization.")

    ax1 = plt.subplot(221)
    ax1.imshow(canvas_list[0].canvas)
    ax1.set_title('GT', fontsize=15)
    ax1.axis('off')

    ax2 = plt.subplot(222)
    ax2.imshow(canvas_list[1].canvas)
    ax2.set_title('Preds', fontsize=15)
    ax2.axis('off')

    ax3 = plt.subplot(212)
    ax3.imshow(canvas_list[2].canvas)
    ax3.set_title('GT with pred', fontsize=15)
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, transparent=False, dpi=500)
    plt.clf()
    plt.close()

