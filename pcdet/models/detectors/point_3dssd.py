import torch

from .detector3d_template import Detector3DTemplate
from ...ops.iou3d_nms import iou3d_nms_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class Point3DSSD(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()

        loss = loss_point
        return loss, tb_dict, disp_dict
    
    def init_recall_record(self, metric, **kwargs):
        # initialize gt_num for all classes
        for cur_cls in range(len(self.class_names)):
            metric['gt_num[%s]' % self.class_names[cur_cls]] = 0

        # initialize statistics of all sampling segments
        npoint_list = self.model_cfg.BACKBONE_3D.SA_CONFIG.NPOINT_LIST
        for cur_layer in range(len(npoint_list)):
            for cur_seg in range(len(npoint_list[cur_layer])):
                metric['positive_point_L%dS%d' % (cur_layer, cur_seg)] = 0
                metric['recall_point_L%dS%d' % (cur_layer, cur_seg)] = 0
                for cur_cls in range(self.num_class):
                    metric['recall_point_L%dS%d[%s]' \
                        % (cur_layer, cur_seg, self.class_names[cur_cls])] = 0

        # initialize statistics of the vote layer
        metric['positive_point_candidate'] = 0
        metric['recall_point_candidate'] = 0
        metric['positive_point_vote'] = 0
        metric['recall_point_vote'] = 0
        for cur_cls in range(len(self.class_names)):
            metric['recall_point_candidate[%s]' % self.class_names[cur_cls]] = 0
            metric['recall_point_vote[%s]' % self.class_names[cur_cls]] = 0

    def generate_recall_record(self, box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        # point_coords format: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
        point_list = data_dict['point_coords_list']  # ignore raw point input
        npoint_list = self.model_cfg.BACKBONE_3D.SA_CONFIG.NPOINT_LIST
        assert len(point_list) == len(npoint_list)

        cur_points_list = []
        for cur_layer in range(npoint_list.__len__()):
            cur_points = point_list[cur_layer]
            bs_idx = cur_points[:, 0]
            bs_mask = (bs_idx == batch_index)
            cur_points = cur_points[bs_mask][:, 1:4]
            cur_points_list.append(cur_points.split(npoint_list[cur_layer], dim=0))

        base_points = data_dict['point_candidate_coords']
        vote_points = data_dict['point_vote_coords']
        bs_idx = base_points[:, 0]
        bs_mask = (bs_idx == batch_index)
        base_points = base_points[bs_mask][:, 1:4]
        vote_points = vote_points[bs_mask][:, 1:4]

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        # initialize recall_dict
        if recall_dict.__len__() == 0:
            recall_dict = {'gt_num': 0}
            for cur_thresh in thresh_list:
                recall_dict['recall_roi_%s' % (str(cur_thresh))] = 0
                recall_dict['recall_rcnn_%s' % (str(cur_thresh))] = 0
            self.init_recall_record(recall_dict)  # init customized statistics

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            # backbone
            for cur_layer in range(len(npoint_list)):
                for cur_seg in range(len(npoint_list[cur_layer])):
                    box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                        cur_points_list[cur_layer][cur_seg].unsqueeze(dim=0),
                        cur_gt[None, :, :7].contiguous()
                    ).long().squeeze(dim=0)
                    box_fg_flag = (box_idxs_of_pts >= 0)
                    recall_dict['positive_point_L%dS%d' % (cur_layer, cur_seg)] += box_fg_flag.long().sum().item()
                    box_recalled = box_idxs_of_pts[box_fg_flag].unique()
                    recall_dict['recall_point_L%dS%d' % (cur_layer, cur_seg)] += box_recalled.size(0)

                    box_recalled_cls = cur_gt[box_recalled, -1]
                    for cur_cls in range(self.num_class):
                        recall_dict['recall_point_L%dS%d[%s]' % (cur_layer, cur_seg, self.class_names[cur_cls])] += \
                            (box_recalled_cls == (cur_cls + 1)).sum().item()

            # candidate points
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                base_points.unsqueeze(dim=0), cur_gt[None, :, :7]
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            recall_dict['positive_point_candidate'] += box_fg_flag.long().sum().item()
            box_recalled = box_idxs_of_pts[box_fg_flag].unique()
            recall_dict['recall_point_candidate'] += box_recalled.size(0)

            box_recalled_cls = cur_gt[box_recalled, -1]
            for cur_cls in range(self.num_class):
                recall_dict['recall_point_candidate[%s]' % self.class_names[cur_cls]] += \
                    (box_recalled_cls == (cur_cls + 1)).sum().item()

            # vote points
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                vote_points.unsqueeze(dim=0), cur_gt[None, :, :7]
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            recall_dict['positive_point_vote'] += box_fg_flag.long().sum().item()
            box_recalled = box_idxs_of_pts[box_fg_flag].unique()
            recall_dict['recall_point_vote'] += box_recalled.size(0)

            box_recalled_cls = cur_gt[box_recalled, -1]
            for cur_cls in range(self.num_class):
                recall_dict['recall_point_vote[%s]' % self.class_names[cur_cls]] += \
                    (box_recalled_cls == (cur_cls + 1)).sum().item()

            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['recall_rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['recall_rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['recall_roi_%s' % str(cur_thresh)] += roi_recalled

            cur_gt_class = cur_gt[:, -1]
            for cur_cls in range(self.num_class):
                cur_cls_gt_num = (cur_gt_class == cur_cls + 1).sum().item()
                recall_dict['gt_num'] += cur_cls_gt_num
                recall_dict['gt_num[%s]' % self.class_names[cur_cls]] += cur_cls_gt_num

        return recall_dict
    
    def disp_recall_record(self, metric, logger, sample_num, **kwargs):
        gt_num = metric['gt_num']
        gt_num_cls = [metric['gt_num[%s]' % cur_cls] for cur_cls in self.class_names]

        # backbone
        for k in metric.keys():
            if 'positive_point_' in k:  # count the number of positive points
                cur_positive_point = metric[k] / sample_num
                logger.info(k + (': %f' % cur_positive_point))
            elif 'recall_point_' in k and not any(cur_cls in k for cur_cls in self.class_names):
                cur_recall_point = metric[k] / max(gt_num, 1)
                logger.info(k + (': %f' % cur_recall_point))
                for cur_cls in range(len(self.class_names)):
                    cur_recall_point_cls = metric[k + '[%s]' % self.class_names[cur_cls]] / max(gt_num_cls[cur_cls], 1)
                    logger.info('\t- ' + self.class_names[cur_cls] + ': %f' % cur_recall_point_cls)

        # candidate points
        positive_point_candidate = metric['positive_point_candidate'] / sample_num
        logger.info('positive_point_candidate: %f' % positive_point_candidate)
        recall_point_candidate = metric['recall_point_candidate'] / max(gt_num, 1)
        logger.info('recall_point_candidate: %f' % recall_point_candidate)
        for cur_cls in range(len(self.class_names)):
            cur_recall_point_cls = metric['recall_point_candidate' + '[%s]' % self.class_names[cur_cls]] / max(gt_num_cls[cur_cls], 1)
            logger.info('\t- ' + self.class_names[cur_cls] + ': %f' % cur_recall_point_cls)

        # vote points
        positive_point_vote = metric['positive_point_vote'] / sample_num
        logger.info('positive_point_vote: %f' % positive_point_vote)
        recall_point_vote = metric['recall_point_vote'] / max(gt_num, 1)
        logger.info('recall_point_vote: %f' % recall_point_vote)
        for cur_cls in range(len(self.class_names)):
            cur_recall_point_cls = metric['recall_point_vote' + '[%s]' % self.class_names[cur_cls]] / max(gt_num_cls[cur_cls], 1)
            logger.info('\t- ' + self.class_names[cur_cls] + ': %f' % cur_recall_point_cls)
