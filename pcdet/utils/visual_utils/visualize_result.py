import argparse
import os

import numpy as np

import visualize_utils

from pcdet.utils import object3d_kitti
from pcdet.datasets import KittiDataset


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--sample_id', type=int, required=True, help='sample index')
    parser.add_argument('--cfg_file', type=str, required=True, help='dataset config file')
    parser.add_argument('--split', type=str, default='train', help='train or test')
    parser.add_argument('--pred_path', type=str, default='default', help='directory for prediction files')

    args = parser.parse_args()
    return args


def process_boxes(obj_list, calib):
    cls_to_idx = {
        'Car': 1,
        'Pedestrian': 2,
        'Cyclist': 3
    }
    obj_list = [_ for _ in obj_list if _.cls_type in cls_to_idx.keys()]
    cls = np.array([cls_to_idx[obj.cls_type] for obj in obj_list])
    dim = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])
    loc = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
    rot = np.array([obj.ry for obj in obj_list])
    score = np.array([obj.score for obj in obj_list])

    loc_lidar = calib.rect_to_lidar(loc)
    l, h, w = dim[:, 0:1], dim[:, 1:2], dim[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2
    boxes = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rot[..., np.newaxis])], axis=1)

    return boxes, score, cls


def main():
    args = parse_config()

    import yaml
    from pathlib import Path
    from easydict import EasyDict
    dataset_cfg = EasyDict(yaml.load(open(args.cfg_file)))

    ROOT_DIR = (Path(__file__).resolve().parent / '../../').resolve()
    data_path = ROOT_DIR / 'data' / 'kitti'
    print(data_path)

    dataset = KittiDataset(
        dataset_cfg=dataset_cfg,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        root_path=data_path,
        training=False)
    dataset.set_split(args.split)

    sample_idx = '%06d' % args.sample_id
    lidar = dataset.get_lidar(sample_idx)
    calib = dataset.get_calib(sample_idx)

    gt_boxes = None
    pred_boxes, pred_scores, pred_labels = None, None, None

    if args.split == 'train':
        gt_labels = dataset.get_label(sample_idx)
        gt_boxes, _, _ = process_boxes(gt_labels, calib)

    if not args.pred_path == 'default':
        pred_file = Path(args.pred_path) / (sample_idx + '.txt')
        assert pred_file.exists()
        pred_labels = object3d_kitti.get_objects_from_label(pred_file)
        pred_boxes, pred_scores, pred_labels = process_boxes(pred_labels, calib)

    visualize_utils.draw_scenes(lidar, gt_boxes=gt_boxes,
                                ref_boxes=pred_boxes, ref_scores=pred_scores, ref_labels=pred_labels)
    input()


if __name__ == '__main__':
    main()
