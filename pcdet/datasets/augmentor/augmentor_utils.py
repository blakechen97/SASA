import numpy as np
import numba
import warnings

from numba.core.errors import NumbaPerformanceWarning
from ...utils import common_utils
from ...utils import box_utils


def random_flip_along_x(enable_prob, gt_boxes, points):
    """
    Args:
        enable_prob:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[1.0 - enable_prob, enable_prob])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points


def random_flip_along_y(enable_prob, gt_boxes, points):
    """
    Args:
        enable_prob:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[1.0 - enable_prob, enable_prob])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points


def global_rotation(enable_prob, gt_boxes, points, rot_range):
    """
    Args:
        enable_prob:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[1.0 - enable_prob, enable_prob])
    if enable:
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
        gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
        gt_boxes[:, 6] += noise_rotation
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
                np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
                np.array([noise_rotation])
            )[0][:, 0:2]

    return gt_boxes, points


def global_scaling(enable_prob, gt_boxes, points, scale_range):
    """
    Args:
        enable_prob:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    enable = np.random.choice([False, True], replace=False, p=[1.0 - enable_prob, enable_prob])
    if enable:
        noise_scale = np.random.uniform(scale_range[0], scale_range[1])
        points[:, :3] *= noise_scale
        gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


def box_noise(enable_prob, gt_boxes, points, valid_mask=None, extra_width=0.1, sem_labels=None,
              loc_noise_std=[1.0, 1.0, 0.0], scale_range=[1.0, 1.0], rotation_range=[0.0, 0.0], num_try=100):
    """
    Args:
        enable_prob: list of float, prob for enabling center, scale and rotation noise
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        valid_mask: (N), mask to indicate which boxes are valid
        extra_width: points in expanded regions are also permuted
        sem_labels: TODO: support sem_labels
        loc_noise_std: location noise std
        scale_range:
        rotation_range:
        num_try: number of attempts for noise generating
    Returns:
    """
    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
    num_box = gt_boxes.shape[0]
    num_points = points.shape[0]
    enable = np.random.choice([False, True], replace=False, p=[1.0 - enable_prob, enable_prob])
    if enable:
        if valid_mask is None:
            valid_mask = np.ones((num_box,), dtype=np.bool_)

        loc_noise = np.array(loc_noise_std, dtype=gt_boxes.dtype)
        loc_noise = np.random.normal(
            scale=loc_noise, size=[num_box, num_try, 3]
        )
        scale_noise = np.random.uniform(
            scale_range[0], scale_range[1], size=[num_box, num_try]
        )
        rotation_noise = np.random.uniform(
            rotation_range[0], rotation_range[1], size=[num_box, num_try]
        )

        gt_boxes_expand = gt_boxes.copy()
        gt_boxes_expand[:, 3:6] += float(extra_width)

        success_mask = choose_noise_for_box(gt_boxes_expand[:, [0, 1, 3, 4, 6]], valid_mask,
                                            loc_noise, scale_noise, rotation_noise)
        loc_transform = np.zeros((num_box, 3), dtype=gt_boxes.dtype)
        scale_transform = np.ones((num_box,), dtype=gt_boxes.dtype)
        rotation_transform = np.zeros((num_box,), dtype=gt_boxes.dtype)
        for i in range(num_box):
            if success_mask[i] != -1:
                loc_transform[i, :] = loc_noise[i, success_mask[i], :]
                scale_transform[i] = scale_noise[i, success_mask[i]]
                rotation_transform[i] = rotation_noise[i, success_mask[i]]

        gt_corners_expand = box_utils.boxes_to_corners_3d(gt_boxes_expand)
        point_masks = np.zeros((num_box, num_points), dtype=np.bool_)
        for i in range(num_box):
            point_masks[i, :] = box_utils.in_hull(points[:, 0:3], gt_corners_expand[i])

        point_transform_(points, gt_boxes, valid_mask, point_masks, loc_transform, scale_transform, rotation_transform)
        box3d_transform_(gt_boxes, valid_mask, loc_transform, scale_transform, rotation_transform)

    return gt_boxes, points


@numba.njit
def point_transform_(points, gt_boxes, valid_mask, point_masks, loc_transform, scale_transform, rotation_transform):
    num_box = gt_boxes.shape[0]
    num_points = points.shape[0]

    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_T[i], rotation_transform[i], 2)

    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[j, i] == 1:
                    points[i, :3] -= gt_boxes[j, :3]
                    points[i, :3] *= scale_transform[j]
                    points[i:i + 1, :3] = points[i:i + 1, :3] @ rot_mat_T[j]
                    points[i, :3] += gt_boxes[j, :3]
                    points[i, 2] += gt_boxes[j, 5] * (scale_transform[j] - 1) / 2  # ensure box still on the ground
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform


def box3d_transform_(boxes, valid_mask, loc_transform, scale_transform, rotation_transform):
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 3:6] *= scale_transform[i]
            boxes[i, 2] += boxes[i, 5] * (scale_transform[i] - 1) / 2  # ensure box still on the ground
            boxes[i, 6] += rotation_transform[i]
            if boxes.shape[1] > 7:  # rotate [vx, vy]
                boxes[i, 7:9] = common_utils.rotate_points_along_z(
                    np.hstack((boxes[i, 7:9], np.zeros((1,))))[np.newaxis, np.newaxis, :],
                    np.array([rotation_transform[i]])
                )[0][0, 0:2]


@numba.njit
def _rotation_box2d_jit_(corners, angle, rot_mat_T):
    """Rotate 2D boxes.

    Args:
        corners (np.ndarray): Corners of boxes.
        angle (float): Rotation angle.
        rot_mat_T (np.ndarray): Transposed rotation matrix.
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = rot_sin
    rot_mat_T[1, 0] = -rot_sin
    rot_mat_T[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_T


@numba.njit
def _rotation_matrix_3d_(rot_mat_T, angle, axis):
    """Get the 3D rotation matrix.

    Args:
        rot_mat_T (np.ndarray): Transposed rotation matrix.
        angle (float): Rotation angle.
        axis (int): Rotation axis.
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = rot_sin
        rot_mat_T[2, 0] = -rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = rot_sin
        rot_mat_T[1, 0] = -rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = rot_sin
        rot_mat_T[2, 1] = -rot_sin
        rot_mat_T[2, 2] = rot_cos


@numba.njit
def choose_noise_for_box(box2d, valid_mask, loc_noise, scale_noise, rotation_noise):
    """
    Args:
        box2d: (N, 5) [x, y, dx, dy, heading]
        valid_mask:
        loc_noise: (N, M, 3)
        scale_noise: (N, M)
        rotation_noise: (N, M)
    Returns:
        success_mask: unsuccess=-1
    """
    num_box = box2d.shape[0]
    num_try = loc_noise.shape[1]
    box_corners = box2d_to_corner_jit(box2d)
    cur_corners = np.zeros((4, 2), dtype=box2d.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=box2d.dtype)
    success_mask = -np.ones((num_box, ), dtype=np.int64)

    for i in range(num_box):
        if valid_mask[i]:
            for j in range(num_try):
                cur_corners[:] = box_corners[i]
                cur_corners -= box2d[i, :2]
                _rotation_box2d_jit_(cur_corners, rotation_noise[i, j], rot_mat_T)

                cur_corners *= scale_noise[i, j]
                cur_corners += box2d[i, :2] + loc_noise[i, j, :2]
                collision_mat = box_collision_test(
                    cur_corners.reshape(1, 4, 2), box_corners
                )
                collision_mat[0, i] = False
                if not collision_mat.any():
                    success_mask[i] = j
                    box_corners[i] = cur_corners
                    break
    return success_mask


@numba.jit(nopython=True)
def box2d_to_corner_jit(boxes):
    num_box = boxes.shape[0]
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * corners_norm.reshape(
        1, 4, 2)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = rot_sin
        rot_mat_T[1, 0] = -rot_sin
        rot_mat_T[1, 1] = rot_cos
        box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]
    return box_corners


@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    """Convert boxes_corner to aligned (min-max) boxes.

    Args:
        boxes_corner (np.ndarray, shape=[N, 2**dim, dim]): Boxes corners.

    Returns:
        np.ndarray, shape=[N, dim*2]: Aligned (min-max) boxes.
    """
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    """Box collision test.

    Args:
        boxes (np.ndarray): Corners of current boxes.
        qboxes (np.ndarray): Boxes to be avoid colliding.
        clockwise (bool): Whether the corners are in clockwise order.
            Default: True.
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]), axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = corner_to_standup_nd_jit(boxes)
    qboxes_standup = corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (
                min(boxes_standup[i, 2], qboxes_standup[j, 2]) -
                max(boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (
                    min(boxes_standup[i, 3], qboxes_standup[j, 3]) -
                    max(boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for box_l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, box_l, 0]
                            D = lines_qboxes[j, box_l, 1]
                            acd = (D[1] - A[1]) * (C[0] -
                                                   A[0]) > (C[1] - A[1]) * (
                                                       D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] -
                                                   B[0]) > (C[1] - B[1]) * (
                                                       D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (
                                        C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (
                                        D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for box_l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                    boxes[i, k, 0] - qboxes[j, box_l, 0])
                                cross -= vec[0] * (
                                    boxes[i, k, 1] - qboxes[j, box_l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for box_l in range(4):  # point box_l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                        qboxes[j, k, 0] - boxes[i, box_l, 0])
                                    cross -= vec[0] * (
                                        qboxes[j, k, 1] - boxes[i, box_l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret
