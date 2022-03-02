import math
from functools import partial

import tensorflow as tf
import tensorflow.keras.backend as K


def get_yolo_loss(input_shape, num_layers, num_classes):
    def yolo_loss(args):
        #-----------------------------------------------#
        #   labels 标签结果[batch_size, num_gt, 4 + 1]
        #   y_pred 预测结果[[batch_size, 20, 20, num_classes + 5]
        #                   [batch_size, 40, 40, num_classes + 5]
        #                   [batch_size, 80, 80, num_classes + 5]]
        #-----------------------------------------------#
        labels, y_pred = args[-1], args[:-1]
        x_shifts            = []
        y_shifts            = []
        expanded_strides    = []
        outputs             = []
        #-----------------------------------------------#
        # outputs   [[batch_size, 400, num_classes + 5]
        #            [batch_size, 1600, num_classes + 5]
        #            [batch_size, 6400, num_classes + 5]]
        #-----------------------------------------------#
        for i in range(num_layers):
            output          = y_pred[i]
            #-----------------------------------------------#
            #   stride 一个特征点对应原始像素点的数量
            #-----------------------------------------------#
            grid_shape      = tf.shape(output)[1:3]
            stride          = input_shape[0] / tf.cast(grid_shape[0], K.dtype(output))

            #-----------------------------------------------#
            #   根据特征层的高和宽，获得网格点坐标
            #-----------------------------------------------#
            grid_x, grid_y  = tf.meshgrid(tf.range(grid_shape[1]), tf.range(grid_shape[0]))
            grid            = tf.cast(tf.reshape(tf.stack((grid_x, grid_y), 2), (1, -1, 2)), K.dtype(output))
            
            #-----------------------------------------------#
            #   进行解码
            #-----------------------------------------------#
            output          = tf.reshape(output, [tf.shape(y_pred[i])[0], grid_shape[0] * grid_shape[1], -1])
            output_xy       = (output[..., :2] + grid) * stride
            output_wh       = tf.exp(output[..., 2:4]) * stride
            output          = tf.concat([output_xy, output_wh, output[..., 4:]], -1)

            x_shifts.append(grid[..., 0])
            y_shifts.append(grid[..., 1])
            expanded_strides.append(tf.ones_like(grid[..., 0]) * stride)
            outputs.append(output)
        #-------------------------------------------------------------#
        #   n_anchors_all代表了一个图片特征点的数量
        #   当输入是640,640,3的时候，n_anchors_all就是8400
        #   
        #   x_shifts            [1, n_anchors_all]
        #   y_shifts            [1, n_anchors_all]
        #   expanded_strides    [1, n_anchors_all]
        #   outputs             [batch_size, n_anchors_all, num_classes + 5]
        #-------------------------------------------------------------#
        x_shifts            = tf.concat(x_shifts, 1)
        y_shifts            = tf.concat(y_shifts, 1)
        expanded_strides    = tf.concat(expanded_strides, 1)
        outputs             = tf.concat(outputs, 1)
        return get_losses(x_shifts, y_shifts, expanded_strides, outputs, labels, num_classes)
    return yolo_loss


def get_losses(x_shifts, y_shifts, expanded_strides, outputs, labels, num_classes):
    #-----------------------------------------------#
    #   [batch, n_anchors_all, 4] 预测框的坐标
    #   [batch, n_anchors_all, 1] 特征点是否有对应的物体
    #   [batch, n_anchors_all, n_cls] 特征点对应物体的种类
    #-----------------------------------------------#
    bbox_preds  = outputs[:, :, :4]  
    obj_preds   = outputs[:, :, 4:5]
    cls_preds   = outputs[:, :, 5:]  
    
    #------------------------------------------------------------#
    #   labels                      [batch, max_boxes, 5]
    #   tf.reduce_sum(labels, -1)   [batch, max_boxes]
    #   nlabel                      [batch]
    #------------------------------------------------------------#
    nlabel = tf.reduce_sum(tf.cast(tf.reduce_sum(labels, -1) > 0, K.dtype(outputs)), -1)
    total_num_anchors = tf.shape(outputs)[1]

    num_fg      = 0.0
    loss_obj    = 0.0
    loss_cls    = 0.0
    loss_iou    = 0.0
    def loop_body(b, num_fg, loss_iou, loss_obj, loss_cls):
        # num_gt 单张图片的真实框的数量
        num_gt  = tf.cast(nlabel[b], tf.int32)
        #-----------------------------------------------#
        #   gt_bboxes_per_image     [num_gt, 4]
        #   gt_classes              [num_gt]
        #   bboxes_preds_per_image  [n_anchors_all, 4]
        #   obj_preds_per_image     [n_anchors_all, 1]
        #   cls_preds_per_image     [n_anchors_all, num_classes]
        #-----------------------------------------------#
        gt_bboxes_per_image     = labels[b][:num_gt, :4]
        gt_classes              = labels[b][:num_gt,  4]
        bboxes_preds_per_image  = bbox_preds[b]
        obj_preds_per_image     = obj_preds[b]
        cls_preds_per_image     = cls_preds[b]

        def f1():
            num_fg_img  = tf.cast(tf.constant(0), K.dtype(outputs))
            cls_target  = tf.cast(tf.zeros((0, num_classes)), K.dtype(outputs))
            reg_target  = tf.cast(tf.zeros((0, 4)), K.dtype(outputs))
            obj_target  = tf.cast(tf.zeros((total_num_anchors, 1)), K.dtype(outputs))
            fg_mask     = tf.cast(tf.zeros(total_num_anchors), tf.bool)
            return num_fg_img, cls_target, reg_target, obj_target, fg_mask
        def f2():
            gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = get_assignments( 
                gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, obj_preds_per_image, cls_preds_per_image,
                x_shifts, y_shifts, expanded_strides, num_classes, num_gt, total_num_anchors, 
            )
            reg_target  = tf.cast(tf.gather_nd(gt_bboxes_per_image, tf.reshape(matched_gt_inds, [-1, 1])), K.dtype(outputs))
            cls_target  = tf.cast(tf.one_hot(tf.cast(gt_matched_classes, tf.int32), num_classes) * tf.expand_dims(pred_ious_this_matching, -1), K.dtype(outputs))
            obj_target  = tf.cast(tf.expand_dims(fg_mask, -1), K.dtype(outputs))
            return num_fg_img, cls_target, reg_target, obj_target, fg_mask
            
        num_fg_img, cls_target, reg_target, obj_target, fg_mask = tf.cond(tf.equal(num_gt, 0), f1, f2)
        num_fg      += num_fg_img
        loss_iou    += K.sum(1 - box_ciou(reg_target, tf.boolean_mask(bboxes_preds_per_image, fg_mask)))
        loss_obj    += K.sum(K.binary_crossentropy(obj_target, obj_preds_per_image, from_logits=True))
        loss_cls    += K.sum(K.binary_crossentropy(cls_target, tf.boolean_mask(cls_preds_per_image, fg_mask), from_logits=True))
        return b + 1, num_fg, loss_iou, loss_obj, loss_cls
    #-----------------------------------------------------------#
    #   在这个地方进行一个循环、循环是对每一张图片进行的
    #-----------------------------------------------------------#
    _, num_fg, loss_iou, loss_obj, loss_cls = tf.while_loop(lambda b,*args: b < tf.cast(tf.shape(outputs)[0], tf.int32), loop_body, [0, num_fg, loss_iou, loss_obj, loss_cls])
    
    num_fg      = tf.cast(tf.maximum(num_fg, 1), K.dtype(outputs))
    reg_weight  = 5.0
    loss        = reg_weight * loss_iou + loss_obj + loss_cls
    return loss / num_fg

def get_assignments(gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, obj_preds_per_image, cls_preds_per_image, x_shifts, y_shifts, expanded_strides, num_classes, num_gt, total_num_anchors):
    #-------------------------------------------------------#
    #   判断哪些特征点在真实框内部
    #   fg_mask                 [n_anchors_all]
    #   is_in_boxes_and_center  [num_gt, len(fg_mask)]
    #-------------------------------------------------------#
    fg_mask, is_in_boxes_and_center = get_in_boxes_info(gt_bboxes_per_image, x_shifts, y_shifts, expanded_strides, num_gt, total_num_anchors)
    
    #-------------------------------------------------------#
    #   获得在真实框内部的特征点的预测结果
    #   fg_mask                 [n_anchors_all]
    #   bboxes_preds_per_image  [fg_mask, 4]
    #   cls_preds_              [fg_mask, num_classes]
    #   obj_preds_              [fg_mask, 1]
    #-------------------------------------------------------#
    bboxes_preds_per_image  = tf.boolean_mask(bboxes_preds_per_image, fg_mask, axis = 0)
    obj_preds_              = tf.boolean_mask(obj_preds_per_image, fg_mask, axis = 0)
    cls_preds_              = tf.boolean_mask(cls_preds_per_image, fg_mask, axis = 0)
    num_in_boxes_anchor     = tf.shape(bboxes_preds_per_image)[0]

    #-------------------------------------------------------#
    #   计算真实框和预测框的重合程度
    #   pair_wise_ious      [num_gt, fg_mask]
    #-------------------------------------------------------#
    pair_wise_ious      = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image)
    pair_wise_ious_loss = -tf.math.log(pair_wise_ious + 1e-8)
    #-------------------------------------------------------#
    #   计算真实框和预测框种类置信度的交叉熵
    #   cls_preds_          [num_gt, fg_mask, num_classes]
    #   gt_cls_per_image    [num_gt, fg_mask, num_classes]
    #   pair_wise_cls_loss  [num_gt, fg_mask]
    #-------------------------------------------------------#
    gt_cls_per_image    = tf.tile(tf.expand_dims(tf.one_hot(tf.cast(gt_classes, tf.int32), num_classes), 1), (1, num_in_boxes_anchor, 1))
    cls_preds_          = K.sigmoid(tf.tile(tf.expand_dims(cls_preds_, 0), (num_gt, 1, 1))) *\
                          K.sigmoid(tf.tile(tf.expand_dims(obj_preds_, 0), (num_gt, 1, 1)))

    pair_wise_cls_loss  = tf.reduce_sum(K.binary_crossentropy(gt_cls_per_image, tf.sqrt(cls_preds_)), -1)

    #-------------------------------------------------------#
    #   种类比较接近的情况瞎，交叉熵较低
    #   真实框和预测框重合度较高的时候，cost较低
    #   这个特征点是要有对应的真实框的，cost才会低
    #-------------------------------------------------------#
    cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * tf.cast((~is_in_boxes_and_center), K.dtype(bboxes_preds_per_image))

    gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg = dynamic_k_matching(cost, pair_wise_ious, fg_mask, gt_classes, num_gt)
    return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

def get_in_boxes_info(gt_bboxes_per_image, x_shifts, y_shifts, expanded_strides, num_gt, total_num_anchors, center_radius = 2.5):
    #-------------------------------------------------------#
    #   expanded_strides_per_image  [n_anchors_all]
    #   x_centers_per_image         [num_gt, n_anchors_all]
    #   y_centers_per_image         [num_gt, n_anchors_all]
    #-------------------------------------------------------#
    expanded_strides_per_image  = expanded_strides[0]
    x_centers_per_image         = tf.tile(tf.expand_dims(((x_shifts[0] + 0.5) * expanded_strides_per_image), 0), [num_gt, 1])
    y_centers_per_image         = tf.tile(tf.expand_dims(((y_shifts[0] + 0.5) * expanded_strides_per_image), 0), [num_gt, 1])

    #-------------------------------------------------------#
    #   gt_bboxes_per_image_l       [num_gt, n_anchors_all]
    #   gt_bboxes_per_image_r       [num_gt, n_anchors_all]
    #   gt_bboxes_per_image_t       [num_gt, n_anchors_all]
    #   gt_bboxes_per_image_b       [num_gt, n_anchors_all]
    #-------------------------------------------------------#
    gt_bboxes_per_image_l = tf.tile(tf.expand_dims((gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]), 1), [1, total_num_anchors])
    gt_bboxes_per_image_r = tf.tile(tf.expand_dims((gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]), 1), [1, total_num_anchors])
    gt_bboxes_per_image_t = tf.tile(tf.expand_dims((gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]), 1), [1, total_num_anchors])
    gt_bboxes_per_image_b = tf.tile(tf.expand_dims((gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]), 1), [1, total_num_anchors])

    #-------------------------------------------------------#
    #   bbox_deltas     [num_gt, n_anchors_all, 4]
    #-------------------------------------------------------#
    b_l = x_centers_per_image - gt_bboxes_per_image_l
    b_r = gt_bboxes_per_image_r - x_centers_per_image
    b_t = y_centers_per_image - gt_bboxes_per_image_t
    b_b = gt_bboxes_per_image_b - y_centers_per_image
    bbox_deltas = tf.stack([b_l, b_t, b_r, b_b], 2)

    #-------------------------------------------------------#
    #   is_in_boxes     [num_gt, n_anchors_all]
    #   is_in_boxes_all [n_anchors_all]
    #-------------------------------------------------------#
    is_in_boxes     = tf.reduce_min(bbox_deltas, axis = -1) > 0.0
    is_in_boxes_all = tf.reduce_sum(tf.cast(is_in_boxes, K.dtype(gt_bboxes_per_image)), axis = 0) > 0.0

    gt_bboxes_per_image_l = tf.tile(tf.expand_dims(gt_bboxes_per_image[:, 0], 1), [1, total_num_anchors]) - center_radius * tf.expand_dims(expanded_strides_per_image, 0)
    gt_bboxes_per_image_r = tf.tile(tf.expand_dims(gt_bboxes_per_image[:, 0], 1), [1, total_num_anchors]) + center_radius * tf.expand_dims(expanded_strides_per_image, 0)
    gt_bboxes_per_image_t = tf.tile(tf.expand_dims(gt_bboxes_per_image[:, 1], 1), [1, total_num_anchors]) - center_radius * tf.expand_dims(expanded_strides_per_image, 0)
    gt_bboxes_per_image_b = tf.tile(tf.expand_dims(gt_bboxes_per_image[:, 1], 1), [1, total_num_anchors]) + center_radius * tf.expand_dims(expanded_strides_per_image, 0)

    #-------------------------------------------------------#
    #   center_deltas   [num_gt, n_anchors_all, 4]
    #-------------------------------------------------------#
    c_l = x_centers_per_image - gt_bboxes_per_image_l
    c_r = gt_bboxes_per_image_r - x_centers_per_image
    c_t = y_centers_per_image - gt_bboxes_per_image_t
    c_b = gt_bboxes_per_image_b - y_centers_per_image
    center_deltas       = tf.stack([c_l, c_t, c_r, c_b], 2)

    #-------------------------------------------------------#
    #   is_in_centers       [num_gt, n_anchors_all]
    #   is_in_centers_all   [n_anchors_all]
    #-------------------------------------------------------#
    is_in_centers       = tf.reduce_min(center_deltas, axis = -1) > 0.0
    is_in_centers_all   = tf.reduce_sum(tf.cast(is_in_centers, K.dtype(gt_bboxes_per_image)), axis = 0) > 0.0

    #-------------------------------------------------------#
    #   fg_mask                 [n_anchors_all]
    #   is_in_boxes_and_center  [num_gt, fg_mask]
    #-------------------------------------------------------#
    fg_mask = tf.cast(is_in_boxes_all | is_in_centers_all, tf.bool)
    
    is_in_boxes_and_center  = tf.boolean_mask(is_in_boxes, fg_mask, axis = 1) & tf.boolean_mask(is_in_centers, fg_mask, axis = 1)

    return fg_mask, is_in_boxes_and_center

def bboxes_iou(b1, b2):
    #---------------------------------------------------#
    #   num_anchor,1,4
    #   计算左上角的坐标和右下角的坐标
    #---------------------------------------------------#
    b1              = K.expand_dims(b1, -2)
    b1_xy           = b1[..., :2]
    b1_wh           = b1[..., 2:4]
    b1_wh_half      = b1_wh/2.
    b1_mins         = b1_xy - b1_wh_half
    b1_maxes        = b1_xy + b1_wh_half

    #---------------------------------------------------#
    #   1,n,4
    #   计算左上角和右下角的坐标
    #---------------------------------------------------#
    b2              = K.expand_dims(b2, 0)
    b2_xy           = b2[..., :2]
    b2_wh           = b2[..., 2:4]
    b2_wh_half      = b2_wh/2.
    b2_mins         = b2_xy - b2_wh_half
    b2_maxes        = b2_xy + b2_wh_half

    #---------------------------------------------------#
    #   计算重合面积
    #---------------------------------------------------#
    intersect_mins  = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    iou             = intersect_area / (b1_area + b2_area - intersect_area)
    return iou

def dynamic_k_matching(cost, pair_wise_ious, fg_mask, gt_classes, num_gt):
    #-------------------------------------------------------#
    #   matching_matrix     [num_gt, fg_mask]
    #   cost                [num_gt, fg_mask]
    #   pair_wise_ious      [num_gt, fg_mask] 每一个真实框和预测框的重合情况
    #   gt_classes          [num_gt]        
    #   fg_mask             [n_anchors_all]
    #-------------------------------------------------------#
    matching_matrix         = tf.zeros_like(cost)

    #------------------------------------------------------------#
    #   选取iou最大的n_candidate_k个点
    #   获得当前真实框重合度最大的十个预测框的值
    #   重合度的值域是[0, 1]，dynamic_ks的值就是[0, 10]
    #   然后求和，判断应该有多少点用于该框预测
    #   topk_ious           [num_gt, n_candidate_k]
    #   dynamic_ks          [num_gt]
    #   matching_matrix     [num_gt, fg_mask]
    #------------------------------------------------------------#
    n_candidate_k           = tf.minimum(10, tf.shape(pair_wise_ious)[1])
    topk_ious, _            = tf.nn.top_k(pair_wise_ious, n_candidate_k)
    dynamic_ks              = tf.maximum(tf.reduce_sum(topk_ious, 1), 1)
    # dynamic_ks              = tf.Print(dynamic_ks, [topk_ious, dynamic_ks], summarize = 100)
    
    def loop_body_1(b, matching_matrix):
        #------------------------------------------------------------#
        #   给每个真实框选取最小的动态k个点
        #------------------------------------------------------------#
        _, pos_idx = tf.nn.top_k(-cost[b], k=tf.cast(dynamic_ks[b], tf.int32))
        matching_matrix = tf.concat(
            [matching_matrix[:b], tf.expand_dims(tf.reduce_max(tf.one_hot(pos_idx, tf.shape(cost)[1]), 0), 0), matching_matrix[b+1:]], axis = 0
        )
        # matching_matrix = matching_matrix.write(b, K.cast(tf.reduce_max(tf.one_hot(pos_idx, tf.shape(cost)[1]), 0), K.dtype(cost)))
        return b + 1, matching_matrix
    #-----------------------------------------------------------#
    #   在这个地方进行一个循环、循环是对每一张图片进行的
    #-----------------------------------------------------------#
    _, matching_matrix = tf.while_loop(lambda b,*args: b < tf.cast(num_gt, tf.int32), loop_body_1, [0, matching_matrix])

    #------------------------------------------------------------#
    #   anchor_matching_gt  [fg_mask]
    #------------------------------------------------------------#
    anchor_matching_gt = tf.reduce_sum(matching_matrix, 0)
    #------------------------------------------------------------#
    #   当某一个特征点指向多个真实框的时候
    #   选取cost最小的真实框。
    #------------------------------------------------------------#
    biger_one_indice = tf.reshape(tf.where(anchor_matching_gt > 1), [-1])
    def loop_body_2(b, matching_matrix):
        indice_anchor   = tf.cast(biger_one_indice[b], tf.int32)
        indice_gt       = tf.math.argmin(cost[:, indice_anchor])
        matching_matrix = tf.concat(
            [
                matching_matrix[:, :indice_anchor], 
                tf.expand_dims(tf.one_hot(indice_gt, tf.cast(num_gt, tf.int32)), 1), 
                matching_matrix[:, indice_anchor+1:]
            ], axis = -1
        )
        return b + 1, matching_matrix
    #-----------------------------------------------------------#
    #   在这个地方进行一个循环、循环是对每一张图片进行的
    #-----------------------------------------------------------#
    _, matching_matrix = tf.while_loop(lambda b,*args: b < tf.cast(tf.shape(biger_one_indice)[0], tf.int32), loop_body_2, [0, matching_matrix])

    #------------------------------------------------------------#
    #   fg_mask_inboxes  [fg_mask]
    #   num_fg为正样本的特征点个数
    #------------------------------------------------------------#
    fg_mask_inboxes = tf.reduce_sum(matching_matrix, 0) > 0.0
    num_fg          = tf.reduce_sum(tf.cast(fg_mask_inboxes, K.dtype(cost)))

    fg_mask_indices         = tf.reshape(tf.where(fg_mask), [-1])
    fg_mask_inboxes_indices = tf.reshape(tf.where(fg_mask_inboxes), [-1, 1])
    fg_mask_select_indices  = tf.gather_nd(fg_mask_indices, fg_mask_inboxes_indices)
    fg_mask                 = tf.cast(tf.reduce_max(tf.one_hot(fg_mask_select_indices, tf.shape(fg_mask)[0]), 0), K.dtype(fg_mask))

    #------------------------------------------------------------#
    #   获得特征点对应的物品种类
    #------------------------------------------------------------#
    matched_gt_inds     = tf.math.argmax(tf.boolean_mask(matching_matrix, fg_mask_inboxes, axis = 1), 0)
    gt_matched_classes  = tf.gather_nd(gt_classes, tf.reshape(matched_gt_inds, [-1, 1]))

    pred_ious_this_matching = tf.boolean_mask(tf.reduce_sum(matching_matrix * pair_wise_ious, 0), fg_mask_inboxes)
    return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    #-----------------------------------------------------------#
    #   求出预测框左上角右下角
    #   b1_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b1_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    #-----------------------------------------------------------#
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    #-----------------------------------------------------------#
    #   求出真实框左上角右下角
    #   b2_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b2_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    #-----------------------------------------------------------#
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    #-----------------------------------------------------------#
    #   求真实框和预测框所有的iou
    #   iou         (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / K.maximum(union_area, K.epsilon())

    #-----------------------------------------------------------#
    #   计算中心的差距
    #   center_distance (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    #-----------------------------------------------------------#
    #   计算对角线距离
    #   enclose_diagonal (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    ciou = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal ,K.epsilon())
    
    v = 4 * K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1], K.epsilon())) - tf.math.atan2(b2_wh[..., 0], K.maximum(b2_wh[..., 1],K.epsilon()))) / (math.pi * math.pi)
    alpha = v /  K.maximum((1.0 - iou + v), K.epsilon())
    ciou = ciou - alpha * v
    return ciou

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func
