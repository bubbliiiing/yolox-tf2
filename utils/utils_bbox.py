import tensorflow as tf
from tensorflow.keras import backend as K


#---------------------------------------------------#
#   对box进行调整，使其符合真实图片的样子
#---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        #-----------------------------------------------------------------#
        new_shape = K.round(image_shape * K.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

#---------------------------------------------------#
#   图片预测
#---------------------------------------------------#
def DecodeBox(outputs,
            num_classes,
            input_shape,
            max_boxes       = 100,
            confidence      = 0.5,
            nms_iou         = 0.3,
            letterbox_image = True):
            
    image_shape = K.reshape(outputs[-1],[-1])
    outputs     = outputs[:-1]
    #---------------------------#
    #   获得batch_size
    #---------------------------#
    bs      = K.shape(outputs[0])[0]

    grids   = []
    strides = []
    #---------------------------#
    #   获得三个有效特征层的高宽
    #---------------------------#
    hw      = [K.shape(x)[1:3] for x in outputs]
    #----------------------------------------------#
    #   batch_size, 80, 80, 4 + 1 + num_classes
    #   batch_size, 40, 40, 4 + 1 + num_classes
    #   batch_size, 20, 20, 4 + 1 + num_classes
    #   
    #   6400 + 1600 + 400
    #   outputs batch_size, 8400, 4 + 1 + num_classes
    #----------------------------------------------#
    outputs = tf.concat([tf.reshape(x, [bs, -1, 5 + num_classes]) for x in outputs], axis = 1)
    for i in range(len(hw)):
        #---------------------------#
        #   根据特征层生成网格点
        #   获得每一个有效特征层网格点的坐标
        #---------------------------#
        grid_x, grid_y  = tf.meshgrid(tf.range(hw[i][1]), tf.range(hw[i][0]))
        grid            = tf.reshape(tf.stack((grid_x, grid_y), 2), (1, -1, 2))
        shape           = tf.shape(grid)[:2]

        grids.append(tf.cast(grid, K.dtype(outputs)))
        strides.append(tf.ones((shape[0], shape[1], 1)) * input_shape[0] / tf.cast(hw[i][0], K.dtype(outputs)))
    #---------------------------#
    #   将网格点堆叠到一起
    #---------------------------#
    grids               = tf.concat(grids, axis=1)
    strides             = tf.concat(strides, axis=1)
    #-------------------------------------------#
    #   根据网格点进行解码
    #   box_xy 获得预测框中心归一化后的结果
    #   box_wh 获得预测框宽高归一化后的结果
    #-------------------------------------------#
    box_xy = (outputs[..., :2] + grids) * strides / K.cast(input_shape[::-1], K.dtype(outputs))
    box_wh = tf.exp(outputs[..., 2:4]) * strides / K.cast(input_shape[::-1], K.dtype(outputs))

    #-------------------------------------------#
    #   box_confidence 特征点是否有对应的物体
    #   box_class_probs 特征点物体种类的鹅置信度
    #-------------------------------------------#
    box_confidence  = K.sigmoid(outputs[..., 4:5])
    box_class_probs = K.sigmoid(outputs[..., 5: ])
    #------------------------------------------------------------------------------------------------------------#
    #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条，因此生成的box_xy, box_wh是相对于有灰条的图像的
    #   我们需要对其进行修改，去除灰条的部分。 将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    #   如果没有使用letterbox_image也需要将归一化后的box_xy, box_wh调整成相对于原图大小的
    #------------------------------------------------------------------------------------------------------------#
    boxes       = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    box_scores  = box_confidence * box_class_probs

    #-----------------------------------------------------------#
    #   判断得分是否大于score_threshold
    #-----------------------------------------------------------#
    mask             = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out   = []
    scores_out  = []
    classes_out = []
    for c in range(num_classes):
        #-----------------------------------------------------------#
        #   取出所有box_scores >= score_threshold的框，和成绩
        #-----------------------------------------------------------#
        class_boxes      = tf.boolean_mask(boxes, mask[..., c])
        class_box_scores = tf.boolean_mask(box_scores[..., c], mask[..., c])

        #-----------------------------------------------------------#
        #   非极大抑制
        #   保留一定区域内得分最大的框
        #-----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        #-----------------------------------------------------------#
        #   获取非极大抑制后的结果
        #   下列三个分别是：框的位置，得分与种类
        #-----------------------------------------------------------#
        class_boxes         = K.gather(class_boxes, nms_index)
        class_box_scores    = K.gather(class_box_scores, nms_index)
        classes             = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out      = K.concatenate(boxes_out, axis=0)
    scores_out     = K.concatenate(scores_out, axis=0)
    classes_out    = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    def decode_for_vision(output):
        #---------------------------#
        #   根据特征层生成网格点
        #---------------------------#
        # batch_size, 20, 20, 4 + 1 + num_classes
        bs, hw = np.shape(output)[0], np.shape(output)[1:3]
        # batch_size, 400, 4 + 1 + num_classes
        output          = np.reshape(output, [bs, hw[0] * hw[1], -1])

        #---------------------------#
        #   根据特征层的高和宽
        #   进行网格的构建
        #---------------------------#
        grid_x, grid_y  = np.meshgrid(np.arange(hw[1]), np.arange(hw[0]))
        #------------------------------------#
        #   单张图片，四百个网格点的xy轴坐标
        #   1, 400, 2
        #------------------------------------#
        grid            = np.reshape(np.stack((grid_x, grid_y), 2), (1, -1, 2))
        #------------------------#
        #   根据网格点进行解码
        #   box_xy是预测框的中心
        #   box_wh是预测框的宽高
        #------------------------#
        box_xy  = (output[..., :2] + grid)
        box_wh  = np.exp(output[..., 2:4])

        fig = plt.figure()
        ax  = fig.add_subplot(121)
        plt.ylim(-2,22)
        plt.xlim(-2,22)
        plt.scatter(grid_x,grid_y)
        plt.scatter(0,0,c='black')
        plt.scatter(1,0,c='black')
        plt.scatter(2,0,c='black')
        plt.gca().invert_yaxis()

        ax  = fig.add_subplot(122)
        plt.ylim(-2,22)
        plt.xlim(-2,22)
        plt.scatter(grid_x,grid_y)
        plt.scatter(0,0,c='black')
        plt.scatter(1,0,c='black')
        plt.scatter(2,0,c='black')

        plt.scatter(box_xy[0,0,0], box_xy[0,0,1],c='r')
        plt.scatter(box_xy[0,1,0], box_xy[0,1,1],c='r')
        plt.scatter(box_xy[0,2,0], box_xy[0,2,1],c='r')
        plt.gca().invert_yaxis()

        pre_left    = box_xy[...,0] - box_wh[...,0]/2 
        pre_top     = box_xy[...,1] - box_wh[...,1]/2 

        rect1   = plt.Rectangle([pre_left[0,0],pre_top[0,0]],box_wh[0,0,0],box_wh[0,0,1],color="r",fill=False)
        rect2   = plt.Rectangle([pre_left[0,1],pre_top[0,1]],box_wh[0,1,0],box_wh[0,1,1],color="r",fill=False)
        rect3   = plt.Rectangle([pre_left[0,2],pre_top[0,2]],box_wh[0,2,0],box_wh[0,2,1],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.show()

    #---------------------------------------------#
    #   batch_size, 20, 20, 4 + 1 + num_classes
    #---------------------------------------------#
    feat = np.concatenate([np.random.uniform(-1, 1, [4, 20, 20, 2]), np.random.uniform(1, 3, [4, 20, 20, 2]), np.random.uniform(1, 3, [4, 20, 20, 81])], -1)
    decode_for_vision(feat)
