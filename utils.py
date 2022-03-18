def compute_iou(gt_box, b_box):
    '''
    计算交并比
    :param gt_box: box = [x0,y0,x1,y1] （x0,y0)为左上角的坐标（x1,y1）为右下角的坐标
    :param b_box:
    :return: 
    '''
    width0 = gt_box[2] - gt_box[0]
    height0 = gt_box[3] - gt_box[1]
    width1 = b_box[2] - b_box[0]
    height1 = b_box[3] - b_box[1]
    max_x = max(gt_box[2], b_box[2])
    min_x = min(gt_box[0], b_box[0])
    width = width0 + width1 - (max_x - min_x)
    max_y = max(gt_box[3], b_box[3])
    min_y = min(gt_box[1], b_box[1])
    height = height0 + height1 - (max_y - min_y)
    if width < 0 or height < 0:
        interArea = 0
    else:
        interArea = width * height
    boxAArea = width0 * height0
    boxBArea = width1 * height1
    iou = interArea / boxAArea
    # iou = interArea / (boxAArea + boxBArea - interArea)
    return iou
