import cv2
import numpy as np
import torch
from torchvision.ops import nms

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize image to a 32-pixel-multiple rectangle.
    Translated from Ultralytics implementation for independence.
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return im, ratio, (dw, dh)

def xywh2xyxy(x):
    """Convert nx4 boxes from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescale boxes (xyxy) from img1_shape to img0_shape"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    """Clip boxes (xyxy) to image shape (height, width)"""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    Input: prediction (Batch, 4+nc, anchors) if raw, or (Batch, anchors, 4+nc) if transposed
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction)

    # YOLOv8 Output shape: [BATCH, NO_CLASSES+4, NO_ANCHORS] -> e.g., [1, 84, 8400]
    # We need to transpose to [BATCH, NO_ANCHORS, NO_CLASSES+4] -> e.g., [1, 8400, 84]
    if prediction.shape[2] > prediction.shape[1]:
        # Assume correct shape if 3rd dim is larger, otherwise transpose
        # Typical v8: [1, 84, 8400]
        prediction = prediction.transpose(1, 2)
        
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 4  # number of classes
    nm = prediction.shape[1] - nc - 4
    # ... (prints removed from logic)
    mi = 5 + nc   # mask start index
    xc = prediction[..., 4:].max(2).values > conf_thres  # candidates
    
    # Settings
    min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms
    agnostic = False
    multi_label = False
    
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 4:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)  # NMS
        output[xi] = x[i]

    return output

def draw_detections(image, detections, class_names=None):
    """
    Draw bounding boxes and labels on image.
    detections: list/array of [x1, y1, x2, y2, conf, cls]
    """
    img_copy = image.copy()
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = map(float, det[:6])
        cls_id = int(cls_id)
        
        # Color based on class
        np.random.seed(cls_id)
        color = np.random.randint(0, 255, size=3).tolist()
        
        # Box
        cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Label
        label = f"{class_names[cls_id] if class_names else cls_id} {conf:.2f}"
        t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        c2 = int(x1) + t_size[0], int(y1) - t_size[1] - 3
        cv2.rectangle(img_copy, (int(x1), int(y1)), c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img_copy, label, (int(x1), int(y1) - 2), 0, 0.5, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        
    return img_copy
