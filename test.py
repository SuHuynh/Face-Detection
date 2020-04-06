from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode

parser = argparse.ArgumentParser(description='FaceBoxes')

parser.add_argument('-m', '--trained_model', default='weights/FaceBoxes_epoch_50.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.5, type=float, help='confidence_threshold')
# parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.5, type=float, help='nms_threshold')
parser.add_argument('-s', '--show_image', action="store_true", default=True, help='show detection results')
args = parser.parse_args()

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return np.array(pick, dtype = 'uint32')

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing begin
    test_image_folder = './test_img/'
    test_img_list_path = os.listdir(test_image_folder)

    for i, img_name in enumerate(test_img_list_path):
        image_path = test_image_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        img = cv2.resize(img, (1024, 1024))
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        loc, conf = net(img)  # forward pass
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore or remove low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # using non-maxima-suppression to remove overlap bounding boxes
        pick_idx = non_max_suppression_fast(boxes, args.nms_threshold)
        final_boxes = boxes[pick_idx, :]
        final_scores = scores[pick_idx]

        dets = np.hstack((final_boxes, final_scores[:, np.newaxis])).astype(np.float32, copy=False)

        # show image
        if args.show_image:
            for b in dets:
                text = "{:.3f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] - 6
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

            test_img_out_path = './test_results/' + image_path
            cv2.imwrite(test_img_out_path, img_raw)
            cv2.imshow('res', img_raw)
            cv2.waitKey(0)
