import cv2
import os
import torch
import numpy as np

from network import LSCCNN
from utils import compute_boxes_and_sizes, get_upsample_output, get_box_and_dot_maps, get_boxed_img 

CURR_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
shanghaitechA_weights = 'models/part_a_scale_4_epoch_13_weights.pth'
shanghaitechB_weights = 'models/part_b_scale_4_epoch_24_weights.pth'
chosen_default_weights = os.path.join(CURR_DIR, shanghaitechA_weights)

PRED_DOWNSCALE_FACTORS=(8, 4, 2, 1)
GAMMA=(1, 1, 2, 4)
NUM_BOXES_PER_SCALE=3

def preprocess(img, compress_ratio=1):
    ## Resizing it
    img_h, img_w = img.shape[:2]
    new_img_h = img_h//compress_ratio
    new_img_w = img_w//compress_ratio
    img = cv2.resize(img, (new_img_w, new_img_h))
    ## Making it a multiple of 16
    if new_img_w % 16 or new_img_h % 16:
        img = cv2.resize(img, (new_img_w//16*16, new_img_h//16*16))
    ## Converting to torch tensor
    return torch.from_numpy(img.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0), img

class CrowdCounter(object):
    def __init__(self, 
                weights_path=None, 
                pred_downscale_factors=(8,4,2,1), 
                gamma=(1,1,2,4), 
                num_boxes_per_scale=3, 
                nms_thresh=0.25, 
                output_downscale=2):
        self.model = LSCCNN(name='scale_4')
        if weights_path is None:
            weights_path = chosen_default_weights
        assert os.path.exists(weights_path),'Weights given {} does not exist!'.format(weights_path)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            self.model.forward(torch.from_numpy(np.zeros((1, 3, 224, 224), dtype=np.float32)).cuda())
        print('LSC-CNN loaded and warmed up!')

        self.nms_thresh = nms_thresh
        self.output_downscale = output_downscale

        self.box_sizes, self.box_sizes_flat = compute_boxes_and_sizes(pred_downscale_factors, gamma, num_boxes_per_scale)


    def predict(self, img, compress_ratio=1, nms_thresh=None):
        '''
        :param img: can either be a np.ndarray (not processed yet) or torch.Tensor (already preprocessed)
        '''

        if isinstance(img, np.ndarray):
            img_tensor, _ = preprocess(img, compress_ratio=compress_ratio)
        else:
            img_tensor = img
        assert isinstance(img_tensor, torch.Tensor),'Image Tensor is not a torch.Tensor!'

        with torch.no_grad():
            out = self.model.forward(img_tensor.cuda())
        upsampled_out = get_upsample_output(out, self.output_downscale)

        if nms_thresh is None:
            nms_thresh = self.nms_thresh
        pred_dot_map, pred_box_map = get_box_and_dot_maps(upsampled_out, nms_thresh, self.box_sizes)

        return pred_dot_map, pred_box_map


    def get_count(self, frame, compress_ratio=1, nms_thresh=None):
        pred_dot_map, _ = self.predict(frame, compress_ratio=compress_ratio, nms_thresh=nms_thresh)
        return np.where(pred_dot_map>0)[1].shape[0]

    def visualise_count(self, frame, compress_ratio=1, nms_thresh=None):
        img_tensor, small_img = preprocess(frame, compress_ratio=compress_ratio)
        pred_dot_map, pred_box_map = self.predict(img_tensor, compress_ratio=compress_ratio, nms_thresh=nms_thresh)
        boxed_img = get_boxed_img(small_img, pred_box_map, pred_box_map, pred_dot_map, self.output_downscale, self.box_sizes, self.box_sizes_flat, thickness=2, multi_colours=True)

        return boxed_img, np.where(pred_dot_map>0)[1].shape[0]

if __name__ == '__main__':
    cc = CrowdCounter()
    img_path = 'shanghai_test_subset/IMG/shanghai_IMG_128.jpg'
    assert os.path.exists(img_path)
    img = cv2.imread(img_path)
    show_img, count = cc.visualise_count(img)
    print(img.shape)
    print("Predicted count: {}".format(count))    
    print(show_img.shape)
    cv2.imwrite(os.path.basename(img_path).split('.')[0]+'_crowdcount.png', show_img)

    import scipy.io
    gt_path = 'shanghai_test_subset/GT/shanghai_IMG_128.mat'
    assert os.path.exists(gt_path)
    data_mat = scipy.io.loadmat(gt_path)
    gt_pts = data_mat['image_info'][0][0][0][0][0]
    print('GT count: {}'.format(len(gt_pts)))

    cv2.imshow('LSC-CNN', show_img)
    cv2.waitKey(0)
