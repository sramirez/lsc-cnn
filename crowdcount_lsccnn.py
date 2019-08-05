import cv2
import os
import torch
import numpy as np
from shapely.geometry.polygon import Polygon

from network import LSCCNN
from utils_lsccnn import compute_boxes_and_sizes, get_upsample_output, get_box_and_dot_maps, get_boxed_img 
from drawer import draw_count

CURR_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
shanghaitechA_weights = 'models/part_a_scale_4_epoch_13_weights.pth'
shanghaitechB_weights = 'models/part_b_scale_4_epoch_24_weights.pth'
chosen_default_weights = os.path.join(CURR_DIR, shanghaitechA_weights)

PRED_DOWNSCALE_FACTORS=(8, 4, 2, 1)
GAMMA=(1, 1, 2, 4)
NUM_BOXES_PER_SCALE=3

class CrowdCounter(object):
    def __init__(self, 
                img_w,
                img_h,
                compress_ratio=1.0, 
                omit_scales=[], 
                ignore_polys=[],
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

        self.omit_scales = omit_scales

        self.prepreprocess(img_w, img_h, compress_ratio=compress_ratio, ignore_polys=ignore_polys)

    def prepreprocess(self, img_w, img_h, compress_ratio=1.0, ignore_polys=[]):
        self.ignore_polys = []
        self.ignore_polys_raw = []
        self.new_img_h = int((img_h/compress_ratio)//16*16)
        self.new_img_w = int((img_w/compress_ratio)//16*16)
        new_compress_ratio_w = img_w / self.new_img_w
        new_compress_ratio_h = img_h / self.new_img_h
        for polyst in ignore_polys:
            new_polyst = []
            for x,y in polyst:
                x = int(x/new_compress_ratio_w)
                y = int(y/new_compress_ratio_h)
                new_polyst.append((x,y))
            self.ignore_polys_raw.append(new_polyst)
            self.ignore_polys.append(Polygon(new_polyst))

    def preprocess(self, img):
        ## Resizing it
        img = cv2.resize(img, (self.new_img_w, self.new_img_h))
        ## Converting to torch tensor
        return torch.from_numpy(img.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0), img

    def predict(self, img, nms_thresh=None):
        '''
        :param img: can either be a np.ndarray (not processed yet) or torch.Tensor (already preprocessed)
        '''

        if isinstance(img, np.ndarray):
            img_tensor, _ = self.preprocess(img)
        else:
            img_tensor = img
        assert isinstance(img_tensor, torch.Tensor),'Image Tensor is not a torch.Tensor!'

        with torch.no_grad():
            out = self.model.forward(img_tensor.cuda())
        upsampled_out = get_upsample_output(out, self.output_downscale)

        if nms_thresh is None:
            nms_thresh = self.nms_thresh
        pred_dot_map, pred_box_map = get_box_and_dot_maps(upsampled_out, nms_thresh, self.box_sizes, omit_scales=self.omit_scales)

        return pred_dot_map, pred_box_map

    def get_count(self, frame, nms_thresh=None):
        pred_dot_map, _ = self.predict(frame, nms_thresh=nms_thresh)
        return np.where(pred_dot_map>0)[1].shape[0]

    def visualise_count(self, frame, nms_thresh=None, omit_scales=[], ignore_polys=[]):
        '''
        :param omit_scales: list of scale indices where 0 refer to the smallest BBs and 3 refer to the largest BBs (Yellow: Largest, Red: Medium, Green: Medium-Small, Blue: Smallest)
        :param ignore_polys: list of polygons, each polygon being a list of tuples of (x,y) containing at least 3 points in one polygon. 
        '''
        img_tensor, small_img = self.preprocess(frame)
        pred_dot_map, pred_box_map = self.predict(img_tensor, nms_thresh=nms_thresh)
        boxed_img, recount = get_boxed_img(small_img, pred_box_map, pred_box_map, pred_dot_map, self.output_downscale, self.box_sizes, self.box_sizes_flat, thickness=2, multi_colours=True, omit_scales=self.omit_scales, ignore_polys=self.ignore_polys)
        return boxed_img, recount

if __name__ == '__main__':
    cc = CrowdCounter()
    # img_path = 'shanghai_test_subset/IMG/shanghai_IMG_181.jpg'
    img_path = 'golf_crowd.jpg'
    assert os.path.exists(img_path)
    img = cv2.imread(img_path)
    print('input shape:',img.shape)
    show_img, count = cc.visualise_count(img, compress_ratio=1.3, omit_scales=[])
    # show_img, count = cc.visualise_count(img, compress_ratio=1.1, omit_scales=[])
    # print(img.shape)
    print("Predicted count: {}".format(count))    
    # print(show_img.shape)

    gt_count = None
    # import scipy.io
    # gt_path = 'shanghai_test_subset/GT/shanghai_IMG_181.mat'
    # assert os.path.exists(gt_path)
    # data_mat = scipy.io.loadmat(gt_path)
    # gt_pts = data_mat['image_info'][0][0][0][0][0]
    # print('GT count: {}'.format(len(gt_pts)))
    # gt_count = len(gt_pts)

    draw_count(show_img, count, gt_count=gt_count)
    cv2.imwrite(os.path.basename(img_path).split('.')[0]+'_crowdcount.png', show_img)
    cv2.namedWindow('LSC-CNN', cv2.WINDOW_NORMAL)
    cv2.imshow('LSC-CNN', show_img)
    cv2.waitKey(0)
