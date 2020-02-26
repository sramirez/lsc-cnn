"""
network.py: Consists of the main architecture of LSC-CNN 
Authors       : svp 
"""
from ResnetBackbone import ResnetBackbone
from torchvision.models.resnet import resnet101
import cv2
import torch
import torch.nn as nn
import numpy as np
from utils_lsccnn import compute_boxes_and_sizes, get_upsample_output, get_box_and_dot_maps, get_boxed_img


class LSCCNN(nn.Module):
    def __init__(self, name='scale_4', checkpoint_path = None, output_downscale = 2):
        super(LSCCNN, self).__init__()
        self.name = name
        # OPT: Use torchvision.transforms instead
        means = [104.008, 116.669, 122.675]
        if torch.cuda.is_available():
            self.rgb_means = torch.cuda.FloatTensor(means)
        else:
            self.rgb_means = torch.FloatTensor(means)
        self.rgb_means = torch.autograd.Variable(self.rgb_means, requires_grad=False).unsqueeze(0).unsqueeze(
            2).unsqueeze(3)

        self.relu = nn.ReLU(inplace=True)

        #self.backbone = ResnetBackbone()
        self.backbone = resnet101(pretrained=True)
        self.layer1 = self.backbone.layer1  # 64 in_channels (64, 256)
        self.layer2 = self.backbone.layer2  # 64 in_channels (128, 512)
        self.layer3 = self.backbone.layer3  # 64 in_channels (256, 1024)
        self.layer4 = self.backbone.layer4  # 64 in_channels (512, 2048)

        # New top-down feature modulator (after concat section)
        # prev_level / 2 + actual / 2
        # 2048 / 2 = 1024 (no concat)
        self.convA_0 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.convA_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.convA_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convA_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convA_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convA_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        # (1024/2 + 1024/2) = 1024, concat with itself
        self.convB_0 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.convB_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.convB_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convB_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convB_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convB_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        # (512/2 + 1024/2) = 768
        self.convC_0 = nn.Conv2d(768, 512, kernel_size=3, padding=1)
        self.convC_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.convC_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convC_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convC_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convC_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        # (256/2 + 768/2) = 512
        self.convD_0 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.convD_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.convD_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convD_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convD_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convD_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        # downsample by 2 the direct input from FE in TMF
        self.conv_before_transpose_1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.conv_before_transpose_2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv_before_transpose_3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_before_transpose_4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        # output layer 1: 1024
        # output layer 2: 512
        # in: 1024 / 2 = 512
        # scales: 1/4, 1/8, 1/16, 1/32
        self.transpose_1 = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_1_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        # in: 1024 / 2 = 512
        self.transpose_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_2_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        # in: 512 / 2 = 256
        self.transpose_3 = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.conv_after_transpose_3_1 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)

        self.transpose_4_1_a = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.transpose_4_1_b = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_4_1 = nn.Conv2d(1024, 128, kernel_size=3, padding=1)
        
        self.transpose_4_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.conv_after_transpose_4_2 = nn.Conv2d(512, 128, kernel_size=3, padding=1)

        self.transpose_4_3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_4_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path))

    def compute_box_sizes(self, PRED_DOWNSCALE_FACTORS = (16, 8, 4, 2), GAMMA = (1, 1, 2, 4), NUM_BOXES_PER_SCALE = 3):
        self.BOXES, self.BOX_SIZE_BINS = compute_boxes_and_sizes(PRED_DOWNSCALE_FACTORS, GAMMA, NUM_BOXES_PER_SCALE)
        self.output_downscale = output_downscale

    def _initialize_weights(self):
        if True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(std=0.01, mean=0.0)
                    m.bias.data.zero_()

    def mean_over_multiple_dimensions(self, tensor, axes):
        for ax in sorted(axes, reverse=True):
            tensor = torch.mean(tensor, dim=ax, keepdim=True)
        return tensor

    def max_over_multiple_dimensions(self, tensor, axes):
        for ax in sorted(axes, reverse=True):
            tensor, _ = torch.max(tensor, ax, keepdim=True)
        return tensor

    def forward(self, x):
        mean_sub_input = x - self.rgb_means
        l1 = None
        l2 = None
        l3 = None
        l4 = None

        def resl1_hook(module, input_, output):
            nonlocal l1
            l1 = output

        def resl2_hook(module, input_, output):
            nonlocal l2
            l2 = output

        def resl3_hook(module, input_, output):
            nonlocal l3
            l3 = output

        def resl4_hook(module, input_, output):
            nonlocal l4
            l4 = output

        self.layer1.register_forward_hook(resl1_hook)
        self.layer2.register_forward_hook(resl2_hook)
        self.layer3.register_forward_hook(resl3_hook)
        self.layer4.register_forward_hook(resl4_hook)

        self.backbone.forward(mean_sub_input)  # partial output are saved in local variables
        #l1, l2, l3, l4 = self.backbone.forward(mean_sub_input) # l4 is the last layer

        #################### Stage 1 ##########################
        main_out_l4 = self.relu(self.conv_before_transpose_1(self.relu(l4)))
        main_out_rest = self.convA_5(self.relu(self.convA_4(self.relu(self.convA_3(self.relu(self.convA_2(
            self.relu(self.convA_1(self.relu(self.convA_0(main_out_l4)))))))))))
        if self.name == "scale_1":
            return main_out_rest

        ################## Stage 2 ############################
        main_out_l3 = self.relu(self.conv_before_transpose_2(self.relu(l3)))
        sub1_after_transpose_1 = self.relu(self.conv_after_transpose_1_1(self.relu(self.transpose_1(main_out_l4))))
        sub1_concat = torch.cat((main_out_l3, sub1_after_transpose_1), dim=1)

        sub1_out_rest = self.convB_5(self.relu(self.convB_4(self.relu(self.convB_3(
            self.relu(self.convB_2(self.relu(self.convB_1(self.relu(self.convB_0(sub1_concat)))))))))))
        if self.name == "scale_2":
            return main_out_rest, sub1_out_rest

        ################# Stage 3 ############################
        # take one output from feature extractor and compute sub2_out_conv1, will correspond with 1 -> 2 arrow in fig. 4
        main_out_l2 = self.relu(self.conv_before_transpose_3(self.relu(l2)))
        # transpose and conv output from other TFMs
        sub2_after_transpose_1 = self.relu(self.conv_after_transpose_2_1(self.relu(self.transpose_2(main_out_l3))))
        sub3_after_transpose_1 = self.relu(self.conv_after_transpose_3_1(self.relu(self.transpose_3(main_out_l4))))

        # Concat all and apply all convolution to generate the final output (no. 4)
        sub2_concat = torch.cat((main_out_l2, sub2_after_transpose_1, sub3_after_transpose_1), dim=1)

        sub2_out_rest = self.convC_5(self.relu(self.convC_4(self.relu(self.convC_3(
            self.relu(self.convC_2(self.relu(self.convC_1(self.relu(self.convC_0(sub2_concat)))))))))))

        if self.name == "scale_3":
            return main_out_rest, sub1_out_rest, sub2_out_rest

        ################# Stage 4 ############################
        sub4_out_conv1 = self.relu(self.conv_before_transpose_4(self.relu(l1)))

        # TDF 1
        tdf_4_1_a = self.relu(self.transpose_4_1_a(main_out_l4))
        tdf_4_1_b = self.relu(self.transpose_4_1_b(tdf_4_1_a))
        after_tdf_4_1 = self.relu(self.conv_after_transpose_4_1(tdf_4_1_b))
        
        # TDF 2
        tdf_4_2 = self.relu(self.transpose_4_2(main_out_l3))
        after_tdf_4_2 = self.relu(self.conv_after_transpose_4_2(tdf_4_2))

        # TDF 3
        tdf_4_3 = self.relu(self.transpose_4_3(main_out_l2))
        after_tdf_4_3 = self.relu(self.conv_after_transpose_4_3(tdf_4_3))

        sub4_concat = torch.cat((sub4_out_conv1, after_tdf_4_1, after_tdf_4_2, after_tdf_4_3), dim=1)
        sub4_out_rest = self.convD_5(self.relu(self.convD_4(self.relu(self.convD_3(
            self.relu(self.convD_2(self.relu(self.convD_1(self.relu(self.convD_0(sub4_concat)))))))))))

        if self.name == "scale_4":
            return main_out_rest, sub1_out_rest, sub2_out_rest, sub4_out_rest

    def predict_single_image(self, image, nms_thresh=0.25, thickness=2, multi_colours=True):
        if image.shape[0] % 16 or image.shape[1] % 16:
            image = cv2.resize(image, (image.shape[1]//16*16, image.shape[0]//16*16))
        img_tensor = torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = self.forward(img_tensor.cuda())
        out = get_upsample_output(out, self.output_downscale)
        pred_dot_map, pred_box_map = get_box_and_dot_maps(out, nms_thresh, self.BOXES)
        img_out = get_boxed_img(image, pred_box_map, pred_box_map, pred_dot_map, self.output_downscale,
                                self.BOXES, self.BOX_SIZE_BINS, thickness=thickness, multi_colours=multi_colours)
        return pred_dot_map, pred_box_map, img_out