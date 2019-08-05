import argparse
import os
import cv2
import time
import configparser
import numpy as np

from crowdcount_lsccnn import CrowdCounter
from drawer import draw_count

parser = argparse.ArgumentParser()
parser.add_argument('video', help='video to process')
parser.add_argument('--cfg', help='Config file to specify no count/deadzones')
args = parser.parse_args()

assert os.path.exists(args.video),'Video does not exist!'

basename = os.path.basename(args.video).split('.')[0]
out_dir = basename+'_out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

cap = cv2.VideoCapture(args.video)
frame_w = cap.get(3)
frame_h = cap.get(4)
vid_fps = cap.get(5)
# fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# out_vid = cv2.VideoWriter(basename+'_cc.avi',fourcc, vid_fps, (int(frame_w), int(frame_h)))

assert os.path.exists(args.cfg),'Config file given does not exist!'
config = configparser.ConfigParser()
config.read(args.cfg)
# print(config.sections())
skyline = int(config['SKYLINE']['Y'])
skybuffer = 10
skyline_polygon = [(0-skybuffer,0-skybuffer), (0-skybuffer,skyline), (frame_w-1+skybuffer, skyline), (frame_w-1+skybuffer, 0-skybuffer)]
dead_polygons = []
for poly in config['POLYGONS']:
    polystring = [int(x) for x in config['POLYGONS'][poly].split(',')]
    dead_polygons.append( list(zip(polystring[::2], polystring[1::2])))
print('Num of custom dead polygons: {}'.format(len(dead_polygons)))
dead_polygons.append(skyline_polygon)

compress_ratio = float(config['VIDEO']['CompressRatio'])
assert compress_ratio > 0,'compress ratio given is negative.'
omit_scales =[int(x) for x in config['VIDEO']['OmitScales'].split(',')]
sample_rate_sec = float(config['VIDEO']['SampleRate'])
sample_rate_frame = int(vid_fps * sample_rate_sec)

out_csv = open(basename+'_output.csv','w')
out_csv.write('TimeStamp(sec), FrameCount, PeopleCount\n')

cc = CrowdCounter(frame_w,  frame_h, compress_ratio=compress_ratio, omit_scales=omit_scales, ignore_polys=dead_polygons)
resized_dead_polygons = cc.ignore_polys_raw 

total_dur = 0
frame_count = 0
total_dur_count = 0
# frame_skip = 1
cv2.namedWindow('LSC-CNN', cv2.WINDOW_NORMAL)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # if frame_count%frame_skip != 0:
    #     continue
    if (frame_count-1)%sample_rate_frame != 0:
        continue
    tic = time.time()
    show_img, count = cc.visualise_count(frame,)
    print('Current crowd count: {}'.format(count))
    draw_count(show_img, count, ignore_polys=resized_dead_polygons)
    toc = time.time()
    total_dur += (toc - tic)
    total_dur_count += 1
    # cv2.imwrite(os.path.join(out_dir,'{}.png'.format(frame_count)),frame)
    out_vid.write(show_img)
    out_text = '{},{},{}\n'.format(round(frame_count/vid_fps), frame_count, count)
    # print(out_text)
    out_csv.write(out_text)
    cv2.imshow('LSC-CNN', show_img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    cv2.waitKey(5)

cap.release()
out_vid.release()
out_csv.close()

print('Avrg inference time:{}'.format(total_dur/total_dur_count))