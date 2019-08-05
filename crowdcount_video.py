import argparse
import os
import cv2
import time

from crowdcount_lsccnn import CrowdCounter
from drawer import draw_count

parser = argparse.ArgumentParser()
parser.add_argument('video', help='video to process')
args = parser.parse_args()

assert os.path.exists(args.video),'Video does not exist!'

basename = os.path.basename(args.video).split('.')[0]
out_dir = basename+'_out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

cap = cv2.VideoCapture(args.video)
frame_skip = 1
frame_w = cap.get(3)
frame_h = cap.get(4)
vid_fps = cap.get(5)
# fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out_vid = cv2.VideoWriter(basename+'_cc.avi',fourcc, vid_fps, (int(frame_w), int(frame_h)))
cc = CrowdCounter()

total_dur = 0
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count%frame_skip != 0:
        continue
    tic = time.time()
    show_img, count = cc.visualise_count(frame)
    print('Current crowd count: {}'.format(count))
    draw_count(show_img, count)
    toc = time.time()
    total_dur += (toc - tic)
    # cv2.imshow('', show_img)
    # cv2.imwrite(os.path.join(out_dir,'{}.png'.format(frame_count)),show_img)
    out_vid.write(show_img)
    # if cv2.waitKey(1) & 0xff == ord('q'):
        # break

cap.release()
out_vid.release()

print('Avrg inference time:{}'.format(total_dur/frame_count))