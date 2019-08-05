import cv2
from random import randint 
from copy import deepcopy

color = (255,255,0)

font = cv2.FONT_HERSHEY_DUPLEX
fontColor = (255,255,0)

def draw_count(frame, crowd_count, gt_count=None, alpha=0.5):

    frame_h = frame.shape[0]
    print(frame_h)
    frame_w = frame.shape[1]

    fontScale = max(int(frame_w/525), 1)
    fontThickness = int(fontScale*2)

    if gt_count:
        text = 'NDPeeps Counted: {} (GT:{})'.format(crowd_count, gt_count)
    else:
        text = 'NDPeeps Counted: {}'.format(crowd_count)

    overlay = deepcopy(frame)
    # cv2.rectangle(overlay, (20,25), (20+len(text)*92, 25+30*fontScale), (0,0,255), -1)

    # cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    print('Drawing {}'.format(text))
    print(fontScale)
    cv2.putText(frame, text, (20,frame_h-20), font, fontScale, (255,255,255), fontThickness)
    # cv2.putText(frame, text, (20,150), font, fontScale, fontColor, fontThickness)
