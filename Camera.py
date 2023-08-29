import numpy as np
import cv2
import time
import Utils
import os


def Process_Camera():

    print(os.getcwd())

    camera = cv2.VideoCapture(0)
    h, w = None, None

    with open(os.getcwd()+'\\Labels\\coco.names') as f:
        labels = [line.strip() for line in f]

    network = cv2.dnn.readNetFromDarknet(os.getcwd()+'\\Model\\yolov3.cfg',
                                        os.getcwd()+'\\Model\\yolov3.weights')

    while True:
        _, frame = camera.read()
        if w is None or h is None:
            h, w = frame.shape[:2]
        bounding_boxes,confidences,class_numbers = Utils.get_bonding_box(frame,h,w,labels,network,probability_minimum = 0.5,threshold = 0.3)
        Utils.NMS(frame,bounding_boxes,confidences,class_numbers,labels,probability_minimum = 0.5,threshold = 0.3)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

    pass



