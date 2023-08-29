import numpy as np
import cv2
import time
import os

def get_bonding_box(image_BGR,h,w,labels,network,probability_minimum = 0.5,threshold = 0.3):

    blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)
    # Getting list with names of all layers from YOLO v3 network
    layers_names_all = network.getLayerNames()
    # only output layers
    layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
    # Generating colours for representing every detected object
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    network.setInput(blob)
    output_from_network = network.forward(layers_names_output)

    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]
            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                # of bounding box, its width and height for original image
                box_current = detected_objects[0:4] * np.array([w, h, w, h])
                # that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    return bounding_boxes,confidences,class_numbers



def NMS(image_BGR,bounding_boxes,confidences,class_numbers,labels,probability_minimum = 0.5,threshold = 0.3):

    # with function randint(low, high=None, size=None, dtype='l')
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,probability_minimum, threshold)
    # Defining counter for detected objects
    counter = 1
    # Checking if there is at least one detected object after non-maximum suppression
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Showing labels of the detected objects
            print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))
            # Incrementing counter
            counter += 1
            # Getting current bounding box coordinates,
            # its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            # Preparing colour for current bounding box
            # and converting from numpy array to list
            colour_box_current = colours[class_numbers[i]].tolist()
            # Drawing bounding box on the original image
            cv2.rectangle(image_BGR, (x_min, y_min),
                        (x_min + box_width, y_min + box_height),
                        colour_box_current, 2)
            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                confidences[i])
            # Putting text with label and confidence on the original image
            cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)
            cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
            cv2.imshow('Detections', image_BGR)


    
    pass