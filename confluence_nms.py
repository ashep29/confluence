"""
Author: Andrew Shepley
Contact: asheple2@une.edu.au
Source: Confluence
Methods
a) assign_boxes_to_classes
b) normalise_coordinates
c) confluence_nms
"""

from collections import defaultdict
import numpy as np

def assign_boxes_to_classes(bounding_boxes, classes, scores):
    """
    Parameters: 
       bounding_boxes: list of bounding boxes (x1,y1,x2,y2)
       classes: list of class identifiers (int value, e.g. 1 = person)
       scores: list of class confidence scores (0.0-1.0)
    Returns:
       boxes_to_classes: defaultdict(list) containing mapping to bounding boxes and confidence scores to class
    """
    boxes_to_classes = defaultdict(list)
    for each_box, each_class, each_score in zip(bounding_boxes, classes, scores):
        if each_score >= 0.05:
            boxes_to_classes[each_class].append(np.array([each_box[0],each_box[1],each_box[2],each_box[3], each_score]))
    return boxes_to_classes

def normalise_coordinates(x1, y1, x2, y2,min_x,max_x,min_y,max_y):
    """
    Parameters: 
       x1, y1, x2, y2: bounding box coordinates to normalise
       min_x,max_x,min_y,max_y: minimum and maximum bounding box values (min = 0, max = 1)
    Returns:
       Normalised bounding box coordinates (scaled between 0 and 1)
    """
    x1, y1, x2, y2 = (x1-min_x)/(max_x-min_x), (y1-min_y)/(max_y-min_y), (x2-min_x)/(max_x-min_x), (y2-min_y)/(max_y-min_y)
    return x1, y1, x2, y2

def confluence_nms(bounding_boxes,scores,classes,confluence_thr=0.7,gaussian=True,score_thr=0.05,sigma=0.5):  
    """
    Parameters:
       bounding_boxes: list of bounding boxes (x1,y1,x2,y2)
       classes: list of class identifiers (int value, e.g. 1 = person)
       scores: list of class confidence scores (0.0-1.0)
       confluence_thr: value between 0 and 2, with optimum from 0.5-0.8
       gaussian: boolean switch to turn gaussian decaying of suboptimal bounding box confidence scores (setting to False results in suppression of suboptimal boxes)
       score_thr: class confidence score
       sigma: used in gaussian decaying. A smaller value causes harsher decaying.
    Returns:
       output: A dictionary mapping class identity to final retained boxes (and corresponding confidence scores)
    """
    output = {}
    class_mapping = assign_boxes_to_classes(bounding_boxes, classes, scores)
    for each_class in class_mapping:
        dets = np.array(class_mapping[each_class])
        retain = []
        while dets.size > 0:
            max_idx = np.argmax(dets[:, 4], axis=0)
            dets[[0, max_idx], :] = dets[[max_idx, 0], :]
            retain.append(dets[0, :])
            x1, y1, x2, y2 = dets[0, 0], dets[0, 1], dets[0, 2], dets[0, 3]
    
            min_x = np.minimum(x1, dets[1:, 0])
            min_y = np.minimum(y1, dets[1:, 1])
            max_x = np.maximum(x2, dets[1:, 2])   
            max_y = np.maximum(y2, dets[1:, 3])
    
            x1, y1, x2, y2 = normalise_coordinates(x1, y1, x2, y2,min_x,max_x,min_y,max_y)
            xx1, yy1, xx2, yy2 = normalise_coordinates(dets[1:, 0], dets[1:, 1], dets[1:, 2], dets[1:, 3],min_x,max_x,min_y,max_y)

            md_x1,md_x2,md_y1,md_y2 = abs(x1-xx1),abs(x2-xx2),abs(y1-yy1),abs(y2-yy2) 
            manhattan_distance = (md_x1+md_x2+md_y1+md_y2)

            weights = np.ones_like(manhattan_distance)

            if (gaussian == True):
                gaussian_weights = np.exp(-((1-manhattan_distance) * (1-manhattan_distance)) / sigma)
                weights[manhattan_distance<=confluence_thr]=gaussian_weights[manhattan_distance<=confluence_thr]
            else:
                weights[manhattan_distance<=confluence_thr]=0

            dets[1:, 4] *= weights
            to_keep = np.where(dets[1:, 4] >= score_thr)[0]
            dets = dets[to_keep + 1, :]     
        output[each_class]=retain

    return output
