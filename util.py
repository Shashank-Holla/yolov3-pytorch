from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

# To easily access the predictions of the Yolo output feature map and to have single output processing operation for all the 3 scales.
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    """
    Function to take an detection feature map and turn it into a 2-D tensor, each row of the tensor corresponds to attributes of a bounding box.
    Row 1 - 1st bounding box at (0,0)
    Row 2 - 2nd bounding box at (0,0)
    Row 3 - 3rd bounding box at (0,0) and so on
    .
    Attributes of bounding box- tx, ty, tw, th, p0, p1, p2,... p80

    Inputs:
    prediction
    inp_dim
    anchors
    num_classes
    CUDA
    """

    batch_size = prediction.size(0) 
    stride = inp_dim // prediction.size(2)
    grid_size = prediction.size(2)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    # print("prediction sizes",prediction.size(0), prediction.size(2))
    # print("stride",stride)
    # print("prediction shape",prediction.shape)
    # print("batch size:",batch_size)
    # print("bbox and num_anchors",bbox_attrs, num_anchors)
    # print("grid size:",grid_size)
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    # Anchors are in accordance to the input image. Since feature map dimension is equal to image/stride, dividing achor dim by stride.
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Bounding box predictions. Apply sigmoid function on center_x, center_y and prediction score.
    # bx = sigmoid(tx)
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    # by = sigmoid(ty)
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    # p0 = sigmoid(p0) - objctness score
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid) # Prepares (grid * grid) matrix each

    # Flatten
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # x, y offset repeated anchor number of times.
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    # add offset to bx, by (center of bounding box). x_y_offset = top left co-ordinates of the grid.
    prediction[:,:,:2] += x_y_offset 

    # Applying anchors to the dimensions of the bounding box.
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    # bw (bounding box width) = pw * exp(tw) ; bh (bounding box height) = ph * exp(th) ; pw and ph -> anchor dimensions
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors  

    # Apply sigmoid function to the class scores
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes])) 

    prediction[:,:,:4] *= stride

    return prediction
    
def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res
    
    

def bbox_iou(box1, box2):
    """
    Returns the IoU of the 2 input boxes.
    Input
    box1 : bounding box row indexed
    box2 : multiple rows of bounding boxes.
    
    Output : tensor containing IoUs of the first bounding box with each of the bounding box present in box2
    
    """
    # Get the coordinates of the two bounding boxes.
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the coordinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
 
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou
    
    

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    """
    To apply objectness score thresholding and non-maximum suppression
    """
    # Comparing objectness score to confidence threshold. Set it to 1 if higher, 0 otherwise. Then update objectness score in pred with these values.
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction *= conf_mask
    
    # To find the corners of the bounding box and apply these changes to prediction's bounding box details. 
    box_corner = prediction.new(prediction.shape)
    box_width = prediction[:,:,2]
    box_height = prediction[:,:,3]
    # left corner x
    box_corner[:,:,0] = prediction[:,:,0] - box_width/2
    # bottom corner y
    box_corner[:,:,1] = prediction[:,:,1] - box_height/2
    # right corner x
    box_corner[:,:,2] = prediction[:,:,0] + box_width/2
    # top corner y
    box_corner[:,:,3] = prediction[:,:,1] + box_height/2
    prediction[:,:,:4] = box_corner[:,:,:4] # apply these changes to prediction
    
    # Apply confidence thresholding and NMS one image at a time within a batch.
    batch_size = prediction.size(0)
    
    write = False
    
    for ind in range(batch_size):
        image_pred = prediction[ind]
        # index of the class with max score and the score of that class
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
       
        img_classes = unique(image_pred_[:,-1])
        
        # Non maximum suppression
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0) # number of detections
            
            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break

                except IndexError:
                    break
                
                # Detections with IOU less than threshold is made zero.
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask
                
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      
            #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
                
    try:
        return output
    except:
        return 0
        

#Load class names from provided COCO class file
def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
    
    
    
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas
    
    
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img
    

        
        
    
    
