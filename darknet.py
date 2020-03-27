# Necessary packages.
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from util import *


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

# Yolo detection layer
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

# Reading input image
def get_test_input():
    img = cv2.imread("images/dog-cycle-car.jpg")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W X C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_
        
def parse_cfg(cfgfile):
    """
    Parses the config file and provides list of blocks. Each block describes an element in the neural network.
    Input-  configuration file path.
    Output- list of blocks. Each block is represented as a dictionary. 108 elements.
    """
    file = open(cfgfile, 'r')
    lines = file.read().split("\n")
    lines = [x for x in lines if len(x) > 0] # Removes blank space between blocks
    lines = [x for x in lines if x[0] != '#'] # Removes comments
    lines = [x.rstrip().lstrip() for x in lines] # Removes any white space before or after key elements.
    
    block = {}
    blocks = []
    
    #Saving in the format - {type: net, batch:64, subdivision=16....}
    for line in lines:
        if line[0] == "[":
            if len(block)>0: 
                #indicates previous block data is still present
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1] # Ignore leftmost and rightmost square bracket
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks
    

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList() #Modules contained in the module list are registered.
    prev_filters = 3 # Stores the number of filters of the previous layer. Initialised as 3 for input image (RGB).
    output_filters = [] #To keep track of number of filters of all previous layers.
    # print(blocks[1:])
    #Iterate over the blocks and create torch module. Ignore first key which provides network details.
    for index, block_element in enumerate(blocks[1:]):
        module = nn.Sequential()
        # Check the type of block and create new module and append to module_list.
      
        #Conv layer
        if (block_element["type"] == "convolutional"):
            activation = block_element["activation"]
            try:
                batch_normalize = int(block_element["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
    
            filters = int(block_element["filters"])
            kernel_size = int(block_element["size"])
            stride = int(block_element["stride"])
            padding = int(block_element["pad"])
    
            #Padding amount calculation
            if padding:
                pad = (kernel_size - 1) //2 
            else:
                pad = 0
    
            #Prepare convolutional layer
            conv = nn.Conv2d(in_channels=prev_filters, out_channels=filters, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)
                
            #Add Batch Norm
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
    
            if activation == "leaky":
                actv = nn.LeakyReLU(0.1, inplace=True) # Negative slope of 0.1 for x<0
                module.add_module("leaky_{0}".format(index), actv)
        
        # Check for upsampling layer
        elif (block_element["type"] == "upsample"):
            stride = int(block_element["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode='nearest')
            module.add_module("upsample_{}".format(index), upsample)
    
        # Route layer- Concatenation of required feature maps. Placeholder for the concatenation operation.
        elif (block_element["type"] == "route"):
            block_element['layers'] = block_element['layers'].split(',')
            #Start of the route
            start = int(block_element['layers'][0])
            #If end is present
            try:
                end  = int(block_element['layers'][1])
            except:
                end = 0
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
                route = EmptyLayer()
                module.add_module('route_{0}'.format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        
        #Shortcut layer for skip connection. Addition of feature maps. Placeholder layer.
        elif (block_element['type'] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module('shortcut_{0}'.format(index), shortcut)
    
        # Detection layer (Yolo layer)
        elif (block_element['type'] == "yolo"):
            mask = block_element['mask'].split(',')
            mask = [int(x) for x in mask]
    
            anchors = block_element['anchors'].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask] # Require only those anchor boxes which are indexed by mask
    
            detection = DetectionLayer(anchors)
            module.add_module('detection_{0}'.format(index), detection)
    
        # essential book keeping
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)    
      
    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        
    #Forward pass. Calculates output, transforms output feature map for the upcoming layer.
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {} # Caches output of each layer.
        # modules - dictionary containing all the blocks of the model
        # module_list - nn.module class with individual modules

        write = 0 # After first prediction, write is set to 1 after which further predictions are concatenated.
        for i, module in enumerate(modules):
            module_type = (module["type"])

            # For convolutional and Upsampling layers
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            
            # For Route layer. Feature map format- B C H W
            if module_type == "route":
                layers = module['layers']
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i
                
                if len(layers) == 1:
                    x = outputs[i + layers[0]]

                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            #For shortcut layer
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i + from_]

            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"])
                # Number of label classes 
                num_classes = int(module["classes"])

                #Transformation
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
            outputs[i] = x
            
        return detections
    
    
    # Load model train weights
    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")
        
        # Header information - first 5 int32 values (160 bytes)
        # 1. Major version number  2. Minor version number 3. Subversion number 4,5. Images seen by the network
        header = np.fromfile(fp, dtype=np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        
        # rest of the file (float32) are the weights
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]
                
                if batch_normalize:
                    bn = model[1]
                    
                    num_bn_biases = bn.bias.numel()
                    
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    
                    # Copy the data to the model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Load the weights to the convolutional layers
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
    
    
        


