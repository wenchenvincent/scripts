'''
This script parses the model.summary() from the MLPerf Resnet50 model (resnet50_arch.txt),
and analyzes the convolution layers in the model. Assuming im2col,
it will output the dimensions of matrices if these forward convolutions
are performed using im2col+GEMM
'''
import sys
import re
import math

layers = {} 

#There are cases like the following, but as we are only concerned about Conv layer. We ignore the correctness
#of input layers for these cases
#add (Add)                       (None, 256, 56, 56)  0           bn2a_branch2c[0][0]              
#                                                                 bn2a_branch1[0][0]               
with open(sys.argv[1]) as fi:
    lines = fi.readlines()
    for line in lines:
        match = re.search(r'(\w+)\s+\((\w+)\)\s+\[?\(None,\s*(\d+),\s*(\d+),\s*(\d+)\)\s*(\d+)\s*((\w+)\[0\]\[0\])?',line)
        if match:
            name = match.group(1)
            layer_type = match.group(2)
            shape = (int(match.group(3)), int(match.group(4)), int(match.group(5)))
            num_parameters = int(match.group(6))
            input_layer = match.group(8)
            #print(name, layer_type, shape, num_parameters, input_layer)
            layer = {'name' : name, 'type' : layer_type, 'shape' : shape, 'num_parameters' : num_parameters, 'input' : input_layer}
            layers[name] = layer

        
#conv_layers = [layer for layer in layers.keys() if layer['type']=='Conv2D']
conv_layers = {name : layers[name] for name in layers.keys() if layers[name]['type']=='Conv2D'}
#print(len(conv_layers.keys()))

for layer in conv_layers.values():
    K, H, W = layer['shape']
    C = layers[layer['input']]['shape'][0]
    #print(layer['num_parameters'], K, C)
    F = int( math.sqrt(layer['num_parameters']/(K*C)) )
    GEMM_M = K
    GEMM_K = F*F*C
    GEMM_N = H*W
    print('layer=%s, input_channel=%d, H=%d, W=%d, output_channel=%d, filter_size=%dx%d' %(layer['name'], C, H, W, K, F,F))
    print('After im2col, M=%d, K=%d, N=%d' %(GEMM_M, GEMM_K, GEMM_N))
