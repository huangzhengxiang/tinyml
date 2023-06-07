import numpy as np
import torch

model = torch.load('.\\ptq.pth')

layer_dict = {}
current_layer_list = ['None']
layer_name = ''
for layer_key in model.keys():
    level = layer_key.split('.')
    if level[-1] == 'scales':
        layer_name_list = level[:-2]
        param_name = "w_scales"
    else:
        layer_name_list = level[:-1]
        param_name = level[-1]
    if layer_name_list != current_layer_list:
        current_layer_list = layer_name_list
        layer_name = layer_name_list[0]
        for i in range(1, len(layer_name_list)):
            layer_name += '.'
            layer_name += layer_name_list[i]
        layer_dict[layer_name] = {}
    layer_dict[layer_name][param_name] = model[layer_key]
prev_scale = layer_dict['QuantStub']['scale']
prev_zero = layer_dict['QuantStub']['zero_point']

for layer_key in layer_dict.keys():
    layer = layer_dict[layer_key]
    layer['x_scale'] = prev_scale
    layer['x_zero'] = prev_zero
    prev_scale = layer['scale']
    prev_zero = layer['zero_point']
    layer['y_scale'] = layer['scale']
    layer['y_zero'] = layer['zero_point']
    layer.pop('scale')
    layer.pop('zero_point')
    if 'bias' in layer.keys():
        bias = layer['bias']
        if bias is not None:
            if bias.dtype == np.float32:
                bias=torch.tensor(bias)
                sb = torch.tensor(layer["w_scales"]*layer['x_scale'])
                qb = torch.clamp(bias / sb,min=-2**31,max=2**31-1).floor().int()
                q_bias = qb.numpy() - (layer['x_zero'] * layer['weight']).reshape(1,-1).sum(axis=-1)
                layer['bias'] = q_bias

layer_dict.pop("QuantStub")

torch.save(layer_dict,"ptq_weight.pth")
