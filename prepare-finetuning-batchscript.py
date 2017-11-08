import sys

sys.path.append('/home/simon/Research/lib/caffe/python')
import caffe

import h5py
import numpy as np
import os
import google.protobuf
import google.protobuf.text_format
import uuid
import pyprind
import argparse

parser = argparse.ArgumentParser(description='Prepare fine-tuning of multiscale alpha pooling. The working directory should contain train_val.prototxt of vgg16. The models will be created in the subfolders.')
parser.add_argument('train_imagelist', type=str, help='Path to imagelist containing the training images. Each line should contain the path to an image followed by a space and the class ID.')
parser.add_argument('val_imagelist', type=str, help='Path to imagelist containing the validation images. Each line should contain the path to an image followed by a space and the class ID.')
parser.add_argument('--init_weights', type=str, help='Path to the pre-trained vgg16 model', default='../vgg16_imagenet.caffemodel')
parser.add_argument('--label', type=str, help='Label of the created output folder', default='nolabel')
parser.add_argument('--gpu_id', type=int, help='ID of the GPU to use', default=0)
parser.add_argument('--num_classes', type=int, help='Number of object categories', default=6000)
parser.add_argument('--image_root', type=str, help='Image root folder, used to set the root_folder parameter of the ImageData layer of caffe.', default='/')
args = parser.parse_args()

# Some other parameters, usually you don't need to change this
initial_alpha = 2.0
chop_off_layer = 'relu5_3'
resize_size = 560
crop_size = 560
resolutions = [224,560]
prefix_template = 'res%i/'
num_classes = args.num_classes

caffe.set_device(args.gpu_id)
caffe.set_mode_gpu()

# Create parameter files
# Net
netparams_in = caffe.proto.caffe_pb2.NetParameter()
protofile = 'train_val.prototxt'
google.protobuf.text_format.Merge(open(protofile).read(),netparams_in)

# In[3]:

# Change to working dir
os.chdir('./vgg16-training/')
working_dir = '%s_%s'%(args.label,str(uuid.uuid4()))
try: os.makedirs(working_dir) 
except: pass
os.chdir(working_dir)


# Prepare data layer
lyr = netparams_in.layer
#lyr[0].image_data_param.source = '/home/simon/Research/generic/results/2016-07-01_dataset_filter/train_only_animals_vehicles.txt'
lyr[0].image_data_param.source = args.train_imagelist
lyr[0].image_data_param.root_folder = args.image_root
lyr[0].image_data_param.batch_size = 1
lyr[0].image_data_param.smaller_side_size[0] = resize_size
#lyr[0].image_data_param.smaller_side_size[1] = crop_size
lyr[0].transform_param.crop_size = crop_size
lyr[0].type = 'ImageData'

#lyr[1].image_data_param.source = '/home/simon/Research/generic/results/2016-07-01_dataset_filter/val_only_animals_vehicles.txt'
lyr[1].image_data_param.source = args.val_imagelist
lyr[1].image_data_param.root_folder = args.image_root
lyr[1].image_data_param.batch_size = 1
lyr[1].image_data_param.smaller_side_size[0] = resize_size
#lyr[1].image_data_param.smaller_side_size[1] = crop_size
lyr[1].transform_param.crop_size = crop_size
lyr[1].type = 'ImageData'


# Add batch norm
netparams = caffe.proto.caffe_pb2.NetParameter()
netparams.name = netparams_in.name

alpha_outputs = []


# Input layers
for idx, l in enumerate(netparams_in.layer):
    if l.type in ['ImageData', 'Data']:
        netparams.layer.add()
        netparams.layer[-1].MergeFrom(l)

for idx, l in enumerate(netparams_in.layer):
    if l.type in ['ImageData', 'Data']:
        netparams.layer.add()
        netparams.layer[-1].name = 'zeros'
        netparams.layer[-1].type = 'DummyData'
        netparams.layer[-1].top.append('zeros')
        netparams.layer[-1].dummy_data_param.shape.add()
        netparams.layer[-1].dummy_data_param.shape[0].dim.extend([l.image_data_param.batch_size,1])
        netparams.layer[-1].include.add()
        netparams.layer[-1].include[0].phase = l.include[0].phase


# In[9]:


for res_idx, res in enumerate(resolutions):
    prefix = prefix_template%res 
    netparams.layer.add()
    netparams.layer[-1].name = prefix + netparams_in.layer[0].top[0]
    netparams.layer[-1].type = 'SpatialTransformer'
    netparams.layer[-1].bottom.append(netparams_in.layer[0].top[0])
    netparams.layer[-1].bottom.append('zeros')
    netparams.layer[-1].top.append(netparams.layer[-1].name)
    netparams.layer[-1].st_param.theta_1_1 = 1
    netparams.layer[-1].st_param.theta_1_2 = 0
    netparams.layer[-1].st_param.theta_1_3 = 0
    netparams.layer[-1].st_param.theta_2_1 = 0
    netparams.layer[-1].st_param.theta_2_2 = 1
    #netparams.layer[-1].st_param.theta_2_3 = 0
    netparams.layer[-1].st_param.to_compute_dU = False
    netparams.layer[-1].st_param.output_H = res;
    netparams.layer[-1].st_param.output_W = res;


# In[10]:


for res_idx, res in enumerate(resolutions):
    for idx, l in enumerate(netparams_in.layer):
        if l.type in ['ImageData', 'Data']:
            continue
        netparams.layer.add()
        netparams.layer[-1].MergeFrom(l)
        prefix = prefix_template%res 
        netparams.layer[-1].name = prefix + netparams.layer[-1].name 
        for i in range(len(l.top)):
            netparams.layer[-1].top[i] = prefix + netparams.layer[-1].top[i]
        for i in range(len(l.bottom)):
            netparams.layer[-1].bottom[i] = prefix + netparams.layer[-1].bottom[i]
        for param_idx, p in enumerate(netparams.layer[-1].param):
            p.name = '%s_param%i'%(l.name,param_idx)

        if l.name == chop_off_layer:
            break

    # Add alpha layer
    netparams.layer.add()
    netparams.layer[-1].name = prefix + 'alpha_power'
    netparams.layer[-1].type = 'SignedPower'
    netparams.layer[-1].bottom.append(netparams.layer[-2].top[0])
    netparams.layer[-1].top.append(netparams.layer[-1].name)
    netparams.layer[-1].power_param.power = initial_alpha - 1
    netparams.layer[-1].param.add()
    netparams.layer[-1].param[0].name = 'alpha_power'
    netparams.layer[-1].param[0].lr_mult = 10
    netparams.layer[-1].param[0].decay_mult = 0

    # Add outer product layer
    netparams.layer.add()
    netparams.layer[-1].name = prefix + 'outer_product'
    netparams.layer[-1].type = 'CompactBilinear'
    netparams.layer[-1].bottom.append(netparams.layer[-3].top[0])
    netparams.layer[-1].bottom.append(netparams.layer[-2].top[0])
    netparams.layer[-1].top.append(netparams.layer[-1].name)
    netparams.layer[-1].compact_bilinear_param.num_output = 8192

    alpha_outputs.append(netparams.layer[-1].top[0])


# In[11]:


if len(alpha_outputs)>1:
    netparams.layer.add()
    netparams.layer[-1].name = 'sum'
    netparams.layer[-1].type = 'Eltwise'
    for alpha_out in alpha_outputs:
        netparams.layer[-1].bottom.append(alpha_out)
    netparams.layer[-1].top.append(netparams.layer[-1].name)

if True:
    netparams.layer.add()
    netparams.layer[-1].name = 'root'
    netparams.layer[-1].type = 'SignedPower'
    netparams.layer[-1].bottom.append(netparams.layer[-2].name)
    netparams.layer[-1].top.append(netparams.layer[-1].name)
    netparams.layer[-1].power_param.power = 0.5 #1.0 / (gamma)
    netparams.layer[-1].param.add()
    netparams.layer[-1].param[0].lr_mult = 0
    netparams.layer[-1].param[0].decay_mult = 0

if False:
    # Add reshape for global bn
    netparams.layer.add()
    netparams.layer[-1].name = 'final_dropout'
    netparams.layer[-1].type = 'Dropout'
    netparams.layer[-1].bottom.append(netparams.layer[-2].top[0])
    netparams.layer[-1].top.append(netparams.layer[-1].name)
    netparams.layer[-1].dropout_param.dropout_ratio = 0.5

if True:
    netparams.layer.add()
    netparams.layer[-1].name = 'l2'
    netparams.layer[-1].type = 'L2Normalize'
    netparams.layer[-1].bottom.append(netparams.layer[-2].top[0])
    netparams.layer[-1].top.append(netparams.layer[-1].name)

# fc8
netparams.layer.add()
netparams.layer[-1].name = 'fc8_ft'
netparams.layer[-1].type = 'InnerProduct'
netparams.layer[-1].bottom.append(netparams.layer[-2].top[0])
netparams.layer[-1].top.append(netparams.layer[-1].name) 
netparams.layer[-1].inner_product_param.num_output = num_classes
[netparams.layer[-1].param.add() for _ in range(2)]
netparams.layer[-1].param[0].lr_mult = 1
netparams.layer[-1].param[0].decay_mult = 1
netparams.layer[-1].param[1].lr_mult = 2
netparams.layer[-1].param[1].decay_mult = 2

# Accuracy
netparams.layer.add()
netparams.layer[-1].name = 'loss'
netparams.layer[-1].type = 'SoftmaxWithLoss'
netparams.layer[-1].bottom.append(netparams.layer[-2].top[0])
netparams.layer[-1].bottom.append('label')
netparams.layer[-1].top.append(netparams.layer[-1].name) 

# Softmax
netparams.layer.add()
netparams.layer[-1].name = 'Accuracy'
netparams.layer[-1].type = 'Accuracy'
netparams.layer[-1].bottom.append(netparams.layer[-3].top[0])
netparams.layer[-1].bottom.append('label')
netparams.layer[-1].top.append(netparams.layer[-1].name) 
netparams.layer[-1].include.add()
netparams.layer[-1].include[0].phase = 1


# Learning rates and decays and so on
for l in netparams.layer:
    if l.type in ['InnerProduct','Convolution','Scale']:
        [l.param.add() for _ in range(2 - len(l.param))]
        l.param[0].lr_mult = 1
        l.param[0].decay_mult = 1
        l.param[1].lr_mult = 2
        l.param[1].decay_mult = 2
    if l.type in ['InnerProduct']:
        l.inner_product_param.weight_filler.type = "gaussian"
        l.inner_product_param.weight_filler.ClearField('std')
        l.inner_product_param.weight_filler.std = 0.01
        l.inner_product_param.bias_filler.type = "constant"
        l.inner_product_param.bias_filler.value = 0.0
    if l.name in ['fc8_ft']:
        l.inner_product_param.weight_filler.type = "gaussian"
        l.inner_product_param.weight_filler.std = 0.000000001
        l.inner_product_param.bias_filler.type = "constant"
        l.inner_product_param.bias_filler.value = 0.01
    if l.type in ['Convolution']:
        l.convolution_param.weight_filler.type = "gaussian"
        l.convolution_param.weight_filler.ClearField('std')
        l.inner_product_param.weight_filler.std = 0.01
        l.convolution_param.bias_filler.type = "constant"
        l.convolution_param.bias_filler.value = 0.0
    if l.type == "BatchNorm":
        l.param[0].lr_mult = 0
        l.param[1].lr_mult = 0
        l.param[2].lr_mult = 0
        l.batch_norm_param.ClearField('use_global_stats')
#    if l.name in ['fc6','fc7']:
#        l.inner_product_param.num_output = 2048


num_images = [len([None for _ in open(netparams.layer[i].image_data_param.source,'r')]) for i in [0,1]]
iter_per_epoch = int(num_images[0]/32) 

# Solver
solverfile = 'ft.solver'
params = caffe.proto.caffe_pb2.SolverParameter()
params.net = u'ft.prototxt'
params.test_iter.append(int(len([None for _ in open(netparams.layer[1].image_data_param.source,'rt')]) / lyr[0].image_data_param.batch_size))
params.test_interval = 10000
params.test_initialization = True
params.base_lr = 0.001
params.display = 100
params.max_iter = 200 * iter_per_epoch
params.lr_policy = "fixed"
params.power = 1
#params.stepsize = 100000
#params.gamma = 0.1
#params.momentum = 0.9
params.weight_decay = 0.0005
params.snapshot = 10000
#params.random_seed = 0
params.snapshot_prefix = "ft"
params.net = "ft.prototxt"
params.iter_size = int(8/lyr[0].image_data_param.batch_size)
#params.type = "Nesterov"
assert params.iter_size > 0
open(solverfile,'w').write(google.protobuf.text_format.MessageToString(params))
open(params.net,'w').write(google.protobuf.text_format.MessageToString(netparams))

net_origin = caffe.Net('/home/simon/Data/caffe/vgg16/deploy.prototxt', args.init_weights, caffe.TEST)
net_target = caffe.Net('ft.prototxt',caffe.TEST)

for origin_param in net_origin.params.keys():
    for res in resolutions:
        prefix = prefix_template%res
        target_param = prefix + origin_param
        if target_param in net_target.params:
            for idx in range(len(net_origin.params[origin_param])):
                #print('Copying %s[%i] to %s[%i]'%(origin_param, idx, target_param, idx))
                net_target.params[target_param][idx].data[...] = net_origin.params[origin_param][idx].data

net_target.save('model_init')
del net_origin
del net_target


#Calc the features
def calc_features(net, n_images, blobs):
    n_images = int(0.6*n_images)
    batchsize = net.blobs['data'].data.shape[0]
    feats = dict()
    for blob in blobs:
        out_shape = list(net.blobs[blob].data.shape)
        out_shape[0] = n_images
        print('Will allocate {:.2f} GiB of memory'.format(np.prod(out_shape)*2/1024/1024/1024))
        feats[blob] = np.zeros(tuple(out_shape),dtype=np.float16 if not blob=='label' else np.int32)
    print('Need %.3f GiB'%(np.sum([x.nbytes for x in feats.values()])/1024/1024/1024))
        
    for it in pyprind.prog_bar(range(0,n_images,batchsize),update_interval=10, stream=sys.stderr):
        net.forward()
        for blob in blobs:
            feats[blob][it:it+batchsize,...] = net.blobs[blob].data[:feats[blob][it:it+batchsize,...].shape[0],...]
            
    return [feats[blob] for blob in blobs]

last_blob = [l.bottom[0] for l in netparams.layer if l.type == 'InnerProduct'][-1]

solver = caffe.get_solver('ft.solver')
solver.net.copy_from('model_init')
train_feats,train_labels = calc_features(solver.net,num_images[0],[last_blob,'label'])
del solver

try:
	f = h5py.File('features.h5', "w")
	dset = f.create_dataset("feats", train_feats.shape, dtype='float16', compression="gzip", compression_opts=1)
	dset[...] = train_feats
	dset = f.create_dataset("labels", train_labels.shape, dtype='int32', compression="gzip", compression_opts=1)
	dset[...] = train_labels
	f.close()
except e:
	pass



netparams_fixed = caffe.proto.caffe_pb2.NetParameter()
netparams_fixed.layer.add()
netparams_fixed.layer[-1].name = 'data'
netparams_fixed.layer[-1].type = 'Input'
netparams_fixed.layer[-1].top.append(last_blob)
netparams_fixed.layer[-1].input_param.shape.add()
netparams_fixed.layer[-1].input_param.shape[0].dim.extend((32,) + train_feats.shape[1:])

netparams_fixed.layer.add()
netparams_fixed.layer[-1].name = 'label'
netparams_fixed.layer[-1].type = 'Input'
netparams_fixed.layer[-1].top.append('label')
netparams_fixed.layer[-1].input_param.shape.add()
netparams_fixed.layer[-1].input_param.shape[0].dim.extend((32,))
# Add all layers after fc8
approached_fc8 = False
for l in netparams.layer:
    if l.name == 'fc8_ft':
        l.param[0].lr_mult = 1
        l.param[0].decay_mult = 1
        l.param[1].lr_mult = 1
        l.param[1].decay_mult = 1
        l.inner_product_param.weight_filler.std = 0.0001
        l.inner_product_param.bias_filler.value = 0
    approached_fc8 = approached_fc8 or l.name == 'fc8_ft'
    if approached_fc8:
        netparams_fixed.layer.add()
        netparams_fixed.layer[-1].MergeFrom(l)


# In[42]:
iter_per_epoch = int(iter_per_epoch)
# Solver
solverfile = 'ft_fixed.solver'
params = caffe.proto.caffe_pb2.SolverParameter()
params.net = u'ft_fixed.prototxt'
#params.test_iter.append(1450)
#params.test_interval = 1000
params.test_initialization = False
params.base_lr = 1
params.display = 100
params.max_iter = 360 * iter_per_epoch
params.lr_policy = "multistep"
params.stepvalue.extend([ep * iter_per_epoch for ep in [120,180,240,300]])
#params.power = 1
#params.stepsize = 100000
params.gamma = 0.25
params.momentum = 0.9
params.weight_decay = 0.000005
params.snapshot = 10000000
#params.random_seed = 0
params.snapshot_prefix = "ft_fixed"
params.iter_size = 1
assert params.iter_size > 0
open(solverfile,'w').write(google.protobuf.text_format.MessageToString(params))
open(params.net,'w').write(google.protobuf.text_format.MessageToString(netparams_fixed))

solver = caffe.get_solver('ft_fixed.solver')

# Train
for it in pyprind.prog_bar(range(params.max_iter), stream=sys.stderr):
    train_ids = random.sample(range(train_feats.shape[0]),32)
    solver.net.blobs[last_blob].data[...] = train_feats[train_ids,...]
    solver.net.blobs['label'].data[...] = train_labels[train_ids]
    solver.step(1)

solver.net.save('model_lr')
del solver

solver = caffe.get_solver('ft.solver')
solver.net.copy_from('model_init')
solver.net.copy_from('model_lr')
solver.net.save('model_lr')
