__author__ = 'shengjia'


import os
import argparse
import numpy as np
import sys
sys.path.insert(0, '..')
from find_maxes.loaders import load_imagenet_mean, load_labels, caffe

def read_filelist(filename, max_size):
    path_list = []
    label_list = []
    infile = open(filename)
    count = 0
    while True:
        if max_size is not None and count >= max_size:
            break
        content = infile.readline().split()
        if len(content) == 0:
            break
        elif len(content) == 2:
            path_list.append(content[0])
            label_list.append(int(content[1]))
        else:
            print("Error: received " + str(len(content)) + " items in a line")
            assert False

    return path_list, label_list


# models/caffenet-yos/caffenet-yos-deploy.prototxt models/caffenet-yos/caffenet-yos-weights /home/ubuntu/sdf/images /home/ubuntu/sdf/database_list activation_out conv3,conv4,conv5
# models/caffenet-yos/caffenet-yos-deploy.prototxt models/caffenet-yos/caffenet-yos-weights input_images input_images/list activation_out conv3,conv4,conv5
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loads a pickled NetMaxTracker and outputs one or more of {the patches of the image, a deconv patch, a backprop patch} associated with the maxes.')
    parser.add_argument('--gpu',         action = 'store_true', help = 'Use gpu.')
    parser.add_argument('--num',         type = int, default=None, help = 'Number of images to process')
    parser.add_argument('net_prototxt',  type = str, help = 'Network prototxt to load')
    parser.add_argument('net_weights',   type = str, help = 'Network weights to load')
    parser.add_argument('datadir',       type = str, help = 'Directory to look for files in')
    parser.add_argument('filelist',      type = str, help = 'List of image files to consider, one per line. Must be the same filelist used to produce the NetMaxTracker!')
    parser.add_argument('outdir',        type = str, help = r'Which output directory to use. Files are output into outdir/layer/unit_%%04d/{maxes,deconv,backprop}_%%03d.png')
    parser.add_argument('layers',         type = str, help = 'Which layer to output, separate by comma')
    args = parser.parse_args()

    if args.gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    layers = args.layers.split(',')
    print("Recording layers " + str(layers))

    imagenet_mean = load_imagenet_mean()
    net = caffe.Classifier(args.net_prototxt, args.net_weights,
                           mean=imagenet_mean,
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))

    path_list, label_list = read_filelist(args.filelist, args.num)

    result_array = {}
    for layer in layers:
        layer_result = {'name': layer}
        layer_shape = net.blobs[layer].data.shape
        if len(layer_shape) == 4 or len(layer_shape) == 2:
            layer_result['activation'] = np.ndarray((len(path_list), layer_shape[1]), dtype=float)
        else:
            print("Unknown layer shape")
        result_array[layer] = layer_result

    iter_count = 0
    for path, label in zip(path_list, label_list):
        fullpath = os.path.join(args.datadir, path)
        if not os.path.isfile(fullpath):
            print("Error: file " + fullpath + " not found")
        im = caffe.io.load_image(fullpath)
        net.predict([im], oversample=False)   # Just take center crop
        for layer in layers:
            result_array[layer]['activation'][iter_count:] = net.blobs[layer].data.max(3).max(2).max(0)
        iter_count += 1
        if iter_count % 1000 == 0:
            print("Processing " + str(iter_count) + "-th image")

    print("Finished, saving to file")
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
    for layer in layers:
        np.save(os.path.join(args.outdir, layer), result_array[layer]['activation'])
