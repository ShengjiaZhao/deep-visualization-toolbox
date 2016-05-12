__author__ = 'shengjia'


settings = {}
environment = "REMOTE"

if environment == 'REMOTE':
    settings['image_root'] = '/home/ubuntu/sdf/images/'
    settings['node_root'] = '/home/ubuntu/sdf/results/'
    settings['activation_root'] = '/home/ubuntu/sdf/activations/'
elif environment == 'LOCAL':
    settings['image_root'] = '/home/shengjia/deep-visualization-toolbox/input_images'
    settings['node_root'] = '/home/shengjia/DeepLearning/deep-visualization-toolbox/result_out'
