__author__ = 'shengjia'


settings = {}
settings2 = {}
settings3 = {}
environment = "BOTH"

if environment == 'REMOTE':
    settings['image_root'] = '/home/ubuntu/sdf/images/'
    settings['node_root'] = '/home/ubuntu/sdf/results/'
    settings['activation_root'] = '/home/ubuntu/sdf/activations/'
elif environment == 'LOCAL':
    settings['image_root'] = '/home/shengjia/deep-visualization-toolbox/input_images'
    settings['node_root'] = '/home/shengjia/DeepLearning/deep-visualization-toolbox/result_out'
elif environment == 'REMOTE2':
    settings['image_root'] = '/home/ubuntu/sdg/train/'
    settings['node_root'] = '/home/ubuntu/sdg/results/'
    settings['activation_root'] = '/home/ubuntu/sdg/activations/'
elif environment == 'BOTH':
    settings['image_root'] = '/home/ubuntu/sdf/images/'
    settings['node_root'] = '/home/ubuntu/sdf/results/'
    settings['activation_root'] = '/home/ubuntu/sdf/activations/'

    settings2['image_root'] = '/home/ubuntu/sdg/train/'
    settings2['node_root'] = '/home/ubuntu/sdg/results/'
    settings2['activation_root'] = '/home/ubuntu/sdg/activations/'

    settings3['image_root'] = '/home/ubuntu/sdg/train/'
    settings3['node_root'] = '/home/ubuntu/sdg/sigmoid_results/'
    settings3['activation_root'] = '/home/ubuntu/sdg/sigmoid_activations/'