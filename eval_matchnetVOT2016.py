"""This script evaluates a given matchnet model (including feature net and metric
   net) on a given ubc test set.
   
   Dataset: VOT2016
   Compares first frame with previous frame
   
"""
import sys
sys.path.append("/home/arpita/code/caffe/python")

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from caffe.proto import caffe_pb2
from caffe.io import *
import glob
from matchnet import *
from getpatch import *
import heapq
import operator
import matplotlib.pyplot as plt
import settings


        
# Static variables initialization 
enable_bash = False
first_frame_only = False

def main():
    # Parse input arguments.
    if enable_bash:
        args = ParseArgs()
    else:
        parser = ArgumentParser(description=__doc__,formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument('feature_net_model',help='Feature network description.')
        parser.add_argument('feature_net_params',help='Feature network parameters.')
        parser.add_argument('metric_net_model', help='Metric network description.')
        parser.add_argument('metric_net_params', help='Metric network parameters')
        parser.add_argument('test_dir', help='Patch to the directory of .bmp files.')
        parser.add_argument('ground_truth',help=('Ground Truth in text format.' ))
        parser.add_argument('output_txt',help='Result file containing the predictions.')
        parser.add_argument('--use_gpu',action='store_true',dest='use_gpu',help=('Switch to use gpu.'))
        parser.add_argument('--gpu_id', default=0,type=int,dest='gpu_id',
                            help=('GPU id. Effective only when --use_gpu=True.'))
        args = parser.parse_args(['--use_gpu' ,'--gpu_id=0','models/feature_net.pbtxt','models/yosemite_r_0.01_m_0.feature_net.pb',
                                  'models/classifier_net.pbtxt','models/yosemite_r_0.01_m_0.classifier_net.pb', 
                                  'data/vot2016/godfather','data/vot2016/godfather/groundtruth.txt',
                                  'tmp/predictions.txt'])

    settings.init()
    # Initialize networks.
    feature_net = FeatureNet(args.feature_net_model, args.feature_net_params)
    metric_net = MetricNet(args.metric_net_model, args.metric_net_params)

    if args.use_gpu:
        caffe.set_mode_gpu()
        print "GPU mode"
    else:
        caffe.set_mode_cpu()
        print "CPU mode"
        
    
    
    # Read the image names in the dataset
    image_list = []
    for filename in glob.glob(args.test_dir+'/*.jpg'):
        image_list.append(filename)
    image_list.sort()
    image_list_prev = image_list[:-1]
    image_list = image_list[1:]
    # read ground truth data
    with open(args.ground_truth) as f:
        ground_truth = [[float(x) for x in line.split(",")] for line in f]
    
    
    # track the object for every image in the video
    for i,(image_previous, image_current) in enumerate(zip(image_list_prev,image_list)): 
        input_patches = ReadPatches(ground_truth,i, image_first, image_current \
                        ,first_frame_only)
        
        # Compute features.
        feats = [feature_net.ComputeFeature(input_patches[0]),
                 feature_net.ComputeFeature(input_patches[1])]
          
        # Compute scores.
        scores = metric_net.ComputeScore(feats[0], feats[1])
        
        index = [0,1,2,3,4]
        plot_image = np.array([])      
        for i in index:
            total_arr = np.array([])
            for j in [0,1]:
                arr = input_patches[j][i][0].astype(np.uint8)
                total_arr = np.vstack([total_arr, arr]) if total_arr.size else arr
            plot_image = np.hstack([plot_image, total_arr]) if plot_image.size else total_arr
        plt.imshow(plot_image,cmap = plt.get_cmap('gray'))
        plt.show()
        
        top5_scores = zip(*heapq.nlargest(5, enumerate(scores), key=operator.itemgetter(1)))[0]

        index = list(top5_scores)
        plot_image = np.array([])      
        for i in index:
            total_arr = np.array([])
            for j in [0,1]:
                arr = input_patches[j][i][0].astype(np.uint8)
                total_arr = np.vstack([total_arr, arr]) if total_arr.size else arr
            plot_image = np.hstack([plot_image, total_arr]) if plot_image.size else total_arr
        plt.imshow(plot_image,cmap = plt.get_cmap('gray'))
        plt.show()
        
    
    print "end"
    
    
if __name__ == '__main__':
    main()
