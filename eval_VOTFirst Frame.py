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

import os.path
import cv2
import operator


        
# Static variables initialization 
enable_bash = False
first_frame_only = True


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
        parser.add_argument('test_dir', help='Patch to the Input directory of .bmp files.')
        parser.add_argument('output_dir',help='Patch to the Output directory of .bmp files.')
        parser.add_argument('--use_gpu',action='store_true',dest='use_gpu',help=('Switch to use gpu.'))
        parser.add_argument('--gpu_id', default=0,type=int,dest='gpu_id',
                            help=('GPU id. Effective only when --use_gpu=True.'))
        args = parser.parse_args(['--use_gpu' ,'--gpu_id=0','models/feature_net.pbtxt','models/yosemite_r_0.01_m_0.feature_net.pb',
                                  'models/classifier_net.pbtxt','models/yosemite_r_0.01_m_0.classifier_net.pb', 
                                  'data/vot2016Input','data/vot2016Output'])

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
        
    count_dir = 0;
    for path, subdirs, files in os.walk(args.test_dir):
        if path == "data/vot2016Input":
            continue
        ground_truth_txt = path + "/groundtruth.txt"
        
        # Read the image names in the dataset
        image_list = []
        for filename in glob.glob(path+'/*.jpg'):
            image_list.append(filename)
        image_list.sort()
        image_first = image_list[0]
        image_list = image_list[1:]
            # read ground truth data
        with open(ground_truth_txt) as f:
            ground_truth = [[float(x) for x in line.split(",")] for line in f]
                
        # track the object for every image in the video     
        for i,image_current in enumerate(image_list): 
            if i%3 == 0:
                input_patches = ReadPatches(ground_truth,i, image_first, image_current \
                        ,first_frame_only)
                        
                # Compute features.
                feats = [feature_net.ComputeFeature(input_patches[0]), \
                 feature_net.ComputeFeature(input_patches[1])]
                 
                # Compute scores.
                scores = metric_net.ComputeScore(feats[0], feats[1])
                
                # draw bounding box of values above threshold (0.95) and top value
                top_95 = [m for m,score in enumerate(scores) if score > 0.95]
                index, value = max(enumerate(scores[1:]), key=operator.itemgetter(1))
                if os.path.isfile(image_current):
                    img = cv2.imread(image_current)
                    imgtop = cv2.imread(image_current)
    
                w = 0
                h = 0
                image_str = "data/vot2016Output/"+ os.path.basename(os.path.normpath(path)) \
                                + "/" + os.path.basename(os.path.normpath(image_current))
                imagetop_str = "data/vot2016Output/"+ os.path.basename(os.path.normpath(path)) \
                                + "/top" + os.path.basename(os.path.normpath(image_current))
                for n in top_95:
                    if n == 0:
                        X1,Y1,X2,Y2,X3,Y3,X4,Y4 = ground_truth[i+1]
                        L=[[X1,Y1],[X2,Y2],[X3,Y3],[X4,Y4]]
                        cnt = np.array(L).reshape((-1,1,2)).astype(np.int32)
                        x,y,w,h = cv2.boundingRect(cnt)
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        cv2.rectangle(imgtop,(x,y),(x+w,y+h),(0,255,0),2)
                    else:
                        y = settings.pairs[n-1][0] -h/2
                        x = settings.pairs[n-1][1] -w/2
                        color_scale = 100 + ((scores[n]-0.95)/(0.05)) * 155
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,color_scale),2)
                cv2.imwrite(image_str, img)
                y = settings.pairs[index][0] -h/2
                x = settings.pairs[index][1] -w/2
                cv2.rectangle(imgtop,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.imwrite(imagetop_str, imgtop)
                print "end loop 1"
        print "end loop 2"
    print "end loop 3"
      

    
    
if __name__ == '__main__':
    main()
