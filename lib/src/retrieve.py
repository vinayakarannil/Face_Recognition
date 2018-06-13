"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
from lib.src.facenet import load_data,load_img,load_model,to_rgb
#import lfw
import os
import sys
import math
import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import cv2

from lib.src.align import detect_face
from datetime import datetime
from scipy import ndimage
from scipy.misc import imsave 
from scipy.spatial.distance import cosine
import pickle
#face_cascade = cv2.CascadeClassifier('out/face/haarcascade_frontalface_default.xml')
parser = argparse.ArgumentParser()
    
parser.add_argument('--imagePath', type=str, default='/home/impadmin/Pictures/IMG_20171009_194254_547.jpg',
        help='Path to the data directory containing aligned LFW face patches.')
parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
parser.add_argument('--model', type=str,default='../../dl_face_recognition/trunk/lib/src/ckpt/20170512-110547', 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

args = parser.parse_args()

def similarity(a, b):
        similarity = np.linalg.norm(a-b)
        #print (similarity)
	return(similarity)

def get_top_k_similar(image_data, pred, pred_final, k=9):
        print("total data",len(pred))
        print(image_data.shape)
        #for i in pred:
		#print(i.shape)
                #break
        #os.mkdir('static/result')
        
    # cosine calculates the cosine distance, not similiarity. Hence no need to reverse list
    	top_k_ind = np.argsort([np.linalg.norm(image_data-pred_row) \
                            for ith_row, pred_row in enumerate(pred)])[:k]
        print(top_k_ind)
        
        for i, neighbor in enumerate(top_k_ind):
        	image = ndimage.imread(pred_final[neighbor])
                
                timestr = datetime.now().strftime("%Y%m%d%H%M%S")
                name= timestr+"."+str(i)
                print(name)
                name = 'static/result/image'+"_"+name+'.jpg'
                imsave(name, image)


def align_face(img,pnet, rnet, onet):
                        print("inside align face")
                        print("inside size:",img.size)
		        minsize = 20 # minimum size of face
		        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
		        factor = 0.709 # scale factor
                        #img = misc.imread(image_path)
                        print("before img.size == 0")	
                        if img.size == 0:
                                print("empty array")
				return False,img,[0,0,0,0]
                        print("after img.size == 0")
                        print("before img.ndim<2")
                        if img.ndim<2:
                            print('Unable to align "%s"' % image_path)
                            #text_file.write('%s\n' % (output_filename))
                            #continue
                        print("after img.ndim<2")
                        print("before img.ndim == 2")
                        if img.ndim == 2:
                            print("yes, img.ndim ==2")
                            img = to_rgb(img)
                        print("after img.ndim == 2")
                        img = img[:,:,0:3]
                        print("before detect_face.detect_face")			
			bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        print("after detect_face.detect_face")
                        nrof_faces = bounding_boxes.shape[0]
                        print("nrof_faces",nrof_faces)
                        
                        if nrof_faces==0:
                            print("inside nrof_faces>0")
                            return False,img,[0,0,0,0]
                        else:
                            det = bounding_boxes[:,0:4]
                            det_arr = []
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces>1:
                                if args.detect_multiple_faces:
                                    for i in range(nrof_faces):
                                        det_arr.append(np.squeeze(det[i]))
                                else:
                                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                    det_arr.append(det[index,:])
                            else:
                                det_arr.append(np.squeeze(det))
                            if len(det_arr)>0:
                                    faces = []
                                    bboxes = []
		                    for i, det in enumerate(det_arr):
		                        det = np.squeeze(det)
		                        bb = np.zeros(4, dtype=np.int32)
		                        bb[0] = np.maximum(det[0]-args.margin/2, 0)
		                        bb[1] = np.maximum(det[1]-args.margin/2, 0)
		                        bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
		                        bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
		                        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
		                        scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
		                        misc.imsave("cropped.png", scaled)
                                        faces.append(scaled)
                                        bboxes.append(bb)
		                        print("leaving align face")
		                    return True,faces,bboxes
			

def identify_person(image_vector, feature_array, k=9):
	    top_k_ind = np.argsort([np.linalg.norm(image_vector-pred_row) \
                            for ith_row, pred_row in enumerate(feature_array.values())])[:k]
            result = feature_array.keys()[top_k_ind[0]]
            return result


def recognize_face(sess,pnet, rnet, onet,feature_array):

		    # Read the file containing the pairs used for testing
		    #pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

		    # Get the paths for the corresponding images
		    #paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
		    #path = args.imagePath
		    # Load the model
		    
		    #load_model(args.model)
		    # Get input and output tensors
		    images_placeholder = sess.graph.get_tensor_by_name("input:0")
		    images_placeholder = tf.image.resize_images(images_placeholder,(160,160))
		    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
		    phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
		    
		    #image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
		    image_size = args.image_size
		    embedding_size = embeddings.get_shape()[1]
		    
		    # Run forward pass to calculate embeddings
		    print('Runnning forward pass on LFW images')
		    '''batch_size = args.lfw_batch_size
		    nrof_images = len(paths)
		    nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
		    emb_array = np.zeros((nrof_images, embedding_size))
		    for i in range(nrof_batches):
		        print("batch:",str(i))
		        start_index = i*batch_size
		        end_index = min((i+1)*batch_size, nrof_images)
		        paths_batch = paths[start_index:end_index]'''
		    #pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
		    print("created network")
                    cap = cv2.VideoCapture(-1)
    		    while(True):
			    ret, frame = cap.read()
        		    gray = cv2.cvtColor(frame, 0)
                            
        		    if cv2.waitKey(1) & 0xFF == ord('q'):
		                     cap.release()
		                     cv2.destroyAllWindows()
		                     break
                            if (gray.size > 0):
                                    print(gray.size)
		                    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
				    response, faces,bboxs = align_face(gray,pnet, rnet, onet)
                                    print(response)
                                    if (response == True):
                                            for i, image in enumerate(faces): 
                                                    bb = bboxs[i]
						    images = load_img(image, False, False, image_size)
						    
						    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
						    feature_vector = sess.run(embeddings, feed_dict=feed_dict)
						    #plt.imshow(feature_vector)
						    #plt.show()
						    
						    #print(feature_array.keys())
						    #image_paths = np.load("images.npy")
						    result = identify_person(feature_vector, feature_array,9)
						    print(result.split("/")[2])
						    #for (x,y,w,h) in faces:
						    cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
						    W = int(bb[2]-bb[0])//2
						    H = int(bb[3]-bb[1])//2
						    cv2.putText(gray,"Hello "+result.split("/")[2],(bb[0]+W-(W//2),bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
						    cv2.imshow('img',gray)
			    else:
				    continue
		

 
            #tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array, 
                #actual_issame, nrof_folds=args.lfw_nrof_folds)

            #print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            #print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            #auc = metrics.auc(fpr, tpr)
            #print('Area Under Curve (AUC): %1.3f' % auc)
            #eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            #print('Equal Error Rate (EER): %1.3f' % eer)
            
'''def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--imagePath', type=str, default='/home/gpuuser/Downloads/images/data/IMG_20171018_123324_350.jpg',
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--model', type=str,default='/home/gpuuser/vinayak/models/facenet/src/ckpt/20170512-110547', 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    #parser.add_argument('--lfw_pairs', type=str,
        #help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    #parser.add_argument('--lfw_file_ext', type=str,
        #help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    #parser.add_argument('--lfw_nrof_folds', type=int,
        #help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)'''

#if __name__ == '__main__':
    #main()
