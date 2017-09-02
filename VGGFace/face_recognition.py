import numpy as np
import os
import skimage
import sys
import caffe
import sklearn.metrics.pairwise as pw
import math

#  sys.path.insert(0, '/Downloads/caffe-master/python');
#  load Caffe model 
caffe_root = '/home/chengstone/Downloads/caffe'
sys.path.insert(0, caffe_root+'python')
os.chdir(caffe_root)
caffe.set_mode_cpu()
#solver = caffe.SGDSolver('/home/chengshd/ML/caffe-master/examples/att_faces/attfaces_solver.prototxt')

#caffe.set_mode_gpu();

global net;
net = caffe.Classifier('/home/chengstone/Downloads/caffe/VGGFace/VGG_FACE_deploy.prototxt', '/home/chengstone/Downloads/caffe/VGGFace/VGG_FACE.caffemodel');

def compare_pic(feature1, feature2):
    predicts = pw.cosine_similarity(feature1, feature2);
    return predicts;

def get_feature_new(path):
    global net;
    X = read_image_new(path);
    # test_num = np.shape(X)[0];
    # print test_num;
    out = net.forward_all(data = X);
    #print out
    feature = np.float64(out['fc7']);
    feature = np.reshape(feature, (1, 4096));
    return feature;

def read_image_other(filelist):

    averageImg = [129.1863,104.7624,93.5940]
    X=np.empty((1,3,224,224))
    word=filelist.split('\n')
    filename=word[0]
    im1=skimage.io.imread(filename,as_grey=False)
    #
    image =skimage.transform.resize(im1,(224, 224))*255
    X[0,0,:,:]=image[:,:,0]-averageImg[0]
    X[0,1,:,:]=image[:,:,1]-averageImg[1]
    X[0,2,:,:]=image[:,:,2]-averageImg[2]
    return X

def read_image_new(filepath):
    averageImg = [129.1863, 104.7624, 93.5940];
    X = np.empty((1,3,224,224));
    filename = filepath.split('\n');
    filename = filename[0];
    im = skimage.io.imread(filename, as_grey=False);
    image = skimage.transform.resize(im, (224, 224))*255;
    #mean_blob.shape = (-1, 1); 
    #mean = np.sum(mean_blob) / len(mean_blob);
    X[0,0,:,:] = image[:,:,0] - averageImg[0];
    X[0,1,:,:] = image[:,:,1] - averageImg[1];
    X[0,2,:,:] = image[:,:,2] - averageImg[2];
    return X;

def get_feature(path, mean_blob):
    global net;
    X = read_image(path, mean_blob);
    # test_num = np.shape(X)[0];
    # print test_num;
    out = net.forward_all(data = X);
    feature = np.float64(out['fc256']);
    feature = np.reshape(feature, (1, 256));
    return feature;

def read_image(filepath, mean_blob):
    # averageImg = [129.1863, 104.7624, 93.5940];
    X = np.empty((1,3,128,128));
    filename = filepath.split('\n');
    filename = filename[0];
    im = skimage.io.imread(filename, as_grey=False);
    image = skimage.transform.resize(im, (128, 128))*255;
    mean_blob.shape = (-1, 1); 
    mean = np.sum(mean_blob) / len(mean_blob);
    X[0,0,:,:] = image[:,:] - mean;
    X[0,1,:,:] = image[:,:] - mean;
    X[0,2,:,:] = image[:,:] - mean;
    return X;

#
# __Author__ chengstone
# __WeChat__ 15041746064
# __e-Mail__ 69558140@163.com
# 
#'max accuracy: 0.894666666667', 
#'max threshold: 0.769', 
#'Max Precision: 0.919472913616 599', 
#'Max Recall: 1.0 0', 
#'Final F1Score: 0.224185221039', 
#'Final Precision: 0.868980612883', 
#'Final Recalls: 0.926333333333', 
#'Best Accuracy: 0.893333333333', 
#'Best Thershold: 0.753'

def face_recog_test():
    thershold = 0.85;
    TEST_SUM = 15;
    DATA_BASE = "/home/chengshd/ML/caffe-master/examples/VGGFace/";
    MEAN_FILE = 'mean.binaryproto';
    POSITIVE_TEST_FILE = "positive_pairs_path.txt";
    NEGATIVE_TEST_FILE = "negative_pairs_path.txt";
    
    #mean_blob = caffe.proto.caffe_pb2.BlobProto();
    #mean_blob.ParseFromString(open(MEAN_FILE, 'rb').read());
    #mean_npy = caffe.io.blobproto_to_array(mean_blob);
    
    #print mean_npy.shape
    thresholds = np.zeros(len(np.arange(0.4,1,0.05)))
    Accuracys = np.zeros(len(np.arange(0.4,1,0.05)))
    Precisions = np.zeros(len(np.arange(0.4,1,0.05)))
    #AccuracyReals = np.zeros(len(np.arange(0.4,1,0.05)))
    Recalls = np.zeros(len(np.arange(0.4,1,0.05)))
    F1Score = np.zeros(len(np.arange(0.4,1,0.05)))
    tick = 0
    for thershold in np.arange(0.4, 1, 0.05):
        True_Positive = 0;
        True_Negative = 0;
    	False_Positive = 0;
    	False_Negative = 0;
	print "==============================================="
        # Positive Test
        f_positive = open(DATA_BASE + POSITIVE_TEST_FILE, "r");
        PositiveDataList = f_positive.readlines(); 
        f_positive.close( );
        #f_negative = open(NEGATIVE_TEST_FILE, "r");
        #NegativeDataList = f_negative.readlines(); 
        #f_negative.close( );
        labels = np.zeros(len(PositiveDataList))
        results = np.zeros(len(PositiveDataList))
	thresholds[tick] = thershold
        for index in range(len(PositiveDataList)):
            filepath_1 = PositiveDataList[index].split(' ')[0];
            filepath_2 = PositiveDataList[index].split(' ')[1];
            labels[index] = PositiveDataList[index].split(' ')[2][:-2];
            feature_1 = get_feature_new(DATA_BASE + filepath_1);
            feature_2 = get_feature_new(DATA_BASE + filepath_2);
            result = compare_pic(feature_1, feature_2);
            #print "Two pictures similarity is:%f\n\n"%(result)
            print "%s and %s Two pictures similarity is : %f\n\n"%(filepath_1,filepath_2,result)
            #print "thershold: " + str(thershold);
            if result>=thershold:
                print 'Same person!!!!\n\n'
            else:
                print 'Different person!!!!\n\n'

            if result >= thershold:
                #  print 'Same Guy\n\n'
                #True_Positive += 1;
		results[index] = 1
            else:
                #  wrong
                #False_Positive += 1;
		results[index] = 0
                
	    if labels[index] == 1:
		if results[index] == 1:
			True_Positive += 1;
		else:
			False_Negative += 1;
	    else:
		if results[index] == 1:
			False_Positive += 1;
		else:
			True_Negative += 1;
        #for index in range(len(NegativeDataList)):
        #    filepath_1 = NegativeDataList[index].split(' ')[0];
        #    filepath_2 = NegativeDataList[index].split(' ')[1][:-2];
        #    feature_1 = get_feature(DATA_BASE + filepath_1, mean_npy);
        #    feature_2 = get_feature(DATA_BASE + filepath_2, mean_npy);
        #    result = compare_pic(feature_1, feature_2);
        #    if result >= thershold:
        #        #  print 'Wrong Guy\n\n'
        #        #  wrong
        #        False_Negative += 1;
        #    else:
        #        #  correct
        #        True_Negative += 1; 

	if True_Positive + False_Positive == 0:
		Precisions[tick] = 0
	else:
		Precisions[tick] = float(True_Positive) / (True_Positive + False_Positive)
	#AccuracyReals[tick] = float(True_Positive + True_Negative) / (True_Positive + False_Positive + False_Negative + True_Negative)
	if True_Positive + False_Negative == 0:
		Recalls[tick] = 0
	else:
		Recalls[tick] = float(True_Positive) / (True_Positive + False_Negative)	

	if Precisions[tick] + Recalls[tick] == 0:
		F1Score[tick] = 0
	else:
		F1Score[tick] = (Precisions[tick] * Recalls[tick]) / (2 * (Precisions[tick] + Recalls[tick]))

	acc = float(np.sum((labels == results))) / len(PositiveDataList)
	print 'labels = ',labels
	print 'results = ',results
	Accuracys[tick] = acc
	tick = tick + 1
	print "Accuracy: " + str(float(acc));
        
        print "thershold: " + str(thershold);
        #print "Accuracy: " + str(float(True_Positive + True_Negative)/TEST_SUM) + " %";
        #print "True_Positive: " + str(float(True_Positive)/TEST_SUM) + " %";
        #print "True_Negative: " + str(float(True_Negative)/TEST_SUM) + " %";
        #print "False_Positive: " + str(float(False_Positive)/TEST_SUM) + " %";
        #print "False_Negative: " + str(float(False_Negative)/TEST_SUM) + " %";
    print 'Accuracys: ', Accuracys
    print 'Thresholds: ', thresholds
    print 'Precisions: ', Precisions
    print 'Recalls: ', Recalls
    print 'F1Score: ', F1Score
    #print 'AccuracyReals: ',AccuracyReals

    print 'Max Precision: ', np.max(Precisions), np.where(Precisions == np.max(Precisions))[0][0]
    print 'Max Recall: ', np.max(Recalls), np.where(Recalls == np.max(Recalls))[0][0]

    print "Final Accuracy: ", np.max(Accuracys)
    re = np.where(Accuracys == np.max(Accuracys))
    print 'Final Thershold: ', thresholds[re[0][0]]
    print 'Final F1Score: ', np.max(F1Score)
    re = np.where(F1Score == np.max(F1Score))
    print 'Final Precision: ',Precisions[re[0][0]]
    print 'Final Recalls: ',Recalls[re[0][0]]
    print 'Best Accuracy: ',Accuracys[re[0][0]]
    print 'Best Thershold: ',thresholds[re[0][0]]

if __name__ == '__main__':
    face_recog_test()
