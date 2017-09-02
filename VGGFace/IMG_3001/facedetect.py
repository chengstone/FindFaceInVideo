#!/usr/bin/env python

import numpy as np
import cv2
import cv2.cv as cv
from matplotlib import pyplot as plt
#from video import create_capture
# from common import clock, draw_str
import clock
import subprocess
import os
import sys, getopt
import threading
import signal
import shutil
from argparse import ArgumentParser

from face_recognition import *
from lfw_test_deal import *

help_message = '''
USAGE: facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

DEBUG_FLAG = 1

VIDEO_FLAG = 0

MAIN_PATH = '/home/chengstone/Downloads/caffe/VGGFace/'
CAS_PATH = '/home/chengstone/Downloads/opencv-2.4.13.2/data/haarcascades/'
FRONTAL_FACE = 'haarcascade_frontalface_default.xml'

# IMAGE_FILE_PATH = '/home/chengshd/ML/caffe-master/examples/VGGFace/images/'

TargetPath = MAIN_PATH + 'targets.txt'
camPicture = '/home/chengshd/ML/caffe-master/examples/VGGFace/chengshd/IMG_3167.JPG'#MAIN_PATH + 'IMG_0468.JPG'

VIDEO_HOME = '/home/chengshd/ML/caffe-master/examples/VGGFace/chengshd/'
videoPath = VIDEO_HOME + 'IMG_3170.mp4'    #'IMG_3663.MOV'
#test_obj = None

targetsArr = []
dstArr = []
FoundFace = 0

FaceCropPath = '/home/chengstone/Downloads/SeetaFaceEngine/FaceAlignment/build/'
FA_TEST = './fa_test'
IMAGE_TXT = 'image.txt'

del_threshold = 0.15    #0.25
last_scaleFactor = 1.0655

cascade_fn = CAS_PATH + FRONTAL_FACE#args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
cascade = cv2.CascadeClassifier(cascade_fn)

g_vis = None
dst_rects = None
VGG_THRESHOLD = 0.4    #0.753

createFaceFlag = 0

# cascv_fn = CAS_PATH + 'haarcascade_frontalface_alt.xml'
# cascade_cv = cv2.CascadeClassifier('/home/chengshd/ML/opencv-2.4.13.2/data/haarcascades/haarcascade_frontalface_alt.xml')


def detect(img, cascade):
    # print 'detect IN:'
    rects = cascade.detectMultiScale(img, scaleFactor=1.06559, minNeighbors=4, minSize=(5, 5), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    # print 'draw_rects IN:'
    print rects
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def isOverlap(tmp_rect, rects):
    i = 0
    for x1, y1, x2, y2 in rects:
        if(tmp_rect[0] != x1 and tmp_rect[1] != y1 and tmp_rect[2] != x2 and tmp_rect[3] != y2):
            if(tmp_rect[2] > x1 and x2 > tmp_rect[0] and tmp_rect[3] > y1 and y2 > tmp_rect[1]):
                return i
        i = i + 1
    return -1

def computeRectJoinUnion(rect1, rect2):
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])

    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])

    AJoin = 0
    if(x2 > x1 and y2 > y1):
        AJoin = (x2 - x1) * (y2 - y1)

    A1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    A2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

    AUnion = A1 + A2 - AJoin

    if(AUnion > 0):
        return float(AJoin) / AUnion
    return 0

def procOverlap(rects):
    print 'procOverlap IN:'
    #print rects
    i = 0
    new_rects = []
    del_rects = []
    for x1, y1, x2, y2 in rects:
        tmp_rect = rects[i]
        bOverlap = isOverlap(tmp_rect, rects)
        if(bOverlap > -1):
            print tmp_rect
            print rects[bOverlap], bOverlap
            rectJountUnion = computeRectJoinUnion(tmp_rect, rects[bOverlap])
            print 'rectJountUnion = ', rectJountUnion
            if(rectJountUnion > del_threshold):
                A1 = (tmp_rect[2] - tmp_rect[0]) * (tmp_rect[3] - tmp_rect[1])
                A2 = (rects[bOverlap][2] - rects[bOverlap][0]) * (rects[bOverlap][3] - rects[bOverlap][1])
                if(A1 < A2):
                    new_rects.append(tmp_rect)
                    #np.delete(rects, bOverlap, 0)
                    del_rects.append(rects[bOverlap])
                    print 'A1 < A2'
                #else:
            else:
                new_rects.append(tmp_rect)
        else:
            new_rects.append(tmp_rect)
        #print 'i = ', i
        #print 'new_rects = ', new_rects
        #print new_rects[i], i
        i = i + 1
    
    del_idx = []
    for node in del_rects:
        j = 0
        for new_node in new_rects:
            #print new_node[0],new_node[1],new_node[2],new_node[3]
            if(node[0] == new_node[0] and node[1] == new_node[1] and node[2] == new_node[2] and node[3] == new_node[3]):
                del_idx.append(j)
            #if(node == new_node):
            #    print node,' is equel.'
            j = j + 1
    print del_idx

    for idx in del_idx:
        del new_rects[idx]

    #print new_rects
    print 'procOverlap done.'
    return np.array(new_rects)

def ProcImage(imagePath, modifyPath):
    print 'ProcImage IN:'
    #print help_message

    #cascade_fn = CAS_PATH + FRONTAL_FACE#args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    #cascade = cv2.CascadeClassifier(cascade_fn)

    #args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    #try: video_src = video_src[0]
    #except: video_src = 0
    #args = dict(args)
    # print CAS_PATH + FRONTAL_FACE
    # commands = CAS_PATH + FRONTAL_FACE
    # commands = subprocess.list2cmdline(commands)
    		
    #nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")

    		
    #nested = cv2.CascadeClassifier(nested_fn)

    #cam = create_capture(video_src, fallback='synth:bg=../cpp/lena.jpg:noise=0.05')

    #while True:
        #ret, img = cam.read()
    img = cv2.imread(imagePath)
    print imagePath
    sp = img.shape
    while(sp[0] > 768 + 512 or sp[1] > 1024 + 1024):    #sp[0]: height sp[1]: width sp[3]: tongdao
        img = cv2.pyrDown(img)
        sp = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # t = clock().utc()
    rects = detect(gray, cascade)
    vis = img.copy()
    print rects
    		#print rects.shape#, rects.dtype, rects
    new_rects = procOverlap(rects)
    print new_rects.shape
    
    if DEBUG_FLAG == 1:
    	draw_rects(vis, new_rects, (0, 255, 0))
    #draw_rects(vis, rects, (0, 255, 0))
    #for x1, y1, x2, y2 in new_rects:
    #    roi = gray[y1:y2, x1:x2]
    #    vis_roi = vis[y1:y2, x1:x2]
            #subrects = detect(roi.copy(), nested)
            #draw_rects(vis_roi, subrects, (255, 0, 0))
    # dt = clock.utc() - t

        # draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
    # text = 'time: %.1f ms' % (dt*1000)
    #position = (20, 20)
    #color = (0, 0, 255)
    # cv2.putText(vis, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 1)
    #print file_name.split('.')[-1]
    #cv2.imwrite(parent + 'modify/' + file_name.split('.')[0] + '_modify.' + file_name.split('.')[-1], vis)
    if(modifyPath[-1] != '/'):
        modifyPath = modifyPath + '/'
    cv2.imwrite(modifyPath + imagePath.split('/')[-1].split('.')[0] + '_modify.' + imagePath.split('/')[-1].split('.')[-1], vis)
    
    #for x1, y1, x2, y2 in new_rects:
        
    # cv2.imshow('facedetect', vis)
    # cv2.waitKey(5)
    #     if 0xFF & cv2.waitKey(5) == 27:
    #         break
    # cv2.destroyAllWindows()

def facedetect_test():
    #file_name = 'IMG_0468.JPG'
    #file_name = '2.jpg'
    #file_names = ['81.jpg','84.jpg','92.jpg','94.jpg','104.jpg','116.jpg','194.jpg','203.jpg','1.jpg','2.jpg','IMG_0468.JPG']
    #file_names = ['81.jpg','84.jpg','104.jpg','IMG_0468.JPG']
    file_names = ['qingyansi.jpg']    

    #for parent, dirnames, filenames in os.walk(IMAGE_FILE_PATH):
    #    #print parent,dirnames,filenames
    #	for file_name in filenames:
    #    	imagePath = parent + file_name
    for file_name in file_names:
    	imagePath = MAIN_PATH + file_name
        ProcImage(imagePath, MAIN_PATH)

def checkFile(filepath):
    path = ''
    for field in filepath.split('/'):
        if len(field) > 0:
            path = path + '/' + field
            #print path, os.path.exists(path)
            if field == filepath.split('/')[-1]:
                # print path, path.find('.')
                if path.find('.') != -1:
                    if os.path.exists(path) == False:
                        os.mknod(path)
                elif os.path.exists(path) == False:
                    # print path
                    os.mkdir(path)
            elif os.path.exists(path) == False:
                os.mkdir(path)

def returnFaceImg_Dst_2(imagePath):
    # print 'returnFaceImg_Dst IN:'

    global FoundFace
    global dst_rects
    global g_vis

    dstpath = MAIN_PATH + "output/"
    corp_size = 224

    imagetxt_file = open(FaceCropPath + IMAGE_TXT, 'w')
    imagetxt_file.writelines(imagePath + ' ' + dstpath + ' ' + str(corp_size) + '\n')
    imagetxt_file.close()

    checkFile(dstpath)

    if createFaceFlag == 1:
        os.chdir(FaceCropPath)
        os.system(FA_TEST)
    # cv2.waitKey(500)
    # print 'here'

    FoundFace = 0
    new_rects = []
    t_imgsArr = []
    for parent, dirnames, filenames in os.walk(dstpath):
        for f_file in filenames:
            # if "result" in filenames:
            if f_file.find("result") == -1:
                print parent, dirnames, f_file
                t_imgsArr.append(get_feature_new(parent + f_file))
                #qingyansi_crop_224_0_333_715_425_807.jpg
                X1 = int(f_file.split('.')[0].split('_')[-4])
                Y1 = int(f_file.split('.')[0].split('_')[-3])
                X2 = int(f_file.split('.')[0].split('_')[-2])
                Y2 = int(f_file.split('.')[0].split('_')[-1])
                new_rects.append([X1, Y1, X2, Y2])
                FoundFace = 1
            # else:
            #     img = cv2.imread(parent + f_file)
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(img)
                # plt.show()
                # cv2.namedWindow(f_file)
                # cv2.imshow(f_file, img)
                # cv2.waitKey(5)
    dst_rects = np.array(new_rects)

    img = cv2.imread(imagePath)
    print 'img.dtype = ', img.dtype
    sp = img.shape
    while (sp[0] > 768 + 512 or sp[1] > 1024 + 1024):  # sp[0]: height sp[1]: width sp[3]: tongdao
        img = cv2.pyrDown(img)
        sp = img.shape

    vis = img.copy()
    g_vis = vis

    if DEBUG_FLAG == 1:
        draw_rects(vis, dst_rects, (255, 0, 0))

    print np.array(t_imgsArr).shape
    return np.array(t_imgsArr)


def returnFaceImg_Dst(imagePath):
    # print 'returnFaceImg_Dst IN:'

    global FoundFace
    global dst_rects
    global g_vis
    img = cv2.imread(imagePath)
    print 'img.dtype = ', img.dtype
    #aa = cv.CloneMat(np.fromarrays(img))
    #print aa.dtype

    # cv.CreateMat(img)


    #print imagePath
    sp = img.shape
    while(sp[0] > 768 + 512 or sp[1] > 1024 + 1024):    #sp[0]: height sp[1]: width sp[3]: tongdao
        img = cv2.pyrDown(img)
        sp = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    rects = detect(gray, cascade)
    vis = img.copy()
    g_vis = vis

    print 'dst rects =',rects,vis.dtype,vis.shape
    if len(rects) != 0:
        print imagePath + ' Face found'
        FoundFace = 1
        new_rects = procOverlap(rects)
        print new_rects.shape

        # vis = vis[new_rects[0][0],new_rects[0][1],new_rects[0][2] - new_rects[0][0], new_rects[0][3] - new_rects[0][1]]
        # vis2 = vis[new_rects[0][0]:new_rects[0][2],new_rects[0][1]:new_rects[0][3],:]

        # plt.imshow(vis)
        # # plt.imshow(vis2)
        # plt.show()
        # cv2.namedWindow("dstImage")
        # cv2.imshow("dstImage", vis)
        # cv2.waitKey(5)

        t_dstsArr = []
        for rect in new_rects:
            print rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]
            # if rect[2]<rect[3]:
            #     cv.SetImageROI(vis,(rect[0]+10, rect[1]+10,rect[2]-100,rect[2]-100))
            # else:
            #     cv.SetImageROI(vis,(rect[0]+10, rect[1]+10,rect[3]-100,rect[3]-100))
            # #
            # dst=cv.CreateImage((224,224), 8, 3)
            # cv.Resize(vis,dst,cv.CV_INTER_LINEAR)
            #cv.SaveImage(imagePath.split('/')[-2] + 'temp.bmp',dst)
            print imagePath.split('/')[-2]
            vis2 = vis[rect[1]:rect[3], rect[0]:rect[2], :]
            # vis2 = vis[rect[2]:rect[0], rect[3]:rect[1], :]
            cv2.imwrite("./tmp.jpg", vis2)
            # aaa = cv2.imread("./tmp.jpg")
            # plt.imshow(aaa)
            # plt.show()
            t_dstsArr.append(get_feature_new("./tmp.jpg"))

        if DEBUG_FLAG == 1:
            draw_rects(vis, new_rects, (255, 0, 0))
            # plt.imshow(vis)
            # plt.show()

        dst_rects = new_rects
        print np.array(t_dstsArr).shape
        return np.array(t_dstsArr)

    else:
        FoundFace = 0
        print imagePath + ' Face not found'
        return []

# def detect_cv(img, cascade_cv):
#     rects = cv.HaarDetectObjects(img, cascade_cv, cv.CreateMemStorage(), 1.1, 2,cv.CV_HAAR_DO_CANNY_PRUNING, (255,255))#
#     if len(rects) == 0:
#         return []
#     result = []
#     #
#     for r in rects:
#         result.append((r[0][0], r[0][1], r[0][0]+r[0][2], r[0][1]+r[0][3]))
#     #
#     # if result[0][2]> 300 and result[0][3] > 300 and result[0][2]< 500 and result[0][3] < 500:
#     #     return result
#     # else:
#     #     return []
#     return result

# def draw_rects_cv(img, rects, color):
#     if rects:
#         for i in rects:
#             cv.Rectangle(img, (int(rects[0][0]), int(rects[0][1])),(int(rects[0][2]),int(rects[0][3])),cv.CV_RGB(0, 255, 0), 1, 8, 0)#

# def returnFaceImg_Dst___(imagePath):
#     img = cv.LoadImage(imagePath)
#
#     src = cv.CreateImage((img.width, img.height), 8, 3)
#     cv.Resize(img, src, cv.CV_INTER_LINEAR)
#
#     gray = cv.CreateImage((img.width, img.height), 8, 1)
#     cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
#     cv.EqualizeHist(gray, gray)
#     # rects = detect_cv(gray, cascade_cv)
#     # # face_rect = rects
#     #
#     # if DEBUG_FLAG == 1:
#     #     draw_rects_cv(src, rects, (0, 255, 0))
#
#     # cv.ShowImage('DeepFace Wang_jun_qian', src)

#/home/chengshd/ML/caffe-master/examples/att_faces/s40/2.jpg /home/chengshd/ML/caffe-master/examples/att_faces_new/s40 224
#fullpath dstpath cropSize
def returnFaceImg(imagePath, dstpath, corp_size):
    # print 'returnFaceImg IN:'
    imagetxt_file = open(FaceCropPath + IMAGE_TXT, 'w')
    imagetxt_file.writelines(imagePath + ' ' + dstpath + ' ' + str(corp_size) + '\n')
    imagetxt_file.close()

    checkFile(dstpath)

    if createFaceFlag == 1:
        os.chdir(FaceCropPath)
        os.system(FA_TEST)
    #cv2.waitKey(500)
    #print 'here'
    
    t_targetsArr = []
    for parent, dirnames, filenames in os.walk(dstpath):
        for f_file in filenames:
            #if "result" in filenames:
            if f_file.find("result") == -1:
                print parent, dirnames, f_file
                t_targetsArr.append(get_feature_new(parent + f_file))
            else:
                img = cv2.imread(parent + f_file)
                plt.subplot(1, 2, 1)
                b, g, r = cv2.split(img)
                img2 = cv2.merge([r, g, b])
                if VIDEO_FLAG != 1:
                    plt.imshow(img2)
                # plt.show()
                #cv2.namedWindow(f_file)
                #cv2.imshow(f_file, img)
                #cv2.waitKey(5)

    print np.array(t_targetsArr).shape
    return np.array(t_targetsArr)

def getTargetFace():
    # print 'getTargetFace IN:'
    global targetsArr
    if os.path.exists(TargetPath) == False:
        print TargetPath + ' File not found'
        exit(0)

    fileObj = open(TargetPath)
    fileObjDataList = fileObj.readlines(); 
    fileObj.close()

    #for line in fileObj:
    #    print line
    for index in range(len(fileObjDataList)):    
        line = fileObjDataList[index]
        print index, line[:-1]
        if os.path.exists(line[:-1]) == False:
            print TargetPath + ' File not found'
        else:
            targetsArr.append(returnFaceImg(line[:-1], MAIN_PATH + line[:-1].split('/')[-1].split('.')[0] + '/', 224))

def getDstFace_2():
    # print 'getDstFace IN:'
    global dstArr
    dstArr.append(returnFaceImg_Dst_2(camPicture))
    # print dstArr

def getDstFace():
    # print 'getDstFace IN:'
    global dstArr
    dstArr.append(returnFaceImg_Dst(camPicture))
    # print dstArr

def draw_single_rect(img, rect, color):
    # print 'draw_single_rect IN:'
    # for x1, y1, x2, y2 in rect:
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)

# def show_img():
#     while True:
#         # if g_vis:
#         plt.imshow(g_vis)
#         plt.show()

# thread_show_img = threading.Thread(target=show_img)

def prediction():
    print 'prediction IN:'
    global targetsArr
    global dstArr
    global dst_rects

    # t_dstArr = np.array(dstArr)
    # print dstArr
    # try:
    if 1 == 1:
        if FoundFace == 1:
        # if len(dstArr[0]) != 0:
            dstShape = dstArr[0].shape
            print 'dst shape: ', dstShape[0]

            targetShape = targetsArr[0].shape
            print 'target shape: ', targetShape[0]

            results = np.zeros([dstShape[0], targetShape[0]])
            print '===========predict results: ==========='
            for i in range(dstShape[0]):
                for j in range(targetShape[0]):
                    # print results[i][j]
                    results[i][j] = compare_pic(dstArr[0][i], targetsArr[0][j])
                    print results[i][j]

            # print dstArr[0][0].shape, dstArr[0][0]
            # result = compare_pic(feature_1, feature_2);
            for i in range(dstShape[0]):
                for j in range(targetShape[0]):
                    if results[i][j] >= VGG_THRESHOLD:
                        draw_single_rect(g_vis, dst_rects[i], (0, 255, 0))
                    if DEBUG_FLAG == 1:
                        if results[i][j] >= VGG_THRESHOLD:
                            pen = (0, 255, 0)
                        else:
                            pen = (255, 0, 0)
                        cv2.putText(g_vis, str(round(results[i][j], 2)), (dst_rects[i][0], dst_rects[i][1] - 7), cv2.FONT_HERSHEY_DUPLEX, 0.8, pen)
            cv2.imwrite(MAIN_PATH + "tmp.jpg", g_vis)
            plt.subplot(1, 2, 2)
            b, g, r = cv2.split(g_vis)
            img2 = cv2.merge([r, g, b])
            # plt.figure(num=1)
            plt.imshow(img2)
            if VIDEO_FLAG != 1:
                plt.show()
    # except:
    #     print 'except! FoundFace = ', FoundFace
    # print dst_rects , dst_rects.shape, dst_rects[1]
    # if result >= VGG_THRESHOLD:
    #     #  print 'Same Guy\n\n'
    #     # True_Positive += 1;
    #     results[index] = 1
    #     else:
    #     #  wrong
    #     # False_Positive += 1;
    #     results[index] = 0

def findPersionByImage_2():
    global targetsArr
    global dstArr
    getTargetFace()
    #print targetsArr[0].shape

    getDstFace_2()
    # print dstArr[0].shape

    prediction()
    print 'done'


def findPersionByImage():
    global targetsArr
    global dstArr
    getTargetFace()
    #print targetsArr[0].shape

    getDstFace()
    # print dstArr[0].shape

    prediction()
    print 'done'
    #fileObj.close()
    #facedetect_test()
    # thread_show_img.start()

# class InputTimeoutError(Exception):
#     pass
#
# def interrupted(signum, frame):
#     raise InputTimeoutError

def findPersionByVideo():
    global targetsArr
    global dstArr
    global camPicture
    getTargetFace()

    videoCapture = cv2.VideoCapture(videoPath)
    fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    framesCount = videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    # videoWriter = cv2.VideoWriter(MAIN_PATH + 'out/' + videoPath.split('/')[-1], cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, size)
    videoWriter = cv2.VideoWriter(MAIN_PATH + 'out/test.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'),
                                  fps, size)
    print 'videoWrite = ' + MAIN_PATH + 'out/test.avi'

    # signal.signal(signal.SIGALRM, interrupted)
    # signal.alarm(1)

    success, frame = videoCapture.read()

    print 'frame.shape = ', frame.shape
    tick = 0
    while success:    # and tick < 10
        tick = tick + 1
        print 'Current frame : ' + str(tick) + ' / ' + str(framesCount)

        dstArr = []
        # try:
        #     cmds = raw_input()
        # except InputTimeoutError:
        #     cmds = ""
        #
        # if cmds == 'over':
        #     break

        cv2.imwrite(MAIN_PATH + "frame_tmp.jpg", frame)
        camPicture = MAIN_PATH + "frame_tmp.jpg"
        getDstFace()
        # getDstFace_2()
        prediction()

        print 'g_vis.shape = ', g_vis.shape
        cv2.imshow("find persion by video", g_vis)
        # cv2.imshow("find persion by video", frame)
        cv2.waitKey(1000/int(fps))
        videoWriter.write(g_vis)
        # videoWriter.write(frame)
        success, frame = videoCapture.read()
    videoCapture.release()
    videoWriter.release()
    # signal.alarm(0)
    print 'findPersionByVideo done'

if __name__ == '__main__':
    global VIDEO_FLAG
    parser = ArgumentParser()  
      
    parser.add_argument('--content',  
            dest='content', help='image/video path',  
            metavar='CONTENT', required=True)  #
    #parser.add_argument('--step', type=int, default = 20,  
    #        dest='step', help='the video step you want use',  
    #        metavar='STEP')  
    #parser.add_argument('--folder', 
    #        dest='folder', help='the videos and pictures around here',  
    #        metavar='FOLDER')  
    #parser.add_argument('--threshold', type=float, default = 0.8,  
    #        dest='threshold', help='the videos and pictures threshold',  
    #        metavar='THRESHOLD')  
    
    options = parser.parse_args()  
      
    content = options.content # the image/video name you want to detect  

    if content.split('.')[-1] in ['jpg', 'JPG', 'jpeg', 'bmp', 'png']:
            print content
            camPicture = content
            findPersionByImage()
    elif content.split('.')[-1] in \
        ['mov', 'MOV', 'rm', 'RM', \
         'rmvb', 'RMVB', 'mp4', 'MP4', \
         'avi', 'AVI', 'wmv', 'WMV', \
         '3gp', '3GP', 'mpeg', 'MPEG', \
         'mkv', 'MKV']:
        print content
        VIDEO_FLAG = 1
        videoPath = content
        findPersionByVideo()
    #findPersionByImage()
    # findPersionByVideo()
    # findPersionByImage_2()

