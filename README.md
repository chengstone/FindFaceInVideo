# FindFaceInVideo
This is a Deep Learning practice Demo which can find person in the video by human face.人脸识别的小demo，通过待识别的人脸图像在视频影像中找人。


__Author__ chengstone

__WeChat__ 15041746064

__e-Mail__ 69558140@163.com


本Demo完成于2017年3月末4月初。

程序使用了VGG的模型参数，这里没有上传，需要你另外下载，共两个文件（VGG_FACE_deploy.prototxt，VGG_FACE.caffemodel）放到VGGFace目录下。
我当时的下载地址如下：
http://101.96.8.164/www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz

下面是在图片中找人的output：
![image](https://github.com/chengstone/FindFaceInVideo/raw/master/VGGFace/IMG_0468_modify_2.jpg)
![image](https://github.com/chengstone/FindFaceInVideo/raw/master/VGGFace/IMG_3030_modify_2.jpg)
![image](https://github.com/chengstone/FindFaceInVideo/raw/master/VGGFace/IMG_3664_modify_2.jpg)
![image](https://github.com/chengstone/FindFaceInVideo/raw/master/VGGFace/IMG_3747.JPG)
![image](https://github.com/chengstone/FindFaceInVideo/raw/master/VGGFace/IMG_3748.JPG)
![image](https://github.com/chengstone/FindFaceInVideo/raw/master/VGGFace/IMG_3751.JPG)
![image](https://github.com/chengstone/FindFaceInVideo/raw/master/VGGFace/IMG_3746.JPG)

视频中找人的output在文件夹out中，有4个avi文件。

# 目录结构：

facedetect.py:主程序，注意在使用时要将里面的路径改写成你本地的路径。

face_recognition.py:提供两个人脸特征比较的功能。
这个程序来源于https://github.com/HolmesShuan/DeepID-I-Reimplement，我做了修改，感兴趣的同学可以BeyondCompare差分一下。

lfw_test_deal.py:来源于https://github.com/hqli/face_recognition。实际上好像并没有用到，可能是我当时做测试时用过，有点记不清了。

out文件夹:用来存放视频找人的输出结果。

tmp.jpg:是在图片中根据提供的待找人的人脸进行找人时的输出。

以图片文件名命名的文件夹:比如IMG_3588，qingyansi等文件夹是对图片进行人脸识别测试时的输出，只关注是否找到人脸，而不进行是否是我们要找的人的判断。

chengshd文件夹:用来存放我个人的测试用例，包含图片和视频。

targets.txt:待查找人的图片路径列表。

其他文件:基本上都是我测试时的输出。

# 程序的使用:

本程序还要用到另外一个程序，请一起下载https://github.com/chengstone/SeetaFaceEngine

本Demo分为两个工程：SeetaFaceEngine和VGGFace。
其中SeetaFaceEngine来源于https://github.com/seetaface/SeetaFaceEngine，我们只使用其中的FaceDetection和FaceAlignment，
主要用来做待查找人图片中的人脸检测和定位。其中FaceAlignment/src/test/face_alignment_test.cpp我做了修改，使程序能够以命令行的方式支持多样化的人脸识别。
VGGFace用来做人脸的比较，这里也使用了opencv的人脸检测，主要用于在目标图片中的多人脸的检测。

# 运行准备：

## 一、环境依赖，请提前安装好：

opencv2
caffe
Python2.7

## 二、build顺序：

1.先编译SeetaFaceEngine下面的FaceDetection，编译方法可以参见里面的readme，大致命令是：

mkdir build

cd build

cmake ..

make

2.然后将编译好的文件（以我的电脑编译好后为例是facedet_test，libseeta_facedet_lib.dylib这两个文件），
FaceDetection的.h文件face_detection.h，seeta_fa_v1.1.bin和seeta_fd_frontal_v1.0.bin几个文件拷贝到FaceAlignment的build目录下，
然后编译，编译命令跟build FaceDetection一样。

mkdir build

cd build

cmake ..

make

编译好后的文件名（以我的电脑编译好后为例）：fa_test

3.主要使用的就是FaceAlignment。
这个程序的命令行格式是：

第一种: fa_test 源图片全路径 目标保存文件夹路径 [图片大小变幻的像素数]

其中[图片大小变幻的像素数]可以省略，省略的话默认是不缩放图片的大小。

举例：fa_test /path/to/123.jpg /path/to/folder 32

将图片123.jpg识别出的人脸图片输出到/path/to/folder文件夹中，缩放后的尺寸是32 * 32的大小。

第二种：fa_test 保存源图片全路径列表的文件路径

其中[存源图片全路径列表的文件路径]是一个文本文件，里面是若干行第一种命令格式的字符串，用于批量识别图片。

举例：fa_test /path/to/image.txt

其中image.txt里面的内容类似如下内容，可以是一行也可以是多行：

/Users/chengstone/Downloads/ML/att_faces/s40/9.jpg /Users/chengstone/Downloads/ML/att_faces_new/s40 64
/Users/chengstone/Downloads/ML/att_faces/s40/8.jpg /Users/chengstone/Downloads/ML/att_faces_new/s40 64
/Users/chengstone/Downloads/ML/att_faces/s40/7.jpg /Users/chengstone/Downloads/ML/att_faces_new/s40 64
/Users/chengstone/Downloads/ML/att_faces/s40/6.jpg /Users/chengstone/Downloads/ML/att_faces_new/s40 64

这个程序有时并不能准确识别人脸。。。

第三种：fa_test

如果不传入参数，默认在程序当前路径下读取image.txt文件，文件内的格式参见上面的说明。这个是本Demo的使用方式。

## 三、开始视频/图片找人：

主要代码在VGGFace中的facedetect.py中。
这里使用了caffe的VGG模型用来做人脸的特征提取和特征比较。
由于我自己训练的模型准确率太低（采用的DeepID模型，可能是我没有实现好，而且我的训练集也不太够），只好使用VGG公开的模型参数。

facedetect.py的使用方法：

1.先修改VGGFace/targets.txt文件，里面是图片路径的列表，每一个图片必须只有一个人。
如果只有一个图片意味着只找这一个人，就是我说的待查找人。多个的话就是在视频和图片中找多个人。

2.facedetect.py的命令行参数：

python ./facedetect.py —-content 图片和视频的全路径

其中[图片和视频的全路径]是我们要查找的目标。

举例1：python ./facedetect.py —-content /path/to/123.jpg

举例2：python ./facedetect.py —-content /path/to/456.mov

# 识别过程：

有个概念我解释一下，targets.txt文件中列出的是要找的人，可以是一个也可以是多个人。
比如是我的照片，意味着要在视频或图片中找出我，查找依据就是根据targets.txt文件中我的照片查找。

例如：/home/chengshd/ML/caffe-master/examples/VGGFace/chengshd/IMG_3588.JPG

然后facedetect.py会先调用fa_test，将targets.txt文件中指定人物的照片做人脸识别，就是把照片中的人脸先找出来。
facedetect.py会把调用fa_test所用的参数写进fa_test同级目录下的image.txt文件中。

例如：/home/chengshd/ML/caffe-master/examples/VGGFace/chengshd/IMG_3588.JPG /home/chengshd/ML/caffe-master/examples/VGGFace/chengshd/IMG_3588 224

结果就是把识别出的人脸图片放在以targets.txt文件中待查找人的图片文件名作为目录的目录下。

例如：VGGFace/IMG_3588下：IMG_3588_crop_224_0_145_460_652_967.JPG文件就是fa_test识别出的人脸。

接下来使用刚刚fa_test识别出的人脸，根据调用facedetect.py时传入的参数，在图片或者视频中找人。

这里facedetect.py使用opencv的cv2.CascadeClassifier做人脸识别。

你一定会问我为什么使用两种不同的方式识别人脸，我经过多次测试发现cv2.CascadeClassifier和SeetaFace的人脸识别效果都不是很完美，当前这个组合还可以一用。

当然，一定有更好的方式替代当前这个人脸识别的方案。比如采用卷积神经网络的方式进行人脸识别，这个我没有继续深入去做。

在给定图片或视频中使用cv2.CascadeClassifier找出人脸，然后根据fa_test识别出的人脸图片做特征的比较。

这里使用的是VGG的模型做的人脸特征值比较。按理说相似度越高越好，经过我的测试，我将相似度阈值（VGG_THRESHOLD）设置成了0.4。

就是说相似度大于40%，我就认为是同一个人。这个值设的确实有点小了，奈何我用自己的照片去测试，很少出现相似度大于85%以上的时候。。。。

但是一旦两个人不相似，相似度很低，大都小于1%，所以0.4这个值还是堪用的。

更多内容请参考代码，Enjoy！
