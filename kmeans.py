from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

# 忽略所有的GPUs，tf的随机森林不需要

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 导入 MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
full_data_x = mnist.train.images

# 变量
# 训练的步骤
num_steps = 50
# 每一个batch的样本大小
batch_size = 1024
# 聚类的数目
k = 25
# 0-9十个数字
num_classes = 10
# 每一个图像都是28*28像素

# 输入图像
X = tf.placeholder(tf.float32, shape=[None, num_features])
# 标签
Y = tf.placeholder(tf.float32, shape=[None, num_features])

# K-Means聚类的参数
kmeans = KMeans(inputs=X, num_clusters=k,
                distance_metric='cosine', use_mini_batch=True)
# 建立KMeans计算图
training_graph = kmeans.training_graph()
if len(training_graph) > 6:
