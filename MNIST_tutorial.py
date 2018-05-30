import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# None表示此张量的第一个维度可以是任何长度的
x = tf.placeholder("float", [None, 784])
# 一个Variable代表一个可修改的张量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])
# 用 tf.log 计算 y 的每个元素的对数。
# 我们把 y_ 的每一个元素和 tf.log(y_) 的对应元素相乘。最后，用 tf.reduce_sum 计算张量的所有元素的总和。
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 添加一个操作来初始化创建的变量
init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

# 训练模型

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# tf.argmax能给出某个 tensor 对象在某一维上的其数据最大值所在的索引值
# 对模型进行评估
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={
      x: mnist.test.images, y_: mnist.test.labels}))
