import tensorflow as tf

# TensorFlow是编程系统，使用图来表示编程任务，图中的节点被称为op，下面的constant()和matmul()也被称为op

matrix1 = tf.constant([[3., 3.]])

matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrix1, matrix2)

# 构造完图后，必须启动图

sess = tf.Session()

# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数.
# 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回
# 矩阵乘法 op 的输出.
#
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
#
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
#
# 返回值 'result' 是一个 numpy `ndarray` 对象.

result = sess.run(product)

print(result)
sess.close()

# TensorFlow 将图形定义转换成分布式执行的操作, 以充分利用可用的计算资源(如 CPU 或 GPU).
# 一般你不需要显式指定使用 CPU 还是 GPU, TensorFlow 能自动检测. 如果检测到 GPU, TensorFlow 会尽可能地利用找到的第一个 GPU 来执行操作.

# with tf.Session() as sess:
#   with tf.device("/gpu:1"):
#     matrix1 = tf.constant([[3., 3.]])
#     matrix2 = tf.constant([[2.],[2.]])
#     product = tf.matmul(matrix1, matrix2)
#     ...
