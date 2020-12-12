# import tensorflow as tf
import tensorflow.compat.v1 as tf
# t = tf.random.uniform([5,30],-1,1)
# x = tf.Variable(t)

# print(t)
# print(x)
# print(tf.debugging.set_log_device_placement(True))

tf.compat.v1.disable_eager_execution()

# x = [[2.]]
# print(x)
# m = tf.matmul(x, x)
# print("hello, {}".format(m))

a = tf.constant([[1, 2],
                 [3, 4]])

# a[0,0] = 55
# print(a)
# b = tf.add(a, 1)
# print(b)

# w = tf.Variable([[1.0]])
# print(w)
# with tf.GradientTape() as tape:
#   loss = w * w

# print(loss)
# grad = tape.gradient(loss, w)
# print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)

# variable_a = tf.constant(3.0)
# variable_b = tf.constant(2.0)
# with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
#     tape.watch(variable_a)
#     y = variable_a ** 2  # Gradients will be available for `variable_a`.
#     z = variable_b ** 3  # No gradients will be available since `variable_b` is
#     # not being watched.
#     ya = tape.gradient(y,variable_a)
#     yb = tape.gradient(y,variable_b)
# print(ya)
# print(yb)
# x = tf.constant([1,2,3])
# y= tf.constant([11,22,33])
# dd = {x.ref():y}
# print(dd)


# print(tf.zeros(dtype=tf.float32, shape=[None, 10] ))
# t1 = 1
# t2 = 3
# print(tf.concat(1, [t1, t2]))

c_prev = tf.zeros((7, 25))
x = tf.placeholder(tf.float32, [None, 25])
x_hat = x - tf.sigmoid(c_prev)
print(tf.concat([x, x_hat], 1))