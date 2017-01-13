import tensorflow as tf

# Build computation graph
x1 = tf.constant(5)
x2 = tf.constant(6)
y = tf.mul(x1, x2)
print y

# Build session and run. Let's secc 3 ways to do this.
#1. create session and close manually
sess = tf.Session()
print sess.run(y)
sess.close()

# 2. second way is to run session using "with". No need to close
with tf.Session() as sess:
	print(sess.run(y))

# 3. Save output and access it outside
with tf.Session() as sess:
	output = sess.run(y)

print output
