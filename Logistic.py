import tensorflow as tf
import numpy as np

# Normalize x values
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)

# data is the data set with y in the last column
# num_x is the number of the variables
def logRegression(data,num_x,learn_rate=0.01):

    y_data = np.array([num[num_x] for num in data])
    x_data = np.array([num[0:num_x] for num in data])
        
    sample_train = normalize_cols(sample_train)
    sample_test = normalize_cols(sample_test)

    train_indices = np.random.choice(len(x_data), int(0.7*len(x_data)), replace=False)
    sample_train = x_data[train_indices]
    label_train = y_data[train_indices]
    test_indices = np.array(list(set(range(len(x_data))) - set(train_indices)))
    sample_test = x_data[test_indices]
    label_test = y_data[test_indices]

    # Build a model
    A = tf.Variable(np.random.rand(8, 1).astype(np.float32))
    b = tf.Variable(np.random.rand(1, 1).astype(np.float32))
    x_ = tf.placeholder(tf.float32, [None, 8])
    y_ = tf.placeholder(tf.float32, [None, 1])

 

    kernel_model=tf.add(tf.matmul(x_, A), b)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=kernel_model, labels=y_))

    train = tf.train.GradientDescentOptimizer(learn_rate)
    train2 = train.minimize(loss)

    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    prediction = tf.round(tf.sigmoid(kernel_model))
    predictions_correct = tf.cast(tf.equal(prediction, y_), tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)

    batch_size =int(len(label_train)*0.3)

    for i in range(10000):
        rand_index = np.random.choice(len(label_train), size=batch_size)
        rand_x = sample_train[rand_index]
        rand_y = np.transpose([label_train[rand_index]])
        sess.run(train2, feed_dict={x_: rand_x, y_: rand_y})
    acc_data = sess.run(accuracy, feed_dict={x_: sample_test, y_: np.transpose([label_test])})
    print("Accuracy: ", acc_data)