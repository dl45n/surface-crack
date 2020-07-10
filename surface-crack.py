import numpy as np
from tensorflow.python.framework import ops
import h5py
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import math
import tensorflow as tf

DATADIR = #"D:/Surface/Train"

CATEGORIES = ["Plain", "Cracked"]

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)  # create path to plain and pothole
    for img in os.listdir(path):  # iterate over each image per plain and pothole
        img_array = cv2.imread(os.path.join(path,img))  # convert to array
        plt.imshow(img_array)  # graph it
        plt.show()  # display!

        break  #to display just one picture
    break 


IMG_SIZE = 150
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array)
plt.show()


training_data = []
def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  # create path
        class_num = CATEGORIES.index(category)  # get the classificatio

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])            # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
create_training_data()
print(len(training_data))

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1]) # just to check

train_X_orig = []
Y_train = []
for features, label in training_data:
    train_X_orig.append(features)
    Y_train.append(label)

train_X_orig = np.array(train_X_orig).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
train_X_flatten = train_X_orig.reshape(train_X_orig.shape[0], -1).T
print ("train_X_flatten shape: " + str(train_X_flatten.shape))

X_train = train_X_flatten / 255

print(X_train.shape)


DATADIR2 = #"D:/Surface/Test"

CATEGORIES2 = ["Plain", "Cracked"]

for category in CATEGORIES2:
    path = os.path.join(DATADIR2,category)  # create path to plain and pothole
    for img in os.listdir(path):  # iterate over each image per plain and pothole
        img_array2 = cv2.imread(os.path.join(path,img))  # convert to array
        plt.imshow(img_array2)  # graph it
        plt.show()  # display!

        break  # to display just one picture
    break 


IMG_SIZE = 150
new_array2 = cv2.resize(img_array2, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array2)
plt.show()

test_data = []
def create_test_data():
    for category in CATEGORIES2:  

        path = os.path.join(DATADIR2,category)  # create path to plain and pothole
        class_num2 = CATEGORIES2.index(category)  # get the classification  (0 or a 1). 0=plain 1=pothole

        for img in tqdm(os.listdir(path)):  # iterate over each image per plain and pothole
            try:
                img_array2 = cv2.imread(os.path.join(path,img))  # convert to array
                new_array2 = cv2.resize(img_array2, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                test_data.append([new_array2, class_num2])            # add this to our test_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
create_test_data()

random.shuffle(test_data)

for sample in test_data[:10]:
    print(sample[1]) #just to check

test_X_orig = []
Y_test = []
for features, label in test_data:
    test_X_orig.append(features)
    Y_test.append(label)
test_X_orig = np.array(test_X_orig).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

test_X_flatten = test_X_orig.reshape(test_X_orig.shape[0], -1).T
print ("test_X_flatten shape: " + str(test_X_flatten.shape))

X_test = test_X_flatten / 255


def placeholder(n_x):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(1, None), name='Y') # add a second argument for multi-class classification and use it in place of 1 for 'Y' placeholder
    return X, Y


def init_param():
    W1 = tf.get_variable("W1", [20,67500], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [20,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [15,20], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [15,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1,15], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return parameters


def for_prop(X, parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)                           
    A1 = tf.nn.relu(Z1)                             
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                  
    A2 = tf.nn.relu(Z2) 
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    
    return Z3


def comp_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)) 
    #0.001*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)) #use if you want to ass L2 regularization
    
    return cost


def random_mini_batches(X, Y, mini_batch_size = 16, seed = 0):
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]      
    mini_batches = []
        
    # Step 1: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[:, k * mini_batch_size: (k+1)*mini_batch_size]
        mini_batch_Y = np.array([Y[k * mini_batch_size: (k+1)*mini_batch_size]])
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = np.array([Y[num_complete_minibatches * mini_batch_size:]])
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 200, minibatch_size = 16, print_cost = True):

    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                                                   
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = placeholder(n_x)

    # Initialize parameters
    parameters = init_param()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = for_prop(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = comp_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

    
        predicted = tf.nn.sigmoid(Z3)
        correct_pred = tf.equal(tf.round(predicted), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        
        '''# Calculate the correct predictions                                        # for multi-class classification 
        correct_prediction = tf.equal(A3, Y)

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: np.array([Y_train])}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: np.array([Y_test])}))'''
        
        print('Train Accuracy:', sess.run([accuracy, tf.round(predicted)], feed_dict={X: X_train, Y: np.array([Y_train])}))
        print('Test Accuracy:', sess.run([accuracy, tf.round(predicted)], feed_dict={X: X_test, Y: np.array([Y_test])}))
        
        return parameters


parameters = model(X_train, Y_train, X_test, Y_test)





