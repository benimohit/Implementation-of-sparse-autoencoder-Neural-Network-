#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:57:08 2019

@author: mohitbeniwal
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import fmin_l_bfgs_b

def normalizeData(patches):
    # Squash data to [0.1, 0.9] since we use sigmoid as the activation
    # function in the output layer
    
    # Remove DC (mean of images). 
    patches = patches -np.mean(patches)
    
    # Truncate to +/-3 standard deviations and scale to -1 to 1
    pstd = 3 * np.std(patches)
    patches = np.maximum(np.minimum(patches, pstd), -pstd) / pstd;
    
    # Rescale from [-1,1] to [0.1,0.9]
    patches = (patches + 1) * 0.4 + 0.1;
    return patches

    
def sampleIMAGES():
    images=loadmat('IMAGES')
    images=images['IMAGES']# load images from disk 
    # sampleIMAGES
    # Returns 10000 patches for training
    patchsize = 8  # we'll use 8x8 patches 
    numpatches = 10000
    # Initialize patches with zeros.  Your code will fill in this matrix--one
    # column per patch, 10000 columns. 
    patches = np.zeros((patchsize*patchsize, numpatches));
    ## ---------- YOUR CODE HERE --------------------------------------
    for i in range(numpatches):
        image_index=np.random.randint(10)
        patch_pos=np.random.randint(504)#as the maximum start position for any patch can exceed(512-8)=504
        patch=images[patch_pos:patch_pos+patchsize,patch_pos:patch_pos+patchsize,image_index]
        patches[:,i]=patch.reshape(patchsize*patchsize)
    #  Instructions: Fill in the variable called "patches" using data 
    #  from IMAGES.  
    #  
    #  IMAGES is a 3D array containing 10 images
    #  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
    #  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
    #  it. (The contrast on these images look a bit off because they have
    #  been preprocessed using using "whitening."  See the lecture notes for
    #  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
    #  patch corresponding to the pixels in the block (21,21) to (30,30) of
    #  Image 1

    
    ## ---------------------------------------------------------------
    # For the autoencoder to work well we need to normalize the data
    # Specifically, since the output of the network is bounded between [0,1]
    # (due to the sigmoid activation function), we have to make sure 
    # the range of pixel values is also bounded between [0,1]
    patches = normalizeData(patches);
    return patches

def display_network(A,file_name):
    # This function visualizes filters in matrix A. Each column of A is a
    # filter. We will reshape each column into a square image and visualizes
    # on each cell of the visualization panel. 
    # All other parameters are optional, usually you do not need to worry
    # about it.
    # opt_normalize: whether we need to normalize the filter so that all of
    # them can have similar contrast. Default value is true.
    # opt_graycolor: whether we use gray as the heat map. Default is true.
    # cols: how many columns are there in the display. Default value is the
    # squareroot of the number of columns in A.
    # opt_colmajor: you can switch convention to row major for A. In that
    # case, each row of A is a filter. Default value is false.
    # rescale
        # rescale
    A = A - np.mean(A)

    # compute rows, cols
    L, M = A.shape
    sz = int(np.sqrt(L))
    buf = 1

    rows = cols = int(np.sqrt(M))
    while rows*cols < M: 
        rows+=1

    # initialize the picture matrix
    array = np.ones((rows*(sz+buf) + buf, cols*(sz+buf) + buf))

    # fill up the matrix with image values
    row_cnt = col_cnt = 0
    for i in range(M):
        clim = np.max(abs(A[:,i])) # for normalizing the contrast
        x, y = row_cnt*(sz+buf) + buf, col_cnt*(sz+buf) + buf
        array[x : x+sz, y : y+sz] = A[:,i].reshape((sz,sz)) / clim
        col_cnt += 1
        if col_cnt >= cols:
            row_cnt += 1
            col_cnt = 0
            
    fig = plt.figure()
    plt.imshow(array, cmap='gray', interpolation='nearest')
    fig.savefig(file_name)
    plt.show()

def initializeParameters(hiddenSize, visibleSize):
    ## Initialize parameters randomly based on layer sizes.
    r  = np.sqrt(6) / np.sqrt(hiddenSize+visibleSize+1);   # we'll choose weights uniformly from the interval [-r, r]
    W1 = np.random.rand(hiddenSize, visibleSize) * 2 * r - r;
    W2 = np.random.rand(visibleSize, hiddenSize) * 2 * r - r;
    
    b1 = np.zeros((hiddenSize, 1));
    b2 = np.zeros((visibleSize, 1));
    
    # Convert weights and bias gradients to the vector form.
    # This step will "unroll" (flatten and concatenate together) all 
    # your parameters into a vector, which can then be used with minFunc. 
    theta = np.hstack((W1.ravel() , W2.ravel() , b1.ravel() , b2.ravel()))
    return theta

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda_, sparsityParam, beta, data):
    # visibleSize: the number of input units (probably 64) 
    # hiddenSize: the number of hidden units (probably 25) 
    # lambda_: weight decay parameter
    # sparsityParam: The desired average activation for the hidden units (denoted in the lecture
    #                           notes by the greek alphabet rho_cap, which looks like a lower-case "p").
    # beta: weight of sparsity penalty term
    # data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
      
    # The input theta is a vector (because minFunc expects the parameters to be a vector). 
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
    # follows the notation convention of the lecture notes. 
    
    W1 = theta[:hiddenSize*visibleSize].reshape(hiddenSize, visibleSize);
    W2 = theta[hiddenSize*visibleSize:2*hiddenSize*visibleSize].reshape(visibleSize, hiddenSize);
    b1 = theta[2*hiddenSize*visibleSize:2*hiddenSize*visibleSize+hiddenSize].reshape(hiddenSize,1);
    b2 = theta[2*hiddenSize*visibleSize+hiddenSize:].reshape(visibleSize,1);
    
    # Cost and gradient variables (your code needs to compute these values). 
    # Here, we initialize them to zeros. 
    cost = 0;
    W1grad = np.zeros(W1.shape); 
    W2grad = np.zeros(W2.shape);
    b1grad = np.zeros(b1.shape); 
    b2grad = np.zeros(b2.shape);
    
    m = data.shape[1]
    # forward prop
    input_laper=data
    input_layer_out=sigmoid(np.dot(W1,data)+b1)
    hidden_layer_out=sigmoid(np.dot(W2,input_layer_out)+b2)

    error = input_laper - hidden_layer_out

    # calculate rho_cap.
    rho_cap = 1 / m * np.sum(input_layer_out,1).reshape(-1,1)
    
     # compute cost
    mean_squared_error = 1 / m * np.sum(error**2)
    regularization_part = lambda_ / 2 * sum([np.sum(W1**2), np.sum(W2**2)])
    kl_divergence = (sparsityParam * np.log(sparsityParam / rho_cap) + (1 - sparsityParam)
    * np.log((1 - sparsityParam) / (1 - rho_cap)))
    cost = 0.5 * mean_squared_error + regularization_part + beta * np.sum(kl_divergence)

    # backprop with rho_cap
    delta3 = -(input_laper - hidden_layer_out) * hidden_layer_out * (1 - hidden_layer_out)
    delta2 = ((np.dot(W2.T,delta3) + beta * (-sparsityParam / rho_cap + (1 - sparsityParam)
    / (1 - rho_cap))) * input_layer_out * (1 - input_layer_out))

    W2grad = np.dot(delta3,input_layer_out.T)
    W1grad = np.dot(delta2,input_laper.T)
    b2grad = np.sum(delta3,1).reshape(-1,1)
    b1grad = np.sum(delta2,1).reshape(-1,1)

    W2grad = 1/m * W2grad + lambda_ * W2
    W1grad = 1/m * W1grad + lambda_ * W1
    b2grad = 1/m * b2grad
    b1grad = 1/m * b1grad
    # roll up cost and gradients to a vector format (suitable for minFunc)
    grad = np.hstack([W1grad.ravel(), W2grad.ravel(), b1grad.ravel(), b2grad.ravel()])

    return cost, grad

def simpleQuadraticFunction(x):
    value = x[0]**2 + 3*x[0]*x[1]
    grad = np.zeros(2)
    grad[0] = 2*x[0] + 3*x[1]
    grad[1] = 3*x[0]
    return value, grad

def computeNumericalGradient(j, theta):
    # numgrad = computeNumericalGradient(J, theta)
    # theta: a vector of parameters
    # J: a function that outputs a real-number. Calling y = J(theta) will return the
    # function value at theta. 
      
    # Initialize numgrad with zeros
    n=theta.size
    numgrad = np.zeros(n);
    e = 1e-4
    theta_pos = theta + np.eye(n) * e
    theta_neg = theta - np.eye(n) * e 
    for i, x in enumerate(theta):
        numgrad[i] = (j(theta_pos[i,:])[0] - j(theta_neg[i,:])[0]
                       ) / (2.0 * e)
        
    return numgrad
    # Instructions: 
    # Implement numerical gradient checking, and return the result in numgrad.  
    # (See Section 2.3 of the lecture notes.)
    # You should write code so that numgrad(i) is (the numerical approximation to) the 
    # partial derivative of J with respect to the i-th input argument, evaluated at theta.  
    # I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
    # respect to theta(i).
    #                
    # Hint: You will probably want to compute the elements of numgrad one at a time. 
    
def checkNumericalGradient():
    # This code can be used to check your numerical gradient implementation 
    # in computeNumericalGradient.m
    # It analytically evaluates the gradient of a very simple function called
    # simpleQuadraticFunction (see below) and compares the result with your numerical
    # solution. Your numerical gradient implementation is incorrect if
    # your numerical solution deviates too much from the analytical solution.
      
    # Evaluate the function and gradient at x = [4; 10]; (Here, x is a 2d vector.)
    x = np.array([4,10])
    value, grad = simpleQuadraticFunction(x);
    
    # Use your code to numerically compute the gradient of simpleQuadraticFunction at x.
    # (The notation "@simpleQuadraticFunction" denotes a pointer to a function.)
    numgrad = computeNumericalGradient(simpleQuadraticFunction, x);
    
    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar. 
    disp=np.vstack(([numgrad,  grad])).T;
    #print(disp)
    print('Left Numerical Gradient and Right Analytical Gradient:\n '+str(disp));
    
    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # in computeNumericalGradient.m, then diff below should be 2.1452e-12 
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    #print(diff); 
    print('Difference between numerical and analytical gradient (should be < 1e-9):\n '+str(diff));

def train():
    ## STEP 0: Here we provide the relevant parameters values that will
    #  allow your sparse autoencoder to get good filters; you do not need to 
    #  change the parameters below.
    
    visibleSize = 8*8;   # number of input units 
    hiddenSize = 25;     # number of hidden units 
    sparsityParam = 0.01;   # desired average activation of the hidden units.
                         # (This was denoted by the Greek alphabet rho_cap, which looks like a lower-case "p",
    		     #  in the lecture notes). 
    lambda_ = 0.0001;     # weight decay parameter       
    beta = 3;            # weight of sparsity penalty term       
    ##======================================================================
    ## STEP 1: Implement sampleIMAGES
    #
    #  After implementing sampleIMAGES, the display_network command should
    #  display a random sample of 200 patches from the dataset
    
    patches = sampleIMAGES();
    #plot and save in a pdf with name given as parameter
    display_network(patches[:,1:200],"random_200.pdf")
    
    
    #  Obtain random parameters theta
    theta = initializeParameters(hiddenSize, visibleSize);
    
    ##======================================================================
    ## STEP 2: Implement sparseAutoencoderCost
    #
    #  You can implement all of the components (squared error cost, weight decay term,
    #  sparsity penalty) in the cost function at once, but it may be easier to do 
    #  it step-by-step and run gradient checking (see STEP 3) after each step.  We 
    #  suggest implementing the sparseAutoencoderCost function using the following steps:
    #
    #  (a) Implement forward propagation in your neural network, and implement the 
    #      squared error term of the cost function.  Implement backpropagation to 
    #      compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking 
    #      to verify that the calculations corresponding to the squared error cost 
    #      term are correct.
    #
    #  (b) Add in the weight decay term (in both the cost function and the derivative
    #      calculations), then re-run Gradient Checking to verify correctness. 
    #
    #  (c) Add in the sparsity penalty term, then re-run Gradient Checking to 
    #      verify correctness.
    #
    #  Feel free to change the training settings when debugging your
    #  code.  (For example, reducing the training set size or 
    #  number of hidden units may make your code run faster; and setting beta 
    #  and/or lambda to zero may be helpful for debugging.)  However, in your 
    #  final submission of the visualized weights, please use parameters we 
    #  gave in Step 0 above.
    
    cost, grad = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda_, sparsityParam, beta, patches[:,1:10]);
    
    ##======================================================================
    ## STEP 3: Gradient Checking
    #
    # Hint: If you are debugging your code, performing gradient checking on smaller models 
    # and smaller training sets (e.g., using only 10 training examples and 1-2 hidden 
    # units) may speed things up.
    
    # First, lets make sure your numerical gradient computation is correct for a
    # simple function.  After you have implemented computeNumericalGradient.m,
    # run the following: 
    checkNumericalGradient();
    # Now we can use it to check your cost function and derivative calculations
    # for the sparse autoencoder.  
    check_cost = partial(sparseAutoencoderCost,visibleSize=visibleSize,hiddenSize=hiddenSize,lambda_=lambda_,sparsityParam=sparsityParam,beta=beta,data=patches[:,1:10])
    numgrad = computeNumericalGradient(check_cost, theta)
    # Use this to visually compare the gradients side by side
    disp = np.vstack([numgrad, grad]).T
    print(disp)
    
    # Compare numerically computed gradients with the ones obtained from backpropagation
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
     # Should be small. In our implementation, these values are
     # usually less than 1e-9.
     # When you got this working, Congratulations!!! 
    print("The Difference in Gradients is: "+str(diff))

    ## STEP 4: After verifying that your implementation of
    #  sparseAutoencoderCost is correct, You can start training your sparse
    #  autoencoder with minFunc (L-BFGS).
    
    #  Randomly initialize the parameters
    theta = initializeParameters(hiddenSize, visibleSize)

    partialCost = partial(sparseAutoencoderCost,visibleSize=visibleSize,
                                    hiddenSize=hiddenSize,
                                    lambda_=lambda_,
                                    sparsityParam=sparsityParam,
                                    beta=beta,
                                    data=patches)
    # Here, we use L-BFGS to optimize our cost
    # function. Generally, for minFunc to work, you
    # need a function pointer with two outputs: the
    # function value and the gradient. In our problem,
    # sparseAutoencoderCost.m satisfies this.
    # Maximum number of iterations of L-BFGS to run
    opttheta, cost, info = fmin_l_bfgs_b(partialCost, theta, maxiter=400, disp=1)
    #print(info)

    ## STEP 5: Visualization
    W1 = opttheta[:hiddenSize*visibleSize].reshape(hiddenSize, visibleSize)
    display_network(W1.T,"final_visualization.pdf")
    
if __name__=='__main__':
    train()
