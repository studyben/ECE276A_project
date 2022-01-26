'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
from generate_rgb_data import read_pixels
from scipy.stats import multivariate_normal

class PixelClassifier():

  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    self.x1Mean = [0, 0, 0]
    self.x2Mean = [0, 0, 0]
    self.x3Mean = [0, 0, 0]
    self.x1Cov = [0, 0, 0]
    self.x2Cov = [0, 0, 0]
    self.x3Cov = [0, 0, 0]
    folder = 'data/training'
    self.x1 = read_pixels(folder + '/red', verbose=True)
    self.x2 = read_pixels(folder + '/green')
    self.x3 = read_pixels(folder + '/blue')
    #y1, y2, y3 = np.full(x1.shape[0], 1), np.full(x2.shape[0], 2), np.full(x3.shape[0], 3)
    #self.X, self.y = np.concatenate((x1, x2, x3)), np.concatenate((y1, y2, y3))

  def training(self):
    self.x1Cov = np.cov(np.matrix.transpose(self.x1))
    self.x2Cov = np.cov(np.matrix.transpose(self.x2))
    self.x3Cov = np.cov(np.matrix.transpose(self.x3))
    self.x1Mean = np.mean(np.matrix.transpose(self.x1), axis=1)
    self.x2Mean = np.mean(np.matrix.transpose(self.x2), axis=1)
    self.x3Mean = np.mean(np.matrix.transpose(self.x3), axis=1)

  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
    
    # Just a random classifier for now
    # Replace this with your own approach
    y = []
    for xs in X:
      xs1 = multivariate_normal.pdf(xs, mean=self.x1Mean, cov=self.x1Cov)
      xs2 = multivariate_normal.pdf(xs, mean=self.x2Mean, cov=self.x2Cov)
      xs3 = multivariate_normal.pdf(xs, mean=self.x3Mean, cov=self.x3Cov)
      print(np.argmax([xs1, xs2, xs3]))
      y = np.append(y, np.argmax([xs1, xs2, xs3]))
    
    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

