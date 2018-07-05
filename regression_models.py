
# coding: utf-8

# # Linear Models
# This file is for test and creation of linear/logistic regression models
# 1. Linear_regression built - testing to be done
# 2. Logistic_regression to be built as a class
# 3. Neural_network - simple perceptron to be built as class

import numpy as np
from scipy.optimize import fmin_cg #fmin_cg to train neural network

"""
 feature scaling
"""
def normalize(X):
    # normalize across features
    m,n = X.shape
    for col in range(n):
        X[:,col] = ((X[:,col]-np.min(X[:,col]))/(np.max(X[:,col])-np.min(X[:,col])))
    return X
def standardize(X):
    # standardize across features
    m,n = X.shape
    for col in range(n):
        X[:,col] = (X[:,col]-np.mean(X[:,col]))/np.std(X[:,col])
    return X
# Creation of linear regression class:
class Linear_regression:
    """Linear regression model
    Parameters
    -----------
    alpha : float, default 0.01
        learning rate of gradient descent
    degree_accuracy : float, default 0.05
        degree of accuracy that linear
        regression is looking for during 
        gradient descent
    feature_scaling : string, default None
        options: None,'normalize'
        """
    def __init__(self,alpha=0.01, degree_accuracy=0.05,feature_scaling=None):
        self.alpha = alpha
        self.degree_accuracy = degree_accuracy
        self.thetas_set = False
        self.feature_scaling_options = [None,'normalize']
        assert feature_scaling in self.feature_scaling_options,"no such feature scaling option"
        self.feature_scaling = feature_scaling
        self.theta = np.array([])
    def generate_theta(self,X):
        self.theta = np.random.rand(X.shape[1],1)
    def add_intercept(self,X):
        return np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
    def coefficients(self):
        # prints coefficients
        print(self.theta[:-1,:])
    def intercept(self):
        # print intercept
        print(self.theta[-1,:])
    def computeCost(self,X,y,theta):
        # Cost function
        m,n=X.shape
        J=0
        X=(np.dot(X,theta))
        J=sum(np.power(X-y,2))
        J*=(1.0/(2.0*m))
        return J
    def fit(self,X,y):
        """fits model
        Parameters
        --------------
        X : numpy array
            Array should be independent variables.
            Shape must be m * n, where m is cases
            and n is features
        y : numpy array
            Array should be dependent variable
            Shape should be m * 1, where m is cases
        Returns
        --------------
        theta : numpy array
            These are the coefficients of the linear
            regression. This will be stored in the 
            model for use in predict"""
        
        alpha=self.alpha
        degree_accuracy=self.degree_accuracy
        if self.thetas_set == False:
            if self.feature_scaling == 'normalize':
                X=normalize(X)
            X = self.add_intercept(X)
            self.generate_theta(X)
            self.thetas_set = True
        theta = self.theta
        h=0
        m,n=X.shape
        J_history=[]
        J=999.0
        iterations=0
        print(theta)
        while J>=degree_accuracy:
            h=np.dot(X,theta)
            #print('h1: {0}'.format(h))
            h=np.dot(X.T,h-y)/float(m)
            #print('h2: {0}'.format(h))
            theta=theta-(h*alpha)
            #print(theta)
            J=self.computeCost(X,y,theta)
            J_history.append(J)
            iterations+=1
            if iterations > 2 and J > J_history[-2]:
                print('J increasing, reducing alpha to: {0}'.format(alpha/2))
                self.generate_theta(X)
                self.alpha = alpha/2
                theta = self.fit(X,y)
                return theta
        #print(J_history)
        print('Returned Thetas are: {0}'.format(theta))
        print('degree of accuracy: %s' % J_history[-1])
        print('number of iterations: %s' % iterations)
        self.theta = theta
        return theta
    def predict(self,X):
        """Use this function to predict y
        from a set of X data
        Parameters
        ------------
        X : numpy array
            independent variables, shaped
            m * n, where m is case and n is 
            feature
        """
        if self.feature_scaling == 'normalize':
            X=normalize(X)
        X = self.add_intercept(X)
        return np.dot(X,self.theta)
class Logistic_regression:
    """Linear regression model
    Parameters
    -----------
    alpha : float, default 0.01
        learning rate of gradient descent
    degree_accuracy : float, default 0.05
        degree of accuracy that linear
        regression is looking for during 
        gradient descent
    feature_scaling : string, default None
        options: None,'normalize','standardize'
        """
    def __init__(self,alpha=0.01,degree_accuracy=0.05,feature_scaling=None):
        self.alpha = alpha
        self.degree_accuracy=degree_accuracy
        self.feature_scaling=feature_scaling
    def cost_function(self,X,y,theta):
        m,n=X.shape
        j=(1/m)*(-np.dot(y.T,np.log(self.g(X,theta)))-np.dot((self.one-y).T,np.log(1-self.g(X,theta))))
        return sum(j[0])
    def gradient_descent(self,alpha,theta,X,y):
        m,n=X.shape
        theta=theta-(self.alpha/m)*np.dot(X.T,self.g(X,theta)-y)
        return theta
    def g(self,X,theta):
        return 1/(1+np.power(self.test_e,-np.dot(X,theta)))
    def fit(self,X,y,max_iterations=500):
        """fits model
        Parameters
        --------------
        X : numpy array
            Array should be independent variables.
            Shape must be m * n, where m is cases
            and n is features
        y : numpy array
            Array should be dependent variable
            Shape should be m * 1, where m is cases
        max_iterations : integer, defaut 500
            Can amend the maximum iterations of 
            gradient descent before model finishes
        """
        if self.feature_scaling=='normalize':
            X=normalize(X)
        elif self.feature_scaling=='standardize':
            X=standardize(X)
        self.X=X
        self.y=y
        m,n=X.shape
        iteration=0
        self.theta = np.random.rand(n,1)
        self.test_e=np.zeros((m,1))
        self.test_e+=np.exp(1)
        self.one=np.ones((m,1))
        while not(np.absolute(self.cost_function(X,y,self.theta))<=self.degree_accuracy or iteration>=max_iterations):
            self.theta = self.gradient_descent(self.alpha,self.theta,X,y)
            iteration+=1
        if iteration>=max_iterations:
            pass
            print('Iterations exceeded. Model may return incorrect values')
        else:
            pass
            print('Model fit.')
    def coef(self):
        print(self.theta)
    def predict(self,X):
        """Predicts outcomes based on fitted model
        Parameters
        --------------
        X : numpy array
            inputs in shape m x n,
            where m is sample cases and
            n is features
        Returns 
        --------------
        prediction : numpy array
            This is the list of predicted
            outcomes based on fitted model
        """
        if self.feature_scaling=='normalize':
            X=normalize(X)
        elif self.feature_scaling=='standardize':
            X=standardize(X)
        prediction = self.g(X,self.theta)
        prediction[prediction<0.5]=0
        prediction[prediction>0.5]=1
        return prediction
    def accuracy(self):
        pred=self.predict(self.X)
        accu = y[y==pred].shape[0]/y.shape[0]
        print('Accuracy: {0}'.format(accu))
        return accu

class Neural_network:
    """Neural network object. This is a simple
    perceptron with only one hidden layer
    Parameters
    --------------
    hidden_layer_size : integer, default 20
        This is the size of the hidden layer
    epsilon : float, default 0.12
        float used in random number generation
        to decide start point for coefficients
    lamb : float, default 1
        lambda used for regularization of
        gradient descent
    feature_scaling : string, default None
        options: None, 'normalize','standardize'
    """
    def __init__(self,hidden_layer_size=20,epsilon=0.12,lamb=1,feature_scaling=None):
        self.hidden_layer_size=hidden_layer_size
        self.epsilon=epsilon
        self.lamb=lamb
        self.X=np.array([[]])
        self.y=np.array([[]])
        self.theta=np.array([[]])
        self.input_layer_size=5
        self.output_layer_size=1
        self.feature_scaling_options = [None,'normalize','standardize']
        assert feature_scaling in self.feature_scaling_options,"no such feature scaling option"
        self.feature_scaling=feature_scaling
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    def sigmoidGradient(self,z):
        return np.multiply(self.sigmoid(z),(1-self.sigmoid(z)))
    def initialise_thetas(self,input_layer_size,hidden_layer_size,output_layer_size):
        # Creates initial coefficients for each node
        theta1=np.random.rand(input_layer_size+1,hidden_layer_size)
        theta2=np.random.rand(hidden_layer_size+1,output_layer_size)
        theta=np.array([theta1,theta2])
        theta=self.theta_flatten(theta)*2*self.epsilon-self.epsilon
        return theta
    def theta_flatten(self,theta):
        # flattens for use in minimization
        theta_t=theta[:]
        theta=np.array([])
        #fmin_cg requires a gradient to be (m,0) dimensions
        for x in theta_t:
            theta=np.concatenate((theta,x.flatten()),0)
        #theta=theta.reshape(len(theta),0)
        #print(theta.dtype)
        return theta
    def theta_unflatten(self,theta,input_layer_size,hidden_layer_size,output_layer_size):
        # unflattens coefficients for general use
        theta1=theta[:(input_layer_size+1)*hidden_layer_size].reshape((input_layer_size+1),hidden_layer_size)
        theta2=theta[(input_layer_size+1)*hidden_layer_size:].reshape(hidden_layer_size+1,output_layer_size)
        return theta1, theta2
    def costFunction(self,theta,X,y,input_layer_size,hidden_layer_size,output_layer_size,lamb):
        # Calculate the cost function
        m,n=X.shape
        theta1,theta2=self.theta_unflatten(theta,input_layer_size,hidden_layer_size,output_layer_size)
        one=np.ones((m,1))
        a1=np.concatenate((one,X),1)
        a2=np.concatenate((one,self.sigmoid(np.dot(a1,theta1))),1)
        sig=self.sigmoid(np.dot(a2,theta2))
        cost=np.multiply(-y,np.log(sig))-np.multiply((1-y),np.log(1-sig))
        theta1_bias=theta1[1:,:]
        theta2_bias=theta2[1:,:]
        J=(1/m)*sum(sum(cost))+(lamb/(2*m))*(sum(sum(np.square(theta1_bias)))+sum(sum(np.square(theta2_bias))))
        return J
    def nnGradient(self,theta,X,y,input_layer_size,hidden_layer_size,output_layer_size,lamb):
        # calculate gradient for use in gradient descent
        m,n=X.shape
        theta1,theta2=self.theta_unflatten(theta,input_layer_size,hidden_layer_size,output_layer_size)
        one=np.ones((m,1))
        a1=np.concatenate((one,X),1)
        a2=np.concatenate((one,self.sigmoid(np.dot(a1,theta1))),1)
        sig=self.sigmoid(np.dot(a2,theta2))
        d3=sig-y
        d2=np.dot(d3,theta2.T)
        z2=self.sigmoidGradient(np.concatenate((one,np.dot(a1,theta1)),1))
        d2=np.multiply(d2,z2)
        delta1=np.dot(a1.T,d2[:,1:])
        delta2=np.dot(a2.T,d3)
        one=np.ones((1,hidden_layer_size))
        theta1=np.concatenate((one,theta1[1:,:]),0)
        one=np.ones((1,output_layer_size))
        theta2=np.concatenate((one,theta2[1:,:]),0)
        t1_grad=(1/m)*delta1+(lamb/m)*theta1
        t2_grad=(1/m)*delta2+(lamb/m)*theta2
        grad=self.theta_flatten([t1_grad,t2_grad])
        #print(grad.shape)
        return grad
    def predict(self,X):
        """Predicts outcomes based on fitted model
        Parameters
        --------------
        X : numpy array
            Sample used for prediction based on trained
            model. Dimensions must be: m x n where m 
            is samples and n is features
            """
        if self.feature_scaling == 'normalize':
            X=normalize(X)
        elif self.feature_scaling == 'standardize':
            X=standardize(X)
        m,n=X.shape
        theta1,theta2=self.theta_unflatten(self.theta,self.input_layer_size,self.hidden_layer_size,self.output_layer_size)
        one=np.ones((m,1))
        a1=np.concatenate((one,X),1)
        a2=np.concatenate((one,self.sigmoid(np.dot(a1,theta1))),1)
        sig=self.sigmoid(np.dot(a2,theta2))
        sig[sig>0.5]=1
        sig[sig<0.5]=0
        #print((sig[sig==y].shape[0]/X.shape[0])/output_layer_size)
        return sig
    def fit(self,X,y):
        """Fits the model to the training data
        Parameters
        --------------
        X : numpy array
            shape is m x n, where m is sample
            and n is features
        y : numpy array, 
            shape must be m x u, where m is
            samples and u is categorical features.
            This must be an array of integers 0 or 1 
            representing boolean data"""
        if self.feature_scaling == 'normalize':
            X=normalize(X)
        elif self.feature_scaling == 'standardize':
            X=standardize(X)
        m,n=X.shape
        self.input_layer_size=n
        self.hidden_layer_size=n+1
        m,n=y.shape
        self.output_layer_size=n
        self.theta=self.initialise_thetas(self.input_layer_size,self.hidden_layer_size,self.output_layer_size)
        arg=X,y,self.input_layer_size,self.hidden_layer_size,self.output_layer_size,self.lamb
        self.theta=fmin_cg(self.costFunction,x0=self.theta, fprime= self.nnGradient,args=arg)
        print('Training complete')