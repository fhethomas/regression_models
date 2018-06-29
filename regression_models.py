
# coding: utf-8

# # Linear Models
# This file is for test and creation of linear/logistic regression models
# 1. Linear_regression built - testing to be done
# 2. Logistic_regression to be built as a class
# 3. Neural_network - simple perceptron to be built as class

import numpy as np
# Creation of linear regression class:
class Linear_regression:
    def __init__(self,alpha=0.01, degree_accuracy=0.05):
        self.alpha = alpha
        self.degree_accuracy = degree_accuracy
        self.thetas_set = False
    def generate_theta(self,X):
        self.theta = np.random.rand(X.shape[1],1)
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
        return np.dot(X,self.theta)


