
# coding: utf-8

# # Linear Models
# This file is for test and creation of linear/logistic regression models
# 1. Linear_regression built - testing to be done
# 2. Logistic_regression to be built as a class
# 3. Neural_network - simple perceptron to be built as class

import numpy as np

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