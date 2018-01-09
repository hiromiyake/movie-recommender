#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 18:25:49 2017

@author: hiro

Cleaned functionized version of collaborative filtering code
using raw data from movielens website
Data used is the latest (as of 10/2016)
https://grouplens.org/datasets/movielens/
https://grouplens.org/datasets/movielens/latest/
Small: 100,000 ratings and 1,300 tag applications applied to 
9,000 movies by 700 users. Last updated 10/2016.
"""

import numpy as np
import numpy.matlib
import numpy.random
from scipy.optimize import minimize

## Define the cost function and its derivative
## Will be used in the movierec function below
## Returns a number J and array grad
## grad will be fed to an optimization function
def moviereccostfunc(params, *args):
    Y, R, num_users, num_movies, num_features, lam = args
    X = np.reshape(params[:num_movies*num_features], (num_movies, num_features))
    Theta = np.reshape(params[(num_movies*num_features):], (num_users, num_features))
    
    J = 0
    X_grad = numpy.matlib.zeros(X.shape)
    Theta_grad = numpy.matlib.zeros(Theta.shape)

    J = (0.5*np.sum(np.sum(np.square(np.multiply((np.dot(X,Theta.T)-Y),R)))) + 
                  0.5*lam*np.sum(np.sum(np.square(Theta))) +
                              0.5*lam*np.sum(np.sum(np.square(X))))
    
    X_grad = (X_grad + np.dot((np.multiply((np.dot(X,Theta.T) - Y),R)),Theta) + 
                            lam*X)
    Theta_grad = (Theta_grad + (np.dot(X.T,(np.multiply((np.dot(X,Theta.T)-Y),R)))).T + 
                                    lam*Theta)

    grad = np.concatenate((X_grad.flatten().T, Theta_grad.flatten().T), axis = 0)
    grad = np.asarray(grad).flatten()
    
    return J, grad

def movierec(my_ratings_dict, randseed = 0, prednum = 10, lam = 10, num_features = 10):
    ## Extract the movieId, title pairs
    ## There are 9125 movies
    ## Note the movieId are NOT consecutive numbers
    ## Use a dictionary for the movieId, title pairs
    file = open("movies.csv","r")
    moviedict = {}
    movienum = 0
    firstline = True
    idtonumdict = {}
    ## NOTE: numtoiddict starts from 0
    ## So for Toy Story (1995) with itemId = 1 has num = 0
    ## Likewise idtonumdict the num starts from 0
    numtoiddict = {}
    for line in file:
        if firstline:
            firstline = False
        else:
            infoline = line.split("\"")
            if len(infoline) == 1:
                infoline = infoline[0].split(",", 2)
            else:
                infoline[0] = infoline[0].split(",")[0]
            moviedict[int(infoline[0])] = infoline[1]
            idtonumdict[int(infoline[0])] = movienum
            numtoiddict[movienum] = int(infoline[0])
            movienum += 1
    
    ## infolist = [number of users, number of movies]
    infolist = [671, movienum]
    
    ## Extract the userId, movieId, rating triplets
    ## There are 671 users with consecutive userId starting with userId = 1
    ## So userId can directly be used as the index for the matrix
    ## matrices have dimensions movies x users
    Y = np.matrix(numpy.matlib.zeros((infolist[1],infolist[0])))
    R = np.matrix(numpy.matlib.zeros((infolist[1],infolist[0])))
    file = open("ratings.csv","r")
    usernum = 0
    for line in file:
        if usernum == 0:
            usernum += 1
        else:
            infoline = line.split(",")
            Y[idtonumdict[int(infoline[1])], int(infoline[0]) - 1] = float(infoline[2])
            R[idtonumdict[int(infoline[1])], int(infoline[0]) - 1] = 1
    
    my_ratings = np.matrix(numpy.matlib.zeros([movienum,1]))
    my_ratings_b = np.matrix(numpy.matlib.zeros([movienum,1]))
    
    ## Print name of movie and rating provided by the user for examination
    for i in my_ratings_dict:
        my_ratings[idtonumdict[i],0] = float(my_ratings_dict[i])
        my_ratings_b[idtonumdict[i],0] = 1
        print 'Rated {0} for {1}'.format(my_ratings_dict[i], moviedict[i])
    
    ## Add the user provided ratings to the other ratings provided in data
    Y = np.concatenate((my_ratings, Y), axis = 1)
    R = np.concatenate((my_ratings_b, R), axis = 1)
    
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    #num_features = 10
    
    np.random.seed(randseed)
    X = np.matrix(numpy.random.randn(num_movies, num_features))
    Theta = np.matrix(numpy.random.randn(num_users, num_features))
    initial_parameters = np.concatenate((X.flatten().T, Theta.flatten().T), axis = 0)
    initial_parameters = np.asarray(initial_parameters).flatten()
    #lam = 10
    args = (Y, R, num_users, num_movies, num_features, lam)
                       
    theta = minimize(moviereccostfunc, initial_parameters, args = args, method='CG',
                     jac = True, options = {'maxiter': 100})
    
    X = np.matrix(np.reshape(theta.x[:num_movies*num_features], (num_movies, num_features)))
    Theta = np.matrix(np.reshape(theta.x[(num_movies*num_features):], (num_users, num_features)))
    
    p = np.matrix(np.dot(X,Theta.T))
    my_predictions=p[:,0]
    
    my_predictions = np.asarray(my_predictions).flatten()
    predsortind = np.argsort(-my_predictions)
    
    ## The approach below does not print predicted ratings for movies
    ## in which the user has already provided ratings
    print 'Top {0} recommendations for you:'.format(prednum)
    i, j = 0, 0
    while i < prednum:
        if not (numtoiddict[predsortind[j]] in my_ratings_dict):
            print '{2}. Predicting rating {0:.2f} for movie {1}'.format(my_predictions[predsortind[j]], moviedict[numtoiddict[predsortind[j]]],i+1)
            i += 1
        j += 1
