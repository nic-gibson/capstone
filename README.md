# Capstone Project - Imperial College Business School Machine Learning & AI

## Overview



This is a black box optimisation project using Bayesian Optimisation.  There are eight functions to be processed with increasing dimensionality  (running from two to 8). The purpose is to optimise a Gaussian Processor in order to emulate the behaviour of the hidden function as closely as possible. 

Bayesian Optimisation is useful in both the machine learning and physical worlds. It is useful when it is expensive to carry out the real operation (a function call or a real world activity such as biological testing). In a SAAS environment this can be useful when identifying optimal configurations for cloud infrastructure/software as measuring the effectiveness of infrastructure would normally require creation and configuration of that infrastructure which is a costly process.

## Inputs & Outputs

At the starting point, we received a certain number of inputs to and outputs from the hidden functions (see below). Every output is a single scalar. Inputs are constrained to be in the range [0, 1]. The initial datasets contained between ten and forty input and output points

| Function No | Degree | Initial Points |
| ----------: | -----: | -------------: |
| 1 | 2 | 10 |
| 2 | 2 | 10 | 
| 3 | 3 | 15 | 
| 4 | 4 | 30 | 
| 5 | 4 | 20 |
| 6 | 5 | 20 |
| 7 | 6 | 30 |
| 8 | 8 | 40 |


## Challenge Objectives

This is a maximisation problem so the objective is to maximise each of the 8 hidden functions. The task is constrained by the lack of information on the nature of the functions. The task starts with a certain number of initial points supplied and, each week, the y value from a proposed Xn values can be requested - once per week. Each function may have entirely different features and some may be much more noisy than others. In the higher order functions (6, 7 and 8) the provided set of initial values provides a very small proportion of the total volume of the problem space. The only known constraint on the Xn values is that they are all between 0.0 and 1.0 


## Technical Approach 

Rather than choosing a single approach, I built multiple acquisition functions and set of tools designed to test those. I built the following acquisition functions

* UCB 
* maximum variance
* pure exploit using just the mean 
* probability of improvement
* expected improvement

I use the initial data to experiment and identify potentially useful acquisition functions. I built tools to use the initial data as training and validation sets and compared each functions performance. I followed a similar approach with the values of hyperparameters (_xi_ and _kappa_). Each week I review the acquisition functions and their hyperparameters to determine if either needs to be changed. At each stage I've used some visulation tools and leave one out evalution to test if I'm still following the right approach to whether exploitation or exploration is required. 

