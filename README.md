    Section 1: project overview
        Briefly describe the BBO capstone project and its purpose.
        What is the overall goal of the BBO capstone project? Why is it relevant in real-world ML? What’s the high-level idea?
        How would this BBO capstone project support you in your current or future career?
    Section 2: inputs and outputs

        Clearly state what your model receives and returns.
        What are the inputs (query format, dimensions, constraints, etc.)? What is the expected output (response value, performance signal, etc.)? Include example formats, if possible.
    Section 3: challenge objectives

        Outline what you are trying to achieve within the BBO capstone project.
        Is the goal to minimise or maximise the function(s)? What constraints or limitations must you consider (e.g. number of queries, response delay and unknown function structure)?
    Section 4: technical approach

        Describe the strategies you used across your first three query submissions. You’re encouraged to treat this section as a living record – continue updating it as your approach evolves throughout the BBO capstone project.
        What ML methods or heuristics do you use? Will you model the unknown function? Would you consider using SVMs, regressions or Bayesian techniques? 
        How do you balance exploration and exploitation? What makes your approach thoughtful or unique?



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




## Technical Approach 

