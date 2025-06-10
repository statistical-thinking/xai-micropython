---
title: 'XAI-MICROPYTHON: Explainable Artificial Intelligence with MicroPython'
tags:
  - Explainable Artificial Intelligence
  - Statistics
  - Machine Learning
  - Deep Learning
  - MicroPython
authors:
  - name: Dennis Klinkhammer
    orcid: 0000-0003-1011-5517
    affiliation: 1
affiliations:
 - name: Statistical Thinking (Germany), www.statistical-thinking.de, info (at) statistical-thinking.de
   index: 1
date: 10 June 2025
bibliography: paper.bib
---

# Summary

XAI-MICROPYTHON is a MicroPython based framework which emphasizes the fundamentals of statistics, machine learning and deep learning, since they are increasingly driving innovation across various fields [@LeCun2015-hn]. These fundamentals can also be transferred on microcontrollers [@Ray2022-oq], for educational purpose or e.g. in order to process sensor data [@Cioffi2020-lq]. Since microcontrollers have limited computational resources, a particularly efficient implementation of these methods is required [@Delnevo2023-ux; @Wulfert2024-qf]. However, it is the efficient implementation that provides insights into the fundamentals of statistics, machine learning and deep learning in terms of explainable artificial intelligence that makes this approach also relevant for educational purpose [@Haque2023-du; @Meske2022-ke]. Coding from scratch in MicroPython allows for a streamlined and tailored approach [@Delnevo2023-ux] and provides users with transparent insights into underlying principles and the resulting code, which makes XAI-MICOPYTHON suitable for didactic application [@Collier2024-ld; @Verma2022-ie; @Scherer2016-fy].

# Target group
XAI-MICROPYTHON draws on well-known exercise data sets and common methods to give (academic) teachers a quick start in designing their own teaching materials. In addition, XAI-MICROPYTHON can of course also be used by self-learners.

# Statement of need

The methods presented with XAI-MICROPYTHON can actually be easily implemented in Python using libraries such as SciKit-Learn. However, the aim of XAI-MICROPYTHON is not only to call up and apply already existing functions, but also to understand their methodological foundations via programming from scratch in MicroPython and without recourse to previous libraries [@Kong2022-sk]. A previous investigation of skill requirements in artificial intelligence and machine learning job advertisements [@Verma2022-ie] served as a guide for the development of XAI-MICROPYTHON. It is intended to provide insights into basic statistical principles, some machine learning algorithms as well as the structure and functioning of neural networks as basis for artificial intelligence [@Schmidt2020-ak; @Frank2020-rw; @Scherer2016-fy]. Since XAI-MICROPYTHON is based on the MicroPython programming language, it can be used on microcontrollers like the Raspberry Pi Pico 2 [@Sakr2021-mc] or it can be called up directly via Jupyter Lite as a live demo in the browser. Especially the step-by-step visibility in MicroPython helps users to identify and understand the mathematical and statistical foundations of artificial intelligence [@Kong2022-sk]. This is in accordance with the primary goals of explainable artificial intelligence [@Haque2023-du; @Schmidt2020-ak] and provides insights into artificial intelligence related responsibilities [@Collier2024-ld; @Frank2020-rw] as well as practical experiences [@Li2021-bn]. Therefore, XAI-MICROPYTHON shall promote necessary innovation and problem-solving skills in accordance with the scientific discourse [@Verma2022-ie].

# Basic elements
XAI-MICROPYTON introduces to mean, variance, standard deviation, covariance and correlation as elements of univariate and bivariate statistics, which are necessary for the implementation of machine learning algorithms like single and multiple linear regressions. Simple classification tasks will be highlighted via multiple logistic regression, followed by k-Means as algorithm, as well as the concept of the within cluster sum of squares, for the demonstration of clustering. Furthermore, factor analysis will be implemented in MicroPython in order to demonstrate dimensionality reduction. In particular, the division into four different machine learning application areas is based on the classification of SciKit-Learn (regression, classification, clustering and dimensionality reduction). Finally, a pre-trained neural network and a self-learning neural network will be implemented in MicroPython in order to highlight some common activation functions as well as the underlying principles of layers and neurons. Therefore, XAI-MICROPYTHON is an underlying framework, which can be extended independently by (academic) teachers with further algorithms and methods and is intended to grow via the community guidelines provided.

# References
