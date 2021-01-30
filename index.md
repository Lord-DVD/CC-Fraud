# Comparative analysis of most commonly used ML algorithms to test efficiency of them to detect fraudulent credit card transactions  
  
## Introduction  

<p> The aim of this project is to build a classifier that can detect fraudulent credit card transactions using
several machine learning algorithms such as logistic regression, decision trees, artificial neural network
and gradient boosting. And determine which algorithm gives best results for given use case scenario and
should be recommended in real life application using real life credit card transaction data. </p>  

## Algorithms tested  

- Logistic Regression  
- Decision Tree  
- Artificial Nueral Network  
- Gradient Boosting  

## Method used to compare the models  

Area under the ROC(Receiver Operator Characteristic) curve.
<p> It is a graphical plot used to show the diagnostic ability of binary classifiers. Such as in our case fraud/not fraud. ROC is created using plotting the True positive rate (TPR) against the false positive rate (FPR). Hence it shows the trade-off between sensitivity and specificity. Hence closer the curve is to top left of the graph better fit of the model and closer it is to the 45-degree line, less accurate the fit. </p>  

## Dataset Used  
  
I have used real-life credit card transaction dataset from <b>data-flair</b> website and is available in this repository. It contains a log of 284807 real life credit card transactions and has all of them identified as fraudulent or not. </p>  

#### For complete workigs please refer to the [pdf](https://github.com/Lord-DVD/CC-Fraud/blob/main/Testing_Different_ML_Algorithms.pdf) file. 
