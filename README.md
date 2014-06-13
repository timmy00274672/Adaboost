Adaboost
========
A 2-class problem is considered. The data set X consists of N=100 2-D vectors. Of 
these, 50 stem from class w1, and the other 50 data vectors stem from class w2.

![question](https://raw.githubusercontent.com/timmy00274672/Adaboost/master/img/question.jpg)

1. 	Use the Adaboost algorithm on the data set X to generate a strong classifier. 
	Assume the maximum number of base classifiers is 1000.
2.	Classify the vectors of the training set X using the classifier in (1). 
	Compute the classification error of P when only the first t weak classifiers 
	are taken into account, t=1, 2, 3,…,T_max. 
	Plot P versus the number of weak classifiers.
3.	Generate a test data set Z using the specification of X. 
	Classify the test data set Z via the designed classifier and 
	plot the error P versus the number of weak classifiers used.

## Howto
Just run the `Demo.m`.

## Result

![result](https://raw.githubusercontent.com/timmy00274672/Adaboost/master/img/result.jpg)

## Reference

* [Classic AdaBoost Classifier](http://www.mathworks.com/matlabcentral/fileexchange/27813-classic-adaboost-classifier)