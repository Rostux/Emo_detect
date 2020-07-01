# Emo_detect
Face recoginition and emotion detection using 2 hidden layer MLP with Xavier initialization, a Support Vector Machine(SVM) and Logstic Regression.

Multiple classifiers have been used on many different feature spaces to get an idea of which one works good. I have used a 2 hidden layer Multi Layer Perceptron(MLP) with Xavier Initialization, a Support Vector Machine(SVM) and Logistic Regression.

Several feature spaces are used in which some are combinations. For all the datasets, all these datasets work sufficiently good on all feature spaces except PCA and kernel PCA spaces. PCA and kernel PCA do not capture the different classes as in PCA, we remove the components irrespective of the importance as we do not use labels for PCA or KPCA. LDA overcomes this by using the labels to do a better job in dimensionality reduction and further classification. The best combination was found to be SVM on RESNET.

Similarly, we use labelled images and a multi layer perceptron to classify the different emotions. This could be used to detect emotions of people in certain areas and allow authorities to monitor suspicious individuals if they consistently show negative emotions.
