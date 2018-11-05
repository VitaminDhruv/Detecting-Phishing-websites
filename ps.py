#2)	Naïve Bayesian Classifer- 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

import numpy as np
from sklearn.naive_bayes import GaussianNB
import time


def load_data():
    

    #Get the data
    training_data = np.genfromtxt(r'Dataset.csv', delimiter=',', dtype=np.int32)

    inputs = training_data[:,:-1]         # Get the inputs - All rows and all columns except the last one 

    outputs = training_data[:,-1]         # Get the labels

    # Divide the data set into training and testing. Total=2456
    #  Training dataset (1500 rows)
    #  Training dataset (956 rows) 	
    training_inputs = inputs[:1500]       #  Select first 1500 rows (0-1499) excluding last column
    training_outputs = outputs[:1500]     #  Select first 1500 rows (0-1499) with only last column
    testing_inputs = inputs[1500:]		  #  Select remaining rows (1500-2455) excluding last column
    testing_outputs = outputs[1500:]      #  Select remaining rows (1500-2455) with only last column

    # Return the four arrays
    return training_inputs, training_outputs, testing_inputs, testing_outputs
   

if __name__ == '__main__':        # Entry point of the program
    start_time = time.time()
    train_inputs, train_outputs, test_inputs, test_outputs = load_data()      # get  the data 
	
    classifier= GaussianNB()
  
    classifier.fit(train_inputs, train_outputs)       # Train the classifier model
    
    predictions = classifier.predict(test_inputs)      # make the predictions on testing data
	
    confusionmatrix=confusion_matrix(test_outputs,predictions)       # Create a confusion matrix 
   
    accuracy = 100.0 * accuracy_score(test_outputs, predictions)     # Calculate the accuracy
    print ("The accuracy of your Naive Bayesian classifier on testing data is: " + str(round(accuracy,2))+ "%")
    print("confusion matrix=\n",confusionmatrix)
	
    error=(1-accuracy/100.0)*100.0
    print("The error rate of the Naive Bayesian classifier on testing data is: " + str(round(error,2)) + "%")
	
    report=classification_report(test_outputs,predictions)
    print("The classification report is:\n "+ str(report))
    print("Time = %s seconds " % (time.time() - start_time))

    
    