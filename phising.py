from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

import numpy as np

#from sklearn import tree
#from sklearn.tree import export_graphviz
##import pydotplus
#import graphviz

def load_data():
    

    #Get the data
    training_data = np.genfromtxt(r'Mohammad14JulyDS_1.csv', delimiter=',', dtype=np.int32)

    
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
    train_inputs, train_outputs, test_inputs, test_outputs = load_data()      # get  the data 

    classifier = tree.DecisionTreeClassifier()        # Create a decision tree classifier model using scikit-learn
   # classifier=lr()
  
    classifier.fit(train_inputs, train_outputs)       # Train the classifier model
    
    predictions = classifier.predict(test_inputs)      # make the predictions on testing data
	
    confusionmatrix=confusion_matrix(test_outputs,predictions)       # Create a confusion matrix 
   
    accuracy = 100.0 * accuracy_score(test_outputs, predictions)     # Calculate the accuracy
    print ("The accuracy of your decision tree on testing data is: " + str(round(accuracy,2))+ "%")
    print("confusionmatrix=\n",confusionmatrix)
	
    error=(1-accuracy/100.0)*100.0
    print("The error rate of the decision tree on testing data is: " + str(round(error,2)) + "%")
	
    report=classification_report(test_outputs,predictions)
    print("The classification report is:\n "+ str(report))
#
#    dot_data = tree.export_graphviz(classifier, out_file=None,
#	                      feature_names=["Having_IP_Address","URL_Length","Shortening_service","@","//","Prefix_suffix",
#						  "having Subdomain","SSLfinal_state", "Domain_Registration_length","favicon","port","HTTPS_Token","Request_URL",
#						  "URL_of_Anchor","Links_in_Tags","SFH","Submitiing_to_email","Abnormal_URL","Redirect"," On_mouseover",
#						  "Rightclick","popup","iframe","ageofdomain","DNS","web_traffic","pagerank",
#						  "google_index","links","statistical_report"],class_names=["legitimate","Malicious"],
#                         filled=True, rounded=True,  
#                         special_characters=True)  
#    graph = graphviz.Source(dot_data)  
#    graph.render("map",view=True)
    
    