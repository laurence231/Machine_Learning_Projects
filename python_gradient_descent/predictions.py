from csv import reader
#This is the perceptron tutorial test.


#CSV loader
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


#to convert string module to float
def str_column_to_float(dataset,column):
    for row in dataset:
        row[column] = float(row[column].strip())  #strip removes whitespace etc

def str_column_to_int(dataset,column):




#This is a function which produces a prediction with weights

def predict(row, weights):
    activation = weights[0]       #this is the bias
    for i in range(len(row)-1):
        activation+= weights[i+1]*row[i]      #this adds the weight*the value on to the bias
    return 1.0 if activation>=0 else 0.0
    #######################################################
    #activation = (w1 * X1) + (w2 * X2) + bias effectively
    #######################################################
#here is a test dataset for us to work with

#Stochastic gradient descent algorithm
#this is used to calculate the weight values

def train_weights(train, learning_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    #print(weights, ' this is the weight')
    for epoch in range(n_epoch):                        #this loop iterates over each successive epoch
        sum_error = 0.0                                 #this reinitialises the error each time
        for row in (train):
            prediction = predict(row, weights)    #this produces the prediction of the result
            #print(prediction, 'is the prediction')
            error = row[-1] - prediction           #this compares prediction with the result
            sum_error += error**2
            weights[0]=weights[0]+learning_rate*error  #this updates the bias - if there is an error, then the weights are updated!
            #print(weights[0],'this is the bias')
            for i in range(len(row)-1):
                weights[i+1] = weights[i+1] + learning_rate*error*row[i]  #this loop updates the weights within the epoch if there is an error
                #print(weights[i+1], 'this is the weight of the variable %s' %(i+1))
        print('epoch = %s, learning rate = %s, error = %s'%(epoch, learning_rate,sum_error))
    return(weights)

#CSV file loader




# of the form [col1, col2, expected output]
#dataset = [[2.7810836,2.550537003,0],#

#	[1.465489372,2.362125076,0],
#	[3.396561688,4.400293529,0],
#	[1.38807019,1.850220317,0],
#	[3.06407232,3.005305973,0],
#	[7.627531214,2.759262235,1],
#	[5.332441248,2.088626775,1],
#	[6.922596716,1.77106367,1],
#	[8.675418651,-0.242068655,1],
#	[7.673756466,3.508563011,1]]

#Weight is of the form [bias, weight for col 1, weight for col 2].
learning_rate = 0.1
n_epoch = 5
weights = train_weights(dataset, learning_rate, n_epoch)

print(weights)



