import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sheet = "Sheet2"  # clean database to train this MLP

rg = np.random.default_rng()
df = pd.read_excel('dataframe.xlsx', sheet_name=['Sheet1', 'Sheet2'])

#  these are the hidden nodes and output node activated values
nodeOutput = []
finalOutput = 0

# these will hold the changing values for biases and weights
weightHidden = []  # all the weights from inputs to hidden nodes
biasHidden = []  # all biases from hidden nodes
weightOutput = []  # all weights from hidden nodes to the output
biasOutput = 0  # output bias

# these are the last measured weight changes
boDiff = 0
woDiff = []
bhDiff = []
whDiff = []

def standardise(raw, min, max):
    return ((raw-min)/(max-min))*0.8 + 0.1

def destandardise(stan, min, max):
    return ((stan -0.1)/0.8)*(max-min) + min

def generate_weights(n):
    weights = np.random.uniform(-2/inputNum, 2/inputNum, n)
    return weights

def sigmoid(x):
    return 1/(1+np.exp(-x))

def forward_pass(row):
    global nodeOutput, finalOutput
    nodeOutput.clear()
    finalOutput = biasOutput
    for node in range(hiddenNodes):
        # get activated outputs from all hidden nodes:
        wsum = biasHidden[node]
        for predictor in range(inputNum):
            wsum = wsum + (stanData[row][predictor] * weightHidden[predictor][node])
        activated = sigmoid(wsum)
        nodeOutput.append(activated)

        # get activated output of the output node
        finalOutput = finalOutput + (activated * weightOutput[node])
    finalOutput = sigmoid(finalOutput)
    # print(nodeOutput)
    # print(finalOutput)

def backward_pass(row):
    global weightHidden, biasHidden, weightOutput, biasOutput, boDiff, woDiff, bhDiff, whDiff
    correct = stanData[row][inputNum]
    deltaOutput = (correct - finalOutput)*(finalOutput*(1-finalOutput))
    delta = []
    for node in range(hiddenNodes):
        delta.append(weightOutput[node] * deltaOutput * (nodeOutput[node] * (1-nodeOutput[node])))

    temp = biasOutput
    biasOutput = biasOutput + (lr * deltaOutput) + (0.9*boDiff) # update output bias
    boDiff = biasOutput - temp

    for node in range(hiddenNodes):  # update weights going to the output and hidden node biases
        temp = weightOutput[node]
        weightOutput[node] = weightOutput[node] + (lr * deltaOutput * nodeOutput[node]) + (0.9*woDiff[node])
        woDiff[node] = weightOutput[node] - temp

        temp = biasHidden[node]
        biasHidden[node] = biasHidden[node] + (lr * delta[node]) + (0.9*bhDiff[node])
        bhDiff[node] = biasHidden[node] - temp

        for predictor in range(inputNum):
            temp = weightHidden[predictor][node]
            weightHidden[predictor][node] = weightHidden[predictor][node] + (lr * delta[node] * stanData[row][predictor]) + (0.9*whDiff[predictor][node])
            whDiff[predictor][node] = weightHidden[predictor][node] - temp

def get_error(row):
    observed = df[sheet].iloc[row][inputNum]
    modelled = destandardise(finalOutput, min[inputNum], max[inputNum])
    return np.square(modelled - observed)



if __name__ == '__main__':
    # inputNum = int(input("How many inputs?"))
    hiddenNodes = int(input("How many hidden nodes?"))
    epochs = int(input("How many epochs?"))
    lrstart = float(input("What is the starting learning rate?"))
    lrend = float(input("What is the ending learning rate?"))
    records = len(df[sheet])
    fields = len(df[sheet].columns)
    inputNum = fields - 1
    lr = 0.1  # just a default value

    dataset = []
    for i in range(records):
        dataset.append(i)
    np.random.shuffle(dataset)
    trainSet = dataset[:int(records / 10 * 6)]
    valSet = dataset[int(records / 10 * 6):int(records / 10 * 8)]
    testSet = dataset[int(records / 10 * 8):]
    # print(dataset)
    # print(len(dataset))
    # print(trainSet)
    # print(len(trainSet))
    # print(valSet)
    # print(len(valSet))
    # print(testSet)
    # print(len(testSet))

    # here we do our standardisation
    min = []
    max = []
    trainval = trainSet + valSet
    # print(trainval)
    for col in range(len(df[sheet].columns)):
        min.append(1000)
        max.append(-1000)
        for row in trainval:
            if df[sheet].iloc[row][col] > max[col]:  # updates max
                max[col] = df[sheet].iloc[row][col]
            if df[sheet].iloc[row][col] < min[col]:  # updates min
                min[col] = df[sheet].iloc[row][col]
    # print(min)
    # print(max)

    # now with our predictor mins and maxs of the validation set we can standardise each column of the testing set
    stanData = []
    for row in range(records):
        stanRecord = []
        for col in range(fields):
            stan = standardise(df[sheet].iloc[row][col], min[col], max[col])
            stanRecord.append(stan)
            # print(df['Sheet1'].iloc[row][col])
        stanData.append(stanRecord)
        #print(stanRecord)
    # print(stanData)

    # begin training

    # randomise initial weights and biases AND set default weight changes for momentum
    weightHidden = []
    for inp in range(inputNum):
        weightHidden.append(generate_weights(hiddenNodes))
        whDiff.append(hiddenNodes*[0])
    biasHidden = generate_weights(hiddenNodes)
    bhDiff += hiddenNodes*[0]
    weightOutput = generate_weights(hiddenNodes)
    woDiff += hiddenNodes*[0]
    biasOutput = generate_weights(1)[0]
    boDiff = 0

    # for x in weightHidden:
    #     print(x)
    # print(biasHidden)
    # print(weightOutput)
    # print(biasOutput)
    # print(df['Sheet1'])

    #do initial epoch before first error reading
    for row in trainSet:
        forward_pass(row)
        backward_pass(row)

    print("TRAINING SET")
    trainAcc = []
    valAcc = []
    for cycle in range(epochs):
        print(lr)
        lr = lrend + (lrstart - lrend)*(1-(1/(1+np.exp(10-(20*cycle/epochs)))))

        rmse = 0
        for row in trainSet:
            forward_pass(row)
            rmse = rmse + get_error(row)
            backward_pass(row)
        rmse = np.sqrt(rmse/len(trainSet))
        trainAcc.append(rmse)
        print(rmse)

        rmse = 0
        for row in valSet:
            forward_pass(row)
            rmse = rmse + get_error(row)
            # backward_pass(row)
        rmse = np.sqrt(rmse / len(valSet))
        valAcc.append(rmse)
        print(rmse)


    # weightHidden = []
    # for inp in range(inputNum):
    #     weightHidden.append(generate_weights(hiddenNodes))
    # biasHidden = generate_weights(hiddenNodes)
    # weightOutput = generate_weights(hiddenNodes)
    # biasOutput = generate_weights(1)[0]


    # for row in valSet:
    #     forward_pass(row)
    #     backward_pass(row)

    # print("VALIDATION SET")
    # valAcc = []
    # for cycle in range(epochs):
        # rmse = 0
        # for row in valSet:
        #     forward_pass(row)
        #     rmse = rmse + get_error(row)
        #     backward_pass(row)
        # rmse = np.sqrt(rmse / len(valSet))
        # valAcc.append(rmse)
        # print(rmse)
        # for x in weightHidden:
        #     print(x)
        # print(biasHidden)
        # print(weightOutput)
        # print(biasOutput)
        # print(finalOutput)  # last rows modelled output
        # print(rmse)
        #lrow = row
    # print("observed:")
    # (stanData[lrow][fields-1])  # last rows observed output

    plt.plot(range(1, epochs+1), trainAcc, 'g', label='Training')
    plt.plot(range(1, epochs + 1), valAcc, 'b', label='Validation')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



