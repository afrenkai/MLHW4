import numpy as np
import matplotlib.pyplot as plt
IM_WIDTH = 48
NUM_INPUT = IM_WIDTH**2
NUM_HIDDEN = 20
NUM_OUTPUT = 1

def relu (z):
    return np.maximum(0, z)


def forward_prop (x, y, W1, b1, W2, b2):
    # TODO: finish me
    z = W1@x + b1[:, np.newaxis]
    h = relu(z)
    yhat = W2 @ h + b2
    loss = 0.5 * np.mean((yhat - y) ** 2)
    return loss, x, z, h, yhat

def relu_prime(z):
    return (z>0).astype(float)

def back_prop (X, y, W1, b1, W2, b2):
    n = X.shape[1]
    print(W2.shape)
    print(W1.shape)
    loss, x, z, h, yhat = forward_prop(X, y, W1, b1, W2, b2)
    print(f"yhat shape: {yhat.shape}, y shape: {y.shape}")
    gradyhat = (yhat - y) / n
    gradW2 = gradyhat @ h.T
    gradb2 = np.sum(gradyhat, axis = 1, keepdims = True)
    print(gradyhat.shape)
    g_t = (gradyhat@W2.T) * relu_prime(z.T)
    gradW1 = g_t.T @ x.T
    gradb1 = g_t
    return gradW1, gradb1, gradW2, gradb2

def train (trainX, trainY, W1, b1, W2, b2, testX, testY, epsilon = 1e-2, batchSize = 64, numEpochs = 1000):
    n = trainX.shape[1]
    print(trainY)

    for epoch in range(numEpochs):
        indices = np.random.permutation(n)
        trainX, trainY = trainX[:, indices], trainY[:, indices]

        for i in range(0, n, batchSize):
            batchX = trainX[:, i:i+batchSize]
            batchY = trainY[:, i:i+batchSize]

            gradW1, gradb1, gradW2, gradb2 = back_prop(batchX, batchY, W1, b1, W2, b2)
            W1 -= epsilon * gradW1
            b1 -= epsilon * gradb1.squeeze()
            W2 -= epsilon * gradW2
            b2 -= epsilon * gradb2.squeeze

            if epoch % 100 == 0:
                loss, _, _, _, _ = forward_prop(testX, testY, W1, b1, W2, b2)
                print(f'epoch: {epoch}: Loss: {loss}')
    # TODO: finish me
    return W1, b1, W2, b2

def show_weight_vectors (W1):
    # Show weight vectors in groups of 5.
    for i in range(NUM_HIDDEN//5):
        plt.imshow(np.hstack([ np.pad(np.reshape(W1[idx,:], [ IM_WIDTH, IM_WIDTH ]), 2, mode='constant') for idx in range(i*5, (i+1)*5) ]), cmap='gray'), plt.show()
    plt.show()

def loadData (which, mu = None):
    images = np.load("age_regression_X{}.npy".format(which)).reshape(-1, 48**2).T
    labels = np.load("age_regression_y{}.npy".format(which))

    if which == "tr":
        mu = np.mean(images)

    # TODO: you may wish to perform data augmentation (e.g., left-right flipping, adding Gaussian noise).

    return images - mu, labels, mu

def checkGradient():
    testW1 = np.load("testW1.npy")
    testb1 = np.load("testb1.npy")
    testW2 = np.load("testW2.npy")
    testb2 = np.load("testb2.npy")
    oneSampleX = np.load("oneSampleX.npy")
    oneSampley = np.load("oneSampley.npy")
    gradW1, gradb1, gradW2, gradb2 = back_prop(np.atleast_2d(oneSampleX).T, oneSampley, testW1, testb1, testW2, testb2)
    correctGradW1 = np.load("correctGradW1OnSample.npy")
    correctGradb1 = np.load("correctGradb1OnSample.npy")
    correctGradW2 = np.load("correctGradW2OnSample.npy")
    correctGradb2 = np.load("correctGradb2OnSample.npy")
    # The differences should all be <1e-5
    assert (np.sum(np.abs(gradW1 - correctGradW1))) <= 1e-5
    assert (np.sum(np.abs(gradb1 - correctGradb1))) <= 1e-5
    assert (np.sum(np.abs(gradW2 - correctGradW2))) <= 1e-5
    assert (np.sum(np.abs(gradb2 - correctGradb2))) <= 1e-5
    print('all test cases passed')


 
if __name__ == "__main__":
    # Load data
    
    trainX, trainY, mu = loadData("tr")
    testX, testY, _ = loadData("te", mu)

    # Check the gradient value for correctness.
    # Note: the gradients shown below assume 20 hidden units.
    checkGradient()

    # Initialize weights to reasonable random values
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = np.mean(trainY)

    # Train NN
    W1, b1, W2, b2 = train(trainX, trainY, W1, b1, W2, b2, testX, testY)
