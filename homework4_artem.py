# pyright: reportMissingImports = false
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
IM_WIDTH = 48
NUM_INPUT = IM_WIDTH**2
NUM_HIDDEN = 20
NUM_OUTPUT = 1

def relu (z):
    return np.maximum(0, z)

def half_mse(yhat, y, n, W1, W2, alpha):
    loss = 0.5 * np.mean((y - yhat) ** 2)
    reg = (alpha /2) * (np.sum(W1**2) + np.sum(W2**2))
    return loss + reg
def forward_prop (x, y, W1, b1, W2, b2, alpha):
    #print(f'W1 shape: {W1.shape}')
    #print(f'b1 shape: {b1.shape}')
    #print(f'W2 shape: {W2.shape}')
    #print(f'b2 shape: {b2.shape}')
    n = x.shape[1]
    z = W1@x + b1[:, np.newaxis]
    h = relu(z)
    yhat = W2 @ h + b2
    loss = half_mse(yhat, y, n, W1, W2, alpha)
    return loss, x, z, h, yhat

def back_prop (X, y, W1, b1, W2, b2, alpha):
    n = X.shape[1]
    #print(y.shape)
    loss, x, z, h, yhat = forward_prop(X, y, W1, b1, W2, b2, alpha)
    dfdyhat = (yhat - y)
    #print(f'h shape: {h.shape}')
    gradW2 = (dfdyhat @ h.T) / n + alpha * W2
    gradb2 = np.sum(dfdyhat, axis = 1)/n
    g = (W2.T @ (yhat-y) ) * (z > 0)
    gradW1 = (g @ x.T)/n + alpha * W1
    gradb1 = np.sum(g, axis =1)/n
    #print(f'gradb1 shape {b1.shape}')
    return gradW1, gradb1, gradW2, gradb2, loss


def train (trainX, trainY, W1, b1, W2, b2, epsilon = 1e-4, batchSize = 64, numEpochs = 1000, alpha = 1e-4, clipval = 1.0):
    n = trainX.shape[1]
    losses = []
    for epoch in tqdm(range(numEpochs)):
        epoch_loss = 0
        indices = np.random.permutation(n) 
        trainX, trainY = trainX[:, indices], trainY[:, indices]
        for i in range(0, n, batchSize):
            batchX = trainX[:, i:i+batchSize]
            batchY = trainY[:, i:i+batchSize]
            gradW1, gradb1, gradW2, gradb2, loss = back_prop(batchX, batchY, W1, b1, W2, b2, alpha)
            gradW1 = np.clip(gradW1, -clipval, clipval)
            gradb1 = np.clip(gradb1, -clipval, clipval)
            gradW2 = np.clip(gradW2, -clipval, clipval)
            gradb2 = np.clip(gradb2, -clipval, clipval)
            epoch_loss += loss / (n // batchSize)
            W1 -= epsilon * gradW1
            b1 -= epsilon * gradb1
            W2 -= epsilon * gradW2
            b2 -= epsilon * gradb2
            
        losses.append(epoch_loss)
        # if epoch % 100 == 0:
        #   print(f' losses of all x batches in this epoch: {losses[epoch]}')
        #
        print(f'Epoch: {epoch + 1}/{numEpochs}, Loss: {epoch_loss}')
    return W1, b1, W2, b2


def tune_params(trainX, trainY, testX, testY, lrs, bs, hu, numEpochs, alphas):
    best_loss = float('inf')
    best_params = []
    bestW1, bestb1, bestW2, bestb2 = None, None, None, None

    for epsilon in lrs:
        for b in bs:
            for n in numEpochs:
                for h in hu:
                    for a in alphas:
                    # Initialize weights to reasonable random values
                        W1 = 2*(np.random.random(size=(h, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
                        b1 = 0.01 * np.ones(h)
                        W2 = 2*(np.random.random(size=(NUM_OUTPUT, h))/n**0.5) - 1./h**0.5
                        b2 = np.mean(trainY)
                    #train
                        W1_tr, b1_tr, W2_tr, b2_tr = train(trainX, trainY, W1, b1, W2, b2, epsilon, b, n, a)
                    #eval using built fwd pass
                        test_loss, _, _, _, _ = forward_prop(testX, testY, W1, b1, W2, b2, 0.)
                        if test_loss < best_loss:
                            print(f'new best loss {test_loss}, obtained w params: {b} batch size, {h} hidden units, {epsilon} learning rate, {a} regularization and {n} epochs')
                            best_loss = test_loss
                            best_params = [epsilon, b, n,a,h]
                            bestW1, bestb1, bestW2, bestb2 = W1_tr, b1_tr, W2_tr, b2_tr
    
    bp = {'Learning Rate': best_params[0],
          'Batch Size': best_params[1],
          'Num Epochs': best_params[2],
          'Regularization Strength': best_params[3],
          'Hidden Size': best_params[4],
          }
    return bp, best_loss, bestW1, bestb1, bestW2, bestb2

def show_weight_vectors (W1, h):
    # Show weight vectors in groups of 5.
    for i in range(h//5):
        plt.imshow(np.hstack([ np.pad(np.reshape(W1[idx,:], [ IM_WIDTH, IM_WIDTH ]), 2, mode='constant') for idx in range(i*5, (i+1)*5) ]), cmap='gray'), plt.show()
    plt.show()

def loadData (which, mu = None,aug = True):
    images = np.load("age_regression_X{}.npy".format(which)).reshape(-1, 48**2).T
    labels = np.load("age_regression_y{}.npy".format(which)).reshape(1, -1)

    if which == "tr":
        mu = np.mean(images)
    images = images-mu

    
    # TODO: you may wish to perform data augmentation (e.g., left-right flipping, adding Gaussian noise).
    #
    #
    if aug and which == 'tr':
        n = images.shape[1]
        imgflip = images.reshape(IM_WIDTH, IM_WIDTH, -1)[:, ::-1, :].reshape(IM_WIDTH **2, -1)
        flip_idx = np.random.choice(n, size = n//2, replace = False)
        images[:, flip_idx] = imgflip[:, flip_idx]
        noise = np.random.normal(0, 1e-5, images.shape)
        images+= noise
    return images, labels, mu



def checkGradient():
    testW1 = np.load("testW1.npy")
    testb1 = np.load("testb1.npy")
    testW2 = np.load("testW2.npy")
    testb2 = np.load("testb2.npy")
    oneSampleX = np.load("oneSampleX.npy")
    oneSampley = np.load("oneSampley.npy")
    
    assert testW1.shape == (20, NUM_INPUT)
    #print('testW1 properly shaped')
    assert testb1.shape == (20,)
    #print('testb1 properly shaped')
    assert testW2.shape == (1, 20)
    #print('testW2 works properly')
    
    gradW1, gradb1, gradW2, gradb2, _= back_prop(np.atleast_2d(oneSampleX).T, oneSampley, testW1, testb1, testW2, testb2, alpha = 0.)
    correctGradW1 = np.load("correctGradW1OnSample.npy")
    correctGradb1 = np.load("correctGradb1OnSample.npy")
    correctGradW2 = np.load("correctGradW2OnSample.npy")
    correctGradb2 = np.load("correctGradb2OnSample.npy")
    # The differences should all be <1e-5
    
    assert (np.sum(np.abs(gradW1 - correctGradW1))) <= 1e-5
    #print(np.sum(np.abs(gradb1 - correctGradb1)))
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
    # W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    # b1 = 0.01 * np.ones(NUM_HIDDEN)
    # W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    # b2 = np.mean(trainY)

    # Train NN
    #W1, b1, W2, b2 = train(trainX, trainY, W1, b1, W2, b2, testX, testY)
    #loss, _, _, _, _ = forward_prop(testX, testY, W1, b1, W2, b2)
    #print(f'Test Loss for untuned model: {loss:.4f}')


    # declare param lists as provided in hw
    #lrs = [1e-3, 1e-4, 1e-5]
    #bs = [63, 128, 256]
    #hu = [10, 20, 30]
    #num_epochs = [100, 200]
    #alphas = [0, 1e-3, 1e-4]
    #best_params, best_loss, bestW1, bestb1, bestW2, bestb2 = tune_params(trainX, trainY, testX, testY, lrs, bs, hu, num_epochs, alphas)
    #print(f'best params: {best_params} best test loss: {best_loss}')


    # initialize weights to reasonable random values
    best_lr = 0.001
    best_bs = 256
    best_epoch = 200
    best_hidden = 30
    best_alpha = 0.0001
    W1 = 2*(np.random.random(size=(best_hidden, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(best_hidden)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, best_hidden))/best_hidden**0.5) - 1./best_hidden**0.5
    b2 = np.mean(trainY)


    # Train NN
    W1, b1, W2, b2 = train(trainX, trainY, W1, b1, W2, b2, best_lr, best_bs, best_epoch, best_alpha, 1.)
    loss, _, _, _, _ = forward_prop(testX, testY, W1, b1, W2, b2, alpha = best_alpha)
    print(f'Test Loss for model w good params: {loss:.4f}')
    show_weight_vectors(W1, best_hidden)


