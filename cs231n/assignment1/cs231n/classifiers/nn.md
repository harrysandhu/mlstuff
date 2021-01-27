
    Wx -> y
    find W

a. svm, softmax
    1. FORWARD: find the loss,
    2. BACKWARD: grad on the loss function wrt parameters (how parameters effect loss)
    3. update parameters in the direction of negative grad to minimize loss
            

b. NN

    FORWARD: (X) ->  W1 (X)  ---softmax loss with Y1-->  |(Z)|  ---> W2(Z)  --softmax loss with Y2-> |L|
                                                                                                                                                  

    S = W2 max(0, W1x)   # 2 layer => learn  W1 and W2
    
    S = W3 (max (0, W2 (max (0, W1X))))   # layer



    We want an activation function.
    - sigmoid
    - tanh
    - relu
    - leaky relu

    then we want a loss function
    - 