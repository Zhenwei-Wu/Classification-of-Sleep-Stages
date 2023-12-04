# Define the model function and softmax cost function here
def model(X, W, W_c):
    # Apply shared nonlinear transformation
    f_X = np.tanh(np.dot(X, W))
    # Apply class-specific linear combination
    return (np.dot(f_X, W_c[:-1]) + W_c[-1])  # Use W_c[-1] as bias for W_c


# Define the softmax cost function
def softmax_cost(W_combined, X, y, num_features, num_classes, num_units, class_weights):
    
    # Corrected reshaping of weights including bias terms
    W = W_combined[:num_features * num_units + num_units].reshape((num_features + 1), num_units)  # +1 for bias term in W
    W_c = W_combined[num_features * num_units + num_units:].reshape((num_units + 1), num_classes)  # +1 for bias term in W_c
    
    # Compute the model predictions
    f_X = np.tanh(np.dot(X, W))  # X already includes the bias term
    
    # Apply softmax with log smoothing
    logits = np.dot(f_X, W_c[:-1]) + W_c[-1]
    preds = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    preds /= np.sum(preds, axis=1, keepdims=True)
    preds = np.clip(preds, 1e-9, 1 - 1e-9)  # Prevent 0 or 1 in predictions
    log_preds = np.log(preds[np.arange(len(y)), y])
    
    # Apply class weights
    weighted_log_preds = class_weights[y] * np.log(preds[np.arange(len(y)), y])
    cost = -np.sum(weighted_log_preds) / len(y)
    return cost