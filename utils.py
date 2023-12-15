def accuracy_score(y_true, y_pred):
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length."
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true)
    return accuracy
def preprocess(data):
    X = data['评论'].tolist()
    y_dict = {0: 0,  4: 1}
    y = data['情绪'].map(y_dict).tolist()
    return X,y