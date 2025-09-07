def accuracy(y_true, y_pred):
    return (y_true == y_pred).float().mean().item()

def precision(y_true, y_pred):
    true_positives = ((y_true == 1) & (y_pred == 1)).sum().item()
    predicted_positives = (y_pred == 1).sum().item()
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def recall(y_true, y_pred):
    true_positives = ((y_true == 1) & (y_pred == 1)).sum().item()
    actual_positives = (y_true == 1).sum().item()
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0