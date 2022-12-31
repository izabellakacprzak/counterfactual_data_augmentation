from sklearn import metrics

def pretty_print_evaluation(y_pred, y_true, labels):
    confusion_matrix = get_confusion_matrix(y_pred, y_true)
    classification_report = get_classification_report(y_pred, y_true, labels)

    print("-------------------------------------")
    print("----------CONFUSION MATRIX-----------")
    print("-------------------------------------")
    print(confusion_matrix)
    print("\n")

    print("-------------------------------------")
    print("-------CLASSIFICATION METRICS--------")
    print("-------------------------------------")
    print(classification_report)
    print("\n")



def accuracy(confusion_matrix):
    return confusion_matrix.diagonal()/confusion_matrix.sum(axis=1)

def get_confusion_matrix(y_pred, y_true):
    return metrics.confusion_matrix(y_true, y_pred)

def get_classification_report(y_pred, y_true, labels):
    return metrics.classification_report(y_true, y_pred, digits=len(labels))