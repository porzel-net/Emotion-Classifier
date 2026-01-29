##########################################IsabelaLabus##############################
def hold_out_split (inputs, labels, test_size = 0.2):
    
    dataset_size = len(inputs)

    split = int (dataset_size * (1.0 - test_size))

    inputs_train = inputs[:split]
    labels_train = labels[:split]

    inputs_test = inputs[split:]
    labels_test = labels[split:]

    return inputs_train, inputs_test, labels_train, labels_test


def accuracy (true_labels, pred_labels):

    correct_count = 0.0

    total_samples = len(true_labels)

    for i in range (total_samples):

        if true_labels[i] == pred_labels[i]:
            correct_count = correct_count + 1.0

    return correct_count / total_samples


def macro_precision(true_labels, pred_labels, number_emotions):

    total_precision = 0.0

    for emotion in range (number_emotions):

        true_positives = 0.0
        false_positives = 0.0

        for i in range(len(true_labels)):
            if pred_labels[i] == emotion:
                if true_labels[i] == emotion:
                    true_positives = true_positives + 1.0
                else:
                    false_positives = false_positives + 1.0

        if true_positives + false_positives == 0.0:
            precision_emotion = 0.0
        else:
            precision_emotion = true_positives / (true_positives + false_positives)

        total_precision = total_precision + precision_emotion

    return total_precision / number_emotions

def macro_recall (true_labels, pred_labels, number_emotions):

    total_recall = 0.0

    for emotion in range (number_emotions):
        true_positives = 0.0
        false_negatives = 0.0

        for i in range (len (true_labels)):
            if true_labels[i] == emotion:
                if pred_labels[i] == emotion:
                    true_positives = true_positives + 1.0
                else:
                    false_negatives = false_negatives + 1.0

        if true_positives + false_negatives == 0.0:
            recall_classes = 0.0

        else:
            recall_classes = true_positives / (true_positives + false_negatives)
        total_recall = total_recall + recall_classes

    return total_recall / number_emotions

def macro_f1 (true_labels, pred_labels, number_emotions):

    total_f1 = 0.0

    for emotion in range (number_emotions):

        true_positives = 0.0
        false_positives = 0.0
        false_negatives = 0.0

        for i in range(len(true_labels)):
            if pred_labels[i] == emotion and true_labels[i] == emotion:
                true_positives = true_positives + 1.0
            elif pred_labels[i] == emotion and true_labels[i] != emotion:
                false_positives = false_positives + 1.0
            elif pred_labels[i] != emotion and true_labels[i] == emotion:
                false_negatives = false_negatives + 1.0

        if true_positives + false_positives == 0.0:
            precision  = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0.0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0.0:
            classes_f1 = 0.0
        else:
            classes_f1 = 2.0 * precision * recall / (precision + recall)

        total_f1 = total_f1 + classes_f1

    return total_f1 / number_emotions

#######################################IsabelaLabus###########################################

