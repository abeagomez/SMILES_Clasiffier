import utils
import numpy as np
import nn_utils as nn
from keras.preprocessing import sequence
np.random.seed(6)

"""
This script trains and shows the results of our last proposal of network
for every target
"""
for i in range(12):
    target_index = i
    print("OUTPUT FOR TARGET: %d" %(i+1))
    smiles = utils.get_smiles_as_vectors()
    original_targets = utils.get_filled_targets()

    target = [target[target_index] for target in original_targets]
    targets_vector = [[target[i] for i in range(
        len(target)) if i != target_index] for target in original_targets]
    smiles = [smiles[i] + targets_vector[i] for i in range(len(smiles))]
    x_end = smiles.copy()
    y_end = target.copy()
    pos_s = utils.get_positive_samples(target_index, smiles, original_targets)
    x_train, y = utils.over_sampling_data_set(0, pos_s, smiles, target)

    x_train, x_test, y_train, y_test = nn.training_data(x_train, y)
    print("Train")
    model = nn.build_cnn_model()
    history = model.fit(x_train, y_train,
                        epochs=10,
                        validation_data=(x_test, y_test),
                        verbose=1)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    predicted = utils.get_target_values(model.predict(x_test, verbose=1))
    expected = utils.get_target_values(y_test)

    t_p, t_n, f_p, f_n = utils.evaluation_variables(predicted, expected)

    print("")
    print("Accuracy: %f" % ((t_p + t_n)/(t_p + t_n + f_p + f_n)))
    print("Precision: %f" % (t_p/(t_p + f_p)))
    print("Recall: %f" % (t_p/(t_p + f_n)))

    print("")
    print("EVALUATING THE ORIGINAL DATASET")
    max_len = len(max(x_end, key=len))
    x_train = sequence.pad_sequences(x_end, maxlen=max_len)
    x = [[[float(i)] for i in x] for x in x_train]
    x = np.array(x)
    x = x.reshape(x.shape[0], max_len, 1)
    predicted = utils.get_target_values(model.predict(x, verbose=1))

    t_p, t_n, f_p, f_n = utils.evaluation_variables(predicted, y_end)

    print("")
    print("Accuracy: %f" % ((t_p + t_n)/(t_p + t_n + f_p + f_n)))
    print("Precision: %f" % (t_p/(t_p + f_p)))
    print("Recall: %f" % (t_p/(t_p + f_n)))
    print("")


