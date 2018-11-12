import utils
import numpy as np
import nn_utils as nn
# fix random seed for reproducibility
np.random.seed(7)

"""
CHANGE THE TARGET INDEX HERE
The target_index is the way to indicate the target we want to predict.
The value of the target_index can be between 0 and 11
"""
target_index = 0

smiles = utils.get_smiles_as_vectors()
original_targets = utils.get_filled_targets()

target = [target[target_index] for target in original_targets]
pos_s = utils.get_positive_samples(target_index, smiles, original_targets)
x_train, y = utils.over_sampling_data_set(0, pos_s, smiles, target)

x_train, x_test, y_train, y_test = nn.training_data(x_train, y)
print("Train")
model = nn.build_lstm_model()
history= model.fit(x_train, y_train,
                epochs=10,
                validation_data=(x_test, y_test),
                verbose=1)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

nn.show_metrics(history)
