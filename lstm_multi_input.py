import utils
import nn_utils as nn

smiles = utils.get_smiles_as_vectors()
original_targets = utils.get_filled_targets()
target_index = 0
target = [target[target_index] for target in original_targets]
targets_vector = [[target[i] for i in range(
    len(target)) if i != target_index] for target in original_targets]
smiles = [smiles[i] + targets_vector[i] for i in range(len(smiles))]
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

nn.show_metrics(history)

predicted = utils.get_target_values(model.predict(x_test, verbose=1))
expected = utils.get_target_values(y_test)

utils.evaluation_variables(predicted, expected)


