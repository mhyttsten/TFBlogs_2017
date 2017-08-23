import tensorflow as tf
import collections
import os
import urllib
import sys

# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
print("TensorFlow version: {}".format(tf_version))
assert "1.3" <= tf_version, "TensorFlow r1.3 or later is needed"

# Windows users: You only need to change PATH, rest is platform independent
PATH = "/tmp/tf_dataset_and_estimator_apis"

# Load Training and Test dataset files
PATH_DATASET = PATH +os.sep + "dataset"
FILE_TRAIN =   PATH_DATASET + os.sep + "boston_train.csv"
FILE_TEST =    PATH_DATASET + os.sep + "boston_test.csv"
URL_TRAIN =   "http://download.tensorflow.org/data/boston_train.csv"
URL_TEST =    "http://download.tensorflow.org/data/boston_test.csv"
def downloadDataset(url, file):
    if not os.path.exists(PATH_DATASET):
        os.makedirs(PATH_DATASET)

    if not os.path.exists(file):
        data = urllib.urlopen(url).read()
        with open(file, "w") as f:
            f.write(data)
            f.close()

downloadDataset(URL_TRAIN, FILE_TRAIN)
downloadDataset(URL_TEST, FILE_TEST)

tf.logging.set_verbosity(tf.logging.INFO)

# The CSV fields in the files
dataset_fields = collections.OrderedDict([
    ('CrimeRate',           [0.]),
    ('LargeLotsRate',       [0.]),
    ('NonRetailBusRate',    [0.]),
    ('NitricOxidesRate',    [0.]),
    ('RoomsPerHouse',       [0.]),
    ('Older1940Rate',       [0.]),
    ('Dist2EmployeeCntr',   [0.]),
    ('PropertyTax',         [0.]),
    ('StudentTeacherRatio', [0.]),
    ('MarketValueIn10k',    [0.])
])

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def my_input_fn(file_path, repeat_count):
    def decode_csv(line):
        parsed = tf.decode_csv(line, list(dataset_fields.values()))
        return dict(zip(dataset_fields.keys(), parsed))

    dataset = (
        tf.contrib.data.TextLineDataset(file_path) # Read text line file
            .skip(1) # Skip header row
            .map(decode_csv) # Transform each elem by applying decode_csv fn
            .shuffle(buffer_size=1000) # Obs: buffer_size is read into memory
            .repeat(repeat_count) #
            .batch(128)) # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features = iterator.get_next()
    batch_labels = batch_features.pop('MarketValueIn10k')
    return batch_features,\
           batch_labels

feature_names = dataset_fields.copy() # Create a list of our input features
del feature_names['MarketValueIn10k'] # Remove the label field

# Create the feature_columns, which specifies the input to our model
# All our input features are numeric, so use numeric_column for each one
feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

# Create a deep neural network regression classifier
# Use the DNNRegressor canned estimator
classifier = tf.estimator.DNNRegressor(
    feature_columns=feature_columns, # The input features to our model
    hidden_units=[10, 10], # Two layers, each with 10 neurons
    model_dir=PATH) # Checkpoints etc are stored here

# Train our model, use the previously function my_input_fn
# Input to training is a file with training example
# Stop training after 2000 batches have been processed
classifier.train(
    input_fn=lambda: my_input_fn(FILE_TRAIN, None),
    steps=2000)

# Evaluate our model using the examples contained in FILE_TEST
# Return value will contain evaluation_metrics such as: loss & average_loss
evaluate_result = classifier.evaluate(
    input_fn=lambda: my_input_fn(FILE_TEST, None),
    steps=2000)
print("Evaluation results")
for key in evaluate_result:
    print("   {}, was: {}".format(key, evaluate_result[key]))

# Model evaluated, now use it to predict some house prices
# Let's predict the examples in FILE_TEST
predict_result = classifier.predict(
    input_fn=lambda: my_input_fn(FILE_TEST, 1))
for x in predict_result:
    print x
