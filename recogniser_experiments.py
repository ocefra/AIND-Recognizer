#! /home/gizmo/anaconda3/envs/aind/bin/python

import numpy as np
# import pandas as pd
# import pickle
import cloudpickle
from asl_data import AslDb

import warnings
import timeit
# from hmmlearn.hmm import GaussianHMM
from my_model_selectors import SelectorCV, SelectorBIC, SelectorDIC
from my_recognizer import recognize
# from asl_utils import show_errors
from asl_utils import wer, write_errors_to_file

########################################
######## DATA PREPARATION ##############
########################################

# Initialise the database
print("Initialising database...")
asl = AslDb() 


# Make additional features

# N.B. The feature-making functions below may not be completely uniform because
# they were done as piecewise exercises, and only later collected here and
# adapted.


########################################
## 'GROUND' FEATURES

def make_ground_feature(side_initial, coord):
    side = 'right' if side_initial == 'r' else 'left'
    col_name = side + '-' + coord
    nose_col_name = 'nose-' + coord
    ground_col_name = 'grnd-' + side_initial + coord
    asl.df[ground_col_name] = asl.df[col_name] - asl.df[nose_col_name]

print("Making ground features...")
for side_initial in ['r', 'l']:
    for coord in ['x', 'y']:
        make_ground_feature(side_initial, coord)

# Collect the ground feature names into a list
features_ground = [f for f in list(asl.df) if f.startswith('grnd')]


########################################
## NORMALISED FEATURES

features_to_norm = [side + '-' + coord for side in ['right', 'left'] \
                                       for coord in ['x', 'y']]
features_norm = ['norm-' + side_initial + coord for side_initial in ['r', 'l'] \
                                       for coord in ['x', 'y']]

# Make dfs with means and std for all variables
df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()

def make_aux_cols(feature_name):
    """Add columns for mean and std of feature_name to database."""
    mean_col_name = feature_name + "-mean"
    std_col_name = feature_name + "-std"
    asl.df[mean_col_name] = asl.df['speaker'].map(df_means[feature_name])
    asl.df[std_col_name] = asl.df['speaker'].map(df_std[feature_name])

def normalise_feature(feature_name, norm_feature_name):
    mean_col_name = feature_name + "-mean"
    std_col_name = feature_name + "-std"
    asl.df[norm_feature_name] = (asl.df[feature_name] - asl.df[mean_col_name]) / asl.df[std_col_name]

print("Making normalised features...")
for i in range(len(features_to_norm)):
    make_aux_cols(features_to_norm[i])
    normalise_feature(features_to_norm[i], features_norm[i])


########################################
## POLAR COORDINATES

def make_polar_coords(side):
    side_initial = side[0]
    r_col_name = 'polar-' + side_initial + 'r'
    r_theta_name = 'polar-' + side_initial + 'theta'
    x_col_name = 'grnd-' + side_initial + 'x'
    y_col_name = 'grnd-' + side_initial + 'y'
    asl.df[r_col_name] = np.sqrt(np.square(asl.df[x_col_name]) + np.square(asl.df[y_col_name]))
    # Swap x and y (cf. project instructions)
    asl.df[r_theta_name] = np.arctan2(asl.df[x_col_name], asl.df[y_col_name])

print("Making polar coordinate features...")
for side in ['left', 'right']:
    make_polar_coords(side)

features_polar = [f for f in list(asl.df) if f.startswith('polar')]


########################################
## DELTA FEATURES

def make_delta(side, coord):
    side_initial = side[0]
    col_name = side + '-' + coord
    delta_col_name = 'delta-' + side_initial + coord
    asl.df[delta_col_name] = asl.df[col_name].diff().fillna(value=0)

print("Making delta features...")
for side in ['left', 'right']:
    for coord in ['x', 'y']:
        make_delta(side, coord)

features_delta = ['delta-' + side_initial + coord for side_initial in ['r', 'l'] \
                                                  for coord in ['x', 'y']]


########################################
## CUSTOM FEATURES

### Difference between coordinates of right and left hand in same frame
def make_hands_diff(coord):
    """Make column for difference between right and left 'coord' (x or y)."""
    diff_col_name = 'r' + coord + '-l' + coord
    right_col_name = 'right-' + coord
    left_col_name = 'left-' + coord
    asl.df[diff_col_name] = asl.df[right_col_name] - asl.df[left_col_name]

print("Making custom_1 features...")
for coord in ['x', 'y']:
    make_hands_diff(coord)

features_custom_1 = ['r' + coord + '-' + 'l' + coord for coord in ['x', 'y']]


### Delta between nose coordinates and between difference in hands coordinates, between consecutive frames
def make_delta(feature):
    delta_col_name = 'delta-' + feature
    asl.df[delta_col_name] = asl.df[feature].diff().fillna(value=0)

feat_for_delta = ['nose-' + coord for coord in ['x', 'y']] + features_custom_1

print("Making custom_2 features...")
for feat in feat_for_delta:
    make_delta(feat)

features_custom_2 = ['delta-' + f for f in feat_for_delta]


### Scaled version of original delta features 
def scale_feature(feature_name):
    scaled_feature_name = 'scaled-' + feature_name
    feat = asl.df[feature_name]
    feat_min = feat.min()
    feat_max = feat.max()
    asl.df[scaled_feature_name] = (feat - feat_min) / (feat_max - feat_min)

print("Making custom_3 features...")
for feat in features_delta:
    scale_feature(feat)

features_custom_3 = ['scaled-' + f for f in features_delta]


### Scaled version of difference between right and left hand coordinates, in same frame
print("Making custom_4 features...")
for feat in features_custom_1:
    scale_feature(feat)

features_custom_4 = ['scaled-' + f for f in features_custom_1]


### Scaled version of delta between nose coordinates and between difference in hands
# coordinates, between consecutive frames
print("Making custom_5 features...")
for feat in features_custom_2:
    scale_feature(feat)

features_custom_5 = ['scaled-' + f for f in features_custom_2]


### Normalised polar coordinates
# Recompute means and std dfs to include the new features
df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()

features_custom_6 = ['norm-' + f for f in features_polar]

print("Making custom_6 features...")
for i in range(len(features_polar)):
    make_aux_cols(features_polar[i])
    normalise_feature(features_polar[i], features_custom_6[i])


#######################################
# SAVE THE PROCESSED DATA
# Collect all feature name sets
suffixes = ['ground', 'delta', 'norm', 'polar']
all_feature_set_names = ['features_' + suffix for suffix in suffixes]
custom_feature_set_names = ['features_custom_' + str(i) for i in range(1, 7)]
all_feature_set_names += custom_feature_set_names
all_features = {f_set_name: eval(f_set_name) for f_set_name in all_feature_set_names}


# Save the data
print("Saving data...")
# pickle.dump(asl, open('asl.pickle', 'wb'))
# pickle.dump(all_features, open('features.pickle', 'wb'))
cloudpickle.dump(asl, open('asl_cloudpickled', 'wb'))
cloudpickle.dump(all_features, open('features_cloudpickled', 'wb'))



########################################
###### RECOGNISER EXPERIMENTS ##########
########################################

def train_all_words(features, model_selector):
    """Train on full training set."""
    training = asl.build_training(features)
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word, n_constant=3).select()
        model_dict[word]=model
    return model_dict


model_selectors = ['SelectorCV', 'SelectorBIC', 'SelectorDIC']

# Keep track of WER and training time for all feature-selector combinations
wer_all = {}
training_times_all = {}

# Experiments: try feature-selector combinations
print("Training and evaluating recogniser...")
start_all = timeit.default_timer()

for feature_set in all_features:
    print("===============")
    print("*** " + feature_set + " ***")
    features = all_features[feature_set] # get the actual list of feature names
    print("Building test set...")
    test_set = asl.build_test(features)
    print("Number of test set items: {}".format(test_set.num_items))
    print("Number of test set sentences: {}".format(len(test_set.sentences_index)))
    print("=====")

    for model_selector_name in model_selectors:
        start_model_selector = timeit.default_timer()
        print("  " + model_selector_name)
        
        # Get the actual object
        model_selector = eval(model_selector_name)

        # Initialise dictionaries for results and for time
        wer_all[feature_set] = {}
        training_times_all[feature_set] = {}

        # Learn model on training data
        print("    Training word models...")
        models = train_all_words(features, model_selector)
        end_model_selector = timeit.default_timer()
        time_model_selector = end_model_selector - start_model_selector
        print("        training time: {} seconds".format(time_model_selector))
        training_times_all[feature_set][model_selector_name] = time_model_selector

        # Recognise on test data
        print("    Evaluating models...")
        probabilities, guesses = recognize(models, test_set)

        # Evaluate performance on test data
        wer_this = wer(guesses, test_set)
        print("    WER {}".format(wer_this))
        wer_all[feature_set][model_selector_name] = wer_this

        # Write recognised and correct text to file
        result_file = 'recognition_' + feature_set + '_' + model_selector_name
        write_errors_to_file(guesses, test_set, result_file)
        print()

end_all = timeit.default_timer()
time_all = end_all - start_all
print("===============")
print("total time: {} seconds".format(time_all))

# pickle.dump(wer_all, open('wer_all.pickle', 'wb'))
# pickle.dump(training_times_all, open('training_times_all.pickle', 'wb'))
cloudpickle.dump(wer_all, open('wer_all_cloudpickled', 'wb'))
cloudpickle.dump(training_times_all, open('training_times_all_cloudpickled', 'wb'))

