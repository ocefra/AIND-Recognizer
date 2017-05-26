#! /home/gizmo/anaconda3/envs/aind/bin/python
import warnings
import numpy as np
from my_model_selector import SelectorBIC, SelectorDIC, SelectorCV
from my_recognizer import recognize
from asl_utils import show_errors
from asl_data import AslDb
from hmmlearn.hmm import GaussianHMM

asl = AslDb() # initializes the database

# Make 'ground' features
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']


# Normalised features
features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
features_to_norm = ['right-x', 'right-y', 'left-x', 'left-y']

df_means = asl.df.groupby('speaker').mean()
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
df_std = asl.df.groupby('speaker').std()

def make_aux_cols(feature_name):
    """Add columns for mean and std to database."""
    mean_col_name = feature_name + "-mean"
    std_col_name = feature_name + "-std"
    asl.df[mean_col_name] =         asl.df['speaker'].map(df_means[feature_name])
    asl.df[std_col_name] = asl.df['speaker'].map(df_std[feature_name])

for feature in features_to_norm:
    make_aux_cols(feature)

def normalise_feature(feature_name, normalised_feature_name):
    mean_col_name = feature_name + "-mean"
    std_col_name = feature_name + "-std"
    asl.df[normalised_feature_name] = (asl.df[feature_name] - asl.df[mean_col_name]) / asl.df[std_col_name]

for i in range(len(features_to_norm)):
    normalise_feature(features_to_norm[i], features_norm[i])


# #### Polar coordinates

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

def make_polar_coords(side):
    side_initial = side[0]
    r_col_name = 'polar-' + side_initial + 'r'
    r_theta_name = 'polar-' + side_initial + 'theta'
    x_col_name = 'grnd-' + side_initial + 'x'
    y_col_name = 'grnd-' + side_initial + 'y'
    asl.df[r_col_name] = np.sqrt(np.square(asl.df[x_col_name]) + np.square(asl.df[y_col_name]))
    # Swap x and y (cf. instructions)
    asl.df[r_theta_name] = np.arctan2(asl.df[x_col_name], asl.df[y_col_name])

for side in ['left', 'right']:
    make_polar_coords(side)


# #### Delta features

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

def make_delta(side, coord):
    side_initial = side[0]
    col_name = side + '-' + coord
    #col_name = 'norm' + '-' + side_initial + coord # on normalised version
    delta_col_name = 'delta-' + side_initial + coord
    asl.df[delta_col_name] = asl.df[col_name].diff().fillna(value=0)

for side in ['left', 'right']:
    for coord in ['x', 'y']:
        make_delta(side, coord)


# #### Custom features

features_custom_1 = ['rx-lx', 'ry-ly']
features_custom_2 = ['scaled-rx-lx', 'scaled-ry-ly']
features_custom_3 = ['delta-rx-lx', 'delta-ry-ly',
                     'delta-nose-x', 'delta-nose-y']
features_custom_4 = ['scaled-delta-rx-lx', 'scaled-delta-ry-ly',
                     'scaled-delta-nose-x', 'scaled-delta-nose-y']
features_custom_5 = ['scaled-delta-rx', 'scaled-delta-ry',
                     'scaled-delta-lx', 'scaled-delta-ly']
features_custom_6 = ['norm-polar-rr', 'norm-polar-rtheta',
                     'norm-polar-lr', 'norm-polar-ltheta']

def make_hands_diff(coord):
    """Make column for difference between right and left 'coord' (x or y)."""
    diff_col_name = 'r' + coord + '-l' + coord
    right_col_name = 'right-' + coord
    left_col_name = 'left-' + coord
    asl.df[diff_col_name] = asl.df[right_col_name] - asl.df[left_col_name]

for coord in ['x', 'y']:
    make_hands_diff(coord)

def make_delta(feature):
    delta_col_name = 'delta-' + feature
    asl.df[delta_col_name] = asl.df[feature].diff().fillna(value=0)

for feat in ['rx-lx', 'ry-ly', 'nose-x', 'nose-y']:
    make_delta(feat)

def scale_feature(feature_name):
    scaled_feature_name = 'scaled-' + feature_name
    feat = asl.df[feature_name]
    feat_min = feat.min()
    feat_max = feat.max()
    asl.df[scaled_feature_name] = (feat - feat_min) / (feat_max - feat_min)

all_features = asl.df.columns.values.tolist()
delta_features = [feat for feat in all_features if feat.startswith('delta')]

for delta_feature in delta_features:
    scale_feature(delta_feature)

for feature in ['rx-lx', 'ry-ly']:
    scale_feature(feature)

# Recompute means and std dfs to include the new features.
df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()

features_to_norm = [f for f in list(asl.df) if f.startswith('polar') \
                    and not f.endswith('mean') and not f.endswith('std')]
# features_norm is taken and must not be reassigned because it is used in the test
features_norm_2 = ['norm-' + f for f in features_to_norm]

def make_aux_cols(feature_name):
    mean_col_name = feature_name + "-mean"
    std_col_name = feature_name + "-std"
    asl.df[mean_col_name] = asl.df['speaker'].map(df_means[feature_name])
    asl.df[std_col_name] = asl.df['speaker'].map(df_std[feature_name])

for feature in features_to_norm:
    make_aux_cols(feature)

for i in range(len(features_to_norm)):
    normalise_feature(features_to_norm[i], features_norm_2[i])






# ### Recognizer
# ##### Train the full training set

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                               n_constant=3).select()
        model_dict[word]=model
    return model_dict


# ### Recogniser implementation: experiments

features_custom_names = ['custom_' + str(i) for i in range(1, 7)]
features_other_names = ['ground', 'polar', 'delta', 'norm']
features_all_names = features_other_names + features_custom_names
features_for_experiments = ['features_' + name for name in all_names]

model_selectors = ['SelectorCV', 'SelectorBIC', 'SelectorDIC']

# Keep track of WER for all feature-selector combinations.
wer_all = {}

start = timeit.default_timer()

for features_name in features_for_experiments:
    print("===============")
    print(features_name, end=" --- ")
    features = eval(features_name) # get the actual object
    print("Building test set...")
    test_set = asl.build_test(features)
    print("Number of test set items: {}".format(test_set.num_items))
    print("Number of test set sentences: {}".format(len(test_set.sentences_index)))

    for model_selector_name in model_selectors:
        print("  " + model_selector_name)
        # Get the actual object
        model_selector = eval(model_selector_name)
        # Initialise dictionary for results
        wer_all[features_name] = {}
        # Learn model on training data
        print("    Training word models...")
        models = train_all_words(features, model_selector)
        # Recognise on test data
        print("    Evaluating models...")
        probabilities, guesses = recognize(models, test_set)
        # Evaluate performance on test data
        wer = show_errors(guesses, test_set)
        print("    WER {}".format(wer))
        wer_all[features_name][model_selector_name] = wer
        print()

end = timeit.default_timer()
total_time = end - start
print("total time: {}".format(total_time))

# NOTE: I have modified the show_errors function so that now, in addition to displaying the recognised text alongside the correct one, it also returns a value, namely the WER. I needed this for my experiments, in order to compare my feature-selector combinations.

