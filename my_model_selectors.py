import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # number of features
        d = len(self.X[0])
        # number of observations
        N = len(self.X)
        lowest_bic, best_model = math.inf, None
        for n in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(num_states=n)
            try:
                logL = model.score(self.X, self.lengths)
                # number of parameters
                p = n * n + 2 * n * d - 1
                bic = -2 * logL + p * math.log(N)
                if bic < lowest_bic:
                    lowest_bic, best_model = bic, model
            except:
                pass
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    https://discussions.udacity.com/t/dic-criteria-clarification-of-understanding/233161/2
    We are trying to find the model that gives a high likelihood(small negative
    number) to the original word and low likelihood(very big negative number) to
    the other words. So the DIC score is

    DIC = log(P(original world)) - average(log(P(otherwords)))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        largest_dic, best_model = -math.inf, None

        #print("TRAINING WORD {}".format(self.this_word))
        for n in range(self.min_n_components, self.max_n_components+1):
            #print("    ====================")
            #print("    {} STATES".format(n))
            try:
                # Train model for current word.
                model = self.base_model(num_states=n)
                logL = model.score(self.X, self.lengths)
                #print("    MODEL TRAINED AND EVALUATED")
                
                # Evaluate the model (logL) on all other words and compute the average logL.
                # Initialise counts needed for computing the average.
                sum_other_logL = 0 
                num_other_scored = 0 # successfully trained and evaluated
                
                other_words = self.words.keys()
                for other_word in other_words:
                    other_word_X, other_word_lengths = self.hwords[other_word]
                    try:
                        sum_other_logL += model.score(other_word_X, other_word_lengths)
                        num_other_scored += 1
                        #print("    model evaluated on {}".format(other_word))
                    except:
                        #print("    model could not be evaluated on {}".format(other_word))
                        pass
                avg_other_logL = sum_other_logL / num_other_scored
                dic = logL - avg_other_logL
                #print("  DIC = {}".format(dic))
                #print()
                if dic > largest_dic:
                    largest_dic, best_model = dic, model
            except:
                pass
                #print("    MODEL COULD NOT BE TRAINED OR EVALUATED")
                #print()
        #print("BEST DIC: {}".format(largest_dic))
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        highest_avg_test_logL, best_n = -math.inf, None

        for n in range(self.min_n_components, self.max_n_components+1):
            # Split the training data into k folds.
            k = 3
            # Do not split if fewer observations than folds.
            if(len(self.sequences) < k):
                break
            split_method = KFold(n_splits=k, shuffle=False, random_state=self.random_state)
            # Initialise counts needed for computing the average logL on the test folds.
            sum_test_logL = 0 
            num_test_scored = 0 # successfully trained and scored
            # Make each fold the test fold in turn.
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                # Fit an HMM model on the training folds.
                try:
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state,
                                        verbose=False).fit(train_X, train_lengths)
                    # Score the model on the test fold.
                    sum_test_logL += model.score(test_X, test_lengths)
                    num_test_scored += 1
                except:
                    pass
            avg_test_logL = sum_test_logL / num_test_scored
            if avg_test_logL > highest_avg_test_logL:
                highest_avg_test_logL, best_n = avg_test_logL, n

        # Once we have our optimal number of states, train a model on the full
        # training set with that optimal number of states as a parameter, and
        # return that model. If no optimal number of states has been found,
        # simply return a model trained on the full training set with the
        # selector's constant number of states.
        return self.base_model(best_n) if best_n is not None else self.base_model(self.n_constant)

