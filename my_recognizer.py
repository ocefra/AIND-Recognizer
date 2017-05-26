import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]

    Clarification:
    cf. https://discussions.udacity.com/t/recognizer-implementation/234793/22
    len(guesses) = len(probabilities) = number of tokens to classify, i.e. number of tokens in test_set
    But len(probabilities[0]) = len(probabilities[1]) = ... =
            = len(probabilities[num_tokens-1]) = number of words (i.e. types) in the vocabulary
        """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # For every token index in `test_set`
    for token_idx in range(test_set.num_items):
        # GET PROBABILITIES FOR CLASSES
        probas = dict()
        # For every class (i.e. `word_type` in `models.keys()`)
        for word_type in models.keys():
            # Get the token's data
            X, lengths = test_set.get_item_Xlengths(token_idx)
            model = models[word_type]
            try:
                # Compute the logL of the token under that word type's model.
                logL = model.score(X, lengths)
            except:
                # If model could not be scored:
                logL = float("-inf")
            # Store the logL in `probabilities`
            probas[word_type] = logL
        probabilities.append(probas)

        # GET THE MOST LIKELY CLASS
        # Store the word with max logL `guesses`.
        max_logL = max(probas.values())
        for word_type, logL in probas.items():
            if logL == max_logL:
                guesses.append(word_type)
                break
    # Return probabilities, guesses
    return probabilities, guesses

