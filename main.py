import nltk
from nltk.corpus import gutenberg
from sklearn.model_selection import StratifiedShuffleSplit
from nltk import classify, DecisionTreeClassifier
from collections import Counter
import spacy
from tools import *
nltk.download('gutenberg')
nltk.download('punkt')
nlp = spacy.load('en_core_web_md')

author1_train, author1_test = gutenberg.sents('austen-emma.txt') + gutenberg.sents('austen-persuasion.txt'), gutenberg.sents('austen-sense.txt')
author2_train, author2_test = gutenberg.sents('shakespeare-caesar.txt') + gutenberg.sents('shakespeare-hamlet.txt'), gutenberg.sents('shakespeare-macbeth.txt')


all_sents = [(sent, "austen") for sent in author1_train] + [(sent, "shakespeare") for sent in author2_train]

values = [author for (sent, author) in all_sents]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_train_set = []
strat_pretest_set = []
for train_index, pretest_index in split.split(all_sents, values):
    strat_train_set = [all_sents[index] for index in train_index]
    strat_pretest_set = [all_sents[index] for index in pretest_index]

categories = ["austen", "shakespeare"]
test_set = [(sent, "austen") for sent in author1_test] + [(sent, "shakespeare") for sent in author2_test]

train_features = [(get_features(sents), label) for (sents, label) in strat_train_set]
pretest_features = [(get_features(sents), label) for (sents, label) in strat_pretest_set]
test_features = [(get_features(sents), label) for (sents, label) in test_set]

words = []
for (sents, label) in strat_train_set:
    words = extract_words(sents, words)

counts = Counter(words)
maximum = float(13414)
selected_words = []
for item in counts.items():
    count = float(item[1])
    if count > 200 and count/maximum < 0.2:
        selected_words.append(item[0])

train_features = [(get_features_2(sents, selected_words), label) for (sents, label)
                  in strat_train_set]
pretest_features = [(get_features_2(sents, selected_words), label) for (sents, label)
                    in strat_pretest_set]
test_features = [(get_features_2(sents, selected_words), label) for (sents, label)
                 in test_set]

classifier = DecisionTreeClassifier.train(train_features)

print (f"Accuracy on the training set = {str(classify.accuracy(classifier, train_features))}")
print (f"Accuracy on the pretest set = {str(classify.accuracy(classifier, pretest_features))}")
print (f"Accuracy on the test set = {str(classify.accuracy(classifier, test_features))}")
