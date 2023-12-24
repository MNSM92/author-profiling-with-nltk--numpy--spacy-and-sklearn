import nltk
from nltk.corpus import gutenberg
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from numpy import arange
import spacy
from tools import *
nltk.download('gutenberg')
nltk.download('punkt')
nlp = spacy.load('en_core_web_md')
gutenberg_data = gutenberg.fileids()

author1_train = gutenberg.sents('austen-emma.txt') + gutenberg.sents('austen-persuasion.txt')
print (author1_train)
print (len(author1_train))
author1_test = gutenberg.sents('austen-sense.txt')
print (author1_test)
print (len(author1_test))

author2_train = gutenberg.sents('shakespeare-caesar.txt') + gutenberg.sents(
    'shakespeare-hamlet.txt')
print (author2_train)
print (len(author2_train))
author2_test = gutenberg.sents('shakespeare-macbeth.txt')
print (author2_test)
print (len(author2_test))
statistics(gutenberg, gutenberg_data)

all_sents = [(sent, "austen") for sent in author1_train]
all_sents += [(sent, "shakespeare") for sent in author2_train]
print (f"Dataset size = {str(len(all_sents))} sentences")

values = [author for (sent, author) in all_sents]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_train_set = []
strat_pretest_set = []
for train_index, pretest_index in split.split(all_sents, values):
    strat_train_set = [all_sents[index] for index in train_index]
    strat_pretest_set = [all_sents[index] for index in pretest_index]


categories = ["austen", "shakespeare"]
rows = []
rows.append(["Category", "Overall", "Stratified train", "Stratified pretest"])
for cat in categories:
    rows.append([cat, f"{cat_proportions(all_sents, cat):.6f}",
                f"{cat_proportions(strat_train_set, cat):.6f}",
                f"{cat_proportions(strat_pretest_set, cat):.6f}"])

columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i])
                  for i in range(0, len(row))))


test_set = [(sent, "austen") for sent in author1_test]
test_set += [(sent, "shakespeare") for sent in author2_test]

rows = []
rows.append(["Category", "Overall", "Stratified train", "Stratified pretest", "Test"])
for cat in categories:
    rows.append([cat, f"{cat_proportions(all_sents, cat):.6f}",
                f"{cat_proportions(strat_train_set, cat):.6f}",
                f"{cat_proportions(strat_pretest_set, cat):.6f}",
                f"{cat_proportions(test_set, cat):.6f}"])

columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i])
                  for i in range(0, len(row))))


words = []

def extract_words(text, words):
    words += set([word for word in text])
    return words

for (sents, label) in strat_train_set:
    words = extract_words(sents, words)

counts = Counter(words)
print(len(set(counts)))

percentages = {}
maximum = float(13414)

# Let's explore the document frequency bands
for item in counts.items():
    perc = float(item[1])/maximum
    for freq in arange(0.00, 0.05, 0.0125):
        if perc>=freq and perc<=freq+0.0125:
            freq_range = str(freq)[:6] + "%-" + str(freq+0.0125)[:6] + "%"
            percentages[freq_range] = percentages.get(freq_range, 0) + 1
    for freq in arange(0.05, 1.00, 0.05):
        if perc>=freq and perc<=freq+0.05:
            freq_range = str(freq)[:4] + "%-" + str(freq+0.05)[:4] + "%"
            percentages[freq_range] = percentages.get(freq_range, 0) + 1

# Print out these frequency bands
for key in sorted(percentages.keys()):
    print(key + " texts: " + str(percentages.get(key)) + " words")
