import nltk
from nltk.corpus import gutenberg
nltk.download('gutenberg')


author1_train = gutenberg.sents('austen-emma.txt') + gutenberg.sents('austen-persuasion.txt')
print(author1_train)
print(len(author1_train))

author1_test = gutenberg.sents('austen-sense.txt')
print (author1_test)
print (len(author1_test))