def statistics(gutenberg, gutenberg_data):
    for work in gutenberg_data:
        num_chars = len(gutenberg.raw(work))
        num_words = len(gutenberg.words(work))
        num_sents = len(gutenberg.sents(work))
        num_vocab = len(set(w.lower() for w in gutenberg.words(work)))
        print(round(num_chars/num_words),
             round(num_words/num_sents),
             round(num_words/num_vocab),
             work)


def cat_proportions(data, cat):
    count = 0
    for item in data:
        if item[1]==cat:
            count += 1
    return float(count) / float(len(data))


def get_features(text):
    features = {}
    word_list = [word for word in text]
    for word in word_list:
        features[word] = True
    return features


def get_features_2(text, selected_words):
    features = {}
    word_list = [word for word in text]
    for word in word_list:
        if word in selected_words:
            features[word] = True
    return features


def extract_words(text, words):
    words += set([word for word in text])
    return words









