from random import shuffle
from numpy import random, array, exp
from pickle import load, dump

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.utils import to_categorical


class StoryLSTM(Model):
    NUMBER_OF_WORDS: int = 15_987
    MAX_SIZE_OF_WORD: int = 60

    def __init__(self, length=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if length:
            self.NUMBER_OF_WORDS = length

        self.model = Sequential([
            Embedding(self.NUMBER_OF_WORDS, 10, input_length=self.MAX_SIZE_OF_WORD),
            LSTM(100),
            Dropout(0.2),
            Dense(self.NUMBER_OF_WORDS, activation='softmax')
        ])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


class TextUtils:
    __MAX_SIZE_OF_WORD: int = 60
    __RULES = {
        '.': ' DOT ', ',': ' COMMA ', '?': ' QUESTION ', '!': ' EXCLAMATION ',
        '-': ' DASH ', '=': ' ', '(': '', '\n': ' NEWLINE ', ')': '', ':': '',
        ';': ' DOT ', '\'': '', '\"': ' QUOTE ', '„': ' QUOTE ', '”': ' QUOTE ',
        '*': ''
    }

    def generate_dataset(self, text, save_vocab = False):
        text = self.__replace_punctuation(text=text)
        words = self.__get_unique_words(text)

        if save_vocab:
            print(f'Number of unique words: {len(words)}')
            TextUtils.__save_pickles(words=words)

        X, y = self.__tokenize(text, word_to_index=self.__get_w2i(words))
        c = list(zip(X, to_categorical(y, num_classes=len(words))))
        shuffle(c)
        X, y = zip(*c)

        return array(X), array(y)

    def __replace_punctuation(self, text):
        for key, value in self.__RULES.items():
            text = text.replace(key, value)

        return text

    def __tokenize(self, text, word_to_index):
        X, y, step = [], [], 1
        text_parsed_tokenized_nums = [word_to_index[word] for word in text.split()]

        X_append, y_append = X.append, y.append
        for i in range(0, len(text_parsed_tokenized_nums) - self.__MAX_SIZE_OF_WORD, step):
            X_append(text_parsed_tokenized_nums[i:i + self.__MAX_SIZE_OF_WORD])
            y_append(text_parsed_tokenized_nums[i + self.__MAX_SIZE_OF_WORD])

        return X, y

    @staticmethod
    def __save_pickles(words):
        with open('vocab/w2i.pkl', 'wb') as fp:
            dump(TextUtils.__get_w2i(words=words), file=fp)

        with open('vocab/i2w.pkl', 'wb') as fp:
            dump(TextUtils.__get_i2w(words=words), file=fp)

    @staticmethod
    def __get_w2i(words):
        return dict((c, i) for i, c in enumerate(words))

    @staticmethod
    def __get_i2w(words):
        return dict((i, c) for i, c in enumerate(words))

    @staticmethod
    def __get_unique_words(text):
        return sorted(list(set(text.split())))


class TextGenerator:
    __MAX_SIZE_OF_WORD: int = 60
    __RULES = {
        'COMMA': ',', 'DOT': '.', 'QUESTION': '?', 'NEWLINE': '\n', 'EXCLAMATION': '!', 'DASH': '-', 'QUOTE': '"'
    }

    def __init__(self, model_filepath):
        self.__init_model(model_filepath)
        self.__i2w = TextGenerator.__load_vocabulary('vocab/i2w.pkl')
        self.__w2i = TextGenerator.__load_vocabulary('vocab/w2i.pkl')

    def generate_story(self, prompt, length):
        self.__generate(prompt=prompt, length=length)

    def __generate(self, prompt, length):
        seed_tokens = prompt.lower().split()
        seed = []

        for t in seed_tokens:
            seed.append(self.__w2i[t])

        for i in range(self.__MAX_SIZE_OF_WORD - len(seed_tokens)):
            seed.insert(0, self.__w2i['DOT'])

        print(prompt, end=' ')
        input_ = seed

        for i in range(length):
            pred = self.__model.predict(array([input_]), verbose=0)

            best_word = TextGenerator.__get_best_word(pred)
            translated = self.__i2w[best_word]

            if translated in self.__RULES.keys():
                print(self.__RULES[translated], end=' ')
            else:
                print(translated, end=' ')

            input_ = input_[1:]
            input_.append(best_word)

    def __init_model(self, model_filepath):
        self.__model = StoryLSTM()
        self.__model.built = True
        self.__model.load_weights(filepath=model_filepath)

    @staticmethod
    def __load_vocabulary(filepath):
        with open(filepath, 'rb') as fp:
            return load(fp)

    @staticmethod
    def __get_best_word(result, top_n=10):
        arr = result[0].copy()
        indices = arr.argsort()[-top_n - 1:][::-1]

        arr = sorted(arr)[::-1][-top_n - 1:]
        return random.choice(indices, p=[exp(v) / sum(exp(arr)) for v in arr])
