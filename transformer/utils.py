from pickle import load

from tensorflow import cast, constant, shape, bool, expand_dims, int32, tile, range, reshape, concat
from numpy import random, array, exp

from keras.models import Model, Sequential
from keras import layers


class StoryTransformer(Model):
    class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super().__init__()
            self.att = layers.MultiHeadAttention(num_heads, embed_dim)
            self.ffn = Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ])
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(rate)
            self.dropout2 = layers.Dropout(rate)

        def call(self, inputs, training=None, mask=None):
            seq_len = shape(inputs)[1]

            out1 = self.norm1(
                inputs + self.dropout1(self.att(inputs, inputs, attention_mask=self.causal_attention_mask(
                    shape(inputs)[0], seq_len, seq_len, bool
                )))
            )
            return self.norm2(out1 + self.dropout2(self.ffn(out1)))

        @staticmethod
        def causal_attention_mask(batch_size, n_dest, n_src, dtype):
            return tile(
                reshape(
                    cast(range(n_dest)[:, None] >= range(n_src) - n_src + n_dest, dtype), [1, n_dest, n_src]
                ),
                concat([expand_dims(batch_size, -1), constant([1, 1], dtype=int32)], 0)
            )

    class TokenAndPositionEmbedding(layers.Layer):
        def __init__(self, max_len, vocab_size, embed_dim):
            super().__init__()
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
            self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

        def call(self, x, training=None, mask=None):
            return self.token_emb(x) + self.pos_emb(range(start=0, limit=shape(x)[-1], delta=1))

    NUMBER_OF_WORDS: int = 24_278
    MAX_SIZE_OF_WORD: int = 60
    EMBED_DIM: int = 256
    NUM_HEADS: int = 2
    FEED_FORWARD_DIM: int = 256

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        inputs = layers.Input(shape=(self.MAX_SIZE_OF_WORD,), dtype=int32)
        embedding_layer = self.TokenAndPositionEmbedding(self.MAX_SIZE_OF_WORD, self.NUMBER_OF_WORDS, self.EMBED_DIM)

        x = self.TransformerBlock(self.EMBED_DIM, self.NUM_HEADS, self.FEED_FORWARD_DIM)(embedding_layer(inputs))
        self.model = Model(inputs=inputs, outputs=[layers.Dense(self.NUMBER_OF_WORDS)(x), x])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


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
        tokens = [self.__w2i[word] for word in TextGenerator.remove_punctuation(prompt).split()]
        num_tokens_generated, tokens_generated = 0, []

        while num_tokens_generated <= length:
            index = len(tokens) - 1
            index, x = self.__get_tokens(index, tokens)

            y, _ = self.__model.predict(array([x]), verbose=0)
            sample_token = self.__get_best_word(y[0][index])

            tokens_generated.append(sample_token)
            tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)

        text = ' '.join([
            self.__RULES[self.__i2w[w_ind]] if self.__i2w[w_ind] in self.__RULES.keys() else self.__i2w[w_ind]
            for w_ind in tokens
        ])
        print(f'Story:\n {text}\n')

    def __get_tokens(self, index, start_tokens):
        pad_len = self.__MAX_SIZE_OF_WORD - len(start_tokens)

        if pad_len < 0:
            return self.__MAX_SIZE_OF_WORD - 1, start_tokens[:self.__MAX_SIZE_OF_WORD]
        elif pad_len > 0:
            return index, start_tokens + [0] * pad_len

        return index, start_tokens

    def __init_model(self, model_filepath):
        self.__model = StoryTransformer()
        self.__model.built = True
        self.__model.load_weights(filepath=model_filepath)

    @staticmethod
    def __load_vocabulary(filepath):
        with open(filepath, 'rb') as fp:
            return load(fp)

    @staticmethod
    def __get_best_word(result, top_n=10):
        arr = result.copy()
        indices = arr.argsort()[-top_n - 1:][::-1]

        arr = sorted(arr)[::-1][-top_n - 1:]
        return random.choice(indices, p=[exp(v) / sum(exp(arr)) for v in arr])

    @staticmethod
    def remove_punctuation(input_string):
        from string import punctuation
        return input_string.translate(str.maketrans('', '', punctuation))
