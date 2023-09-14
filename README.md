# piat_244820
PIAT - Patryk Janowski, 244820

# LSTM

## Zbudowanie obrazu
```shell
cd lstm
docker build -t lstm:latest .
```

## Uruchomienie obrazu
```shell
docker run -it lstm:latest
```

# Transformer
## Zbudowanie obrazu

```shell
cd transformer
docker build -t transformer:latest .
```

## Uruchomienie obrazu
```shell
docker run -it transformer:latest
```

# Materiały źródłowe
* LSTM - [Character-level text generation with LSTM](https://keras.io/examples/generative/lstm_character_level_text_generation/)
* Transformer - [Text generation with a miniature GPT
](https://keras.io/examples/generative/text_generation_with_miniature_gpt/)