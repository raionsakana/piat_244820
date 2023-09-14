from utils import TextGenerator

text_generator = TextGenerator(model_filepath='weights/model.h5')

print('Model LSTM\n\n')

while True:
    try:
        prompt = input('Podaj prompt: ')
        length = int(input('Podaj długość bajki: '))

        if len(prompt) == 0 or length < 1:
            raise ValueError()

        text_generator.generate_story(prompt=prompt, length=int(length))
    except ValueError:
        print('Podane wartości są nieprawidłowe. Spróbuj jeszcze raz.\n\n')
