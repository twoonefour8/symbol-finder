import cv2
import pickle
from keras.models import load_model
import vocabulary


# пути к моделям
num_model = './models/num.model'
rus_model = './models/rus.model'
num_letter_model = './models/num_letter_glyphs_model.model'
gost_model = './models/gost.model'

# пути к классам
num_classes = 'classes/num.pickle'
rus_classes = 'classes/rus.pickle'
num_letter_classes = 'classes/num_letter_glyphs_classes.pickle'
gost_classes = 'classes/gost.pickle'


# для правильной кодировки кириллицы d в пути изображения
def predict(letter):
    # загружаем модель и бинаризатор меток
    model = load_model(gost_model)
    lb = pickle.loads(open(gost_classes, 'rb').read())

    # меняем его размер на необходимый
    image = cv2.resize(letter, (64, 64))

    # масштабируем значения пикселей к диапазону [0, 1]
    image = image.astype('float') / 255.0
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # делаем предсказание на изображении
    preds = model.predict(image)
    # находим индекс метки класса с наибольшей вероятностью соответствия
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]

    symbol = vocabulary.symbols.get(label)
    percent = f'{preds[0][i] * 100:.2f}%'

    return symbol, percent
