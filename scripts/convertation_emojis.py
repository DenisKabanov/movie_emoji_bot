#===============================================================================
# файл для конвертации эмодзи в вектора для дальнейшей работы
#===============================================================================

import numpy as np # для работы с массивами
import spacy # для обработки текста (токенизация, лемматизация, тегирование части речи)
import emoji # для работы с эмодзи
import json # для сохранения и загрузки объектов
import re # регулярные выражения для работы с текстом
from tqdm import tqdm # для отслеживания прогресса
from navec import Navec # для векторизации текста
from settings import * # импорт параметров
from utils import * # импортируем вспомогательные функции

emojis = {"emojis": [], "names": [], "vectors": []} # словарь с данными про эмодзи вида

nlp_model = spacy.load('ru_core_news_sm') # загружаем модель для обработки текста ("ru_core_news_sm" — обучена для русского языка)
navec = Navec.load(CONVERTATION_MODELS_DIR + "navec/navec_hudlit_v1_12B_500K_300d_100q.tar") # загружаем вектора

for e in tqdm(emoji.EMOJI_DATA.keys()): # идём по всем эмодзи
    emoji_text = emoji.EMOJI_DATA[e]["ru"].lower() # берём название эмодзи на русском (например ":золотая медаль:")
    emoji_text = emoji_text.replace("_", " ").replace("-", " ") # заменяем символы "_" и "-" на пробел
    emoji_text = re.sub(r"[,.…?“/!@#$1234567890#—ツ►๑۩۞۩•*”˜˜”*°°*`)(:«»\"]", '', emoji_text) # удаляем ненужные символы

    emoji_vector = np.zeros(300, dtype=np.float32) # начальный вектор для кодирования эмодзи
    emoji_tokens = nlp_model(emoji_text) # обрабатываем название эмодзи (получаем токены с леммами...)
    try:
        for token in emoji_tokens: # получаем токены из названия и идём по ним
            if token.lemma_ in navec: # если лемма токена известна векторизатору
                emoji_vector += navec[token.lemma_] # добавляем вектор леммы
            elif token.text in navec: # если текст токена известен векторизатору (не лемма, так как в лемме буква "е" заменяется на "ё")
                emoji_vector += navec[token.text] # добавляем вектор леммы
            elif re.sub(r"[аеиоуыя]и", lambda m: m.group(0)[0] + "й", token.lemma_) in navec: # если в токене вместо "й" (а перед ней гласные) идёт "и", то пробуем заменить
                emoji_vector += navec[re.sub(r"[аеиоуыя]и", lambda m: m.group(0)[0] + "й", token.lemma_)] # добавляем вектор слова (не леммы)
            else:
                raise RuntimeError("Word not in model!") # пропускаем эмодзи, если не смогли полностью преобразовать его в вектор
    except RuntimeError as error:
        if str(error) == "Word not in model!":
            continue # переходим к следующему эмодзи
        else:
            raise
    emojis["emojis"].append(e) # добавляем символ эмодзи
    emojis["names"].append(emoji.EMOJI_DATA[e]["ru"]) # добавляем название эмодзи
    emojis["vectors"].append(emoji_vector.tolist()) # сохраняем посчитанный вектор эмодзи (не делим на число токенов, так как нам всё равно будет нужен угол)

with open(DATA_DIR + 'emojis.json', 'w') as f: # открываем файл для записи ('w')
    json.dump(emojis, f) # сохраняем в него данные о эмодзи и соответствующих векторах