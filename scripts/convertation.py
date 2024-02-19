#===============================================================================
# файл для конвертации описания мультфильмов в эмодзи
#===============================================================================

import numpy as np # для работы с массивами
import spacy # для обработки текста (токенизация, лемматизация, тегирование части речи)
import json # для сохранения и загрузки объектов
import re # регулярные выражения для работы с текстом
from tqdm import tqdm # для отслеживания прогресса
from navec import Navec # для векторизации текста
from nltk.corpus import stopwords # стоп-слова
from settings import * # импорт параметров
from utils import * # импорт вспомогательных функций


# загружаем всё необходимое для работы
with open(DATA_DIR + CONV_NAME + ".json", 'r') as f: # открываем файл для чтения ('r')
    movies = json.load(f) # загружаем данные о мультфильмах
with open(DATA_DIR + "emojis.json", 'r') as f: # открываем файл для записи ('w')
    emojis = json.load(f) # загружаем данные о эмодзи
nlp_model = spacy.load('ru_core_news_sm') # загружаем модель для обработки текста ("ru_core_news_sm" — обучена для русского языка)
stop_words = stopwords.words('russian') # список стоп-слов для русского языка
for add_stop in ADD_STOP_WORDS: # идём по дополнительным стоп-словам
    stop_words.append(add_stop) # добавляем стоп-слово
navec = Navec.load(CONVERTATION_MODELS_DIR + "navec/navec_hudlit_v1_12B_500K_300d_100q.tar") # загружаем вектора


# конвертация текста в эмодзи
for movie in tqdm(movies): # идём по мультфильмам
    # токенизируем текст
    movie["tokens"] = [] # список под токены мультфильма
    for section in ["description", "title"]: # какие данные будут токенизированы
        text = movie[section].lower() # приводим текст к нижнему регистру
        text = re.sub(r"[,.…?“/!@#$1234567890#—ツ►๑۩۞۩•*”˜˜”*°°*`)(]", '', text) # удаляем ненужные символы из текста
        tokens = nlp_model(text) # обрабатываем текст моделью (заготавливаем "токены")
        for token in tokens: # идём по полученным токенам
            if token.lemma_ not in stop_words: # проверяем, входит ли токен (его лемма) в список стоп-слов
                movie["tokens"].append([token.text, token.lemma_, token.pos_]) # добавляем токен к данным о мультфильме (само слово, его лемма, тег части речи)

    # считаем вектора для токенов (для леммы слова или него самого)
    movie["vectors"] = [] # вектор под токены описания
    for token in movie["tokens"]: # идём по токенам описания
        if token[1] in navec: # если лемма токена известна векторизатору
            movie["vectors"].append([navec[token[1]].tolist(), token[1]]) # добавляем запись [вектор леммы, лемма]
        elif token[0] in navec: # если текст токена известен векторизатору (не лемма, так как в лемме буква "е" заменяется на "ё")
            movie["vectors"].append([navec[token[0]].tolist(), token[0]]) # добавляем запись [вектор слова, слово]

    # конвертируем полученные вектора в эмодзи
    movie["emojis"] = [] # заготавливаем место под эмодзи
    for vector in movie["vectors"]: # идём по векторам токенов
        similarity = cosine_similarity(emojis["vectors"], vector[0]) # считаем косинусную близость вектора токена (vector[0], под 1 идёт сам токен) ко всем векторам эмодзи
        most_similar = np.argmax(similarity) # находим индекс наибольшего получившегося косинуса угла (самое вероятное эмодзи)
        if (similarity[most_similar] > MIN_SIMILARITY) and (emojis["emojis"][most_similar] not in [e[0] for e in movie["emojis"]]): # если эмодзи проходит порог на схожесть со словом и его ещё нет в списке эмодзи (или список эмодзи пуст) рассматриваемого фильма (movie["emojis"][:,0] — добавленные значки эмодзи)
            movie["emojis"].append([emojis["emojis"][most_similar], similarity[most_similar], vector[1]]) # добавляем запись о самом вероятном эмодзи для слова, соответствующий ему угол и само слово
    
movies = [movie for movie in movies if len(movie["emojis"]) >= MIN_EMOJIS] # оставляем только те мультфильмы, у которых есть хотя бы 2 эмодзи


# сохранение
with open(DATA_DIR + CONV_NAME + "_emoj.json", 'w') as f: # открываем файл для записи ('w')
    json.dump(movies, f) # сохраняем в него данные о фильмах