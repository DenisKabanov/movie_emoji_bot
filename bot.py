import numpy as np # для работы с матрицами
import emoji as em # для конвертации текста в эмодзи
import random # для случайного выбора
import json # для загрузки данных
import telebot # для работы с ботом

import os # для переменных окружения (токена бота)
from dotenv import load_dotenv # для загрузки переменных окружения


with open('./data/movies.json', 'r') as f: # открываем файл для чтения ('r')
    movies = json.load(f) # загружаем данные о фильмах


weights = {} # словарь с весами для пользователя
suggested_answers = {} # список уже предложенных ответов для пользователя

# if weights.sum() == 0: # если у всех фильмов нулевая вероятность — они все были хотя бы раз загаданы
#     weights = np.ones() # возвращаем изначальные веса

load_dotenv() # загрузка переменных окружения
token = os.getenv('TOKEN')
bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start']) # декоратор для обработки команд бота
def get_command(message):
    bot.send_message(chat_id=message.chat.id, text=f"Приветствую вас в шашем телеграм боте {message.chat.id}!")

@bot.message_handler(content_types=['text']) # декоратор для обработки текстовых сообщений
def get_text(message):
    message_text = message.text.lower() # приводим пришедший текст к нижнему регистру
    if message_text == "загадай":
        if message.chat.id not in weights.keys(): # если вероятности для фильмов ещё не определены у пользователя
            weights[message.chat.id] = np.ones(shape=(len(movies))) # задаём их равнозначными (размера len(movies))
        elif weights[message.chat.id].sum() == 0: # если вероятности всех фильмов занулились из-за долгой игры
            weights[message.chat.id] = np.ones(shape=(len(movies))) # возвращаем из к исходному значению

        movie_number = random.choices(np.arange(len(movies)), k=1, weights=weights[message.chat.id])[0] # случайным образом выбираем номер фильма (в зависимости от весов weights)
        weights[message.chat.id][movie_number] = 0 # зануляем вероятность выбора этого же фильма у пользователя
        movie = movies[movie_number]["title"] # берём фильм под выбранным номером

        bot.send_message(chat_id=message.chat.id, text=f"Привет {movie} {em.emojize(':purple_heart:')}, чем я могу тебе помочь?")
    # elif message_text == "/help":
        

bot.polling(none_stop=True, interval=0) # бесконечный (none_stop=True не отключает бота при возникновении ApiException) цикл обращений (с интервалом в interval=0) бота к телеграм серверу для проверки наличия новых сообщений