#===============================================================================
# файл для запуска телеграм бота
#===============================================================================

import numpy as np # для работы с массивами
import random # для случайного выбора
import time # для работы с временем
import json # для сохранения и загрузки объектов
import re # регулярные выражения для работы с текстом
import os # для переменных окружения (токена бота)
import telebot # для работы с ботом
from telebot import apihelper, types # для прокси и клавиатуры
from dotenv import load_dotenv # для загрузки переменных окружения
from settings import * # импорт переменных

# apihelper.proxy = { 'https': '165.22.189.59:999'} # прокси

load_dotenv() # загрузка переменных окружения
token = os.getenv('TOKEN') # берём токен бота из переменных окружения
bot = telebot.TeleBot(token) # API для взаимодействия с ботом

with open(DATA_DIR + DATA_FILE, 'r') as f: # открываем файл для чтения ('r')
    movies = json.load(f) # загружаем данные о мультфильмах

weights = {} # словарь с весами для пользователя
chosen_movie = {} # словарь с загаданным мультфильмом для пользователя


markup_start = types.ReplyKeyboardMarkup(resize_keyboard=True) # клавиатура (resize_keyboard=True — под число кнопок)
markup_start.add(types.KeyboardButton("загадай")) # кнопка на клавиатуре, отправляет *текст* как сообщение

markup_give_a_hint = types.ReplyKeyboardMarkup(resize_keyboard=True) # клавиатура (resize_keyboard=True — под число кнопок)
markup_give_a_hint.add(types.KeyboardButton("подскажи категории")) # кнопка на клавиатуре, отправляет *текст* как сообщение
markup_give_a_hint.add(types.KeyboardButton("подскажи год")) # кнопка на клавиатуре, отправляет *текст* как сообщение
markup_give_a_hint.add(types.KeyboardButton("подскажи режиссёра")) # кнопка на клавиатуре, отправляет *текст* как сообщение
markup_give_a_hint.add(types.KeyboardButton("подскажи известных актёров")) # кнопка на клавиатуре, отправляет *текст* как сообщение
markup_give_a_hint.add(types.KeyboardButton("подскажи слоган")) # кнопка на клавиатуре, отправляет *текст* как сообщение
markup_give_a_hint.add(types.KeyboardButton("подскажи часть названия")) # кнопка на клавиатуре, отправляет *текст* как сообщение

@bot.message_handler(commands=['start']) # декоратор для обработки команд бота
def get_command(message):
    chosen_movie[message.chat.id] = None # делаем пустую запись для пользователя о загаданном фильме
    weights[message.chat.id] = np.ones(shape=(len(movies))) # задаём веса (вероятности) равнозначными (размера len(movies))
    bot.send_message(chat_id=message.chat.id, text=f"Приветствую вас в нашем телеграм боте, да начнём же угадывать!", reply_markup=markup_start) # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)

@bot.message_handler(commands=['help']) # декоратор для обработки команд бота
def get_command(message):
    bot.send_message(chat_id=message.chat.id, text=f"Поддерживаемые сообщения:\n\
1) 'загадай' — для загадывания мультфильма\n\
2) 'подскажи *', где * это слово из (категории, год, режиссёра, известных актёров, слоган, часть названия) — для подсказки\n\
3) 'сдаюсь' — чтобы признать поражение😡 и узнать загаданный мультфильм\n\
4) ну и конечно же само название мультфильма, если он был загадан") # отправляем сообщение (text) в чат (chat_id) 

@bot.message_handler(content_types=['text']) # декоратор для обработки текстовых сообщений
def get_text(message):
    message_text = message.text.lower().strip() # приводим пришедший текст к нижнему регистру и убираем лишние пробелы

    if message.chat.id not in chosen_movie.keys(): # обработка случая, когда не прописана команда /start (нет записи о пользователе)
        chosen_movie[message.chat.id] = None # делаем пустую запись для пользователя о загаданном фильме
        weights[message.chat.id] = np.ones(shape=(len(movies))) # задаём веса (вероятности) равнозначными (размера len(movies))

    if message_text == "загадай":
        if chosen_movie[message.chat.id] is not None: # если мультфильм уже был загадан
            bot.send_message(chat_id=message.chat.id, text=f"Вам уже был загадан мультфильм, отправьте 'подскажи *' (категории, год, режиссёра, известных актёров, слоган, часть названия), чтобы получить подсказку или 'сдаюсь', чтобы узнать загаданный мультфильм.", reply_markup=markup_give_a_hint)  # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)
        else: # если мультфильм ещё не был загадан
            if weights[message.chat.id].sum() == 0: # если вероятности всех фильмов занулились из-за долгой игры
                weights[message.chat.id] = np.ones(shape=(len(movies))) # возвращаем их к исходному значению

            movie_number = random.choices(np.arange(len(movies)), k=1, weights=weights[message.chat.id])[0] # случайным образом выбираем номер фильма (в зависимости от весов weights)
            weights[message.chat.id][movie_number] = 0 # зануляем вероятность выбора этого же фильма у пользователя
            chosen_movie[message.chat.id] = movies[movie_number] # запоминаем мультфильм для определённого пользователя

            bot.send_message(chat_id=message.chat.id, text=f"Загадал... \n{''.join([e[0] for e in movies[movie_number]['emojis']])}") # отправляем сообщение (text) в чат (chat_id) 
            if len(movies[movie_number]["hint"]):
                bot.send_message(chat_id=message.chat.id, text=f"Небольшая подсказка, в названии есть цифры: {' '.join(movies[movie_number]['hint'])}.") # отправляем сообщение (text) в чат (chat_id) 
            # bot.send_message(chat_id=message.chat.id, text=f"{chosen_movie[message.chat.id]['title']}") # DEBUG

    elif message_text == "сдаюсь":
        if chosen_movie[message.chat.id] is None: # если мультфильм ещё не был загадан
            bot.send_message(chat_id=message.chat.id, text=f"Уже сдаёшься?! Я же ещё ничего не загадал...", reply_markup=markup_start) # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)
        else: # если мультфильм уже был загадан
            bot.send_message(chat_id=message.chat.id, text=f"Был загадан мультфильм '{chosen_movie[message.chat.id]['title']}'.", reply_markup=markup_start) # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)
            chosen_movie[message.chat.id] = None # очищаем запись о загаданном мультфильме

    elif message_text.find("подскажи") != -1:
        if chosen_movie[message.chat.id] is None: # если мультфильм ещё не был загадан
            bot.send_message(chat_id=message.chat.id, text=f"Подсказка: сначала нужно отправить 'загадай'!", reply_markup=markup_start) # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)
        else: # если мультфильм уже был загадан
            if message_text == "подскажи": # если не был передан параметр для подсказки
                bot.send_message(chat_id=message.chat.id, text=f"Я могу подсказать следующее:", reply_markup=markup_give_a_hint) # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)
            if message_text.find("категории") != -1:
                bot.send_message(chat_id=message.chat.id, text=f"Категории: {', '.join(chosen_movie[message.chat.id]['categories'])}.", reply_markup=markup_give_a_hint)  # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)
            if message_text.find("год") != -1:
                bot.send_message(chat_id=message.chat.id, text=f"Год: {chosen_movie[message.chat.id]['year']}.", reply_markup=markup_give_a_hint)  # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)
            if message_text.find("режиссёра") != -1:
                bot.send_message(chat_id=message.chat.id, text=f"Режиссёры: {', '.join(chosen_movie[message.chat.id]['directors'])}.", reply_markup=markup_give_a_hint)  # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)
            if message_text.find("известных актёров") != -1:
                bot.send_message(chat_id=message.chat.id, text=f"Известные актёры: {', '.join(chosen_movie[message.chat.id]['actors'])}.", reply_markup=markup_give_a_hint)  # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)
            if message_text.find("слоган") != -1:
                bot.send_message(chat_id=message.chat.id, text=f"Слоган: {chosen_movie[message.chat.id]['tagline']}.", reply_markup=markup_give_a_hint)  # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)
            if message_text.find("часть названия") != -1:
                start = random.randrange(len(chosen_movie[message.chat.id]["title"])) # случайно выбираем номер, с какой буквы выведем ответ 
                end = random.randrange(len(chosen_movie[message.chat.id]["title"])) # случайно выбираем номер, по какую букву выведем ответ 
                start, end = min(start, end), max(start, end) # сортируем, чтобы end не был раньше старта
                hint = "" # для подсказки части слова
                for i in range(len(chosen_movie[message.chat.id]["title"])): # идём по числу букв в загаданном названии
                    if start <= i <= end: # если буква в нужном интервале
                        hint += chosen_movie[message.chat.id]["title"][i] # добавляем её саму
                    else: # иначе
                        hint += "*" # зашифровываем букву
                bot.send_message(chat_id=message.chat.id, text=f"Часть названия: '{hint}'.", reply_markup=markup_give_a_hint) # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)

    else: # обрабатываем переданное название
        if chosen_movie[message.chat.id] is None: # если мультфильм ещё не был загадан
            bot.send_message(chat_id=message.chat.id, text=f"Сначала нужно загадать мультфильм, для этого отправь 'загадай'!", reply_markup=markup_start)  # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)
        else: # если мультфильм уже был загадан
            pred = re.sub(r"[ ,.…?!:;/@#$#\-—ツ►๑۩۞۩•*”˜˜”*°°*`'“\"()\\]", '', message_text) # удаляем ненужные символы из переданного названия
            true = chosen_movie[message.chat.id]["title"].lower().strip() # приводим название к нижнему регистру и убираем лишние пробелы
            true = re.sub(r"[ ,.…?!:;/@#$#\-—ツ►๑۩۞۩•*”˜˜”*°°*`'“\"()\\]", '', true) # удаляем ненужные символы из настоящего названия

            if pred == true: # если ответ совпал 
                bot.send_message(chat_id=message.chat.id, text=f"Правильно! Вы угадали мультфильм '{chosen_movie[message.chat.id]['title']}'!", reply_markup=markup_start)  # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)
                chosen_movie[message.chat.id] = None # очищаем запись о загаданном мультфильме
            else: # если ответ не совпал
                correct_letters = 0 # число совпавших букв
                for i in range(min(len(pred), len(true))): # идём по минимальной длине
                    if pred[i] == true[i]: # сравниваем буквы на позициях
                        correct_letters += 1 # увеличиваем число совпадений
                bot.send_message(chat_id=message.chat.id, text=f"Букв совпало {correct_letters}. Попытайся ещё раз!")  # отправляем сообщение (text) в чат (chat_id) с добавлением клавиатуры (reply_markup)
        
while True: # бесконечный цикл, так как telebot может оборвать подключение 
    try: # пытаемся
        bot.polling(none_stop=True, interval=0) # бесконечный (none_stop=True не отключает бота при возникновении ApiException) цикл обращений (с интервалом в interval=0) бота к телеграм серверу для проверки наличия новых сообщений

    except Exception as e: # ловим ошибку
        print("Looks like we got a telebot exception:", e) # выводи ошибку
        time.sleep(10) # ждём 10 секунд и снова обращаемся к боту
