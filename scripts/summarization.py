#===============================================================================
# файл для суммаризации текста выбранной (MODEL_NAME) моделью
#===============================================================================

import json # для сохранения файлов
import torch # для работы с моделями
from tqdm import tqdm # для отслеживания прогресса
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # модель и токенизатор для суммаризации
from settings import * # импорт параметров


# данные
with open(DATA_DIR + 'movies_base.json', 'r') as f: # открываем файл для чтения ('r')
    movies = json.load(f) # загружаем данные


# загрузка модели
# model = T5ForConditionalGeneration(torch.load(SUMMARIZATION_MODELS_DIR + USE_SAVED_MODEL + "/config.pth")) # создаём модель с сохранённой конфигурацией
    
# model = AutoModelForSeq2SeqLM.from_config(torch.load(SUMMARIZATION_MODELS_DIR + MODEL_NAME + "/config.pth")) # создаём модель с сохранённой конфигурацией
    
# model.load_state_dict(torch.load(SUMMARIZATION_MODELS_DIR + MODEL_NAME +"/state_dict.pth")) # загружаем предобученные веса модели
    
model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODELS_DIR + MODEL_NAME) # загружаем сохранённую модель
model.eval() # переводим модель в режим оценивания


# токенизатор
tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODELS_DIR + MODEL_NAME) # загружаем предобученный токенайзер


# суммаризация
for movie in tqdm(movies): # идём по мультфильмам
    input_ids = tokenizer( # токенизируем текст
        [movie["description"]], # текст, что будет токенизирован
        max_length=MAX_LENGTH, # максимальное число токенов
        add_special_tokens=True, # добавлять или нет специальные токены при кодировании последовательности (строки)
        padding="max_length", # добавление токенов паддинга до достижения максимальной длины
        truncation=True, # усечение числа токенов, если их больше max_length
        return_tensors="pt", # тип возвращаемого тензора (pytorch)
    )["input_ids"] # берём данные "input_ids" (сами токены)

    output_ids = model.generate( # генерируем суммаризацию
        input_ids=input_ids, # передаём входные токены
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE # n-grams данного размера могут возникнуть лишь раз
    )[0] # [0] — берём вложение

    movie["description"] = tokenizer.decode(output_ids, skip_special_tokens=True) # декодируем получившиеся токены в слова с пропуском специальных токенов (skip_special_tokens=True)


# сохранение
with open(DATA_DIR + f'movies_{MODEL_NAME}.json', 'w') as f: # открытие файла для записи ('w')
    json.dump(movies, f) # сохраняем в него данные о фильмах
