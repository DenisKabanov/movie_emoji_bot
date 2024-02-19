#===============================================================================
# файл, содержащий цикл обучения модели
#===============================================================================

import numpy as np # для работы с массивами
import evaluate # для метрик
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer # для работы с моделью (токенизация, обучение...)
from datasets import load_from_disk, Dataset, DatasetDict # для работы с датасетом
from settings import * # импорт параметров


# датасет
dataset = load_from_disk("./data/MultiSim_ru") # загружаем сохранённый датасет
data_train = Dataset.from_dict(dataset["train"][:min(TRAIN_DATASET_SIZE, len(dataset["train"]))]) # берём определённую часть датасета для обучения (так как весь датасет слишком большой)
data_test = Dataset.from_dict(dataset["test"][:min(TEST_DATASET_SIZE, len(dataset["test"]))]) # берём определённую часть датасета для оценки (так как весь датасет слишком большой)
dataset = DatasetDict({"train": data_train, "test": data_test}) # собираем датасеты обратно


# загрузка модели
# model = T5ForConditionalGeneration(torch.load(SUMMARIZATION_MODELS_DIR + USE_SAVED_MODEL + "/config.pth")) # создаём модель из сохранённой конфигурацией

# model = AutoModelForSeq2SeqLM.from_config(torch.load(SUMMARIZATION_MODELS_DIR + MODEL_NAME + "/config.pth")) # создаём модель из сохранённой конфигурацией

# model.load_state_dict(torch.load(SUMMARIZATION_MODELS_DIR + MODEL_NAME + "/state_dict.pth")) # загружаем предобученную модель

model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODELS_DIR + MODEL_NAME) # загружаем сохранённую модель
model.train() # переводим модель в режим обучения


# настройки модели
model.config.num_beams = NUM_BEAMS # Number of beams for beam search that will be used by default in the generate
model.config.max_length = MAX_TARGET_TOKEN_COUNT # максимальное число токенов в ответе модели


# токенизатор
tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODELS_DIR + MODEL_NAME, do_lower_case=False, strip_accents=False) # загружаем предобученный токенайзер

def preprocess_function(dataset): # функция для токенизации текста
    # токенизация данных (в датасете появятся солбцы "input_ids" с токенами входа, "attention_mask" и "labels" с ожидаемыми токенами предсказания)
    model_inputs = tokenizer(dataset["original"], text_target=dataset["simple"], max_length=MAX_SOURCE_TOKEN_COUNT, padding=False, truncation=True) # токенизируем данные (без паддинга, но с обрезанием лишних токенов)
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True) # применяем функцию preprocess_function ко всем данным в датасете


# комплектовщик данных
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model) # создаём компановщик данных для передачи их во время обучения


# метрика
metric = evaluate.load("sacrebleu") # загружаем метрику для задачи

def compute_metrics(eval_preds): # функция для подсчёта метрики
    preds, labels = eval_preds # разделяем предсказания 
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True) # декодируем токены предсказаний в слова (предложения)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True) # декодируем токены ожидания в слова (предложения)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels) # считаем значение метрики
    result = {"bleu": result["score"]} # добавляем в список метрик

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds] # считаем число токенов в предсказании
    result["gen_len"] = np.mean(prediction_lens) # считаем среднюю длину  из токенов
    result = {k: round(v, 4) for k, v in result.items()} # округляем числа до 4 знаков после запятой
    return result


# обучение
training_args = Seq2SeqTrainingArguments( # аргументы для обучения
    output_dir=OUTPUT_DIR, # путь, по которому будут сохраняться чекпоинты (и предсказания) во время обучения (ПОЛНЫЙ путь не должен включать кириллицу)
    overwrite_output_dir=OVERWRITE_OUTPUT_DIR, # перезаписывать ли содержимое папки OUTPUT_DIR, если имена будут совпадать
    save_total_limit=SAVE_LIMIT, # сколько максимум хранить чекпоинтов в папке, старые будут просто перезаписываться (None — без лимита)
    num_train_epochs=TRAIN_EPOCHS, # число эпох для обучения
    learning_rate=LEARNING_RATE, # размер шага обучения
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, # число итераций, за которые будет накапливаться градиент для backward pass (обновления параметров модели)
    per_device_train_batch_size=BATCH_SIZE, # размер батча при обучении (число сэмлов, передающееся в модель за одну итерацию)
    per_device_eval_batch_size=BATCH_SIZE, # размер батча при оценивании (число сэмлов, передающееся в модель за одну итерацию)
    evaluation_strategy=EVAL_STRATEGY, # как часто оценивать качество обучения (возможные значения: "no", "steps", "epoch")
    eval_steps=EVAL_STEPS, # через сколько шагов обновления проводить оценку качества (только при использовании EVAL_STRATEGY="steps")
    save_strategy=SAVE_STRATEGY, # как часто сохранять чекпоинты обучения (возможные значения: "no", "steps", "epoch")
    save_steps=SAVE_STEPS, # через сколько шагов обновления сохранять чекпоинт (только при использовании EVAL_STRATEGY="steps")
    logging_strategy=LOGGING_STRATEGY, # как часто выводить лог обучения (возможные значения: "no", "steps", "epoch")
    logging_steps=LOGGING_STEPS,  # через сколько шагов обновления выводить лог (только при использовании EVAL_STRATEGY="steps")
    warmup_steps=WARMUP_STEPS, # количество шагов, используемое для линейного увеличения learning_rate от 0 до LEARNING_RATE
    fp16=FP_16, # работать ли с fp16 вместо fp32
    predict_with_generate=PREDICT_WITH_GENERATE, # использовать ли метод generate для подсчёта генеративных метрик (ROUGE, BLEU)
    load_best_model_at_end=LOAD_BEST_MODEL_AT_END, # возвращать ли лучшую модель в конце обучения
    metric_for_best_model=METRIC_FOR_BEST_MODEL,  # название метрики, по которой будет определяться лучшая модель
    greater_is_better=GREATER_BETTER # для LOAD_BEST_MODEL_AT_END и METRIC_FOR_BEST_MODEL, чтобы определить, должна ли метрика увеличиваться
)

trainer = Seq2SeqTrainer(
    model=model, # обучаемая модель
    args=training_args, # аргументы для обучения
    train_dataset=tokenized_dataset["train"], # датасет для обучения
    eval_dataset=tokenized_dataset["test"], # датасет для оценки
    tokenizer=tokenizer, # токенизатор
    data_collator=data_collator, # компоновщик данных для подгрузки при обучении
    compute_metrics=compute_metrics, # функция для подсчёта метрик
)

trainer.train(CHECKPOINT) # запуск обучения


# сохранение
model.to("cpu") # возвращаем модель на cpu, если она была на cuda во время обучения
model.save_pretrained(SUMMARIZATION_MODELS_DIR + MODEL_NAME + "_finetuned/") # сохраняем модель после обучения
tokenizer.save_pretrained(SUMMARIZATION_MODELS_DIR + MODEL_NAME + "_finetuned/") # сохраняем токенайзер
