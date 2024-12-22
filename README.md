# Project: Fine-Tuned Embedder for Multitask Benchmarking

This repository contains a comprehensive pipeline for fine-tuning embeddings and achieving high performance across 11 defined benchmarks. The project integrates multiple datasets and tasks, leveraging state-of-the-art techniques in semantic similarity and retrieval, specifically tailored for Russian-language tasks.

## Repository Structure

```
main
├── notebooks
│   ├── Brand_Similarity.ipynb              # Generates brand similarity candidates using semantic embeddings.
│   ├── Semi_Automated_Annotation_Triplets.ipynb  # Automates triplet generation for fine-tuning.
│   ├── Fine-tuned_Embedder.ipynb           # Fine-tunes the embedding model for all tasks.
│   ├── WB_Embedder_GigaChat.ipynb          # Fine-tunes the Giga-Embeddings model for benchmarking.
├── README                                  # Comprehensive guide to the repository.
├── requirements.txt                        # Dependencies for the project.
```

## Overview

This project addresses 11 benchmarks focusing on retrieval, classification, and ranking tasks. The primary datasets used include:
- **`data_sampled_30.csv`**: The main dataset for retrieval and classification tasks, preprocessed to include necessary information such as titles, descriptions, and queries.
- **`triplet_candidates.csv`**: Triplets generated for the **WBSimGoods-triplets** task, used to fine-tune the model for similarity and ranking benchmarks.
- **`brand_candidates.csv`**: Brand similarity pairs generated for the **WBBrandSyns** task, providing data to optimize embeddings for synonym identification.

The tasks are implemented and validated using a multi-task learning framework with shared embeddings and task-specific heads.

## Benchmarks

### Retrieval Tasks:
1. **WBQAGoods-all**: Ranking answers with product information.
2. **WBQAGoods-answer**: Ranking answers without product context.
3. **WBQAGoods-description**: Ranking descriptions for product-specific questions.
4. **WBQASupportFacts**: Ranking support facts for product-specific queries.

### Classification Tasks:
5. **WBReviewsClassification**: Classifying product reviews by ratings (1-5).
6. **WBReviewsPolarityClassification**: Classifying reviews as positive or negative.
7. **WBReviewsSummarizationClassification**: Classifying review summaries as good or bad.
8. **WBGoodsCategoriesClassification**: Classifying products into parent categories.
9. **WBGenderClassification**: Determining the gender of users based on purchase history.
10. **WBQASupportShort**: Determining if a response is "yes" or "no" based on context.

### Ranking Tasks:
11. **WBSearchQueryNmRelevance**: Ranking products for search queries.

## Pipelines

### Data Preprocessing
- **Brand Similarity Pipeline**: This pipeline leverages RuBERT embeddings to generate brand similarity candidates for the **WBBrandSyns** task. Semantic embeddings are used to compute cosine similarity between brand names, ensuring high-quality candidate generation.
- **Triplet Annotation Pipeline**: A semi-automated process for generating triplets for the **WBSimGoods-triplets** task. Anchors, positives, and negatives are defined based on semantic relationships and parent categories. The process is augmented with manual validation for higher accuracy.
- **Classification and Retrieval Preprocessing**: The `data_sampled_30.csv` dataset was carefully preprocessed to include relevant features (titles, descriptions, queries) for retrieval and classification tasks. This ensures robust performance across multiple benchmarks.

### Fine-Tuning Pipeline
The fine-tuning pipeline implements a multi-task learning framework:
1. A shared encoder processes all tasks, enabling the model to leverage common representations.
2. Task-specific heads are used for classification, retrieval, and ranking benchmarks.
3. Contrastive loss is applied to optimize embedding spaces for retrieval and ranking tasks.
4. Multi-task training ensures balanced performance across diverse benchmarks.

## Models Evaluated
Two models were evaluated in this project:

1. **`DeepPavlov/rubert-base-cased`**:
   - **Strengths**: 
     - Well-established for Russian-language tasks.
     - Robust performance on classification tasks due to its pretraining on diverse Russian datasets.
   - **Weaknesses**:
     - Limited flexibility for instruction-based retrieval tasks.
     - Requires additional fine-tuning for ranking and similarity benchmarks.

2. **`ai-sage/Giga-Embeddings-instruct`**:
   - **Strengths**:
     - Supports task-specific instructional prefixes, making it ideal for retrieval and ranking tasks.
     - Strong performance in embedding-based similarity tasks like **WBSimGoods-triplets** and **WBBrandSyns**.
   - **Weaknesses**:
     - As a newer model, it may require more experimentation to optimize for classification tasks.
     - Instructional design introduces complexity in dataset preparation.

Both models were fine-tuned and tested against the 11 benchmarks, leveraging the same data pipeline for consistency in evaluation.

## Notebooks

### 1. Brand_Similarity.ipynb
- Uses RuBERT embeddings to generate high-quality brand similarity pairs for the **WBBrandSyns** task.
- Saves results to `brand_candidates.csv`.

### 2. Semi_Automated_Annotation_Triplets.ipynb
- Automates the generation of triplets for the **WBSimGoods-triplets** task.
- Saves results to `triplet_candidates.csv` for manual validation and fine-tuning.

### 3. Fine-tuned_Embedder.ipynb
- Implements multi-task fine-tuning using the datasets:
  - **`data_sampled_30.csv`**
  - **`triplet_candidates.csv`**
  - **`brand_candidates.csv`**
- Fine-tunes embeddings to optimize for all benchmarks.

### 4. WB_Embedder_GigaChat.ipynb
- Focuses on fine-tuning the **`ai-sage/Giga-Embeddings-instruct`** model for retrieval and ranking tasks.
- Compares performance against the RuBERT-based model.

## Installation

### Requirements
- Python 3.8+
- Libraries specified in `requirements.txt`:

```
pandas
sentence-transformers
transformers
torch
tqdm
datasets
```



### Model Outputs
The fine-tuned models and classification/ranking heads are saved in the `fine_tuned_model` directory.

## Future Work
- Add more robust evaluation scripts for each benchmark.
- Explore multilingual fine-tuning for additional language support.
- Integrate real-time deployment pipelines for the fine-tuned model.


# Проект: Дообученная модель для генерации эмбеддингов 

Этот репозиторий содержит комплексный пайплайн для настройки эмбеддингов и достижения высокого уровня производительности по 11 заданным тестам. Проект интегрирует несколько наборов данных и задач, используя передовые техники семантического сопоставления и поиска, специально адаптированные для задач на русском языке.

## Структура репозитория

```
main
├── notebooks
│   ├── Brand_Similarity.ipynb              # Генерация кандидатов на совпадение брендов с использованием семантических эмбеддингов.
│   ├── Semi_Automated_Annotation_Triplets.ipynb  # Полуавтоматическая генерация триплетов для настройки.
│   ├── Fine-tuned_Embedder.ipynb           # Настройка эмбеддинговой модели для всех задач.
│   ├── WB_Embedder_GigaChat.ipynb          # Настройка модели Giga-Embeddings для тестирования.
├── README                                  # Подробное руководство по репозиторию.
├── requirements.txt                        # Зависимости для проекта.
```

## Обзор

Проект рассчитан на 11 тестов, охватвающих задачи поиска, классификации и ранжирования. Основные используемые наборы данных:
- **`data_sampled_30.csv`**: Основной набор данных для задач поиска и классификации, предварительно обработанный для включения необходимой информации, такой как заголовки, описания и запросы.
- **`triplet_candidates.csv`**: Триплеты, созданные для задачи **WBSimGoods-triplets**, используются для настройки модели на задачи сходства и ранжирования.
- **`brand_candidates.csv`**: Пары брендов для задачи **WBBrandSyns**, предоставляющие данные для оптимизации эмбеддингов для идентификации синонимов.

Задачи реализуются и проверяются с использованием многозадачного фреймворка с общими эмбеддингами и головами, специфичными для задач.

## Тесты

### Задачи поиска:
1. **WBQAGoods-all**: Ранжирование ответов с учетом информации о товаре.
2. **WBQAGoods-answer**: Ранжирование ответов без учета контекста товара.
3. **WBQAGoods-description**: Ранжирование описаний товаров для вопросов.
4. **WBQASupportFacts**: Ранжирование фактов поддержки для вопросов, связанных с товарами.

### Задачи классификации:
5. **WBReviewsClassification**: Классификация отзывов о товарах по оценкам (1-5).
6. **WBReviewsPolarityClassification**: Классификация отзывов как положительных или отрицательных.
7. **WBReviewsSummarizationClassification**: Классификация саммари по набору отзывов на хорошие саммари и плохие. 
8. **WBGoodsCategoriesClassification**: Классификация товаров по родительским категориям.
9. **WBGenderClassification**: Определение пола пользователей на основе истории покупок.
10. **WBQASupportShort**: Делит ответы на короткие и длинные.

### Задачи ранжирования:
11. **WBSearchQueryNmRelevance**: Ранжирование товаров по поисковым запросам.

---

## Пайплайны

### Предварительная обработка данных
- **Пайплайн сравнения брендов**: Использует эмбеддинги RuBERT для генерации кандидатов на совпадение брендов для задачи **WBBrandSyns**. Косинусное сходство используется для оценки схожести между названиями брендов.
- **Пайплайн аннотации триплетов**: Полуавтоматическая генерация триплетов для задачи **WBSimGoods-triplets**. Анкеры, позитивные и негативные примеры определяются на основе семантических отношений и родительских категорий.
- **Предварительная обработка классификации и поиска**: Набор данных `data_sampled_30.csv` был предварительно обработан для включения всех необходимых характеристик (заголовки, описания, запросы).

### Пайплайн настройки
1. Общий энкодер используется для обработки всех задач, что упрощает генерализацию.
2. Головы для классификации, поиска и ранжирования.
3. Контрастивная функция потерь оптимизирует пространства эмбеддингов для задач поиска и ранжирования.
4. Многозадачное обучение обеспечивает сбалансированную производительность для всех тестов.

---

## Оцененные модели

1. **`DeepPavlov/rubert-base-cased`**:
   - **Преимущества**:
     - Хорошо зарекомендовал себя для задач на русском языке.
     - Надежная производительность для задач классификации.
   - **Недостатки**:
     - Ограниченная гибкость для задач поиска с использованием инструкций.
     - Требует дополнительной настройки для задач ранжирования.

2. **`ai-sage/Giga-Embeddings-instruct`**:
   - **Преимущества**:
     - Поддержка инструктивных префиксов для задач поиска.
     - Отличная производительность в задачах сходства.
   - **Недостатки**:
     - Новая модель, требующая дополнительной оптимизации.
     - Инструкции усложняют предварительную обработку данных.
