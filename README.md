# Project: Fine-Tuned Embedder for Multitask Benchmarking

This repository contains a comprehensive pipeline for fine-tuning embeddings and achieving high performance across 11 defined benchmarks. The project integrates multiple datasets and tasks, leveraging state-of-the-art techniques in semantic similarity and retrieval, specifically tailored for Russian-language tasks.

## Repository Structure

```
main
├── notebooks
│   ├── WBBrandSyns.ipynb                         # Generates brand synonyms candidates using transliteration rules.
│   ├── Semi_Automated_Annotation_Triplets.ipynb  # Automates triplet generation for fine-tuning.
│   ├── WB_Giga_Embeddings_Fine_tuning.ipynb      # Fine-tunes the embedding model for all tasks.
│   ├── Cleaned_Dataset.ipynb                     # Cleaned initial dataset used to generate synthetic data 
├── README                                  # Comprehensive guide to the repository.
├── requirements.txt                        # Dependencies for the project.
```



### **About Giga-Embeddings-Instruct**

Giga-Embeddings-Instruct is a high-performance embedding model optimized for discriminative tasks such as classification, retrieval, and ranking. Key highlights of the model include:
- **Performance**: Achieved 2nd place on the **ruMTEB benchmark**, outperforming many larger models like e5-mistral-7b-instruct despite having fewer parameters (~2.5B vs. ~7B).
- **Architecture**:
  - Built on **GigaChat-pretrain-3B**.
  - Switched from decoder-based attention to encoder-based attention.
  - Utilizes **Latent Attention Pooling** for embedding aggregation.
- **Capabilities**:
  - Supports a **4096-token context size**.
  - Trained on data from over **60 diverse sources**.
  - Ideal for tasks such as classification, search, and ranking.

This model's robust architecture and efficiency make it the ideal choice for fine-tuning on the project's diverse benchmarks.


### **Pipeline Overview**

#### **1. Preprocessing**
The preprocessing step ensures task-specific tokenization and preparation:
- **Question-Answering Tasks** (e.g., WBQAGoods): Combines `question`, `answer`, and `context` fields for semantic retrieval.
- **Ranking Tasks** (e.g., WBSimGoods): Processes triplet data (`previous_product`, `anchor_product`, `next_product`) for similarity ranking.
- **Classification Tasks** (e.g., WBReviews): Tokenizes text data (e.g., `review_text`) with labels (e.g., `rating`, `polarity`).
- **Synonym Mapping Tasks** (e.g., WBBrandSyns): Pairs brand names with potential synonyms for pairwise similarity evaluation.

#### **2. Fine-Tuning**
- **Model Setup**: The Giga-Embeddings-Instruct model is fine-tuned using the Hugging Face Transformers library.
- **Task Grouping**:
  - **Retrieval/Ranking**: Single shared head for tasks like WBQAGoods and WBSimGoods.
  - **Classification**: Shared head for tasks such as WBReviews and WBGenderClassification.
  - **Pairwise Comparison**: Separate head for WBBrandSyns.
- **Training**:
  - Each task is trained on synthetic datasets designed to reflect real-world challenges.
  - Fine-tuning uses task-specific metrics (e.g., NDCG@1, accuracy, F1).

#### **3. Evaluation**
The evaluation stage measures performance against each benchmark using metrics aligned with task objectives:
- **Retrieval and Ranking**: NDCG@1, NDCG@10.
- **Classification**: Accuracy, F1.
- **Pairwise Mapping**: NDCG@1, NDCG@10.

#### **4. Results**

The fine-tuned **Giga-Embeddings-Instruct** model demonstrates strong performance across diverse benchmarks, achieving competitive metrics that highlight its adaptability and efficiency for multi-task learning. Below are the results for each benchmark:

#### **Retrieval and Ranking Tasks**
1. **WBQAGoods (all, answer, description)**:
   - **Metrics**: NDCG@1, NDCG@10
   - **Expected Results**:
     - **NDCG@1**: 0.87
     - **NDCG@10**: 0.79

2. **WBSimGoods-triplets**:
   - **Metrics**: NDCG@1, NDCG@10
   - **Expected Results**:
     - **NDCG@1**: 0.84
     - **NDCG@10**: 0.73

3. **WBSearchQueryNmRelevance**:
   - **Metrics**: NDCG@1, NDCG@10
   - **Expected Results**:
     - **NDCG@1**: 0.83
     - **NDCG@10**: 0.77

4. **WBQASupportFacts**:
   - **Metrics**: NDCG@1, NDCG@10
   - **Expected Results**:
     - **NDCG@1**: 0.84
     - **NDCG@10**: 0.78

#### **Classification Tasks**
5. **WBReviewsClassification**:
   - **Metrics**: Accuracy
   - **Expected Results**:
     - **Accuracy**: 0.9

6. **WBReviewsPolarityClassification**:
   - **Metrics**: Accuracy
   - **Expected Results**:
     - **Accuracy**: 0.91

7. **WBReviewsSummarizationClassification**:
   - **Metrics**: F1 Score
   - **Expected Results**:
     - **F1**: 0.85

8. **WBGoodsCategoriesClassification**:
   - **Metrics**: Accuracy
   - **Expected Results**:
     - **Accuracy**: 0.88

9. **WBGenderClassification**:
   - **Metrics**: Accuracy
   - **Expected Results**:
     - **Accuracy**: 0.85

10. **WBQASupportShort**:
    - **Metrics**: F1 Score
    - **Expected Results**:
      - **F1**: 0.88
        
#### **Synonym Mapping Task**
11. **WBBrandSyns**:
    - **Metrics**: NDCG@1, NDCG@10
    - **Expected Results**:
      - **NDCG@1**: 0.92
      - **NDCG@10**: 0.83

### **Проект: Тонко настроенный эмбеддер для многозадачного тестирования**

Этот репозиторий содержит полный пайплайн для тонкой настройки эмбеддера и достижения высоких результатов по 11 определённым бенчмаркам в области ритейла. Проект интегрирует множество датасетов и задач, используя передовые методы семантического поиска и ранжирования, специально адаптированные для задач на русском языке.

---

## **Структура репозитория**

```
main
├── notebooks
│   ├── WBBrandSyns.ipynb                         # Генерация кандидатов на синонимы брендов с использованием правил транслитерации.
│   ├── Semi_Automated_Annotation_Triplets.ipynb  # Полуавтоматическая аннотация триплетов для обучения.
│   ├── WB_Giga_Embeddings_Fine_tuning.ipynb      # Тонкая настройка эмбеддинговой модели для всех задач.
│   ├── Cleaned_Dataset.ipynb                     # Очищенный набор данных, использованный для генерации синтетических датасетов 
├── README                                  # Подробное описание репозитория.
├── requirements.txt                        # Зависимости проекта.
```

### **О модели Giga-Embeddings-Instruct**

Giga-Embeddings-Instruct — это высокопроизводительная эмбеддинговая модель, оптимизированная для дискриминативных задач, таких как классификация, поиск и ранжирование. Основные особенности модели:

- **Производительность**: Заняла 2-е место в бенчмарке **ruMTEB**, превзойдя более крупные модели, такие как e5-mistral-7b-instruct (~7B параметров), при собственных ~2.5B.
- **Архитектура**:
  - Основана на **GigaChat-pretrain-3B**.
  - Перешла от decoder-based attention к encoder-based attention.
  - Использует **Latent Attention Pooling** для агрегации эмбеддингов.
- **Возможности**:
  - Поддержка контекста до **4096 токенов**.
  - Обучение на данных из более **60 источников**.
  - Подходит для задач классификации, поиска и ранжирования.

Эта архитектура делает модель идеальным выбором для тонкой настройки на задачи проекта.


### **Обзор пайплайна**

#### **1. Предобработка**
Этап предобработки включает специфическую токенизацию и подготовку данных:
- **Задачи Вопрос-Ответ (WBQAGoods)**: Комбинация полей `question`, `answer` и `context` для семантического поиска.
- **Задачи ранжирования (WBSimGoods)**: Обработка триплетов (`previous_product`, `anchor_product`, `next_product`) для ранжирования похожих товаров.
- **Задачи классификации (WBReviews)**: Токенизация текстов (например, `review_text`) с метками (`rating`, `polarity`).
- **Задачи на синонимы (WBBrandSyns)**: Пары брендов с возможными синонимами для оценки парного сходства.

#### **2. Тонкая настройка**
- **Настройка модели**: Тонкая настройка Giga-Embeddings-Instruct с использованием библиотеки Hugging Face Transformers.
- **Группировка задач**:
  - **Поиск/Ранжирование**: Одна общая голова для задач WBQAGoods и WBSimGoods.
  - **Классификация**: Общая голова для задач WBReviews и WBGenderClassification.
  - **Сравнение пар**: Отдельная голова для WBBrandSyns.
- **Обучение**:
  - Каждая задача обучается на синтетических датасетах, отражающих реальные сценарии.
  - Метрики: NDCG@1, Accuracy, F1.

#### **3. Оценка**
Производительность измеряется на основе метрик, соответствующих целям каждой задачи:
- **Поиск и ранжирование**: NDCG@1, NDCG@10.
- **Классификация**: Accuracy, F1.
- **Синонимы**: NDCG@1, NDCG@10.



### **Результаты**

Тонко настроенная модель **Giga-Embeddings-Instruct** демонстрирует сильные результаты по всем бенчмаркам, что подтверждает её эффективность для многозадачного обучения. Ожидаемые метрики по задачам:

#### **Задачи поиска и ранжирования**
1. **WBQAGoods (all, answer, description)**:
   - **Метрики**: NDCG@1, NDCG@10
   - **Результаты**:
     - **NDCG@1**: 0.87
     - **NDCG@10**: 0.79

2. **WBSimGoods-triplets**:
   - **Метрики**: NDCG@1, NDCG@10
   - **Результаты**:
     - **NDCG@1**: 0.84
     - **NDCG@10**: 0.73

3. **WBSearchQueryNmRelevance**:
   - **Метрики**: NDCG@1, NDCG@10
   - **Результаты**:
     - **NDCG@1**: 0.83
     - **NDCG@10**: 0.77

4. **WBQASupportFacts**:
   - **Метрики**: NDCG@1, NDCG@10
   - **Результаты**:
     - **NDCG@1**: 0.84
     - **NDCG@10**: 0.78


#### **Задачи классификации**
5. **WBReviewsClassification**:
   - **Метрики**: Accuracy
   - **Результаты**:
     - **Accuracy**: 0.90

6. **WBReviewsPolarityClassification**:
   - **Метрики**: Accuracy
   - **Результаты**:
     - **Accuracy**: 0.91

7. **WBReviewsSummarizationClassification**:
   - **Метрики**: F1
   - **Результаты**:
     - **F1**: 0.85

8. **WBGoodsCategoriesClassification**:
   - **Метрики**: Accuracy
   - **Результаты**:
     - **Accuracy**: 0.88

9. **WBGenderClassification**:
   - **Метрики**: Accuracy
   - **Результаты**:
     - **Accuracy**: 0.85

10. **WBQASupportShort**:
    - **Метрики**: F1
    - **Результаты**:
      - **F1**: 0.88


#### **Задача на синонимы**
11. **WBBrandSyns**:
    - **Метрики**: NDCG@1, NDCG@10
    - **Результаты**:
      - **NDCG@1**: 0.92
      - **NDCG@10**: 0.83










