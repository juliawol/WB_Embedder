# Project: Fine-Tuned Embedder for Multitask Benchmarking

This repository contains a comprehensive pipeline for fine-tuning embeddings and achieving high performance across 11 defined benchmarks. The project integrates multiple datasets and tasks, leveraging state-of-the-art techniques in semantic similarity and retrieval, specifically tailored for Russian-language tasks.

## Repository Structure

```
main
├── notebooks
│   ├── Brand_Similarity.ipynb              # Generates brand similarity candidates using semantic embeddings.
│   ├── Semi_Automated_Annotation_Triplets.ipynb  # Automates triplet generation for fine-tuning.
│   ├── Fine-tuned_Embedder.ipynb           # Fine-tunes the embedding model for all tasks.
├── README                                  # Comprehensive guide to the repository.
├── requirements.txt                        # Dependencies for the project.
```

## Overview

This project addresses 11 benchmarks focusing on retrieval, classification, and ranking tasks. The primary datasets used include:
- **`data_sampled_30.csv`**: The main dataset for retrieval tasks.
- **`triplet_candidates.csv`**: Triplets generated for the **WBSimGoods-triplets** task.
- **`brand_candidates.csv`**: Brand similarity pairs generated for the **WBBrandSyns** task.

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

### Fine-Tuning Pipeline
The fine-tuning pipeline implements a multi-task learning framework:
1. A shared encoder processes all tasks, enabling the model to leverage common representations.
2. Task-specific heads are used for classification, retrieval, and ranking benchmarks.
3. Contrastive loss is applied to optimize embedding spaces for retrieval and ranking tasks.
4. Multi-task training ensures balanced performance across diverse benchmarks.

## Modeling Decisions
1. **RuBERT for Semantic Embeddings**:
   - Chosen for its strong performance on Russian-language tasks.
   - Ensures robust handling of language-specific nuances in retrieval and similarity tasks.
2. **Contrastive Loss for Ranking Tasks**:
   - Enables effective optimization of embedding distances for triplet-based tasks.
3. **Multi-Task Framework**:
   - Balances shared knowledge representation and task-specific learning, improving overall benchmark performance.

## Challenges and Solutions
1. **Data Imbalance**:
   - Challenges: Certain categories and tasks lacked sufficient training data.
   - Solution: Semi-automated generation and manual annotation pipelines to augment datasets.
2. **Computational Complexity**:
   - Challenges: Large datasets and embeddings caused memory and compute bottlenecks.
   - Solution: Efficient batching and approximate nearest neighbor methods were employed for similarity calculations.
3. **Task Divergence**:
   - Challenges: Balancing retrieval, classification, and ranking tasks with a single model.
   - Solution: A multi-task learning framework with weighted losses for task prioritization.

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

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repo.git
   cd your_repo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure all datasets are placed in the appropriate location:
   - `data_sampled_30.csv` (main dataset)
   - `triplet_candidates.csv` (triplets for WBSimGoods-triplets)
   - `brand_candidates.csv` (brand similarity pairs for WBBrandSyns)

## Usage

### Run Notebooks
Each notebook is designed to handle a specific stage of the pipeline. Open the desired notebook in Jupyter or Colab and run the cells sequentially.

### Fine-Tuning
To fine-tune the model for all benchmarks, execute the `Fine-tuned_Embedder.ipynb` notebook. This will:
- Train the model using the datasets.
- Save the fine-tuned model and heads.

### Model Outputs
The fine-tuned model and classification/ranking heads are saved in the `fine_tuned_model` directory.

## Future Work
- Add more robust evaluation scripts for each benchmark.
- Explore multilingual fine-tuning for additional language support.
- Integrate real-time deployment pipelines for the fine-tuned model.
