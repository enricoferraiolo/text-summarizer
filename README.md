# Text-Summarizer 📚🤖

## 📖 Overview

Text-Summarizer is an academic project developed for the Natural Language Processing course 2024/2025. It focuses on building and evaluating models for text summarization, offering a comprehensive workflow that covers data preparation, training and inference processes.

## 🔑 Key Features
- Multiple model architectures based on LSTM and GRU with attention mechanisms
- Data preprocessing pipeline for text cleaning and tokenization
- Comprehensive evaluation using various metrics (ROUGE, WER, BERT Score, etc.)
- Inference capabilities for generating summaries from new inputs
- Visualization tools for analyzing model performance

## 📂 Project Structure
```
text-summarizer/
├── architectures/        # Model architecture implementations  
├── report/               # Documentation and analysis of methodology  
├── results/              # Trained models and evaluation results  
├── attention.py          # Attention mechanism implementation  
├── requirements.txt      # Project dependencies  
├── text-summarizer_training.ipynb   # Training workflow notebook  
├── text_summarizer_inference.ipynb  # Inference notebook  
└── utils.py              # Utility functions  
```

## 📦 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/enricoferraiolo/text-summarizer.git
   cd text-summarizer
   ```
2. Install the requirements:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Dataset
The project uses the Amazon Fine Food Reviews dataset to train and evaluate summarization models. This dataset contains product reviews and summaries, making it suitable for abstractive text summarization tasks.
The original data is processed in order to:
- Remove duplicates
- Drop rows with missing values
- Filter out reviews that are too short or too long

### Preprocessing Pipeline
The preprocessing pipeline includes the following steps:
1. Text cleaning (removing HTML tags, special characters, etc.)
2. Tokenization (adding special tokens for start and end of sequences, separating input and output sequences, padding sequences to a fixed length)

## 📐 Model Architectures
The project implements several sequence-to-sequence architectures.

### Base Architecture
All models extend a common BaseModel abstract class that provides:

- Methods for building encoder and decoder components
- Common functionality for inference, saving, and loading models
- Conversion between sequences and text

### Implemented Models
- **Seq2SeqLSTM**: Basic LSTM-based sequence-to-sequence with attention
- **Seq2SeqGRU**: GRU-based sequence-to-sequence with attention
- **Seq2SeqBiLSTM**: Bidirectional LSTM encoder with attention
- **Seq2Seq3BiLSTM**: Triple-stacked Bidirectional LSTM encoder
- **Seq2SeqLSTMGlove**: LSTM model with pre-trained GloVe embeddings

## 📈 Training Process
The training process is implemented in the `text-summarizer_training.ipynb` notebook. It includes:
1. Data Preparation:
   - Splitting data into training and validation sets
   - Tokenizing and padding sequences
2. Hyperparameter Configuration - the user can set how many hyperparameters he wants to tune, not only the hyperparameter value itself but also set different values for each hyperparameter, then the model will be trained with all combinations of these values. You can set:
     - Embedding dimensions
     - Latent dimensions
     - Dropout rates
     - Optimizer
     - Learning rate
     - Batch size
     - Number of epochs
3. Training Callbacks:
   - Early stopping
   - Learning Rate Scheduler for adaptive learning rate adjustment
   - Reduce LR on plateau for optimization
4. Loss Function:
     - Sparse Categorical Crossentropy

## 🚀 Inference
The inference process is implemented in the `text_summarizer_inference.ipynb` notebook. It allows users to input new text and generate summaries using the trained models. 
1. Load the trained model with its weights
2. Preprocess the input text following the same pipeline as during training
3. Generate the summary using the model's `decode_sequence` method

## 📊 Evaluation
### Evaluation Metrics
The models are evaluated using multiple metrics to provide a comprehensive assessment of their performance:
- **ROUGE 1**, **ROUGE 2**, **ROUGE L** (Recall-Oriented Understudy for Gisting Evaluation) for measuring n-gram overlap
- **WER** (Word Error Rate) Measures the minimum number of edits required to transform the predicted summary into the reference summary
- **Cosine Similarity**: Using sentence embeddings to measure semantic similarity.
- **BERT Score**: Measures the similarity between predicted and reference summaries using BERT embeddings.
- **Custom Evaluation**: Combines multiple metrics with weighted importance.

### Evaluation Process
After the training process, the models are evaluated on the validation set using the defined metrics. The results are saved in the `results/` directory for further analysis.
Each model's is inferred on the validation set and the results are saved in a CSV file. The evaluation metrics are calculated and saved in a separate subdirectory for each model.
The evaluation results include:
- ROUGE scores
- WER scores
- BERT scores
- Cosine similarity scores
- Custom evaluation scores
- Plots of the evaluation metrics

The results are saved in the `results/` directory, organized by model name. Each subdirectory contains:
- `csv/` directory with the evaluation results in CSV format
- `histories/` directory with training histories files
- `media/` which contains the plots of the evaluation metrics. It also contains the model's architecture diagram

A `evaluation_metrics/` directory is generated, containing 
the evaluation results for each evaluation metric.
Furthermore, a `evaluation_metrics/table_report.md` file is generated, containing the evaluation results for each model and each evaluation metric. Here we can find the best model for each evaluation metric.

[link](https://deepwiki.com/enricoferraiolo/text-summarizer)