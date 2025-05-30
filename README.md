# Text-Summarizer 📚🤖

## 📖 Overview

Text-Summarizer is an academic project developed for the Natural Language Processing course. It focuses on building and evaluating deep learning models for abstractive text summarization. This project provides a comprehensive workflow from data preparation and model training to inference and evaluation, enabling users to experiment with various sequence-to-sequence architectures.

The primary goal is to generate concise and meaningful summaries from longer texts, such as product reviews, while preserving the original meaning.

## 🔑 Key Features

*   **Multiple Model Architectures**: Implements several advanced sequence-to-sequence models including LSTM, GRU, BiLSTM, and variations with GloVe embeddings and Transformer components. See [Model Architectures](#-model-architectures).
*   **Modular Design**: Built with a base model class, allowing for easy extension and implementation of new architectures. Check out [architectures/BaseModel.py](architectures/BaseModel.py).
*   **Comprehensive Data Preprocessing**: Includes a robust pipeline for cleaning text, handling contractions, removing stopwords, tokenizing, and padding sequences. See the [`prepare_data`](utils.py) function in [utils.py](utils.py).
*   **Hyperparameter Tuning**: Supports grid search for hyperparameter optimization to find the best model configurations.
*   **Rich Evaluation Suite**: Employs a variety of metrics for thorough model assessment, including ROUGE, BERT Score, Cosine Similarity, and a custom weighted evaluation score. See [Evaluation](#-evaluation).
*   **Training and Inference Notebooks**: Provides dedicated Jupyter notebooks for a clear and organized workflow:
    *   [text-summarizer_training.ipynb](text-summarizer_training.ipynb) for model training and evaluation.
    *   [text_summarizer_inference.ipynb](text_summarizer_inference.ipynb) for generating summaries with trained models.
*   **Visualization**: Generates plots for training history (loss curves) and distributions of evaluation scores.

## 📂 Project Structure

```
text-summarizer/
├── architectures/        # Model architecture implementations (e.g., Seq2SeqLSTM.py, Seq2SeqBiLSTM.py)
│   ├── BaseModel.py      # Abstract base class for models
│   └── ...
├── report/               # Project report and related LaTeX files
├── results/              # Stores trained model weights, evaluation scores, plots, and training histories
│   ├── <model_name>/
│   │   ├── csv/          # Evaluation results in CSV format
│   │   ├── histories/    # Training history logs
│   │   ├── media/        # Plots (loss, metrics) and model architecture diagrams
│   │   └── weights/      # Saved model weights
│   └── evaluations_metrics/ # Aggregated evaluation reports across all models
├── .gitignore
├── attention.py          # Implementation of the attention mechanism
├── README.md             # This file
├── requirements.txt      # Python dependencies for CPU
├── requirements_ubuntuGPUcuda128.txt # Python dependencies for Ubuntu GPU with CUDA 12.8
├── text-summarizer_training.ipynb   # Jupyter notebook for training and evaluating models
├── text_summarizer_inference.ipynb  # Jupyter notebook for model inference
└── utils.py              # Utility functions for data preparation, evaluation, plotting, etc.
```

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/text-summarizer.git
    cd text-summarizer
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the requirements:**
    *   For CPU:
        ```bash
        pip install -r requirements.txt
        ```
    *   For Ubuntu GPU with CUDA 12.8:
        ```bash
        pip install -r requirements_ubuntuGPUcuda128.txt
        ```

4.  **Download NLTK stopwords:**
    Run the following Python code once to download the necessary NLTK data:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## 📊 Dataset

The project utilizes the [Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). This dataset contains product reviews and their corresponding summaries, making it suitable for abstractive text summarization tasks.

### Preprocessing Pipeline

The raw data undergoes a series of preprocessing steps, managed by the [`prepare_data`](utils.py) function in [utils.py](utils.py):
1.  **Data Loading**: A subset of the data is loaded.
2.  **Cleaning**:
    *   Removal of duplicate reviews and rows with missing values.
    *   HTML tag removal using BeautifulSoup.
    *   Expansion of contractions (e.g., "don't" to "do not").
    *   Removal of special characters, quotes, and content within parentheses.
    *   Normalization of repeated characters (e.g., "mmmmm" to "mm").
3.  **Tokenization & Filtering**:
    *   Text is converted to lowercase.
    *   Stopword removal (for input text, not summaries).
    *   Removal of short words (length <= 1).
4.  **Length Filtering**: Reviews and summaries are filtered based on word count (`max_text_len` and `max_summary_len` defined in [`prepare_data`](utils.py)).
5.  **Special Tokens**: Start (`sostok`) and end (`eostok`) tokens are added to the summary sequences.
6.  **Tokenization & Padding**:
    *   Keras `Tokenizer` is used to convert text to sequences of integers.
    *   Vocabulary is built based on word frequencies, excluding rare words.
    *   Sequences are padded to `max_text_len` for input and `max_summary_len` for output.
7.  **Data Splitting**: Data is split into training and validation sets.

## 📐 Model Architectures

The project implements several sequence-to-sequence (Seq2Seq) architectures, all inheriting from a common [`BaseModel`](architectures/BaseModel.py) class. This base class provides shared functionality for building encoder-decoder structures, inference, saving/loading models, and sequence-to-text conversion.

### Implemented Models:

*   **[`Seq2SeqLSTM`](architectures/Seq2SeqLSTM.py)**: A standard Seq2Seq model using LSTMs with an attention mechanism.
*   **[`Seq2SeqGRU`](architectures/Seq2SeqGRU.py)**: Similar to `Seq2SeqLSTM`, but uses GRUs instead of LSTMs, also with attention.
*   **[`Seq2SeqBiLSTM`](architectures/Seq2SeqBiLSTM.py)**: Employs a Bidirectional LSTM encoder for richer context capture, coupled with an attention mechanism and LSTM decoder.
*   **[`Seq2Seq3BiLSTM`](architectures/Seq2Seq3BiLSTM.py)**: Features a deeper encoder with three stacked Bidirectional LSTMs and an attention-based LSTM decoder.
*   **[`Seq2SeqLSTMGlove`](architectures/Seq2SeqLSTMGlove.py)**: An LSTM-based Seq2Seq model that initializes its embedding layer with pre-trained GloVe word embeddings.

The attention mechanism is implemented in [attention.py](attention.py).

## 📈 Training Process

The training workflow is orchestrated within the [text-summarizer_training.ipynb](text-summarizer_training.ipynb) notebook.

1.  **Data Preparation**: The [`prepare_data`](utils.py) function is called to process and tokenize the dataset.
2.  **Hyperparameter Configuration**:
    The [`create_hyperparameter_grid`](utils.py) function in [utils.py](utils.py) allows defining a grid of hyperparameters for tuning. This includes:
    *   Embedding dimensions
    *   Latent dimensions (for LSTMs/GRUs)
    *   Dropout rates (for encoder, decoder, and recurrent layers)
    *   Optimizer type (e.g., Adam, RMSprop) and learning rate
    *   Batch size
    *   Number of epochs
3.  **Model Iteration**: The script iterates through specified model classes and hyperparameter combinations.
4.  **Model Compilation & Training**:
    *   Each model instance is compiled with the specified optimizer and loss function (Sparse Categorical Crossentropy).
    *   Callbacks are used during training:
        *   **EarlyStopping**: To prevent overfitting by stopping training if validation loss doesn't improve.
        *   **ReduceLROnPlateau**: To adaptively adjust the learning rate if progress stagnates.
        *   *(Optional: LearningRateScheduler for custom learning rate schedules)*
5.  **Saving Artifacts**:
    *   **Model Weights**: Trained model weights are saved to the `results/<model_name>/weights/` directory.
    *   **Training History**: Loss and validation loss per epoch are plotted and saved to `results/<model_name>/media/graphs/`. The history data is also logged to `results/<model_name>/histories/`.
    *   **Model Architecture**: A diagram of the model architecture can be saved to `results/<model_name>/media/architectures/`.
    *   **Generated Summaries**: After training, summaries are generated for the validation set and saved to `results/<model_name>/csv/`.

## 🚀 Inference

The [text_summarizer_inference.ipynb](text_summarizer_inference.ipynb) notebook demonstrates how to load a trained model and generate summaries for new text inputs or samples from the validation set.

1.  **Load Data and Tokenizers**: The [`prepare_data`](utils.py) function is used to get the necessary tokenizers and data parameters (`max_text_len`, `max_summary_len`, etc.).
2.  **Instantiate Model**: The desired model architecture is instantiated with the same parameters used during training.
3.  **Load Weights**: Pre-trained model weights are loaded from the `results/<model_class_name>/weights/` directory.
4.  **Preprocess Input**: Input text is preprocessed using the same pipeline as the training data (via `x_tokenizer`).
5.  **Generate Summary**: The model's `decode_sequence` method is used to generate the summary. This method typically involves:
    *   Encoding the input text.
    *   Initializing the decoder state with the encoder's final state.
    *   Iteratively predicting the next token in the summary sequence until an end token or maximum length is reached.
6.  **Postprocess Output**: The generated sequence of tokens is converted back to text using `y_tokenizer`.

## 📊 Evaluation

Model performance is assessed using a suite of metrics, calculated in the "Evaluation" section of the [text-summarizer_training.ipynb](text-summarizer_training.ipynb) notebook.

### Evaluation Metrics:

The following functions from [utils.py](utils.py) are used for evaluation:
*   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**:
    *   [`evaluate_rouge`](utils.py): Calculates ROUGE-1, ROUGE-2, and ROUGE-L F1-scores, measuring n-gram overlap between predicted and original summaries.
*   **Cosine Similarity**:
    *   [`evaluate_cosine_similarity`](utils.py): Uses sentence embeddings (e.g., from `paraphrase-MiniLM-L6-v2`) to measure semantic similarity between predicted and original summaries.
*   **BERT Score**:
    *   [`evaluate_bert_score`](utils.py): Computes similarity using contextual embeddings from BERT, comparing predicted and original summaries.
*   **Custom Evaluation (`myevaluation`)**:
    *   [`evaluate_myevalutation`](utils.py): A weighted combination of BERT Score, cosine similarity (predicted vs. original summary), and keyword overlap. The weights are defined within this function.

### Evaluation Process:

1.  **Load Summaries**: For each trained model instance, the generated summaries (predicted vs. original) are loaded from the CSV files in `results/<model_name>/csv/`.
2.  **Calculate Metrics**: Each evaluation metric is calculated for all summary pairs.
3.  **Save Results**:
    *   The detailed scores for each summary pair, along with the mean scores, are saved back to an "evaluated" CSV file (e.g., `*_summaries_evaluated.csv`) in the same directory.
    *   Plots (histograms) for each metric's distribution are generated and saved in `results/<model_name>/media/graphs/<instance_name>/`. These plots are created by functions like [`plot_rouge`](utils.py), [`plot_bert_score`](utils.py), etc.
4.  **Aggregate Reports**:
    *   Individual metric reports (e.g., `evaluation_mean_rouge1.txt`) are generated in `results/evaluations_metrics/`, ranking models by that specific metric.
    *   A summary table in Markdown format (`table_report.md`) is created in `results/evaluations_metrics/`, comparing all model instances across all key metrics. The best score for each metric is highlighted.

## 📜 Report

The project includes a detailed report located in the [report/](report/) directory. The main PDF document is [report/report.pdf](report/report.pdf), which discusses the methodology, experiments, results, and conclusions.
