import os
from keras import backend as K
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.optimizers import Adam, RMSprop


def create_hyperparameter_grid(
    embedding_dim=[300],
    latent_dim=[256],
    encoder_dropout=[0.3],
    encoder_recurrent_dropout=[0.3],
    decoder_dropout=[0.3],
    decoder_recurrent_dropout=[0.3],
    optimizers=[{"class": Adam, "learning_rate": 0.001}],
    epochs=[50],
    batch_size=[64],
):
    """
    Create a permutation grid of hyperparameters to search for the best model configuration.

    Args:
        latent_dim (list): Dimension of the latent space
        embedding_dim (list): Dimension of the word embeddings
        encoder_dropout (list): Encoder dropout
        encoder_recurrent_dropout (list): Encoder recurrent dropout
        decoder_dropout (list): Decoder dropout
        decoder_recurrent_dropout (list): Decoder recurrent dropout
        optimizers (list): List of dictionaries with the optimizer class and learning rate
        epochs (list): Number of epochs
        batch_size (list): Batch size

    Returns:
        list: permutation grid of hyperparameters
    """

    hyperparameter_grid = []

    # Create all combinations of hyperparameters
    for latent_dim_val in latent_dim:
        for embedding_dim_val in embedding_dim:
            for encoder_dropout_val in encoder_dropout:
                for encoder_recurrent_dropout_val in encoder_recurrent_dropout:
                    for decoder_dropout_val in decoder_dropout:
                        for decoder_recurrent_dropout_val in decoder_recurrent_dropout:
                            for optimizer_config in optimizers:
                                for epochs_val in epochs:
                                    for batch_size_val in batch_size:
                                        hyperparameter_grid.append(
                                            {
                                                "latent_dim": latent_dim_val,
                                                "embedding_dim": embedding_dim_val,
                                                "encoder_dropout": encoder_dropout_val,
                                                "encoder_recurrent_dropout": encoder_recurrent_dropout_val,
                                                "decoder_dropout": decoder_dropout_val,
                                                "decoder_recurrent_dropout": decoder_recurrent_dropout_val,
                                                "optimizer_class": optimizer_config[
                                                    "class"
                                                ],
                                                "learning_rate": optimizer_config[
                                                    "learning_rate"
                                                ],
                                                "epochs": epochs_val,
                                                "batch_size": batch_size_val,
                                            }
                                        )

    print(f"Number of hyperparameter combinations: {len(hyperparameter_grid)}")
    return hyperparameter_grid


def plot_rouge(
    df,
    save_path,
    model_name,
    metric="rouge1",
    bins=30,
    remove_zeros_rows=True,
    color="salmon",
    title=None,
):
    # Filter rows with all zero ROUGE scores
    if remove_zeros_rows:
        df = df[
            (df["rouge_scores"].apply(lambda x: x[metric].fmeasure) > 0)
        ]  # Filter rows with all zero ROUGE scores

    # Plot the distribution of ROUGE scores
    plt.hist(
        df["rouge_scores"].apply(lambda x: x[metric].fmeasure),
        bins=bins,
        color=color,
    )
    plt.suptitle(title or f"Distribution of {metric.upper()} scores", fontsize=16)
    plt.xlabel(f"{metric.upper()} score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(
        f"{save_path}/{model_name}_{metric}_scores.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_wer(df, save_path, model_name, bins=30, color="salmon", title=None):
    # Plot the distribution of WER scores
    plt.hist(df["wer_scores"], bins=bins, color=color)
    plt.suptitle(title or "Distribution of WER scores", fontsize=16)
    plt.xlabel("WER score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(
        f"{save_path}/{model_name}_wer_scores.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_cosine_similarity(
    df, save_path, model_name, bins=30, color="salmon", title=None
):
    # Plot the distribution of cosine similarity scores
    plt.hist(df["cosine_similarity"], bins=bins, color=color)
    plt.suptitle(title or "Distribution of cosine similarity scores", fontsize=16)
    plt.xlabel("Cosine similarity", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(
        f"{save_path}/{model_name}_cosine_similarity_scores.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_bert_score(df, save_path, model_name, bins=30, color="salmon", title=None):
    # Plot the distribution of BERT scores
    plt.hist(df["bert_score"], bins=bins, color=color)
    plt.suptitle(title or "Distribution of BERT scores", fontsize=16)
    plt.xlabel("BERT score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(
        f"{save_path}/{model_name}_bert_scores.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_myevaluation(df, save_path, model_name, bins=30, color="salmon", title=None):
    # Plot the distribution of myevaluation scores
    plt.hist(df["myevaluation_scores"], bins=bins, color=color)
    plt.suptitle(title or "Distribution of myevaluation scores", fontsize=16)
    plt.xlabel("Myevaluation score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(
        f"{save_path}/{model_name}_myevaluation_scores.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def evaluate_myevalutation(df_summaries):
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    from nltk.corpus import stopwords
    import numpy as np
    import nltk
    from bert_score import score as bert_scorer

    def get_keywords(text):
        """Estrae parole chiave da un testo"""
        stopwords_set = set(stopwords.words("english"))
        return set(
            [
                word.lower()
                for word in text.split()
                if word.lower() not in stopwords_set and word.isalpha()
            ]
        )

    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Estrai i testi
    predicted_texts = df_summaries["predicted_summary"].tolist()
    original_summary_texts = df_summaries["original_summary"].tolist()
    original_texts = df_summaries["original_text"].tolist()

    # Embedding dei testi
    predicted_embeddings = model.encode(predicted_texts)
    original_summary_embeddings = model.encode(original_summary_texts)
    original_text_embeddings = model.encode(original_texts)

    # Compute cosine similarity
    cs_PS_OT_all = cosine_similarity(predicted_embeddings, original_text_embeddings)
    cs_PS_OS_all = cosine_similarity(predicted_embeddings, original_summary_embeddings)
    # Extract the diagonal elements to get the similarity scores
    cs_PS_OT_list = cs_PS_OT_all.diagonal().tolist()
    cs_PS_OS_list = cs_PS_OS_all.diagonal().tolist()

    scores = []
    keyword_overlap_list = []
    bert_score_list = []

    weights = {
        "cs_PS_OT": 0.0,
        "cs_PS_OS": 0.07,
        "keyword_overlap": 0.03,
        "bert_score": 0.9,
    }

    # Calcola BERTScore in batch per maggiore efficienza
    bert_results = bert_scorer(
        predicted_texts, original_summary_texts, lang="en", verbose=False
    )

    # Itera sui testi per calcolare keyword overlap e combinare gli score
    for i, pred_text in enumerate(predicted_texts):
        # BERTScore
        bert_F1 = bert_results[2][i]

        # Keyword overlap
        orig_sum_text = original_summary_texts[i]
        pred_keywords = get_keywords(pred_text)
        orig_keywords = get_keywords(orig_sum_text)
        union = len(pred_keywords | orig_keywords)
        intersection = len(pred_keywords & orig_keywords)
        ko = intersection / (union + 1e-8)

        # Calcola il punteggio finale combinato
        my_evaluation_score = (
            weights["cs_PS_OT"] * cs_PS_OT_list[i]
            + weights["cs_PS_OS"] * cs_PS_OS_list[i]
            + weights["keyword_overlap"] * ko
            + weights["bert_score"] * bert_F1.mean().item()
        )

        scores.append(my_evaluation_score)
        keyword_overlap_list.append(ko)
        bert_score_list.append(bert_F1.mean().item())

    # Assegna i risultati al DataFrame
    df_summaries["myevaluation_scores"] = scores
    df_summaries["cs_PS_OT"] = cs_PS_OT_list
    df_summaries["cs_PS_OS"] = cs_PS_OS_list
    df_summaries["keyword_overlap"] = keyword_overlap_list
    df_summaries["bert_score"] = bert_score_list

    mean_myevaluation = df_summaries["myevaluation_scores"].mean()

    return df_summaries, mean_myevaluation


def evaluate_rouge(df_summaries):
    from rouge_score import rouge_scorer

    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def compute_rouge_scores(row):
        predicted = row["predicted_summary"]
        original = row["original_summary"]
        return scorer.score(original, predicted)

    # Apply the function to calculate ROUGE scores for each row in the DataFrame
    df_summaries["rouge_scores"] = df_summaries.apply(compute_rouge_scores, axis=1)

    # Compute the mean ROUGE scores
    mean_rouge1 = (
        df_summaries["rouge_scores"].apply(lambda x: x["rouge1"].fmeasure).mean()
    )

    mean_rouge2 = (
        df_summaries["rouge_scores"].apply(lambda x: x["rouge2"].fmeasure).mean()
    )

    mean_rougeL = (
        df_summaries["rouge_scores"].apply(lambda x: x["rougeL"].fmeasure).mean()
    )

    rouge_means_scores = {
        "mean_rouge1": mean_rouge1,
        "mean_rouge2": mean_rouge2,
        "mean_rougeL": mean_rougeL,
    }

    return df_summaries, rouge_means_scores


def evaluate_wer(df_summaries):
    from jiwer import wer

    def compute_wer_scores(row):
        predicted = row["predicted_summary"]
        original = row["original_summary"]
        return wer(original, predicted)

    # Apply the function to calculate WER scores for each row in the DataFrame
    df_summaries["wer_scores"] = df_summaries.apply(compute_wer_scores, axis=1)

    # Compute the mean WER score
    mean_wer = df_summaries["wer_scores"].mean()

    return df_summaries, mean_wer


def evaluate_cosine_similarity(df_summaries):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def compute_cosine_similarity(row):
        predicted = row["predicted_summary"]
        original = row["original_summary"]
        return cosine_similarity(
            sentence_model.encode([original]), sentence_model.encode([predicted])
        )[0][0]

    # Apply the function to calculate cosine similarity for each row in the DataFrame
    df_summaries["cosine_similarity"] = df_summaries.apply(
        compute_cosine_similarity, axis=1
    )

    # Compute the mean cosine similarity
    mean_cosine_similarity = df_summaries["cosine_similarity"].mean()

    return df_summaries, mean_cosine_similarity


def evaluate_bert_score(df_summaries):
    from bert_score import score as bert_scorer

    # Calculate BERTScore
    P, R, F1 = bert_scorer(
        df_summaries["predicted_summary"].tolist(),
        df_summaries["original_summary"].tolist(),
        lang="en",
        verbose=True,
    )

    # Add BERTScore to the DataFrame
    df_summaries["bert_score"] = F1.tolist()

    # Compute the mean BERTScore
    mean_bert_score = df_summaries["bert_score"].mean()

    return df_summaries, mean_bert_score


def generate_summaries(
    model_instance,
    x_training_padded,
    y_training_padded,
    max_text_len,
    n_summaries,
    save_path,
    to_load_summaries=True,
    to_save_summaries=True,
):
    summaries = []
    file_path = os.path.join(save_path, f"{model_instance.name}_summaries.csv")

    if to_load_summaries and os.path.isfile(file_path):
        print(f"Loading summaries from {file_path}...")
        df_summaries = pd.read_csv(file_path)
    else:
        print("Generating summaries...")
        for i in range(0, n_summaries):
            summaries.append(
                model_instance.decode_sequence(
                    x_training_padded[i].reshape(1, max_text_len)
                )
            )

        df_summaries = pd.DataFrame(
            {
                "original_text": [
                    model_instance.seq2text(x_training_padded[i])
                    for i in range(0, n_summaries)
                ],
                "original_summary": [
                    model_instance.seq2summary(y_training_padded[i])
                    for i in range(0, n_summaries)
                ],
                "predicted_summary": summaries,
            }
        )

        if to_save_summaries:
            print(f"Saving summaries to {file_path}...")
            os.makedirs(save_path, exist_ok=True)
            df_summaries.to_csv(file_path, index=False)

    print(df_summaries.head())
    return df_summaries


# This function is used to prepare the data for the model training, so every model's training script will use this function to prepare the data.
def prepare_data():
    """
    #!pip install tensorflow==2.18.0
    #!pip install keras==3.7.0
    """
    import tensorflow as tf

    import numpy as np
    import pandas as pd
    import re
    from bs4 import BeautifulSoup
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from nltk.corpus import stopwords

    from tensorflow.keras.callbacks import EarlyStopping
    import warnings

    pd.set_option("display.max_colwidth", 200)
    warnings.filterwarnings("ignore")

    GDRIVE_PATH = ""
    """
    # mount drive
    from google.colab import drive
    drive.mount('/content/gdrive')
    GDRIVE_PATH="/content/gdrive/MyDrive/Colab Notebooks/dasalvare/"
    """

    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")

    data = pd.read_csv(f"{path}/Reviews.csv", nrows=100000)  # reading only 100k rows

    data.drop_duplicates(subset=["Text"], inplace=True)  # dropping duplicates
    data.dropna(axis=0, inplace=True)  # dropping na

    contraction_mapping = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "this's": "this is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "here's": "here is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
    }

    import nltk

    # Download stopwords if not already downloaded
    nltk.download("stopwords")

    STOP_WORDS = set(stopwords.words("english"))  # set of stopwords

    def clean_text(input_text, remove_stopwords):
        """
        This function cleans the input text based on the following steps:
        - Lowercase the text
        - Remove HTML tags
        - Remove quotes and parentheses content
        - Replace contractions
        - Remove 's
        - Remove any non-alphanumeric characters
        - Normalize multiple letter repetitions (mmmm -> mm)
        - Tokenize the text
        - Remove stopwords
        - Remove words with length <= 1

        Args:
            input_text (str): To clean text.
            remove_stopwords (bool): If True, remove stopwords.

        Returns:
            str: Clean text.
        """

        # Lowercase
        cleaned_text = input_text.lower()

        # Remove HTML tags
        cleaned_text = BeautifulSoup(cleaned_text, "html.parser").text

        # Remove quotes and parentheses content
        cleaned_text = re.sub(r"\([^)]*\)", "", cleaned_text)
        cleaned_text = re.sub('"', "", cleaned_text)

        # Replace contractions
        cleaned_text = " ".join(
            [
                contraction_mapping[word] if word in contraction_mapping else word
                for word in cleaned_text.split()
            ]
        )

        # Remove 's
        cleaned_text = re.sub(r"'s\\b", "", cleaned_text)

        # Remove any non-alphanumeric characters
        cleaned_text = re.sub(r"[^a-zA-Z]", " ", cleaned_text)

        # Normalize multiple letter repetitions
        cleaned_text = re.sub(
            r"[m]{2,}", "mm", cleaned_text
        )  # Since it's a food review dataset there coulde be words mmm mmmm etc.

        # Tokenizzation
        tokens = cleaned_text.split()

        # Remove stopwords
        if remove_stopwords:
            tokens = [word for word in tokens if word not in STOP_WORDS]

        # Remove words with length <= 1
        filtered_tokens = [word for word in tokens if len(word) > 1]

        # Join tokens back to string
        return " ".join(filtered_tokens).strip()

    # Clean text
    cleaned_text = [clean_text(text, remove_stopwords=True) for text in data["Text"]]

    # Clean summary
    cleaned_summary = [
        clean_text(summary, remove_stopwords=False) for summary in data["Summary"]
    ]

    # Add cleaned text and summary to the dataframe
    data["cleaned_text"] = cleaned_text
    data["cleaned_summary"] = cleaned_summary

    # Substituting empty strings with NaN
    data.replace("", np.nan, inplace=True)

    # Dropping rows with NaN values
    data.dropna(axis=0, inplace=True)

    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    # Calculate the length of the text and summary
    data["text_word_count"] = data["cleaned_text"].apply(lambda x: len(x.split()))
    data["summary_word_count"] = data["cleaned_summary"].apply(lambda x: len(x.split()))

    # Dataframe with text and summary length
    length_df = data[["text_word_count", "summary_word_count"]]

    max_text_len = 30  # Max length of text
    max_summary_len = 8  # Max length of summary

    # Count the number of words in text and summary
    data["text_word_count"] = data["cleaned_text"].apply(
        lambda x: len(x.split())
    )  # Count words in text
    data["summary_word_count"] = data["cleaned_summary"].apply(
        lambda x: len(x.split())
    )  # Count words in summary

    # Filter only text and summaries that satisfy the length limits
    filtered_data = data[
        (data["text_word_count"] <= max_text_len)
        & (data["summary_word_count"] <= max_summary_len)
    ]

    # Creare un nuovo DataFrame con i dati filtrati
    df = filtered_data[["cleaned_text", "cleaned_summary"]].rename(
        columns={"cleaned_text": "text", "cleaned_summary": "summary"}
    )

    # Add special tokens to the summary if not already present
    def add_special_tokens(summary_text):
        """
        Add special tokens to the summary if not already present

        Args:
            summary_text (str): Text to modify.

        Returns:
            str: Text with special tokens.
        """
        prefix = "sostok "
        suffix = " eostok"

        # Add the prefix if not already present
        if not summary_text.startswith(prefix):
            summary_text = prefix + summary_text

        # Add the suffix if not already present
        if not summary_text.endswith(suffix):
            summary_text = summary_text + suffix

        return summary_text

    # Apply the function to the summary column
    df["modified_summary"] = df["summary"].apply(add_special_tokens)

    from sklearn.model_selection import train_test_split

    x_training, x_validation, y_training, y_validation = train_test_split(
        np.array(df["text"]),
        np.array(df["modified_summary"]),
        test_size=0.1,
        random_state=0,
        shuffle=True,
    )

    def calculate_rare_words_coverage(tokenizer, threshold=4):
        """
        Calculate the percentage of rare words and the coverage of rare words in the vocabulary.

        Args:
            tokenizer (Tokenizer): The tokenizer object.
            threshold (int): The threshold to consider a word as rare.

        Returns:
            dict: A dictionary containing the percentage of rare words and the coverage of rare words.
        """
        cnt = 0  # Number of rare words
        tot_cnt = 0  # Number of total words
        freq = 0  # Frequency of rare words
        tot_freq = 0  # Total frequency of all words

        for key, value in tokenizer.word_counts.items():
            tot_cnt += 1
            tot_freq += value
            if value < threshold:
                cnt += 1
                freq += value

        # Percentage of rare words
        rare_word_percentage = (cnt / tot_cnt) * 100 if tot_cnt > 0 else 0
        rare_word_coverage = (freq / tot_freq) * 100 if tot_freq > 0 else 0

        return {
            "percentage_of_rare_words": rare_word_percentage,
            "coverage_of_rare_words": rare_word_coverage,
            "cnt": cnt,
            "tot_cnt": tot_cnt,
            "freq": freq,
            "tot_freq": tot_freq,
        }

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # prepare a tokenizer for reviews on training data
    x_tokenizer = Tokenizer()
    x_tokenizer.fit_on_texts(list(x_training))

    # Calculate the percentage of rare words and the coverage of rare words in the vocabulary
    x_rarewords_result = calculate_rare_words_coverage(x_tokenizer, threshold=4)

    # Define the maximum vocabulary size
    max_vocab_size = max(x_rarewords_result["tot_cnt"] - x_rarewords_result["cnt"], 1)

    # Define the tokenizer with the maximum vocabulary size
    x_tokenizer = Tokenizer(num_words=max_vocab_size)

    # Fit the tokenizer on the training data
    x_tokenizer.fit_on_texts(list(x_training))

    # Calculate the percentage of rare words and the coverage of rare words in the vocabulary
    x_tr_seq = x_tokenizer.texts_to_sequences(x_training)
    x_val_seq = x_tokenizer.texts_to_sequences(x_validation)

    # Pad the sequences
    x_training_padded = pad_sequences(x_tr_seq, maxlen=max_text_len, padding="post")
    x_validation_padded = pad_sequences(x_val_seq, maxlen=max_text_len, padding="post")

    # Define the vocabulary size (padding token included)
    x_voc = x_tokenizer.num_words + 1

    # prepare a tokenizer for reviews on training data
    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(list(y_training))

    # Calculate the percentage of rare words and the coverage of rare words in the vocabulary
    y_rarewords_result = calculate_rare_words_coverage(y_tokenizer, threshold=6)

    # Define the maximum vocabulary size
    max_vocab_size = max(y_rarewords_result["tot_cnt"] - y_rarewords_result["cnt"], 1)

    # Define the tokenizer with the maximum vocabulary size
    y_tokenizer = Tokenizer(num_words=max_vocab_size)

    # Fit the tokenizer on the training data
    y_tokenizer.fit_on_texts(list(y_training))

    # Calculate the percentage of rare words and the coverage of rare words in the vocabulary
    y_tr_seq = y_tokenizer.texts_to_sequences(y_training)
    y_val_seq = y_tokenizer.texts_to_sequences(y_validation)

    # Pad the sequences
    y_training_padded = pad_sequences(y_tr_seq, maxlen=max_summary_len, padding="post")
    y_validation_padded = pad_sequences(
        y_val_seq, maxlen=max_summary_len, padding="post"
    )

    # Define the vocabulary size (padding token included)
    y_voc = y_tokenizer.num_words + 1

    # Function to determine wether a sequence contains only START and END tokens
    def is_only_start_end(sequence):
        return (
            np.count_nonzero(sequence) == 2
        )  # Count the number of non-zero elements, if only 2 then delete

    # Create a mask to remove sequences that contain only START and END tokens
    mask_training = [not is_only_start_end(seq) for seq in y_training_padded]
    mask_validation = [not is_only_start_end(seq) for seq in y_validation_padded]

    # Apply the mask to the training and validation data
    x_training_padded = x_training_padded[mask_training]
    y_training_padded = y_training_padded[mask_training]
    x_validation_padded = x_validation_padded[mask_validation]
    y_validation_padded = y_validation_padded[mask_validation]

    return (
        x_voc,
        y_voc,
        x_tokenizer,
        y_tokenizer,
        x_training_padded,
        y_training_padded,
        x_validation_padded,
        y_validation_padded,
        max_text_len,
        max_summary_len,
    )


def generate_model_name_additional_info(additional_info, hyperparams):
    components = [
        additional_info,
        f"optimizer{hyperparams['optimizer_class'].__name__}",
        f"lr{hyperparams['learning_rate']}",
        f"ed{hyperparams['embedding_dim']}",
        f"ld{hyperparams['latent_dim']}",
        f"do{hyperparams['decoder_dropout']}",
        f"drdo{hyperparams['decoder_recurrent_dropout']}",
        f"edo{hyperparams['encoder_dropout']}",
        f"erdo{hyperparams['encoder_recurrent_dropout']}",
        f"batch_size{hyperparams['batch_size']}",
        f"epochs{hyperparams['epochs']}",
    ]

    return "_".join(components)
