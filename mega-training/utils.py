import os
from keras import backend as K
from matplotlib import pyplot as plt
import pandas as pd


def create_hyperparameter_grid():
    from tensorflow.keras.optimizers import Adam, RMSprop

    latent_dim = [256, 1024]
    embedding_dim = [128, 512]
    encoder_dropout = [0.1, 0.4]
    encoder_recurrent_dropout = [0.1, 0.4]
    decoder_dropout = [0.1, 0.4]
    decoder_recurrent_dropout = [0.1, 0.4]
    optimizer = [
        Adam(learning_rate=0.001),
        Adam(learning_rate=0.0005),
        RMSprop(learning_rate=0.001),
        RMSprop(learning_rate=0.0005),
    ]
    epochs = [50]
    batch_size = [128]

    hyperparameter_grid = []

    # Create all the combinations of hyperparameters
    for latent_dim_val in latent_dim:
        for embedding_dim_val in embedding_dim:
            for encoder_dropout_val in encoder_dropout:
                for encoder_recurrent_dropout_val in encoder_recurrent_dropout:
                    for decoder_dropout_val in decoder_dropout:
                        for decoder_recurrent_dropout_val in decoder_recurrent_dropout:
                            for optimizer_val in optimizer:
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
                                                "optimizer": optimizer_val,
                                                "epochs": epochs_val,
                                                "batch_size": batch_size_val,
                                            }
                                        )

    # Print the number of hyperparameter combinations
    print(f"Number of hyperparameter combinations: {len(hyperparameter_grid)}")

    return hyperparameter_grid


def plot_rouge(
    df,
    save_path,
    model_instance,
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
        f"{save_path}/{model_instance.name}_{metric}_scores.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_wer(df, save_path, model_instance, bins=30, color="salmon", title=None):
    # Plot the distribution of WER scores
    plt.hist(df["wer_scores"], bins=bins, color=color)
    plt.suptitle(title or "Distribution of WER scores", fontsize=16)
    plt.xlabel("WER score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(
        f"{save_path}/{model_instance.name}_wer_scores.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_cosine_similarity(
    df, save_path, model_instance, bins=30, color="salmon", title=None
):
    # Plot the distribution of cosine similarity scores
    plt.hist(df["cosine_similarity"], bins=bins, color=color)
    plt.suptitle(title or "Distribution of cosine similarity scores", fontsize=16)
    plt.xlabel("Cosine similarity", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(
        f"{save_path}/{model_instance.name}_cosine_similarity_scores.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


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
