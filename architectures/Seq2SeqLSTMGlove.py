from keras import backend as K
import requests

K.clear_session()

from attention import AttentionLayer
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Embedding,
    Dense,
    Concatenate,
    TimeDistributed,
)
from tensorflow.keras.models import Model
import numpy as np
from architectures.BaseModel import BaseModel
import os


def download_glove(destination_folder="glove", glove_file="glove.6B.zip"):
    # Base URL for GloVe embeddings
    glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Full path for the zip file
    glove_path = os.path.join(destination_folder, glove_file)

    # Check if the file already exists
    if not os.path.exists(glove_path):
        print(f"Downloading GloVe embeddings to {glove_path}...")
        response = requests.get(glove_url, stream=True)
        with open(glove_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    else:
        print("GloVe embeddings already downloaded.")

    # Extract the zip file
    extracted_path = os.path.join(destination_folder, "glove.6B")
    if not os.path.exists(extracted_path):
        print(f"Extracting {glove_path}...")
        import zipfile

        with zipfile.ZipFile(glove_path, "r") as zip_ref:
            zip_ref.extractall(destination_folder)
        print("Extraction complete.")
    else:
        print("GloVe embeddings already extracted.")


class Seq2SeqLSTMGlove(BaseModel):
    def __init__(
        self,
        x_voc,
        y_voc,
        max_text_len,
        max_summary_len,
        x_tokenizer,
        y_tokenizer,
        name="Seq2SeqLSTMGlove",
        latent_dim=300,
        embedding_dim=100,
        encoder_dropout=0.4,
        encoder_recurrent_dropout=0.4,
        decoder_dropout=0.4,
        decoder_recurrent_dropout=0.2,
    ):
        # Set unique parameters for this model
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.name = name
        self.reverse_target_word_index = y_tokenizer.index_word
        self.reverse_source_word_index = x_tokenizer.index_word
        self.target_word_index = y_tokenizer.word_index

        # Load GloVe embeddings
        self.embedding_matrix_x = self.load_glove_embeddings(
            x_tokenizer, x_voc, glove_txt_file_name="glove.6B.100d.txt"
        )
        self.embedding_matrix_y = self.load_glove_embeddings(
            y_tokenizer, y_voc, glove_txt_file_name="glove.6B.100d.txt"
        )

        # Initialize shared layers specific to this class
        self.encoder_embedding = Embedding(
            x_voc,
            self.embedding_dim,
            weights=[self.embedding_matrix_x],
            trainable=False,
            name="encoder_embedding",
        )
        self.decoder_embedding = Embedding(
            y_voc,
            self.embedding_dim,
            weights=[self.embedding_matrix_y],
            trainable=False,
            name="decoder_embedding",
        )
        self.encoder_lstm1, self.encoder_lstm2, self.encoder_lstm3 = (
            self.get_encoder_lstm_layers()
        )
        self.decoder_lstm = self.get_decoder_lstm_layer()
        self.attention_layer = AttentionLayer(name="attention_layer")
        self.decoder_dense = TimeDistributed(
            Dense(y_voc, activation="softmax"), name="decoder_dense"
        )

        # Call the parent class's initializer
        super().__init__(
            x_voc,
            y_voc,
            max_text_len,
            max_summary_len,
            x_tokenizer=x_tokenizer,
            y_tokenizer=y_tokenizer,
            name=self.name,
        )

        # Build models
        self.model = self.build_model()
        self.encoder_model, self.decoder_model = self.build_inference()

    def load_glove_embeddings(
        self, tokenizer, vocab_size, glove_txt_file_name="glove.6B.100d.txt"
    ):
        # Define the path to save GloVe weights
        glove_folder = "./architectures/weightsGLOVE"
        glove_file = "glove.6B.zip"
        glove_txt_file = os.path.join(glove_folder, glove_txt_file_name)

        # Ensure GloVe embeddings are downloaded and extracted
        download_glove(destination_folder=glove_folder, glove_file=glove_file)

        embeddings_index = {}
        with open(glove_txt_file, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = coefs

        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        for word, i in tokenizer.word_index.items():
            if i < vocab_size:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def get_encoder_lstm_layers(self):
        return (
            LSTM(
                self.latent_dim,
                return_sequences=True,
                return_state=True,
                dropout=self.encoder_dropout,
                recurrent_dropout=self.encoder_recurrent_dropout,
                name="encoder_lstm1",
            ),
            LSTM(
                self.latent_dim,
                return_sequences=True,
                return_state=True,
                dropout=self.encoder_dropout,
                recurrent_dropout=self.encoder_recurrent_dropout,
                name="encoder_lstm2",
            ),
            LSTM(
                self.latent_dim,
                return_sequences=True,
                return_state=True,
                dropout=self.encoder_dropout,
                recurrent_dropout=self.encoder_recurrent_dropout,
                name="encoder_lstm3",
            ),
        )

    def get_decoder_lstm_layer(self):
        return LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True,
            dropout=self.decoder_dropout,
            recurrent_dropout=self.decoder_recurrent_dropout,
            name="decoder_lstm",
        )

    def build_encoder(self):
        encoder_inputs = Input(shape=(self.max_text_len,), name="encoder_inputs")
        enc_emb = self.encoder_embedding(encoder_inputs)

        # LSTM layers
        encoder_output1, state_h1, state_c1 = self.encoder_lstm1(enc_emb)
        encoder_output2, state_h2, state_c2 = self.encoder_lstm2(encoder_output1)
        encoder_outputs, state_h, state_c = self.encoder_lstm3(encoder_output2)

        return encoder_inputs, encoder_outputs, state_h, state_c

    def build_decoder(self, encoder_outputs, encoder_states):
        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        dec_emb = self.decoder_embedding(decoder_inputs)

        decoder_outputs, _, _ = self.decoder_lstm(dec_emb, initial_state=encoder_states)

        attn_out, attn_states = self.attention_layer([encoder_outputs, decoder_outputs])
        decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attn_out])

        decoder_outputs = self.decoder_dense(decoder_concat_input)

        return decoder_inputs, decoder_outputs

    def build_model(self):
        encoder_inputs, encoder_outputs, state_h, state_c = self.build_encoder()
        decoder_inputs, decoder_outputs = self.build_decoder(
            encoder_outputs, [state_h, state_c]
        )

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name=self.name)
        model.compile(
            optimizer=self.get_optimizer(),
            loss=self.get_loss(),
            metrics=self.get_metrics(),
        )

        return model

    def build_inference(self):
        encoder_inputs, encoder_outputs, state_h, state_c = self.build_encoder()
        encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_hidden_state_input = Input(shape=(self.max_text_len, self.latent_dim))

        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        dec_emb = self.decoder_embedding(decoder_inputs)
        decoder_outputs, state_h, state_c = self.decoder_lstm(
            dec_emb, initial_state=[decoder_state_input_h, decoder_state_input_c]
        )

        attn_out, attn_states = self.attention_layer(
            [decoder_hidden_state_input, decoder_outputs]
        )
        decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attn_out])
        decoder_outputs = self.decoder_dense(decoder_concat_input)

        decoder_model = Model(
            [
                decoder_inputs,
                decoder_hidden_state_input,
                decoder_state_input_h,
                decoder_state_input_c,
            ],
            [decoder_outputs, state_h, state_c],
        )

        return encoder_model, decoder_model
