# Base model class for all models in the project.

from abc import ABC, abstractmethod
from attention import AttentionLayer
from keras import backend as K

import numpy as np

K.clear_session()

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Embedding,
    Dense,
    Concatenate,
    TimeDistributed,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


class BaseModel(ABC):
    def __init__(
        self,
        x_voc,
        y_voc,
        max_text_len,
        max_summary_len,
        x_tokenizer,
        y_tokenizer,
        name="BaseModel",
    ):
        self.x_voc = x_voc
        self.y_voc = y_voc
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len
        self.y_tokenizer = y_tokenizer
        self.x_tokenizer = x_tokenizer
        self.reverse_target_word_index = self.y_tokenizer.index_word
        self.reverse_source_word_index = self.x_tokenizer.index_word
        self.target_word_index = self.y_tokenizer.word_index
        self.name = name
        self.callbacks = [
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=1,
                patience=2,
                restore_best_weights=True,
            )
        ]
        self.default_lr = 0.001
        self.optimizer = Adam(learning_rate=self.default_lr)
        self.loss = "sparse_categorical_crossentropy"
        self.metrics = ["accuracy"]
        self.model = self.build_model()

    def add_metrics(self, metrics):
        # Check if metrics is a list
        if isinstance(metrics, list):
            # Add each metric to the list only if it is not already in the list
            for metric in metrics:
                if metric not in self.metrics:
                    self.metrics.append(metric)

    def get_metrics(self):
        return self.metrics

    def change_optimizer(self, optimizer):
        self.optimizer = optimizer

        # Update the model with the new optimizer
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )

    def get_optimizer(self):
        return self.optimizer

    def change_loss(self, loss):
        self.loss = loss

        # Update the model with the new loss
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )

    def get_loss(self, loss="sparse_categorical_crossentropy"):
        self.loss = loss
        return self.loss

    def add_callbacks(self, callbacks):
        # Check if callbacks is a list
        if isinstance(callbacks, list):
            # Add each callback to the list only if it is not already in the list
            for callback in callbacks:
                if callback not in self.callbacks:
                    self.callbacks.append(callback)

    def get_callbacks(self):
        return self.callbacks

    def get_model(self):
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )
        return self.model

    def save_model(self, path):
        self.model.save(path)

    def save_encoder(self, path):
        self.encoder_model.save(path)

    def save_decoder(self, path):
        self.decoder_model.save(path)

    # Sequence to summary
    def seq2summary(self, input_seq):
        return " ".join(
            [
                self.reverse_target_word_index[i]
                for i in input_seq
                if i != 0
                and i != self.target_word_index["sostok"]
                and i != self.target_word_index["eostok"]
            ]
        )

    # Sequence to text
    def seq2text(self, input_seq):
        return " ".join(
            [self.reverse_source_word_index[i] for i in input_seq if i != 0]
        )

    # Inference the input sequence
    def decode_sequence(self, input_seq):
        e_out, e_h, e_c = self.encoder_model.predict(input_seq)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.target_word_index["sostok"]

        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + [e_out, e_h, e_c]
            )

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.reverse_target_word_index[sampled_token_index]

            if sampled_token != "eostok":
                decoded_sentence += " " + sampled_token

            if sampled_token == "eostok" or len(decoded_sentence.split()) >= (
                self.max_summary_len - 1
            ):
                stop_condition = True

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            e_h, e_c = h, c

        return decoded_sentence

    @abstractmethod
    def build_encoder(self):
        pass

    @abstractmethod
    def build_decoder(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def build_inference(self):
        pass
