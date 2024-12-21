# Base model class for all models in the project.

from abc import ABC, abstractmethod
from attention import AttentionLayer
from keras import backend as K

K.clear_session()

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Embedding,
    Dense,
    Concatenate,
    TimeDistributed,
)
from tensorflow.keras.models import Model


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
        self.latent_dim = 300
        self.embedding_dim = 100
        self.y_tokenizer = y_tokenizer
        self.x_tokenizer = x_tokenizer
        self.reverse_target_word_index = self.y_tokenizer.index_word
        self.reverse_source_word_index = self.x_tokenizer.index_word
        self.target_word_index = self.y_tokenizer.word_index
        self.name = name
        self.encoder_inputs, self.encoder_outputs, self.state_h, self.state_c = (
            self.build_encoder()
        )
        (
            self.decoder_inputs,
            self.decoder_outputs,
        ) = self.build_decoder(self.encoder_outputs, self.state_h, self.state_c)
        self.model = self.build_model()

    def get_model(self):
        return self.model

    def save_model(self, path):
        self.model.save(path)

    def save_encoder(self, path):
        self.encoder_model.save(path)

    def save_decoder(self, path):
        self.decoder_model.save(path)

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
