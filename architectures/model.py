from keras import backend as K

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

from architectures.BaseModel import BaseModel


class bohmodel(BaseModel):
    def __init__(
        self,
        x_voc,
        y_voc,
        max_text_len,
        max_summary_len,
        lstm_units=512,
        name="bohmodel",
    ):
        self.lstm_units = lstm_units
        self.name = name
        super().__init__(x_voc, y_voc, max_text_len, max_summary_len, name=name)

    def get_model(self):
        return self.model

    def build_encoder(self):
        encoder_inputs = Input(shape=(self.max_text_len,), name="Encoder-Input")
        encoder_embedding = Embedding(
            input_dim=self.x_voc,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name="Encoder-Embedding",
        )(encoder_inputs)
        encoder_lstm, state_h, state_c = LSTM(
            self.lstm_units, return_state=True, name="Encoder-LSTM"
        )(encoder_embedding)

        return encoder_inputs, encoder_lstm, state_h, state_c

    def build_decoder(self, encoder_outputs, state_h, state_c):
        decoder_inputs = Input(shape=(self.max_summary_len,), name="Decoder-Input")
        decoder_embedding = Embedding(
            input_dim=self.y_voc,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name="Decoder-Embedding",
        )(decoder_inputs)
        decoder_lstm = LSTM(
            self.lstm_units,
            return_sequences=True,
            return_state=True,
            name="Decoder-LSTM",
        )
        decoder_dense = Dense(self.y_voc, activation="softmax", name="Output-Dense")

        decoder_state_input_h = Input(shape=(self.lstm_units,), name="State-H-Input")
        decoder_state_input_c = Input(shape=(self.lstm_units,), name="State-C-Input")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_lstm_outputs, state_h, state_c = decoder_lstm(
            decoder_embedding, initial_state=decoder_states_inputs
        )

        decoder_outputs = decoder_dense(decoder_lstm_outputs)

        return decoder_inputs, decoder_outputs

    def build_model(self):
        # Build the encoder
        encoder_inputs, encoder_lstm, state_h, state_c = self.build_encoder()

        # Build the decoder
        decoder_inputs, decoder_outputs = self.build_decoder(
            encoder_outputs=encoder_lstm, state_h=state_h, state_c=state_c
        )

        # Build the model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name=self.name)

        return model
