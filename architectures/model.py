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


class bohmodel:
    def __init__(
        self,
        x_voc,
        y_voc,
        max_text_len,
        max_summary_len,
        lstm_units=512,
        name="bohmodel",
    ):
        self.vocab_size_encoder = x_voc
        self.vocab_size_decoder = y_voc
        self.max_encoder_seq_len = max_text_len
        self.max_decoder_seq_len = max_summary_len
        self.latent_dim = 300
        self.embedding_dim = 100
        self.name = name
        self.lstm_units = lstm_units
        self.encoder_model = self.build_encoder()
        self.decoder_model = self.build_decoder()
        self.model = self.build_model()

    def get_model(self):
        return self.model

    def build_encoder(self):
        encoder_inputs = Input(shape=(self.max_encoder_seq_len,), name="Encoder-Input")
        encoder_embedding = Embedding(
            input_dim=self.vocab_size_encoder,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name="Encoder-Embedding",
        )(encoder_inputs)
        encoder_lstm, state_h, state_c = LSTM(
            self.lstm_units, return_state=True, name="Encoder-LSTM"
        )(encoder_embedding)
        encoder_states = [state_h, state_c]
        return Model(
            inputs=encoder_inputs, outputs=encoder_states, name="Encoder-Model"
        )

    def build_decoder(self):
        decoder_inputs = Input(shape=(self.max_decoder_seq_len,), name="Decoder-Input")
        decoder_embedding = Embedding(
            input_dim=self.vocab_size_decoder,
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
        decoder_dense = Dense(
            self.vocab_size_decoder, activation="softmax", name="Output-Dense"
        )

        decoder_state_input_h = Input(shape=(self.lstm_units,), name="State-H-Input")
        decoder_state_input_c = Input(shape=(self.lstm_units,), name="State-C-Input")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_lstm_outputs, state_h, state_c = decoder_lstm(
            decoder_embedding, initial_state=decoder_states_inputs
        )

        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_lstm_outputs)

        return Model(
            inputs=[decoder_inputs] + decoder_states_inputs,
            outputs=[decoder_outputs] + decoder_states,
            name="Decoder-Model",
        )

    def build_model(self):
        encoder_inputs = Input(shape=(self.max_encoder_seq_len,), name="Encoder-Input")
        encoder_states = self.encoder_model(encoder_inputs)

        decoder_inputs = Input(shape=(self.max_decoder_seq_len,), name="Decoder-Input")
        decoder_embedding = self.decoder_model.get_layer("Decoder-Embedding")(
            decoder_inputs
        )
        decoder_lstm = self.decoder_model.get_layer("Decoder-LSTM")
        decoder_dense = self.decoder_model.get_layer("Output-Dense")

        decoder_lstm_outputs, _, _ = decoder_lstm(
            decoder_embedding, initial_state=encoder_states
        )
        decoder_outputs = decoder_dense(decoder_lstm_outputs)

        # Return the model
        return Model(
            [encoder_inputs, decoder_inputs], decoder_outputs, name=f"{self.name}"
        )
