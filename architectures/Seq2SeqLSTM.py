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


class Seq2SeqLSTM:
    def __init__(self, x_voc, y_voc, max_text_len, max_summary_len, name="Seq2SeqLSTM"):
        self.x_voc = x_voc
        self.y_voc = y_voc
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len
        self.latent_dim = 300
        self.embedding_dim = 100
        self.name = name
        self.model = self.build_model()

    def get_model(self):
        return self.model

    def build_model(self):
        # Encoder
        encoder_inputs = Input(shape=(self.max_text_len,))

        # embedding layer
        enc_emb = Embedding(self.x_voc, self.embedding_dim, trainable=True)(
            encoder_inputs
        )

        # encoder lstm 1
        encoder_lstm1 = LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True,
            dropout=0.4,
            recurrent_dropout=0.4,
        )
        encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

        # encoder lstm 2
        encoder_lstm2 = LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True,
            dropout=0.4,
            recurrent_dropout=0.4,
        )
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

        # encoder lstm 3
        encoder_lstm3 = LSTM(
            self.latent_dim,
            return_state=True,
            return_sequences=True,
            dropout=0.4,
            recurrent_dropout=0.4,
        )
        encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))

        # embedding layer
        dec_emb_layer = Embedding(self.y_voc, self.embedding_dim, trainable=True)
        dec_emb = dec_emb_layer(decoder_inputs)

        decoder_lstm = LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True,
            dropout=0.4,
            recurrent_dropout=0.2,
        )
        decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(
            dec_emb, initial_state=[state_h, state_c]
        )

        # Attention layer
        attn_layer = AttentionLayer(name="attention_layer")
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

        # Concat attention input and decoder LSTM output
        decoder_concat_input = Concatenate(axis=-1, name="concat_layer")(
            [decoder_outputs, attn_out]
        )

        # dense layer
        decoder_dense = TimeDistributed(Dense(self.y_voc, activation="softmax"))
        decoder_outputs = decoder_dense(decoder_concat_input)

        # Return the model
        return Model([encoder_inputs, decoder_inputs], decoder_outputs, name=self.name)
