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


class Seq2SeqLSTM(BaseModel):
    def __init__(self, x_voc, y_voc, max_text_len, max_summary_len):
        self.latent_dim = 300
        self.embedding_dim = 100
        self.name = "Seq2SeqLSTM"
        super().__init__(x_voc, y_voc, max_text_len, max_summary_len, name=self.name)

    def build_encoder(self):
        """Builds the encoder part of the Seq2Seq model."""
        encoder_inputs = Input(shape=(self.max_text_len,), name="encoder_inputs")

        # Embedding layer
        enc_emb = Embedding(
            self.x_voc, self.embedding_dim, trainable=True, name="encoder_embedding"
        )(encoder_inputs)

        # Encoder LSTM layers
        encoder_lstm1 = LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True,
            dropout=0.4,
            recurrent_dropout=0.4,
            name="encoder_lstm1",
        )
        encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

        encoder_lstm2 = LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True,
            dropout=0.4,
            recurrent_dropout=0.4,
            name="encoder_lstm2",
        )
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

        encoder_lstm3 = LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True,
            dropout=0.4,
            recurrent_dropout=0.4,
            name="encoder_lstm3",
        )
        encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

        return encoder_inputs, encoder_outputs, state_h, state_c

    def build_decoder(self, encoder_outputs, state_h, state_c):
        """Builds the decoder part of the Seq2Seq model."""
        decoder_inputs = Input(shape=(None,), name="decoder_inputs")

        # Embedding layer
        dec_emb_layer = Embedding(
            self.y_voc, self.embedding_dim, trainable=True, name="decoder_embedding"
        )
        dec_emb = dec_emb_layer(decoder_inputs)

        # Decoder LSTM
        decoder_lstm = LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True,
            dropout=0.4,
            recurrent_dropout=0.2,
            name="decoder_lstm",
        )
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

        # Attention layer
        attn_layer = AttentionLayer(name="attention_layer")
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

        # Concatenate attention output and decoder LSTM output
        decoder_concat_input = Concatenate(axis=-1, name="concat_layer")(
            [decoder_outputs, attn_out]
        )

        # Dense layer
        decoder_dense = TimeDistributed(
            Dense(self.y_voc, activation="softmax"), name="decoder_dense"
        )
        decoder_outputs = decoder_dense(decoder_concat_input)

        return decoder_inputs, decoder_outputs

    def build_model(self):
        """Builds the complete Seq2Seq model."""
        # Build encoder
        encoder_inputs, encoder_outputs, state_h, state_c = self.build_encoder()

        # Build decoder
        decoder_inputs, decoder_outputs = self.build_decoder(
            encoder_outputs, state_h, state_c
        )

        # Build the model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name=self.name)

        # Return the model
        return model
