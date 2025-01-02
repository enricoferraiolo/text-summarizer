from keras import backend as K

K.clear_session()

from tensorflow.keras.layers import (
    Input,
    Embedding,
    Dense,
    LayerNormalization,
    Dropout,
    Add,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
import tensorflow as tf
from architectures.BaseModel import BaseModel
import numpy as np


# TODO: INFERENZA È ROTTA
class Seq2SeqTransformer(BaseModel):
    def __init__(
        self, x_voc, y_voc, max_text_len, max_summary_len, x_tokenizer, y_tokenizer
    ):
        self.embedding_dim = 128
        self.num_heads = 8
        self.ff_dim = 512
        self.num_transformer_blocks = 4
        self.dropout_rate = 0.1
        self.name = "Seq2SeqTransformer"

        super().__init__(
            x_voc,
            y_voc,
            max_text_len,
            max_summary_len,
            x_tokenizer=x_tokenizer,
            y_tokenizer=y_tokenizer,
            name=self.name,
        )

        self.model = self.build_model()
        self.encoder_model, self.decoder_model = self.build_inference()

    def transformer_block(self, inputs, num_heads, ff_dim, dropout_rate):
        attn_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=self.embedding_dim
        )(inputs, inputs)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = Add()([inputs, attn_output])
        out1 = LayerNormalization(epsilon=1e-6)(out1)

        ffn_output = Dense(ff_dim, activation="relu")(out1)
        ffn_output = Dense(self.embedding_dim)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        out2 = Add()([out1, ffn_output])
        return LayerNormalization(epsilon=1e-6)(out2)

    def build_encoder(self):
        encoder_inputs = Input(shape=(self.max_text_len,), name="encoder_inputs")
        enc_emb = Embedding(self.x_voc, self.embedding_dim, trainable=True)(
            encoder_inputs
        )

        x = enc_emb
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_block(
                x, self.num_heads, self.ff_dim, self.dropout_rate
            )

        return encoder_inputs, x

    def build_decoder(self, encoder_outputs, encoder_states):
        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        dec_emb = Embedding(self.y_voc, self.embedding_dim, trainable=True)(
            decoder_inputs
        )

        x = dec_emb
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_block(
                x, self.num_heads, self.ff_dim, self.dropout_rate
            )
            cross_attn_output = MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.embedding_dim
            )(x, encoder_outputs)
            x = Add()([x, cross_attn_output])
            x = LayerNormalization(epsilon=1e-6)(x)

        decoder_outputs = Dense(self.y_voc, activation="softmax")(x)
        return decoder_inputs, decoder_outputs

    def build_model(self):
        encoder_inputs, encoder_outputs = self.build_encoder()
        decoder_inputs, decoder_outputs = self.build_decoder(encoder_outputs, None)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name=self.name)
        return model

    def build_inference(self):
        encoder_inputs, encoder_outputs = self.build_encoder()
        encoder_model = Model(encoder_inputs, encoder_outputs)

        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        encoder_states = Input(
            shape=(self.max_text_len, self.embedding_dim), name="encoder_states"
        )

        x = Embedding(self.y_voc, self.embedding_dim, trainable=True)(decoder_inputs)
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_block(
                x, self.num_heads, self.ff_dim, self.dropout_rate
            )
            cross_attn_output = MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.embedding_dim
            )(x, encoder_states)
            x = Add()([x, cross_attn_output])
            x = LayerNormalization(epsilon=1e-6)(x)

        decoder_outputs = Dense(self.y_voc, activation="softmax")(x)

        decoder_model = Model(
            [decoder_inputs, encoder_states],
            decoder_outputs,
            name="decoder_model",
        )

        return encoder_model, decoder_model

    def decode_sequence(self, input_seq):
        # Get the encoder states
        e_out = self.encoder_model.predict(input_seq)

        # Initialize the target sequence (with the start token)
        target_seq = np.zeros((1, 1))  # Shape (1, 1) for the initial token
        target_seq[0, 0] = self.target_word_index["sostok"]

        stop_condition = False
        decoded_sentence = ""

        while not stop_condition:
            # Predict the next token in the sequence
            output_tokens = self.decoder_model.predict([target_seq, e_out])

            # Get the most probable next token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.reverse_target_word_index[sampled_token_index]

            # Stop if we predict the end token or exceed max length
            if (
                sampled_token == "eostok"
                or len(decoded_sentence.split()) >= self.max_summary_len
            ):
                stop_condition = True

            decoded_sentence += " " + sampled_token

            # Update the target sequence (for the next token)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

        return decoded_sentence
