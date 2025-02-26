from tensorflow.keras.layers import (
    MultiHeadAttention,
    LayerNormalization,
    Dropout,
    Dense,
)
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np


class TransformerModel(BaseModel):
    def __init__(
        self,
        num_layers=2,
        d_model=128,
        num_heads=4,
        dff=512,
        dropout_rate=0.1,
        *args,
        **kwargs
    ):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        super().__init__(*args, **kwargs)

    def build_encoder(self):
        """Build transformer encoder"""
        encoder_inputs = Input(shape=(self.max_text_len,))

        # Embedding + Positional Encoding
        x = Embedding(self.x_voc, self.d_model)(encoder_inputs)
        x = self.positional_encoding(x)
        x = Dropout(self.dropout_rate)(x)

        # Transformer layers
        for _ in range(self.num_layers):
            x = self.transformer_encoder_layer(x)

        return Model(encoder_inputs, x, name="encoder")

    def build_decoder(self):
        """Build transformer decoder"""
        decoder_inputs = Input(shape=(self.max_summary_len,))
        encoder_outputs = Input(shape=(self.max_text_len, self.d_model))

        # Embedding + Positional Encoding
        x = Embedding(self.y_voc, self.d_model)(decoder_inputs)
        x = self.positional_encoding(x)
        x = Dropout(self.dropout_rate)(x)

        # Transformer layers
        for _ in range(self.num_layers):
            x = self.transformer_decoder_layer(x, encoder_outputs)

        # Final output
        x = Dense(self.y_voc, activation="softmax")(x)
        return Model([decoder_inputs, encoder_outputs], x, name="decoder")

    def build_model(self):
        """Build full transformer model"""
        # Encoder
        encoder_inputs = Input(shape=(self.max_text_len,))
        encoder_outputs = self.build_encoder()(encoder_inputs)

        # Decoder
        decoder_inputs = Input(shape=(self.max_summary_len,))
        decoder_outputs = self.build_decoder()([decoder_inputs, encoder_outputs])

        # Full model
        model = Model(
            [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
        )
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def build_inference(self):
        """Build models for inference"""
        # Encoder for inference
        self.encoder_model = self.build_encoder()

        # Decoder for inference
        decoder_inputs = Input(shape=(1,))  # Single token at a time
        encoder_outputs = Input(shape=(self.max_text_len, self.d_model))

        # Rebuild decoder for inference
        x = Embedding(self.y_voc, self.d_model)(decoder_inputs)
        x = self.positional_encoding(x)

        for _ in range(self.num_layers):
            x = self.transformer_decoder_layer(x, encoder_outputs)

        x = Dense(self.y_voc, activation="softmax")(x)
        self.decoder_model = Model(
            [decoder_inputs, encoder_outputs], x, name="decoder_inference"
        )

    def positional_encoding(self, x):
        """Add positional encoding to embeddings"""
        position = tf.shape(x)[1]
        d_model = x.shape[-1]
        pos = tf.range(position, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))

        angle_rads = pos * angle_rates
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, :, :]

        return x + pos_encoding

    def transformer_encoder_layer(self, x):
        """Single transformer encoder layer"""
        # Self-attention
        attn_output = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model // self.num_heads
        )(x, x)
        attn_output = Dropout(self.dropout_rate)(attn_output)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed forward
        ffn_output = Dense(self.dff, activation="relu")(x)
        ffn_output = Dense(self.d_model)(ffn_output)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        return LayerNormalization(epsilon=1e-6)(x + ffn_output)

    def transformer_decoder_layer(self, x, encoder_outputs):
        """Single transformer decoder layer"""
        # Self-attention
        attn1 = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model // self.num_heads
        )(x, x)
        attn1 = LayerNormalization(epsilon=1e-6)(x + attn1)

        # Encoder-decoder attention
        attn2 = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model // self.num_heads
        )(attn1, encoder_outputs)
        attn2 = LayerNormalization(epsilon=1e-6)(attn1 + attn2)

        # Feed forward
        ffn_output = Dense(self.dff, activation="relu")(attn2)
        ffn_output = Dense(self.d_model)(ffn_output)
        return LayerNormalization(epsilon=1e-6)(attn2 + ffn_output)

    def decode_sequence(self, input_seq):
        """Modified inference method for transformer"""
        # Encode the input
        enc_output = self.encoder_model.predict(input_seq)

        # Initialize decoder input with start token
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.target_word_index["sostok"]

        decoded_sentence = []
        for _ in range(self.max_summary_len):
            output_tokens = self.decoder_model.predict([target_seq, enc_output])

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.reverse_target_word_index[sampled_token_index]

            if sampled_token == "eostok":
                break

            decoded_sentence.append(sampled_token)
            target_seq = np.array([[sampled_token_index]])

        return " ".join(decoded_sentence)
