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
    MultiHeadAttention,
    Add,
    LayerNormalization,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from architectures.BaseModel import BaseModel


class Seq2SeqLSTMTransformer(BaseModel):
    def __init__(
        self, x_voc, y_voc, max_text_len, max_summary_len, x_tokenizer, y_tokenizer
    ):
        # Model parameters
        self.latent_dim = 300
        self.embedding_dim = 100
        self.name = "Seq2SeqLSTMTransformer-3epoch"
        self.num_heads = 8
        self.ff_dim = 512
        self.num_transformer_blocks = 2
        self.dropout_rate = 0.1

        self.reverse_target_word_index = y_tokenizer.index_word
        self.reverse_source_word_index = x_tokenizer.index_word
        self.target_word_index = y_tokenizer.word_index

        # Initialize layers
        self.encoder_embedding = Embedding(
            x_voc, self.embedding_dim, trainable=True, name="encoder_embedding"
        )
        self.decoder_embedding = Embedding(
            y_voc, self.embedding_dim, trainable=True, name="decoder_embedding"
        )
        self.encoder_lstm1, self.encoder_lstm2, self.encoder_lstm3 = (
            self.get_encoder_lstm_layers()
        )
        self.decoder_lstm = self.get_decoder_lstm_layer()
        self.attention_layer = AttentionLayer(name="attention_layer")
        self.decoder_dense = TimeDistributed(
            Dense(y_voc, activation="softmax"), name="decoder_dense"
        )

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

    def get_encoder_lstm_layers(self):
        return (
            LSTM(
                self.latent_dim,
                return_sequences=True,
                return_state=True,
                dropout=0.4,
                recurrent_dropout=0.4,
                name="encoder_lstm1",
            ),
            LSTM(
                self.latent_dim,
                return_sequences=True,
                return_state=True,
                dropout=0.4,
                recurrent_dropout=0.4,
                name="encoder_lstm2",
            ),
            LSTM(
                self.latent_dim,
                return_sequences=True,
                return_state=True,
                dropout=0.4,
                recurrent_dropout=0.4,
                name="encoder_lstm3",
            ),
        )

    def get_decoder_lstm_layer(self):
        return LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True,
            dropout=0.4,
            recurrent_dropout=0.2,
            name="decoder_lstm",
        )

    def build_encoder(self):
        encoder_inputs = Input(shape=(self.max_text_len,), name="encoder_inputs")
        enc_emb = self.encoder_embedding(encoder_inputs)

        # Add transformer blocks before LSTM
        x = enc_emb
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_block(
                x, self.num_heads, self.ff_dim, self.dropout_rate
            )

        # LSTM layers
        encoder_output1, state_h1, state_c1 = self.encoder_lstm1(x)
        encoder_output2, state_h2, state_c2 = self.encoder_lstm2(encoder_output1)
        encoder_outputs, state_h, state_c = self.encoder_lstm3(encoder_output2)

        return encoder_inputs, encoder_outputs, state_h, state_c

    def build_decoder(self, encoder_outputs, encoder_states):
        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        dec_emb = self.decoder_embedding(decoder_inputs)

        # Add transformer blocks before LSTM
        x = dec_emb
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_block(
                x, self.num_heads, self.ff_dim, self.dropout_rate
            )
            # Add cross-attention with encoder outputs
            cross_attn_output = MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.embedding_dim
            )(x, encoder_outputs)
            x = Add()([x, cross_attn_output])
            x = LayerNormalization(epsilon=1e-6)(x)

        decoder_outputs, _, _ = self.decoder_lstm(x, initial_state=encoder_states)

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
        model.compile(optimizer=self.get_optimizer(), loss=self.get_loss(), metrics=self.get_metrics())

        return model

    def build_inference(self):
        encoder_inputs, encoder_outputs, state_h, state_c = self.build_encoder()
        encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_hidden_state_input = Input(shape=(self.max_text_len, self.latent_dim))

        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        dec_emb = self.decoder_embedding(decoder_inputs)

        # Add transformer blocks
        x = dec_emb
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_block(
                x, self.num_heads, self.ff_dim, self.dropout_rate
            )
            cross_attn_output = MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.embedding_dim
            )(x, decoder_hidden_state_input)
            x = Add()([x, cross_attn_output])
            x = LayerNormalization(epsilon=1e-6)(x)

        decoder_outputs, state_h, state_c = self.decoder_lstm(
            x, initial_state=[decoder_state_input_h, decoder_state_input_c]
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
