from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Concatenate,
    Dropout,
    TimeDistributed,
    LayerNormalization,
    AdditiveAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from architectures.BaseModel import BaseModel


class Seq2SeqImproved(BaseModel):
    def __init__(
        self, x_voc, y_voc, max_text_len, max_summary_len, x_tokenizer, y_tokenizer
    ):
        self.latent_dim = 512
        self.embedding_dim = 300
        self.name = "Seq2SeqImproved"

        # Load pre-trained embeddings (if applicable)
        self.encoder_embedding = Embedding(
            x_voc, self.embedding_dim, trainable=True, name="encoder_embedding"
        )
        self.decoder_embedding = Embedding(
            y_voc, self.embedding_dim, trainable=True, name="decoder_embedding"
        )

        self.attention_layer = AdditiveAttention(name="attention_layer")
        self.encoder_lstm = Bidirectional(
            LSTM(
                self.latent_dim,
                return_sequences=True,
                return_state=True,
                dropout=0.3,
                recurrent_dropout=0.3,
                kernel_regularizer=l2(0.01),
                name="bidirectional_encoder_lstm",
            )
        )
        self.decoder_lstm = LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True,
            dropout=0.3,
            recurrent_dropout=0.2,
            kernel_regularizer=l2(0.01),
            name="decoder_lstm",
        )
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

    def build_encoder(self):
        encoder_inputs = Input(shape=(self.max_text_len,), name="encoder_inputs")
        enc_emb = self.encoder_embedding(encoder_inputs)
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = (
            self.encoder_lstm(enc_emb)
        )
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        return encoder_inputs, encoder_outputs, state_h, state_c

    def build_decoder(self, encoder_outputs, encoder_states):
        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        dec_emb = self.decoder_embedding(decoder_inputs)
        decoder_outputs, _, _ = self.decoder_lstm(dec_emb, initial_state=encoder_states)

        attn_out = self.attention_layer([encoder_outputs, decoder_outputs])
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
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def build_inference(self):
        encoder_inputs, encoder_outputs, state_h, state_c = self.build_encoder()
        encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

        decoder_state_input_h = Input(shape=(self.latent_dim * 2,))
        decoder_state_input_c = Input(shape=(self.latent_dim * 2,))
        decoder_hidden_state_input = Input(
            shape=(self.max_text_len, self.latent_dim * 2)
        )

        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        dec_emb = self.decoder_embedding(decoder_inputs)
        decoder_outputs, state_h, state_c = self.decoder_lstm(
            dec_emb, initial_state=[decoder_state_input_h, decoder_state_input_c]
        )

        attn_out = self.attention_layer([decoder_hidden_state_input, decoder_outputs])
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
