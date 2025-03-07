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
    Bidirectional,
)
from tensorflow.keras.models import Model

from architectures.BaseModel import BaseModel


class Seq2SeqBiLSTM(BaseModel):
    def __init__(
        self,
        x_voc,
        y_voc,
        max_text_len,
        max_summary_len,
        x_tokenizer,
        y_tokenizer,
        name="Seq2SeqBiLSTM",
        name_additional_info="",
        latent_dim=300,
        embedding_dim=100,
        encoder_dropout=0.4,
        encoder_recurrent_dropout=0.4,
        decoder_dropout=0.4,
        decoder_recurrent_dropout=0.2,
    ):
        # Set unique parameters for this model
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.name = name + name_additional_info
        self.reverse_target_word_index = y_tokenizer.index_word
        self.reverse_source_word_index = x_tokenizer.index_word
        self.target_word_index = y_tokenizer.word_index
        self.encoder_dropout = encoder_dropout
        self.encoder_recurrent_dropout = encoder_recurrent_dropout
        self.decoder_dropout = decoder_dropout
        self.decoder_recurrent_dropout = decoder_recurrent_dropout

        # Initialize shared layers specific to this class
        self.encoder_embedding = Embedding(
            x_voc, self.embedding_dim, trainable=True, name="encoder_embedding"
        )
        self.decoder_embedding = Embedding(
            y_voc, self.embedding_dim, trainable=True, name="decoder_embedding"
        )
        self.encoder_bilstm = self.get_encoder_bilstm_layer()
        self.decoder_lstm = self.get_decoder_lstm_layer()
        self.attention_layer = AttentionLayer(name="attention_layer")
        self.decoder_dense = TimeDistributed(
            Dense(y_voc, activation="softmax"), name="decoder_dense"
        )

        # Call the parent class's initializer
        super().__init__(
            x_voc,
            y_voc,
            max_text_len,
            max_summary_len,
            x_tokenizer=x_tokenizer,
            y_tokenizer=y_tokenizer,
            name=self.name,
        )

        # Build models
        self.model = self.build_model()
        self.encoder_model, self.decoder_model = self.build_inference()

    def get_encoder_bilstm_layer(self):
        return Bidirectional(
            LSTM(
                self.latent_dim,
                return_sequences=True,
                return_state=True,
                dropout=self.encoder_dropout,
                recurrent_dropout=self.encoder_recurrent_dropout,
                name="encoder_bilstm",
            )
        )

    def get_decoder_lstm_layer(self):
        return LSTM(
            self.latent_dim
            * 2,  # Double the size to match BiLSTM output, since we are using BiLSTM
            return_sequences=True,
            return_state=True,
            dropout=self.decoder_dropout,
            recurrent_dropout=self.decoder_recurrent_dropout,
            name="decoder_lstm",
        )

    def build_encoder(self):
        encoder_inputs = Input(shape=(self.max_text_len,), name="encoder_inputs")
        enc_emb = self.encoder_embedding(encoder_inputs)

        # Bidirectional LSTM layer
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = (
            self.encoder_bilstm(enc_emb)
        )

        # Concatenate states from both directions
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])

        return encoder_inputs, encoder_outputs, state_h, state_c

    def build_decoder(self, encoder_outputs, encoder_states):
        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        dec_emb = self.decoder_embedding(decoder_inputs)

        decoder_outputs, _, _ = self.decoder_lstm(dec_emb, initial_state=encoder_states)

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
        model.compile(
            optimizer=self.get_optimizer(),
            loss=self.get_loss(),
            metrics=self.get_metrics(),
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
