from keras import backend as K

K.clear_session()

from attention import AttentionLayer
from tensorflow.keras.layers import (
    Input,
    GRU,
    Embedding,
    Dense,
    Concatenate,
    TimeDistributed,
)
from tensorflow.keras.models import Model

from architectures.BaseModel import BaseModel


class Seq2SeqGRU(BaseModel):
    def __init__(
        self,
        x_voc,
        y_voc,
        max_text_len,
        max_summary_len,
        x_tokenizer,
        y_tokenizer,
        name="Seq2SeqGRU",
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
        self.encoder_gru1, self.encoder_gru2, self.encoder_gru3 = (
            self.get_encoder_gru_layers()
        )
        self.decoder_gru = self.get_decoder_gru_layer()
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

    def get_encoder_gru_layers(self):
        return (
            GRU(
                self.latent_dim,
                return_sequences=True,
                return_state=True,
                dropout=self.encoder_dropout,
                recurrent_dropout=self.encoder_recurrent_dropout,
                name="encoder_gru1",
            ),
            GRU(
                self.latent_dim,
                return_sequences=True,
                return_state=True,
                dropout=self.encoder_dropout,
                recurrent_dropout=self.encoder_recurrent_dropout,
                name="encoder_gru2",
            ),
            GRU(
                self.latent_dim,
                return_sequences=True,
                return_state=True,
                dropout=self.encoder_dropout,
                recurrent_dropout=self.encoder_recurrent_dropout,
                name="encoder_gru3",
            ),
        )

    def get_decoder_gru_layer(self):
        return GRU(
            self.latent_dim,
            return_sequences=True,
            return_state=True,
            dropout=self.decoder_dropout,
            recurrent_dropout=self.decoder_recurrent_dropout,
            name="decoder_gru",
        )

    def build_encoder(self):
        encoder_inputs = Input(shape=(self.max_text_len,), name="encoder_inputs")
        enc_emb = self.encoder_embedding(encoder_inputs)

        # GRU layers
        encoder_output1, state_h1 = self.encoder_gru1(enc_emb)
        encoder_output2, state_h2 = self.encoder_gru2(encoder_output1)
        encoder_outputs, state_h = self.encoder_gru3(encoder_output2)

        return encoder_inputs, encoder_outputs, state_h

    def build_decoder(self, encoder_outputs, encoder_states):
        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        dec_emb = self.decoder_embedding(decoder_inputs)

        decoder_outputs, _ = self.decoder_gru(dec_emb, initial_state=encoder_states)

        attn_out, attn_states = self.attention_layer([encoder_outputs, decoder_outputs])
        decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attn_out])

        decoder_outputs = self.decoder_dense(decoder_concat_input)

        return decoder_inputs, decoder_outputs

    def build_model(self):
        encoder_inputs, encoder_outputs, state_h = self.build_encoder()
        decoder_inputs, decoder_outputs = self.build_decoder(encoder_outputs, state_h)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name=self.name)
        model.compile(
            optimizer=self.get_optimizer(),
            loss=self.get_loss(),
            metrics=self.get_metrics(),
        )

        return model

    def build_inference(self):
        encoder_inputs, encoder_outputs, state_h = self.build_encoder()
        encoder_model = Model(encoder_inputs, [encoder_outputs, state_h])

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_hidden_state_input = Input(shape=(self.max_text_len, self.latent_dim))

        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        dec_emb = self.decoder_embedding(decoder_inputs)
        decoder_outputs, state_h = self.decoder_gru(
            dec_emb, initial_state=[decoder_state_input_h]
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
            ],
            [decoder_outputs, state_h],
        )

        return encoder_model, decoder_model

    def decode_sequence(self, input_seq):
        import numpy as np

        # Encoder outputs and final hidden state
        e_out, e_h = self.encoder_model.predict(input_seq)

        # Initialize the target sequence with the start token
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.target_word_index["sostok"]

        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            # Decoder returns the next token probabilities and the new hidden state
            output_tokens, h = self.decoder_model.predict([target_seq, e_out, e_h])

            # Get the predicted token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.reverse_target_word_index[sampled_token_index]

            # Stop if end token is reached or max length is exceeded
            if sampled_token != "eostok":
                decoded_sentence += " " + sampled_token

            if sampled_token == "eostok" or len(decoded_sentence.split()) >= (
                self.max_summary_len - 1
            ):
                stop_condition = True

            # Update the target sequence for the next time step
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update the hidden state for the next prediction
            e_h = h

        return decoded_sentence
