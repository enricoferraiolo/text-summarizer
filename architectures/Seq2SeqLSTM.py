from keras import backend as K
import numpy as np

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

# FIXME:
# devo provare a costruire encoder_model e decoder_model quando faccio l'init dell'istanza della classe
# e provare a fare inferenza da li

class Seq2SeqLSTM(BaseModel):
    def __init__(
        self, x_voc, y_voc, max_text_len, max_summary_len, x_tokenizer, y_tokenizer
    ):
        self.latent_dim = 300
        self.embedding_dim = 100
        self.name = "Seq2SeqLSTM"
        self.reverse_target_word_index = y_tokenizer.index_word
        self.reverse_source_word_index = x_tokenizer.index_word
        self.target_word_index = y_tokenizer.word_index
        super().__init__(
            x_voc,
            y_voc,
            max_text_len,
            max_summary_len,
            x_tokenizer=x_tokenizer,
            y_tokenizer=y_tokenizer,
            name=self.name,
        )
        self.encoder_model, self.decoder_model = self.build_inference()

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

    def build_inference(self):
        """Builds the inference models for the encoder and decoder."""
        # Encoder inference model
        encoder_model = Model(
            inputs=self.encoder_inputs,
            outputs=[
                self.encoder_outputs,
                self.state_h,
                self.state_c,
            ],
        )

        # Decoder inference model
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_hidden_state_input = Input(
            shape=(self.max_text_len, self.latent_dim),
        )

        # Get the embeddings of the decoder sequence
        dec_emb_layer = Embedding(self.y_voc, self.embedding_dim, trainable=True)
        dec_emb = dec_emb_layer(self.decoder_inputs)

        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_lstm = LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True,
            dropout=0.4,
            recurrent_dropout=0.2,
        )
        decoder_outputs, state_h, state_c = decoder_lstm(
            dec_emb, initial_state=[decoder_state_input_h, decoder_state_input_c]
        )

        # Attention inference
        attn_layer = AttentionLayer()
        attn_out, attn_states = attn_layer(
            [decoder_hidden_state_input, decoder_outputs]
        )
        decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attn_out])

        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_dense = TimeDistributed(Dense(self.y_voc, activation="softmax"))
        decoder_outputs = decoder_dense(decoder_concat_input)

        # Final decoder model
        decoder_model = Model(
            [self.decoder_inputs]
            + [
                decoder_hidden_state_input,
                decoder_state_input_h,
                decoder_state_input_c,
            ],
            [decoder_outputs] + [state_h, state_c],
        )

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

        return encoder_model, decoder_model

    def decode_sequence(self, input_seq):
        """Decodes an input sequence to generate the output sequence."""
        e_out, e_h, e_c = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.target_word_index["sostok"]

        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + [e_out, e_h, e_c]
            )

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.reverse_target_word_index[sampled_token_index]

            if sampled_token != "eostok":
                decoded_sentence += " " + sampled_token

            if sampled_token == "eostok" or len(decoded_sentence.split()) >= (
                self.max_summary_len - 1
            ):
                stop_condition = True

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            e_h, e_c = h, c

        return decoded_sentence

    def seq2summary(self, input_seq):
        newString = ""
        for i in input_seq:
            if (
                i != 0 and i != self.target_word_index["sostok"]
            ) and i != self.target_word_index["eostok"]:
                newString = newString + self.reverse_target_word_index[i] + " "
        return newString

    def seq2text(self,input_seq):
        newString = ""
        for i in input_seq:
            if i != 0:
                newString = newString + self.reverse_source_word_index[i] + " "
        return newString
