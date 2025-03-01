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
    LayerNormalization,
)
from tensorflow.keras.models import Model

from architectures.BaseModel import BaseModel


class Seq2SeqBiLSTMImproved(BaseModel):
    def __init__(
        self,
        x_voc,
        y_voc,
        max_text_len,
        max_summary_len,
        x_tokenizer,
        y_tokenizer,
        name="Seq2SeqBiLSTMImproved",
        name_additional_info="",
        latent_dim=256,
        embedding_dim=300,
        encoder_dropout=0.4,
        encoder_recurrent_dropout=0.4,
        decoder_dropout=0.4,
        decoder_recurrent_dropout=0.2,
        num_encoder_layers=20,
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
        self.num_encoder_layers = num_encoder_layers

        # Initialize shared layers specific to this class
        self.encoder_embedding = Embedding(
            x_voc, self.embedding_dim, trainable=True, name="encoder_embedding"
        )
        self.decoder_embedding = Embedding(
            y_voc, self.embedding_dim, trainable=True, name="decoder_embedding"
        )
        
        # Build components
        self.encoder_bilstms = self._build_encoder_layers()
        self.decoder_lstm = self._build_decoder_layer()
        self.attention = AttentionLayer(name="attention_layer")
        self.decoder_dense = TimeDistributed(
            Dense(y_voc, activation="softmax"), name="final_dense"
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

    def _build_encoder_layers(self):
        """Create stacked BiLSTM layers with layer normalization"""
        layers = []
        for i in range(self.num_encoder_layers):
            layers.append(
                Bidirectional(
                    LSTM(
                        self.latent_dim,
                        return_sequences=True,
                        return_state=(i == self.num_encoder_layers - 1),
                        dropout=self.encoder_dropout,
                        recurrent_dropout=self.encoder_recurrent_dropout,
                        name=f"enc_bilstm_{i+1}",
                    )
                )
            )
        return layers

    def _build_decoder_layer(self):
        """Create decoder LSTM with layer normalization"""
        return LSTM(
            self.latent_dim * 2,  # Match encoder's concatenated states
            return_sequences=True,
            return_state=True,
            dropout=self.decoder_dropout,
            recurrent_dropout=self.decoder_recurrent_dropout,
            name="dec_lstm",
        )


    def build_encoder(self):
        """Stacked BiLSTM encoder with layer normalization"""
        encoder_inputs = Input(shape=(self.max_text_len,))
        x = self.encoder_embedding(encoder_inputs)

        for i, layer in enumerate(self.encoder_bilstms):
            if i < self.num_encoder_layers - 1:
                x = layer(x)
                x = LayerNormalization()(x)
            else:
                outputs, fh, fc, bh, bc = layer(x)
                states = [
                    Concatenate()([fh, bh]),  # Forward + backward final states
                    Concatenate()([fc, bc]),
                ]
        return encoder_inputs, outputs, states[0], states[1]

    def build_decoder(self, encoder_outputs, initial_states):
        """Decoder with attention and enhanced processing"""
        decoder_inputs = Input(shape=(None,))
        x = self.decoder_embedding(decoder_inputs)

        # LSTM processing
        lstm_out, *state = self.decoder_lstm(x, initial_state=initial_states)
        lstm_out = LayerNormalization()(lstm_out)

        # Attention mechanism
        context, attn_weights = self.attention([encoder_outputs, lstm_out])

        # Combine context with LSTM output
        combined = Concatenate(axis=-1)([lstm_out, context])

        # Intermediate processing
        intermediate = TimeDistributed(Dense(self.latent_dim * 2, activation="relu"))(
            combined
        )
        intermediate = LayerNormalization()(intermediate)

        # Final output
        outputs = self.decoder_dense(intermediate)
        return decoder_inputs, outputs, state

    def build_model(self):
        """Full model assembly"""
        enc_inputs, enc_outputs, state_h, state_c = self.build_encoder()
        dec_inputs, dec_outputs, _ = self.build_decoder(enc_outputs, [state_h, state_c])

        model = Model([enc_inputs, dec_inputs], dec_outputs, name=self.name)
        model.compile(
            optimizer=self.get_optimizer(),
            loss=self.get_loss(),
            metrics=self.get_metrics(),
        )
        return model

    def build_inference(self):
        """Inference models with beam search support"""
        # Encoder inference
        enc_inputs, enc_outputs, state_h, state_c = self.build_encoder()
        enc_model = Model(enc_inputs, [enc_outputs, state_h, state_c])

        # Decoder inference setup
        dec_inputs = Input(shape=(None,))
        enc_out_input = Input(shape=(self.max_text_len, self.latent_dim * 2))
        state_h_input = Input(shape=(self.latent_dim * 2,))
        state_c_input = Input(shape=(self.latent_dim * 2,))

        dec_emb = self.decoder_embedding(dec_inputs)
        dec_out, state_h, state_c = self.decoder_lstm(
            dec_emb, initial_state=[state_h_input, state_c_input]
        )

        # Attention and processing
        context, _ = self.attention([enc_out_input, dec_out])
        combined = Concatenate(axis=-1)([dec_out, context])
        intermediate = TimeDistributed(Dense(self.latent_dim * 2, activation="relu"))(
            combined
        )
        outputs = self.decoder_dense(intermediate)

        dec_model = Model(
            [dec_inputs, enc_out_input, state_h_input, state_c_input],
            [outputs, state_h, state_c],
        )
        return enc_model, dec_model
