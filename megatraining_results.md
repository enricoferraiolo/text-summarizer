ho fatto il training dei modelli:
```python
model_classes = [
    Seq2SeqGRU,
    Seq2SeqLSTM,
    Seq2SeqLSTMGlove,
    Seq2SeqBiLSTM,
    Seq2Seq3BiLSTM,
]
```

con le seguenti configurazioni di parametri:
```python
latent_dim = [256]
embedding_dim = [512]
encoder_dropout = [0.2]
encoder_recurrent_dropout = [0.2]
decoder_dropout = [0.2]
decoder_recurrent_dropout = [0.2]
# Store optimizer configurations as classes and learning rates
optimizers = [
    {"class": Adam, "learning_rate": 0.001},
    {"class": RMSprop, "learning_rate": 0.001},
]
epochs = [50]
batch_size = [128]
```
per scoprire che l'optimizer Adam è il migliore