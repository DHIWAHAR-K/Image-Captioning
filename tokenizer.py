#tokenizer.py
import tensorflow as tf
from preprocess import captions
from configs import VOCABULARY_SIZE, MAX_LENGTH

tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCABULARY_SIZE,
    standardize=None,
    output_sequence_length=MAX_LENGTH)

tokenizer.adapt(captions['caption'])