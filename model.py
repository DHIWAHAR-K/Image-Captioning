#model.py
import tensorflow as tf

IMAGE_SIZE = (299, 299)

def get_cnn_model():
    base_model = tf.keras.applications.EfficientNetB0(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    cnn_out = base_model.output
    cnn_out = tf.keras.layers.Reshape((-1, cnn_out.shape[-1]))(cnn_out)
    cnn_model = tf.keras.Model(base_model.input, cnn_out)
    return cnn_model

def attention_mechanism(inputs, states):
    attention = tf.keras.layers.Attention()([inputs, states])
    context = tf.reduce_sum(attention * inputs, axis=1)
    return context

def get_image_captioning_model(vocab_size, seq_length, embed_dim):
    # Image Model
    image_model = get_cnn_model()
    image_input = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image_input")
    image_features = image_model(image_input)
    
    # Repeat image features for each time step
    image_features = tf.keras.layers.RepeatVector(seq_length)(image_features)
    
    # Caption Model
    caption_input = tf.keras.Input(shape=(seq_length,), name="caption_input")
    caption_embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(caption_input)
    
    # Attention Mechanism
    context = attention_mechanism(image_features, caption_embeddings)
    
    # Decoder LSTM with Attention
    decoder_lstm = tf.keras.layers.LSTM(embed_dim, return_sequences=True)(context)
    decoder_output = tf.keras.layers.Dense(vocab_size, activation="softmax")(decoder_lstm)
    
    model = tf.keras.Model(inputs=[image_input, caption_input], outputs=decoder_output)
    return model
