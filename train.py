#train.py
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from data_processing import load_captions_data, train_val_split, create_vectorization_layer, make_dataset
from model import get_image_captioning_model

# Constants
VOCAB_SIZE = 10000
SEQ_LENGTH = 25
EMBED_DIM = 256
BATCH_SIZE = 64
EPOCHS = 30

# Directory paths
CHECKPOINTS_DIR = 'checkpoints'
MODELS_DIR = 'models'
GRAPHS_DIR = 'graphs'
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'image_captioning_model')

# Create directories if they don't exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# Load and preprocess data
captions_mapping, text_data = load_captions_data("Flickr8k.token.txt")
train_data, valid_data = train_val_split(captions_mapping)
vectorization = create_vectorization_layer(text_data)
train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()), vectorization)
valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()), vectorization)

# Create the model
model = get_image_captioning_model(VOCAB_SIZE, SEQ_LENGTH, EMBED_DIM)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Callbacks for saving checkpoints and early stopping
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(CHECKPOINTS_DIR, 'epoch_{epoch:02d}_val_loss_{val_loss:.2f}.ckpt'),
    save_weights_only=False,
    save_best_only=False,
    save_freq='epoch'
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# Train the model and capture training history
history = model.fit(train_dataset, epochs=EPOCHS,
                    validation_data=valid_dataset,
                    callbacks=[checkpoint_callback, early_stopping_callback])

# Save the final model in different formats
model.save(os.path.join(MODELS_DIR, 'image_captioning_model'))  # TensorFlow SavedModel format
model.save(os.path.join(MODELS_DIR, 'image_captioning_model.h5'))  # .h5 format
model.save(os.path.join(MODELS_DIR, 'image_captioning_model.keras'), save_format='keras')  # .keras format

# Plotting the epoch vs loss graph
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.legend()
plt.savefig(os.path.join(GRAPHS_DIR, 'epoch_vs_loss.png'))
plt.show()
