import os
import numpy as np
import tensorflow as tf 
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, ReLU, 
    GlobalAveragePooling1D, Dense, Input,  
    MultiHeadAttention, LayerNormalization, GRU,
    LSTM, Dropout, Bidirectional
)
from sklearn.metrics import classification_report
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


# 1) Configuration
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()
SEQ_LEN      = 60
FEATURES     = 2        
EPOCHS       = 100
BATCH_SIZE   = 128
TFLITE_MODEL = r'model/drowsiness_sequence_gru.tflite'

# 2) Load the pickled list-of-arrays
data = np.load("features_ear_mar.npz", allow_pickle=True)
X_train, y_train = data['X_train'].tolist(), data['y_train'].tolist()
X_val,   y_val   = data['X_val'].tolist(),   data['y_val'].tolist()
X_test,  y_test  = data['X_test'].tolist(),  data['y_test'].tolist()

# 3) Build sliding-windows per video
def make_sequences(X_list, y_list, seq_len):
    seqs, labels = [], []
    for X_vid, y_vid in zip(X_list, y_list):
        for i in range(len(X_vid) - seq_len + 1):
            seqs.append(X_vid[i:i+seq_len])
            labels.append(y_vid[i+seq_len-1])
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.int32)

Xtr, ytr = make_sequences(X_train, y_train, SEQ_LEN)
Xv,  yv  = make_sequences(X_val,   y_val,   SEQ_LEN)
Xte, yte = make_sequences(X_test,  y_test,  SEQ_LEN)
print("Sequence shapes:", Xtr.shape, ytr.shape, Xv.shape, yv.shape, Xte.shape, yte.shape)


# 4) Model Configurations
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,          # Reduce LR by half
    patience=2,          # Wait 5 epochs with no improvement
    min_lr=1e-6,         # Lower bound
    verbose=1
)

early_stop   = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 5) Build model
def build_model(seq_len, feature_dim):
    inp = Input(shape=(seq_len, feature_dim), name='seq_input')
    x = GRU(64, recurrent_dropout=0.1)(inp)
    x = Dense(32, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inp, out)
    model.compile(tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model(SEQ_LEN, FEATURES)
model.summary()

# 6) Train
model.fit(
    Xtr, ytr,
    validation_data=(Xv, yv),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[lr_scheduler, early_stop]
)

# 7) Evaluate
y_pred = (model.predict(Xte) > 0.5).astype(int).flatten()
print("\nTest Classification Report:")
print(classification_report(yte, y_pred, target_names=["Alert (0)", "Drowsy (1)"]))

# 8) Convert to TFLite with TF-Select for GRU
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

os.makedirs(os.path.dirname(TFLITE_MODEL), exist_ok=True)
with open(TFLITE_MODEL, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Quantized & TF-Select TFLite model saved to {TFLITE_MODEL}")