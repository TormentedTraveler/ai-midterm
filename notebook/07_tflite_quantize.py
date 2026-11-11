"""
Phase 5b (run in env-tflite, the TensorFlow environment):
Rebuild the PyTorch autoencoder in Keras, convert to float32 and int8 TFLite,
and report model size, quantization error and the accuracy drop.

    pip install tensorflow numpy scipy scikit-learn
    python 07_tflite_quantize.py
"""
import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

E = np.load("ae_export.npz", allow_pickle=True)   # labels were saved as an object array
X = E["X"].astype(np.float32)
labels = E["labels"]
is_train = E["is_train"]
d = X.shape[1]

# ---- 1. rebuild in Keras (PyTorch Linear weight is (out,in); Keras kernel is (in,out)) ----
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(d,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(3),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(d),
])
model.set_weights([E["W1"].T, E["b1"], E["W2"].T, E["b2"],
                   E["W3"].T, E["b3"], E["W4"].T, E["b4"]])


def scores_from(recon):
    return ((recon - X) ** 2).mean(axis=1)


keras_scores = scores_from(model.predict(X, verbose=0))
print("Keras vs PyTorch max |score diff|:",
      float(np.abs(keras_scores - E["torch_scores"]).max()))

# ---- 2. convert ----
conv = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_f32 = conv.convert()

rep_idx = np.where(is_train)[0][:500]


def rep_data():
    for i in rep_idx:
        yield [X[i:i + 1]]


conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tflite_i8 = conv.convert()

open("ae_f32.tflite", "wb").write(tflite_f32)
open("ae_int8.tflite", "wb").write(tflite_i8)


# ---- 3. run both on the desktop interpreter ----
def run_tflite(content, data):
    it = tf.lite.Interpreter(model_content=content)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    s_i, z_i = inp["quantization"]
    s_o, z_o = out["quantization"]
    res = []
    for x in data:
        xin = (np.clip(np.round(x / s_i + z_i), -128, 127).astype(np.int8)
               if inp["dtype"] == np.int8 else x.astype(np.float32))
        it.set_tensor(inp["index"], xin[None, :])
        it.invoke()
        y = it.get_tensor(out["index"])[0]
        if out["dtype"] == np.int8:
            y = (y.astype(np.float32) - z_o) * s_o
        res.append(y)
    return np.array(res)


s_f32 = scores_from(run_tflite(tflite_f32, X))
s_i8 = scores_from(run_tflite(tflite_i8, X))


def auc_vs_interictal(s, target):
    m = np.isin(labels, ["interictal", target])
    return roc_auc_score(labels[m] == target, s[m])


print("\n=== Model size ===")
print(f"float32 .tflite : {len(tflite_f32):6d} bytes")
print(f"int8    .tflite : {len(tflite_i8):6d} bytes")

print("\n=== Quantization effect on anomaly scores (all windows) ===")
print("Spearman rank corr float32 vs int8:", round(spearmanr(s_f32, s_i8)[0], 4))
print("mean |score diff| (int8 vs float32):", float(np.abs(s_f32 - s_i8).mean()))
for target in ("ictal", "preictal"):
    a32, a8 = auc_vs_interictal(s_f32, target), auc_vs_interictal(s_i8, target)
    print(f"AUC {target:8s} vs interictal: float32 {a32:.3f} | int8 {a8:.3f} | drop {a32 - a8:+.3f}")

# ---- 4. analytic memory estimate (int8 graph) ----
act = d + 8 + 3 + 8 + d                         # activation tensors, 1 byte each
w = d * 8 + 8 * 3 + 3 * 8 + 8 * d               # int8 weights
b = (8 + 3 + 8 + d) * 4                         # int32 biases
print("\n=== Analytic estimate (int8) ===")
print(f"activations (no buffer reuse, upper bound): {act} bytes")
print(f"weights: {w} bytes | biases: {b} bytes")
print("The real tensor-arena size must be measured in the TFLite Micro build "
      "(interpreter arena_used_bytes()); the runtime overhead is not in this estimate.")


# ---- 5. C array for the firmware ----
def to_c_array(content, name):
    body = ",".join(f"0x{c:02x}" for c in content)
    return (f"alignas(16) const unsigned char {name}[] = {{{body}}};\n"
            f"const unsigned int {name}_len = {len(content)};\n")


open("model_data.h", "w").write("#pragma once\n#include <cstdint>\n" +
                                to_c_array(tflite_i8, "g_ae_model"))
np.savez("scaler.npz", mean=E["mean"], scale=E["scale"])
print("\nwrote ae_f32.tflite, ae_int8.tflite, model_data.h, scaler.npz")