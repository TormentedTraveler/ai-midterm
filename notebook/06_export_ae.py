"""
Phase 5a (run in env-ml, the PyTorch environment):
Train the final autoencoder on ALL interictal windows with the raw 11 features and
export everything the TFLite step needs into ae_export.npz.

    python 06_export_ae.py
"""
import importlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

M = importlib.import_module("02_models_and_threshold")

df = pd.read_csv(M.CSV)
df[M.LOG_FEATS] = np.log1p(df[M.LOG_FEATS])

train = df[df.label == "interictal"]
sc = StandardScaler().fit(train[M.FEATS])
X_all = sc.transform(df[M.FEATS]).astype(np.float32)

ae = M.train_ae(sc.transform(train[M.FEATS]))
scores = M.ae_score(ae, X_all)

sd = {k: v.detach().numpy() for k, v in ae.state_dict().items()}
np.savez("ae_export.npz",
         W1=sd["enc.0.weight"], b1=sd["enc.0.bias"],
         W2=sd["enc.2.weight"], b2=sd["enc.2.bias"],
         W3=sd["dec.0.weight"], b3=sd["dec.0.bias"],
         W4=sd["dec.2.weight"], b4=sd["dec.2.bias"],
         mean=sc.mean_, scale=sc.scale_,
         X=X_all, torch_scores=scores,
         labels=df.label.values.astype(str),
         is_train=(df.label == "interictal").values)
print("saved ae_export.npz | windows:", X_all.shape, "| features:", M.FEATS)