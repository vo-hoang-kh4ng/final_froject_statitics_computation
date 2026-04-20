import os
import pathlib
from collections import Counter
import autocpd

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_docs as tfdocs
from keras import losses, metrics
from sklearn.preprocessing import LabelEncoder
from data_utils.preprocessing import labelSubject, ExtractSubject
from autocpd.neuralnetwork import deep_nn


# %%
# load the real dataset
# root_path = os.path.dirname(autocpd.__file__)
# datapath = pathlib.Path(root_path, "HASC")

google_drive_path = "/content/drive/MyDrive/SampleData/0_sequence/"
datapath = pathlib.Path(google_drive_path)

# set the random seed
np.random.seed(2022)  # numpy seed
tf.random.set_seed(2022)  # tensorflow seed

subjects = [
    "person101",
    "person102",
    "person103",
    "person104",
    "person105",
    "person106",
    "person107",
]

length = 700
size = 15
size0 = 15

# %%
# extract the data
train_subjects = subjects[:3]
print("train subjects", train_subjects)

# extract the change-point from the training dataset
for i, subject in enumerate(train_subjects):
    subject_path = pathlib.Path(datapath, subject)
   
    result = labelSubject(subject_path, length, size, num_trim=100)
    
    if i == 0:
        ts = result["ts"]
        label = result["label"]
        cp = result["cp"]
        id1 = [subject] * cp.shape[0]
    else:
        ts = np.concatenate([ts, result["ts"]], axis=0)
        cp = np.concatenate([cp, result["cp"]])
        label += result["label"]
        id1 += [subject] * result["cp"].shape[0]


# extract the no change-point from the training dataset
for i, subject in enumerate(train_subjects):
    print(subject)
    subject_path = pathlib.Path(datapath, subject)
    result0 = ExtractSubject(subject_path, length, size0)
    if i == 0:
        ts0 = result0["ts"]
        label0 = result0["label"]
        id0 = [subject] * ts0.shape[0]
    else:
        ts0 = np.concatenate([ts0, result0["ts"]], axis=0)
        label0 += result0["label"]
        id0 += [subject] * result0["ts"].shape[0]


ts_train = np.concatenate([ts, ts0], axis=0).copy()
label_train = label + label0
cp_train = cp.copy()
id_train = id1 + id0


TS_train = ts_train.copy()
CP_train = cp_train.copy()
LABEL_train = label_train.copy()

# check the number and frequency of labels
counts = Counter(LABEL_train)
print(counts)
len(counts)

# % add square transformation
tstrain2 = np.square(TS_train)
TS_train = np.concatenate([TS_train, tstrain2], axis=2)

# rescale
datamin = np.min(TS_train, axis=(1, 2), keepdims=True)
datamax = np.max(TS_train, axis=(1, 2), keepdims=True)
TS_train = 2 * (TS_train - datamin) / (datamax - datamin) - 1


le = LabelEncoder()
label_train = le.fit_transform(LABEL_train)


# shuffle the datasets
ind_train = np.random.permutation(TS_train.shape[0])

x_train = TS_train[ind_train, :, :]
y_train = label_train[ind_train]