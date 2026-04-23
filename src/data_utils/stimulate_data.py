
from autocpd.neuralnetwork import general_deep_nn, general_simple_nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

from autocpd.neuralnetwork import compile_and_fit, general_simple_nn
from autocpd.utils import DataGenAlternative, GenDataMean




def plot_samples(x, y, tau, num_samples=4):
    idxs = np.random.choice(len(x), num_samples, replace=False)

    plt.figure(figsize=(12, 8))

    for i, idx in enumerate(idxs):
        plt.subplot(num_samples, 1, i+1)
        plt.plot(x[idx])

        if y[idx][0] == 1:
            plt.axvline(tau[idx], linestyle='--')
            plt.title(f"CP at {tau[idx]}")
        else:
            plt.title("No CP")

    plt.tight_layout()
    plt.show()




def stimulate_data(
    length_ts, sample_size, epsilon=0.05, mean_l=0, tau_bound=2, rho=0, scale=0, ar_model_name='AR0', sigma=1):

  # detection boundary / threshold trong CPD
  B = np.sqrt(8 * np.log(length_ts / epsilon) / length_ts)

  # ngưỡng magnitude change: mu_L - mu_R
  B_bound = np.array([0.5, 1.5])

  # %% main double for loop
  N = int(sample_size / 2)

  #  generate the dataset for alternative hypothesis (have CP)
  np.random.seed(2022)  # numpy seed
  tf.random.set_seed(2022)  # tensorflow seed

  result = DataGenAlternative(
      N_sub=N,
      B=B,
      mu_L=mean_l,
      n=length_ts,
      B_bound=B_bound,
      ARcoef=rho,
      sigma=sigma,
      tau_bound=tau_bound,
      scale=scale,
      ar_model=ar_model_name,
  )

  data_alt = result["data"]

  # vector lưu tất cả vị trí change point
  tau_alt = result["tau_alt"]

  mu_R_alt = result["mu_R_alt"]

  #  generate dataset for null hypothesis
  data_null = GenDataMean(N, length_ts, cp=None, mu=(mean_l, mean_l), sigma=1)

  # concate all data: alt hypo + null hypo
  data_all = np.concatenate((data_alt, data_null), axis=0)
  y_all = np.repeat((1, 0), N).reshape((2 * N, 1))
  tau_all = np.concatenate((tau_alt, np.repeat(0, N)), axis=0)
  mu_R_all = np.concatenate((mu_R_alt, np.repeat(mean_l, N)), axis=0)

  #  generate the training dataset and test dataset
  x_train, y_train, tau_train, mu_R_train = shuffle(
      data_all, y_all, tau_all, mu_R_all, random_state=42
  )

  return x_train, y_train, tau_train, mu_R_train