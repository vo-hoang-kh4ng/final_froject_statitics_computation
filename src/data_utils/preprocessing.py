import os
import posixpath
import numpy as np
import pandas as pd
import warnings


def extract(n1, n2, length, size, ntrim):
    """
    This function randomly extracts samples (consecutive segments) with
    length 'length' from a time series concatenated by two different time
    series with length 'n1' and 'n2' respectively. Argument 'ntrim' controls
    the minimum distance between change-point and start or end point of
    consecutive segment. It returns a dictionary containing two arrays: cp and
    sample. cp is an array of change points. sample is a 2D array where each
    row is the indices of consecutive segment .


    Parameters
    ----------
    n1 : the length of signal before change-point
        _description_
    n2 : int
        the length of time series after change-point
    length : int
        the length of time series segment that we want to extract
    size : int
        the sample size
    ntrim : int
        the number of observations to be trimmed before and after the change-point

    Returns
    -------
    dict
        'cp' is the set of change-points. 'sample' is a matrix of indices
    """
    n = n1 + n2
    ts = np.arange(n)
    if length > n - size:
        warnings.warn(
            "Not enough sample.",
            DeprecationWarning,
        )
    if n1 < ntrim + size or n2 < ntrim + size:
        warnings.warn(
            "One segment has not enough sample.",
            DeprecationWarning,
        )

    cp = np.zeros((size,))
    sample = np.zeros((size, length))
    len_half = length // 2
    if n1 <= n2:
        if n1 >= ntrim + size and n1 <= len_half:
            cp[0] = n1
            s = 0
            sample[0,] = ts[s:length]
            if size == 1:
                return {"cp": cp, "sample": sample}

            s_set = np.random.choice(
                range(max(1, n1 + ntrim - length), min(n1 - ntrim, n - length)),
                (size - 1,),
                replace=False,
            )
            for ind, s in enumerate(s_set):
                cp[ind + 1] = n1 - s
                sample[ind + 1, :] = ts[s : s + length]

            return {"cp": cp, "sample": sample}
        if n1 > len_half:
            cp[0] = len_half
            s = n1 - len_half
            sample[0,] = ts[n1 - len_half : n1 + len_half]
            if size == 1:
                return {"cp": cp, "sample": sample}
            s_set = np.random.choice(
                range(max(0, n1 + ntrim - length), min(n1 - ntrim, n - length)),
                (size - 1,),
                replace=False,
            )
            for ind, s in enumerate(s_set):
                cp[ind + 1] = n1 - s
                sample[ind + 1, :] = ts[s : s + length]

            return {"cp": cp, "sample": sample}
    else:
        if n2 >= ntrim + size and n2 <= len_half:
            cp[0] = length - n2
            s = n - length
            sample[0,] = ts[s:n]
            if size == 1:
                return {"cp": cp, "sample": sample}
            s_set = np.random.choice(
                range(max(0, n1 + ntrim - length), min(n1 - ntrim, n - 1 - length)),
                (size - 1,),
                replace=False,
            )
            for ind, s in enumerate(s_set):
                cp[ind + 1] = n1 - s
                sample[ind + 1, :] = ts[s : s + length]

            return {"cp": cp, "sample": sample}
        if n2 > len_half:
            cp[0] = len_half
            s = n1 - len_half
            sample[0,] = ts[n1 - len_half : n1 + len_half]
            if size == 1:
                return {"cp": cp, "sample": sample}
            s_set = np.random.choice(
                range(max(0, n1 + ntrim - length), min(n1 - ntrim, n - length)),
                (size - 1,),
                replace=False,
            )
            for ind, s in enumerate(s_set):
                cp[ind + 1] = n1 - s
                sample[ind + 1, :] = ts[s : s + length]

            return {"cp": cp, "sample": sample}



def labelTransition(data, label, ind, length, size, num_trim=100):
    """get the transition labels, change-points and time series from one subject

    Parameters
    ----------
    data : DataFrame
        the time series.
    label : DataFrame
        the states of the subject
    ind : scalar
        the index of state
    length : int
        the length of extracted time series
    size : int
        the sample size
    num_trim : int, optional
        the number of observations to be trimmed before and after the change-point, by default 100

    Returns
    -------
    dictionary
        cp: the change-points; ts: time series; label: the transition labels.
    """
    s = label["start"].iloc[ind : ind + 2].to_numpy()
    e = label["end"].iloc[ind : ind + 2].to_numpy()
    state = label["state"].iloc[ind : ind + 2].to_numpy()

    new_label = state[0] + "->" + state[1]

    logical0 = (data["time"] >= s[0]) & (data["time"] <= e[0])
    logical1 = (data["time"] >= s[1]) & (data["time"] <= e[1])

    data_trim = data[logical0 | logical1]

    len0 = sum(logical0)
    len1 = sum(logical1)

    ts_final = np.zeros((size, length, 3))
    label_final = [new_label] * size
    cp_final = np.zeros((size,))

    result = extract(len0, len1, length, size, ntrim=num_trim)
    cp_final = result["cp"].astype("int32")
    sample = result["sample"].astype("int32")

    for i in range(size):
        ts_temp = data_trim.iloc[sample[i, :], 1:4]
        ts_final[i, :, :] = ts_temp.to_numpy()


    return {"cp": cp_final, "ts": ts_final, "label": label_final}




def labelSubject(subject_path, length, size, num_trim=100):
    """
    obtain the transition labels, change-points and time series from one subject.

    Parameters
    ----------
    subject_path : string
        the path of subject data
    length : int
        the length of extracted time series
    size : int
        the sample size
    num_trim : int, optional
        the number of observations to be trimmed before and after the change-point, by default 100

    Returns
    -------
    dictionary
        cp: the change-points; ts: time series; label: the transition labels.
    """
    # get the csv files
    all_files = os.listdir(subject_path)
    csv_files = [f for f in all_files if f.endswith(".csv") and f.startswith("HASC")]

    for ind, fname in enumerate(csv_files):
        print(ind)
        print(fname)
        
        fname_label = fname.replace(".csv", ".label")
        fname_label = posixpath.join(subject_path, fname_label)
        fname = posixpath.join(subject_path, fname)
        
        # load the labels
        label_dataset = pd.read_csv(
            fname_label, comment="#", delimiter=",", names=["start", "end", "state"]
        )
        num_consecutive_states = label_dataset.shape[0]
        
        # load the dataset
        dataset = pd.read_csv(
            fname, comment="#", delimiter=",", names=["time", "x", "y", "z"]
        )
        if num_consecutive_states < 2:
            warnings.warn(
                "The length of times series exceeds the minimum length of two segments. Reduce length or increase the num_trim.",
                DeprecationWarning,
            )

        for i in range(num_consecutive_states - 1):
            result = labelTransition(
                dataset, label_dataset, i, length, size, num_trim=num_trim
            )
            if i == 0:
                ts = result["ts"]
                label = result["label"]
                cp = result["cp"]
            else:
                ts = np.concatenate([ts, result["ts"]], axis=0)
                cp = np.concatenate([cp, result["cp"]])
                label += result["label"]

        if ind == 0:
            ts_ind = ts
            label_ind = label
            cp_ind = cp
        else:
            ts_ind = np.concatenate([ts_ind, ts], axis=0)
            cp_ind = np.concatenate([cp_ind, cp])
            label_ind += label

    return {"cp": cp_ind, "ts": ts_ind, "label": label_ind}



def tsExtract(data_trim, new_label, length, size, len0):
    """
    To extract the labels without change-points

    Parameters
    ----------
    data_trim : DataFrame
        the dataset of one specific state
    new_label : DataFrame
        the label, not transition label.
    length : int
        the length of extracted time series
    size : int
        the sample size
    len0 : int
        the length of time series for one specific state

    Returns
    -------
    dict
        ts: time series; label: the labels.
    """
    ts_final = np.zeros((size, length, 3))
    label_final = [new_label] * size

    sample = np.sort(np.random.choice(range(0, len0 - length), (size,), replace=False))
    for i in range(size):
        ts_temp = data_trim.iloc[sample[i] : sample[i] + length, 1:4]
        ts_final[i, :, :] = ts_temp.to_numpy()

    return {"ts": ts_final, "label": label_final}



def ExtractSubject(subject_path, length, size):
    """
    To extract the null labels without change-points from one subject

    Parameters
    ----------
    subject_path : string
        the path of subject data
    length : int
        the length of extracted time series
    size : int
        the sample size

    Returns
    -------
    dict
        ts: time series; label: the labels.
    """
    # get the csv files
    all_files = os.listdir(subject_path)
    csv_files = list(filter(lambda f: f.endswith(".csv"), all_files))
    csv_files = list(filter(lambda f: f.startswith("HASC"), csv_files))
    for ind, fname in enumerate(csv_files):
        print(ind)
        print(fname)
        fname_label = fname.replace("-acc.csv", ".label")
        fname_label = posixpath.join(subject_path, fname_label)
        fname = posixpath.join(subject_path, fname)
        # load the labels
        label_dataset = pd.read_csv(
            fname_label, comment="#", delimiter=",", names=["start", "end", "state"]
        ).reset_index(drop=True)

        num_consecutive_states = label_dataset.shape[0]
        
        # load the dataset
        dataset = pd.read_csv(
            fname, comment="#", delimiter=",", names=["time", "x", "y", "z"]
        ).reset_index(drop=True)

        ts = np.array([]).reshape((0, length, 3))
        label = []
        
        for i in range(num_consecutive_states):
            s = label_dataset["start"].iloc[i]
            e = label_dataset["end"].iloc[i]
            new_label = label_dataset["state"].iloc[i]
            logical = (dataset["time"] >= s) & (dataset["time"] <= e)
            data_trim = dataset[logical]
            len0 = sum(logical)
            if len0 <= length + size:
                continue
            else:
                result = tsExtract(data_trim, new_label, length, size, len0)
                ts = np.concatenate([ts, result["ts"]], axis=0)
                label += result["label"]

        if ind == 0:
            ts_ind = ts
            label_ind = label
        else:
            ts_ind = np.concatenate([ts_ind, ts], axis=0)
            label_ind += label

    return {"ts": ts_ind, "label": label_ind}