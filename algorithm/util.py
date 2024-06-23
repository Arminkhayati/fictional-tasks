from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sksurv.metrics import integrated_brier_score
import numpy as np
import torch

def mlt_train_test_split(full_dataset, indices, stratify, batch_size, ratio=0.25):
    train_indices, test_indices = train_test_split(
        indices, stratify=stratify, random_state=1, test_size=ratio
    )
    train_dataset, test_dataset = (torch.utils.data.Subset(full_dataset, train_indices),
                                   torch.utils.data.Subset(full_dataset, test_indices))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    for x, y, w, e in train_loader:
        in_features = x.shape[1]
        out_features = len(y)
        break
    return train_loader, test_loader, train_dataset, test_dataset , in_features, out_features


# made by Xinyu
def binarize_and_sum_columns(output_list):
    def binarize_list(input_list):
        tensor = torch.Tensor(input_list)
        # print(input_list.max() == input_list.min())
        binary_tensor = (tensor >= 0.5).float()
        return binary_tensor

    result = binarize_list(output_list[0])
    for i in range(1, len(output_list)):
        binary_column = binarize_list(output_list[i])
        # print(binary_column.max() == binary_column.min())
        result += binary_column

    return result


def brier_score(event_all, Y_all, target_events, Y_true_target, target_predictions):
    dtype = [('cens', bool), ('time', int)]
    # Create the structured array
    all_array = np.array(list(zip(event_all.astype(bool), Y_all.sum(axis=1).cpu())), dtype=dtype)
    target_array = np.array(list(zip(target_events.astype(bool), Y_true_target)), dtype=dtype)
    _times = np.arange(1, Y_true_target.max())
    probs = target_predictions #np.column_stack([item.cpu().numpy() for item in target_predictions])
    return integrated_brier_score(all_array, target_array, probs[:, 1:7], _times)

# def true_values_from_data_loader(data_loader):
#     trues = []
#     statuses = []
#     for _, targets, masks, status in data_loader:
#         true_label = [targets[i] * masks[i] for i in range(len(targets))]
#         trues.append(true_label)
#         statuses.append(status)
#
#     trues = [torch.cat([preds[i] for preds in trues]) for i in range(len(trues[0]))]
#     statuses = torch.cat([status for status in statuses])
#     Y_true = binarize_and_sum_columns(trues)
#     Y_true = Y_true.squeeze()
#
#     return statuses, Y_true

def unique_value_counts(np_array):
    # Convert the PyTorch tensor to a NumPy array
    if torch.is_tensor(np_array):
        np_array = np_array.cpu().numpy()

    # Use np.unique to get unique values and their counts
    unique_values, counts = np.unique(np_array, return_counts=True)

    # Convert the results back to PyTorch tensors (if needed)
    unique_values_tensor = torch.from_numpy(unique_values)
    counts_tensor = torch.from_numpy(counts)
    print("Unique Values:", unique_values_tensor)
    print("Counts:", counts_tensor)
    return unique_values_tensor, counts_tensor

