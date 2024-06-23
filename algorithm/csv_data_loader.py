import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MinMaxScaler

class CSVDataLoader:

    def __init__(self, file_path='Dataset/processdata.csv'):
        self.file_path = file_path
        self.load()
        self.preprocess()

    def load(self):
        data = pd.read_csv('Dataset/processdata.csv', encoding='latin-1')
        date_columns = ['Date.of.Last.Contact', 'Date.of.Diagnostic']
        data[date_columns] = data[date_columns].apply(pd.to_datetime, errors='coerce')
        has_na = data[date_columns].isna().any(axis=1)
        if has_na.any():
            data['Survival_Time'] = (data['Date.of.Last.Contact'] - data['Date.of.Diagnostic']).dt.days
        else:
            data['Survival_Time'] = (data['Date.of.Last.Contact'] - data['Date.of.Diagnostic']).dt.days

        data.loc[:, 'Survival_Time'] = data['Survival_Time'].replace({-1: 0})
        data['indicater'] = np.where(data['Date.of.Death'].isna(), 0, 1)
        columns_to_drop = ['Date.of.Death', 'Date.of.Last.Contact', 'Date.of.Diagnostic']
        data.drop(columns=columns_to_drop, inplace=True)

        self.data = data

    def preprocess(self):
        columns_to_one_hot = ['RCBP.Name', 'Raca.Color', 'State.Civil', 'Code.Profession', 'Name.Occupation',
                              'Status.Address',
                              'City.Address', 'Description.of.Topography', 'Topography.Code', 'Morphology.Description',
                              'Code.of.Morphology', 'Description.of.Disease', 'Illness.Code', 'Diagnostic.means',
                              'Extension',
                              'Type.of.Death']
        # Replace other values that are not in top 9, into "other"
        for column in columns_to_one_hot:
            top_9_values = self.data[column].value_counts().nlargest(9).index
            self.data[column] = self.data[column].where(self.data[column].isin(top_9_values), 'other')

        self.data = pd.get_dummies(self.data, columns=columns_to_one_hot)
        columns_to_binarize = ['Gender', 'Indicator.of.Rare.Case']

        lb = LabelBinarizer()
        for column in columns_to_binarize:
            self.data[column] = lb.fit_transform(self.data[column])

        scaler = MinMaxScaler()
        self.data['Age'] = scaler.fit_transform(self.data[['Age']])


    def get_data(self, Tmax=7500, num_intervals=7):
        X = self.data.drop(['Survival_Time', 'indicater'], axis=1)
        self.time_all = self.data['Survival_Time'].values
        self.event_all = self.data['indicater'].values
        # max_time = self.data['Survival_Time'].max()

        intervals = [(i * (Tmax // num_intervals), (i + 1) * (Tmax // num_intervals)) for i in range(num_intervals)]
        Y = np.zeros((len(self.time_all), num_intervals), dtype=np.int_)
        # Until the event happens, value is 1. after that, it is 0
        for i, time_val in enumerate(self.time_all):
            for j, (left, right) in enumerate(intervals):
                if time_val > right or (left < time_val <= right):
                    Y[i, j] = 1
        Y = torch.Tensor(Y)

        # Creat mask matrix
        W = np.zeros((len(self.time_all), num_intervals), dtype=np.int_)
        # Until the time we know he was alive the value is 1, after that is 0
        for i, (time_val, event_val) in enumerate(zip(self.time_all, self.event_all)):
            for j, (left, right) in enumerate(intervals):
                if event_val == 0 and time_val < left:
                    W[i, j] = 0
                else:
                    W[i, j] = 1
        W = torch.Tensor(W)

        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
        X = torch.tensor(X.values, dtype=torch.float32)

        Y_transform = [Y[:, i:i + 1] for i in range(Y.size(1))]
        W_transform = [W[:, i:i+1] for i in range(W.size(1))]

        return X, Y, Y_transform, W, W_transform, self.time_all, self.event_all














