import torch
from torch.utils.data import Dataset
import pandas as pd


class HeartDiseaseDataset(Dataset):
    def __init__(self, path):
        dataset = pd.read_csv(path)

        # Replace categorical attributes with numerical Labels
        dataset["Sex"] = dataset["Sex"].astype("category")
        dataset["Sex_num_labels"] = dataset["Sex"].cat.codes

        dataset["ChestPainType"] = dataset["ChestPainType"].astype("category")
        dataset["ChestPainType_num_labels"] = dataset["ChestPainType"].cat.codes

        dataset["RestingECG"] = dataset["RestingECG"].astype("category")
        dataset["RestingECG_num_labels"] = dataset["RestingECG"].cat.codes

        dataset["ExerciseAngina"] = dataset["ExerciseAngina"].astype("category")
        dataset["ExerciseAngina_num_labels"] = dataset["ExerciseAngina"].cat.codes

        dataset["ST_Slope"] = dataset["ST_Slope"].astype("category")
        dataset["ST_Slope_num_labels"] = dataset["ST_Slope"].cat.codes

        self.dataset = dataset

    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset.iloc[index]
    
    def stack(self, batch):
        train_params = list()
        insight_params = list()
        labels = list()

        for sample in batch:
            labels.append(torch.tensor(sample["HeartDisease"]))
            train_params.append(torch.tensor([sample[att] for att in ["Age", "Sex_num_labels", "ChestPainType_num_labels", "RestingBP", "Cholesterol", "FastingBS", "RestingECG_num_labels", "MaxHR", "ExerciseAngina_num_labels", "Oldpeak", "ST_Slope_num_labels"]]))
            insight_params.append([sample[att] for att in ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]])

        return train_params, insight_params, labels
    
    def get_data_loaders(self, batch_size: int = 32, shuffle: bool = True, seed: int = 12, split = [0.85, 0.15]):
        train_set, eval_set = torch.utils.data.random_split(self, split, torch.Generator().manual_seed(seed))
        return (torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, collate_fn=self.stack), torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=shuffle, collate_fn=self.stack))