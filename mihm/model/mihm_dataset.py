from torch.utils.data import Dataset
import numpy as np
import torch


class MIHMDataset(Dataset):
    def __init__(self, interactor_var, controlled_vars, interaction_input_vars, label):
        self.interactor_var = interactor_var.astype(np.float32)
        self.controlled_vars = controlled_vars.astype(np.float32)
        self.interaction_input_vars = interaction_input_vars.astype(np.float32)
        self.label = label.astype(np.float32)

    def __len__(self):
        return self.controlled_vars.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to_list()

        sample = {
            "interactor_var": torch.tensor(self.interactor_var[index]),
            "controlled_vars": torch.from_numpy(self.controlled_vars[index, :]),
            "interaction_input_vars": torch.from_numpy(
                self.interaction_input_vars[index, :]
            ),
            "label": torch.tensor(self.label[index]),
        }
        return sample


class PM25Dataset(Dataset):
    def __init__(self, pm25, controlled_vars, interaction_vars, pheno_age):
        self.pm25 = pm25.astype(np.float32)
        self.controlled_vars = controlled_vars.astype(np.float32)
        self.interaction_vars = interaction_vars.astype(np.float32)
        self.pheno_age = pheno_age.astype(np.float32)

    def __len__(self):
        return self.controlled_vars.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to_list()

        sample = {
            "controlled_vars": torch.from_numpy(self.controlled_vars[index, :]),
            "pm25": torch.tensor(self.pm25[index]),
            "label": torch.tensor(self.pheno_age[index]),
            "interaction_vars": torch.tensor(self.interaction_vars[index, :]),
        }
        return sample


class HeatDataset(Dataset):
    def __init__(self, heat, controlled_vars, interaction_vars, pheno_age):
        self.heat = heat.astype(np.float32)
        self.controlled_vars = controlled_vars.astype(np.float32)
        self.interaction_vars = interaction_vars.astype(np.float32)
        self.pheno_age = pheno_age.astype(np.float32)

    def __len__(self):
        return self.controlled_vars.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to_list()

        sample = {
            "controlled_vars": torch.from_numpy(self.controlled_vars[index, :]),
            "heat_index": torch.tensor(self.heat[index]),
            "label": torch.tensor(self.pheno_age[index]),
            "interaction_vars": torch.tensor(self.interaction_vars[index, :]),
        }
        return sample
