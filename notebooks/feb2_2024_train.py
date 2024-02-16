# %%
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from mihm.data.process import (
    multi_cat_to_one_hot,
    binary_to_one_hot,
    standardize_continuous_cols,
    convert_categorical_to_ordinal,
)
from mihm.data.trainutils import train_test_split
from mihm.model.mihm import MIHM, IndexPredictionModel
from mihm.model.mihm_dataset import MIHMDataset
from mihm.model.modelutils import get_index_prediction_weights

# %%
read_cols = [
    "zPCPhenoAge_acc",
    "m_HeatIndex_7d",
    "age2016",
    "female",
    "racethn",
    "eduy",
    "ihs_wealthf2016",
    "pmono",
    "PNK_pct",
    "PBcell_pct",
    "PCD8_Plus_pct",
    "PCD4_Plus_pct",
    "PNCD8_Plus_pct",
    "smoke2016",
    "drink2016",
    "bmi2016",
    "tractdis",
    "urban",
    "mar_cat2",
    "psyche2016",
    "stroke2016",
    "hibpe2016",
    "diabe2016",
    "hearte2016",
    "ltactx2016",
    "mdactx2016",
    "vgactx2016",
    "chd2016",
    "dep2016",
    "adl2016",
    "living2016",
    "division",
]


df = pd.read_stata("../HeatResilience.dta", columns=read_cols)

# %%
categorical_cols = [
    "female",
    "racethn",
    "urban",
    "mar_cat2",
    "psyche2016",
    "stroke2016",
    "hibpe2016",
    "diabe2016",
    "hearte2016",
    "living2016",
    "division",
]
ordinal_cols = ["smoke2016", "drink2016", "ltactx2016", "mdactx2016", "vgactx2016"]
continuous_cols = [
    "eduy",
    "ihs_wealthf2016",
    "age2016",
    "pmono",
    "bmi2016",
    "tractdis",
    "chd2016",
    "dep2016",
    "adl2016",
    "m_HeatIndex_7d",
    "PNK_pct",
    "PBcell_pct",
    "PCD8_Plus_pct",
    "PCD4_Plus_pct",
    "PNCD8_Plus_pct",
]
for c in categorical_cols:
    df[c] = df[c].astype("category")
# categorical = [c for c in df.columns if df[c].dtype == "category"]
# separate binary vs multicategory cols
binary_cats = [c for c in categorical_cols if df[c].nunique() <= 2]
multi_cats = [c for c in categorical_cols if df[c].nunique() > 2]

# %%
# Preprocess df for model
df = binary_to_one_hot(df, binary_cats)  # convert binary to one hot
df = multi_cat_to_one_hot(df, multi_cats)  # convert multi cat to one hot
df = convert_categorical_to_ordinal(df, ordinal_cols)  # convert ordinal to ordinal
df_norm, mean_std_dict = standardize_continuous_cols(
    df, continuous_cols + ordinal_cols
)  # standardize continuous cols
df_norm.dropna(inplace=True)  # drop Nan rows

# %%
df.columns

# %%
input_cols = [
    "female",
    "eduy",
    "ihs_wealthf2016",
    "pmono",
    "bmi2016",
    "age2016",
    "tractdis",
    "mar_cat2",
    "psyche2016",
    "stroke2016",
    "hibpe2016",
    "diabe2016",
    "hearte2016",
    "chd2016",
    "dep2016",
    "adl2016",
    "living2016",
    "smoke2016",
    "drink2016",
    "ltactx2016",
    "mdactx2016",
    "vgactx2016",  # ordinals
    "racethn_0. NHW",
    "racethn_1. NHB",
    "racethn_2. Hispanic",
    "racethn_3. Others",  # multi cats
    "urban_1. urban",
    "urban_2. suurban (code 2)",
    "urban_3. ex-urban",
    "division_Northeast",
    "division_Midwest",
    "division_South",
    "division_West",
    "PNK_pct",
    "PBcell_pct",
    "PCD8_Plus_pct",
    "PCD4_Plus_pct",
    "PNCD8_Plus_pct",
]
controlled_cols = [
    "m_HeatIndex_7d",
    "pmono",
    "PNK_pct",
    "PBcell_pct",
    "PCD8_Plus_pct",
    "PCD4_Plus_pct",
    "PNCD8_Plus_pct",
]
interaction_predictors = [
    "female",
    "racethn_0. NHW",
    "racethn_1. NHB",
    "racethn_2. Hispanic",
    "racethn_3. Others",
    "eduy",
    "ihs_wealthf2016",
    "bmi2016",
    "tractdis",
    "mar_cat2",
    "psyche2016",
    "stroke2016",
    "hibpe2016",
    "diabe2016",
    "hearte2016",
    "chd2016",
    "dep2016",
    "adl2016",
    "living2016",
    "smoke2016",
    "drink2016",
    "ltactx2016",
    "mdactx2016",
    "vgactx2016",
    "urban_1. urban",
    "urban_2. suurban (code 2)",
    "urban_3. ex-urban",
    "division_Northeast",
    "division_Midwest",
    "division_South",
    "division_West",
]

# %%
# interactor
heat_cont_np = df_norm["m_HeatIndex_7d"].to_numpy()
# controlled vars
controlled_vars_np = df_norm[controlled_cols].to_numpy()
# interaction input vars
interaction_vars_np = df_norm[interaction_predictors].to_numpy()
# dependent var (label)
pheno_epi_np = df_norm["zPCPhenoAge_acc"].to_numpy()

# %%
num_elems, _ = controlled_vars_np.shape
print("number of data points: {}".format(num_elems))

# %%
# split to train and test
train_idx, test_idx = train_test_split(num_elems, 0.7)
train_heat_cont = heat_cont_np[train_idx]
train_controlled_vars = controlled_vars_np[train_idx]
train_interaction_vars = interaction_vars_np[train_idx]
train_pheno_epi = pheno_epi_np[train_idx]

test_heat_cont = torch.from_numpy(heat_cont_np[test_idx].astype(np.float32))
test_controlled_vars = torch.from_numpy(controlled_vars_np[test_idx].astype(np.float32))
test_interaction_vars = torch.from_numpy(
    interaction_vars_np[test_idx].astype(np.float32)
)
test_pheno_epi = torch.from_numpy(pheno_epi_np[test_idx].astype(np.float32))

# %%
# create dataset
train_dataset = MIHMDataset(
    train_heat_cont, train_controlled_vars, train_interaction_vars, train_pheno_epi
)
dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# %% [markdown]
# ## Load model and train

# %%
interaction_var_size = interaction_vars_np.shape[1]
controlled_var_size = controlled_vars_np.shape[1]
hidden_layer_sizes = [50, 10, 1]
model = MIHM(
    interaction_var_size,
    controlled_var_size,
    hidden_layer_sizes,
    include_interactor_bias=True,
    dropout=0.5,
)

# %%
torch.manual_seed(0)
mseLoss = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.1)
epochs = 300

# %%
model.eval()
with torch.no_grad():
    predicted_epi, predicted_index = model(
        test_interaction_vars, test_heat_cont, test_controlled_vars
    )
    loss_test = mseLoss(predicted_epi, test_pheno_epi)
    print("Testing Loss: {}".format(loss_test.item()))
early_stop_trigger_counter = 0
early_stop_tolerance = 0.2

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, sample in enumerate(dataloader):
        optimizer.zero_grad()
        # forward pass
        predicted_epi, predicted_index = model(
            sample["interaction_input_vars"],
            sample["interactor_var"],
            sample["controlled_vars"],
        )
        label = torch.unsqueeze(sample["label"], 1)
        loss = mseLoss(predicted_epi, label)

        # backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # print average loss for epoch
    epoch_loss = running_loss / len(dataloader)

    # evaluation on test set
    model.eval()
    with torch.no_grad():
        predicted_epi, predicted_interaction = model(
            test_interaction_vars, test_heat_cont, test_controlled_vars
        )
        loss_test = mseLoss(predicted_epi, test_pheno_epi)
    print("Epoch {}/{} done!".format(epoch + 1, epochs))
    print("Training Loss: {}".format(epoch_loss))
    print("Testing Loss: {}".format(loss_test.item()))
    # early stopping
    if loss_test.item() > epoch_loss + early_stop_tolerance:
        early_stop_trigger_counter += 1
        if early_stop_trigger_counter > 5:
            print("Early stopping triggered!")
            break

# %%
all_interaction_vars_tensor = torch.from_numpy(interaction_vars_np.astype(np.float32))
model.eval()
predicted_index = model.get_resilience_index(all_interaction_vars_tensor)
all_resilience_index = predicted_index.detach().numpy()

# %%
plt.hist(all_resilience_index, bins=30)

# %%
torch.save(model.state_dict(), "../checkpoints/Feb2_heat_model.pth")

# %%
