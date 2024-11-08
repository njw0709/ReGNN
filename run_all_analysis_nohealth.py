# %% [markdown]
# ## Prep data

# %%
import pandas as pd
from mihm.data.process import (
    multi_cat_to_one_hot,
    binary_to_one_hot,
    standardize_cols,
    convert_categorical_to_ordinal,
)
from mihm.data.dataset import MIHMDataset
import os

# %%
from mihm.hyperparam.preprocess import preprocess

# %%
data_path = "./Mar1_HeatResilience.dta"
# read model and rename cols
outcome_col = "Pheno Age Accel."
outcome_original_name = "zPCPhenoAge_acc"

read_cols = [
    "zPCPhenoAge_acc",
    "zPCHorvath1_acc",
    "zPCHorvath2_acc",
    "zPCHannum_acc",
    "zPCGrimAge_acc",
    "zDunedinPACE_acc",
    "PC_age",
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
    "dep2016",
    "adl2016",
    "living2016",
    "division",
]

rename_dict = {
    "zPCPhenoAge_acc": "Pheno Age Accel.",
    "zPCHorvath1_acc": "Horvath1 Accel.",
    "zPCHorvath2_acc": "Horvath2 Accel.",
    "zPCHannum_acc": "Hannum Accel.",
    "zPCGrimAge_acc": "GrimAge Accel.",
    "zDunedinPACE_acc": "DunedinPACE Accel.",
    "PC_age": "PC age",
    "m_HeatIndex_7d": "mean heat index over 7d",
    "age2016": "age",
    "female": "female",
    "racethn": "race/ethnicity",
    "eduy": "education (in years)",
    "ihs_wealthf2016": "household wealth (ihs)",
    "smoke2016": "smoking status",
    "drink2016": "drinking status",
    "bmi2016": "bmi",
    "tractdis": "tract disadvantage",
    "urban": "urbanicity",
    "mar_cat2": "marital status",
    "psyche2016": "psychiatric conditions",
    "stroke2016": "stroke",
    "hibpe2016": "hypertension",
    "diabe2016": "diabetes",
    "hearte2016": "heart disease",
    "ltactx2016": "light activity",
    "mdactx2016": "moderate activity",
    "vgactx2016": "vigorous activity",
    "dep2016": "depressive symptoms",
    "adl2016": "adl limitations",
    "living2016": "living alone",
    "division": "census division",
}

interactor_col = "mean heat index over 7d"
controlled_cols = [
    "mean heat index over 7d",
    "pmono",
    "PNK_pct",
    "PBcell_pct",
    "PCD8_Plus_pct",
    "PCD4_Plus_pct",
    "PNCD8_Plus_pct",
]
interaction_predictors = [
    "female",
    "education (in years)",
    "household wealth (ihs)",
    "smoking status",
    "drinking status",
    "bmi",
    "tract disadvantage",
    "marital status",
    "light activity",
    "moderate activity",
    "vigorous activity",
    "adl limitations",
    "living alone",
    "race/ethnicity_1. NHB",
    "race/ethnicity_2. Hispanic",
    "race/ethnicity_3. Others",
    "urbanicity_2. suurban (code 2)",
    "urbanicity_3. ex-urban",
    "census division_Midwest",
    "census division_South",
    "census division_West",
]

# define variable types for preprocessing
categorical_cols = [
    "female",
    "race/ethnicity",
    "urbanicity",
    "marital status",
    "psychiatric conditions",
    "stroke",
    "hypertension",
    "diabetes",
    "heart disease",
    "living alone",
    "census division",
]
ordinal_cols = [
    "smoking status",
    "drinking status",
    "light activity",
    "moderate activity",
    "vigorous activity",
    "adl limitations",
]
continuous_cols = [
    "education (in years)",
    "household wealth (ihs)",
    "age",
    "bmi",
    "tract disadvantage",
    "depressive symptoms",
    "adl limitations",
    "mean heat index over 7d",
    "pmono",
    "PNK_pct",
    "PBcell_pct",
    "PCD8_Plus_pct",
    "PCD4_Plus_pct",
    "PNCD8_Plus_pct",
    "Pheno Age Accel.",
]

# %%
df_orig, heat_dataset = preprocess(
    data_path,
    read_cols,
    rename_dict,
    categorical_cols,
    ordinal_cols,
    continuous_cols,
    interactor_col,
    outcome_col,
    controlled_cols,
    interaction_predictors,
)

# %%
print("Mean and std of standardized predictors: ", heat_dataset.mean_std_dict)

# %% [markdown]
# ### Split data into train test validation set

# %%
from mihm.data.trainutils import train_test_val_split

# %%
train_idx, test_idx, val_idx = train_test_val_split(len(heat_dataset))
print("Train size: ", len(train_idx))
print("Test size: ", len(test_idx))
print("Val size: ", len(val_idx))

# %%
train_heat_dataset = heat_dataset.get_subset(train_idx)
test_heat_dataset = heat_dataset.get_subset(test_idx)
val_heat_dataset = heat_dataset.get_subset(val_idx)
train_heat_dataset.outcome_original_name = outcome_original_name

# %% [markdown]
# ### Train

# %%
from mihm.hyperparam.train import train_mihm, test_mihm
import torch

# %%
hidden_layer_sizes = [50, 10, 1]
vae = True
svd = True
k_dims = 20
epochs = 300
batch_size = 500
lr = 0.003
weight_decay = 0.1
shuffle = True
eval = True
all_interaction_predictors = heat_dataset.to_tensor(device="cuda")[
    "interaction_predictors"
]
model_name = outcome_original_name

# %%
model, traj_data = train_mihm(
    train_heat_dataset,
    val_heat_dataset,
    hidden_layer_sizes,
    vae,
    svd,
    k_dims,
    epochs,
    batch_size,
    lr,
    weight_decay,
    shuffle=shuffle,
    evaluate=eval,
    df_orig=df_orig,
    all_interaction_predictors=all_interaction_predictors,
    file_id=0,
    use_stata=True,
    return_trajectory=True,
)

# %%
val_loss = test_mihm(model, test_heat_dataset.to_tensor(device="cuda"))
print("Validation loss: ", val_loss)

# %%
results_dir = "/home/namj/projects/heat_air_epi/notebooks/Mar1_2024_traj_results"
import pickle

with open(os.path.join(results_dir, "traj_data_{}.pkl".format(model_name)), "wb") as f:
    pickle.dump(traj_data, f)
torch.save(model, os.path.join(results_dir, "model_{}.pt".format(model_name)))

# %% [markdown]
# ## graph trajectory of the model

# %%
import matplotlib.pyplot as plt
import numpy as np
import os

# %%
with open(os.path.join(results_dir, "traj_data_{}.pkl".format(model_name)), "rb") as f:
    traj_data_loaded = pickle.load(f)

# %%
figpath = results_dir
fig, ax = plt.subplots()
x_data = np.arange(1, len(traj_data_loaded))
train_loss = [data["train_loss"] for data in traj_data_loaded[1:]]
test_loss = [data["test_loss"] for data in traj_data_loaded[1:]]
p_val = [data["interaction_pval"] for data in traj_data_loaded[1:]]
ax.plot(x_data, train_loss, label="train loss", color="orange")
ax.plot(x_data, test_loss, label="test loss", color="purple")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax2 = ax.twinx()
ax2.plot(x_data, p_val, label="p val", color="blue")
ax2.set_ylabel("p val", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")
fig.legend(["train_loss", "test_loss", "p_val"], loc="upper right")
ax.set_title("{}".format(model_name))
fig.savefig(os.path.join(figpath, "traj_{}.png".format(model_name)))


# %%
bias = (
    model.interactor_bias.cpu().detach().numpy()
    * heat_dataset.mean_std_dict["mean heat index over 7d"][1]
    + heat_dataset.mean_std_dict["mean heat index over 7d"][0]
)
print("Interactor bias: {} ".format(bias))

# %% [markdown]
# ## Model prediction

# %%
from mihm.model.mihm import IndexPredictionModel
from mihm.model.modelutils import get_index_prediction_weights
import matplotlib.pyplot as plt
import torch

# %%
model_index = model.get_index_prediction_model()
model_index.cuda().eval()

# %%
# get all interaction vars from the dataset
all_interaction_vars_tensor = heat_dataset.to_tensor(device="cuda")[
    "interaction_predictors"
]
mean_predicted_index = model_index(all_interaction_vars_tensor)
all_resilience_index = mean_predicted_index.detach().cpu().numpy()

# %%
fig, ax = plt.subplots()
ax.hist(all_resilience_index, bins=50)
ax.set_xlabel("Predicted vulnerability index")
ax.set_ylabel("Counts")
ax.set_title("Vulnerability index score distribution using {}".format(model_name))
fig.savefig(os.path.join(figpath, "vulnerability_index_dist_{}.png".format(model_name)))

# %%
predicted_index, predicted_index_std = model_index.get_index_mean_std(
    all_interaction_vars_tensor
)

# %%
predicted_index_std = predicted_index_std.detach().cpu().numpy()
fig, ax = plt.subplots()
ax.hist(predicted_index_std, bins=50)
ax.set_xlabel("Predicted vulnerability index uncertainty (std)")
ax.set_ylabel("Counts")
ax.set_title("Vulnerability index uncertainty distribution using {}".format(model_name))
fig.savefig(
    os.path.join(figpath, "vulnerability_index_std_dist_{}.png".format(model_name))
)

# %%
df_orig = pd.read_stata(data_path)
df_orig["vul_index"] = all_resilience_index

# %%
save_name = "./data/HeatResilience_{}.dta".format(model_name)
df_orig.to_stata(save_name, write_index=False)

# %% [markdown]
# ## Confirm significance using linear regression on Stata

# %%
import stata_setup

stata_setup.config("/usr/local/stata17", "se")
from pystata import stata
import pandas as pd

# %%
set_cmd = "set scheme burd"
graph_set_cmd = 'graph set window fontface "Arial Narrow"'
stata.run(set_cmd)
stata.run(graph_set_cmd)

# %%
load_cmd = 'use "{}", clear'.format(save_name)
stata.run(load_cmd)

# %%
# regress_cmd = "regress zPCPhenoAge_acc m_HeatIndex_7d age2016 i.female i.racethn eduy ihs_wealthf2016 pmono PNK_pct PBcell_pct PCD8_Plus_pct PCD4_Plus_pct PNCD8_Plus_pct smoke2016 i.drink2016 bmi2016 tractdis i.urban i.mar_cat2 psyche2016 stroke2016 hibpe2016 diabe2016 hearte2016 i.ltactx2016 i.mdactx2016 i.vgactx2016 dep2016 adl2016 i.living2016 i.division c.m_HeatIndex_7d#c.vul_index"
regress_cmd_2 = "regress {} m_HeatIndex_7d c.m_HeatIndex_7d#c.vul_index pmono PNK_pct PBcell_pct PCD8_Plus_pct PCD4_Plus_pct PNCD8_Plus_pct age2016 i.female i.racethn eduy ihs_wealthf2016 i.smoke2016 i.drink2016 bmi2016 tractdis i.urban i.mar_cat2 i.psyche2016 i.stroke2016 i.hibpe2016 i.diabe2016 i.hearte2016 dep2016 ltactx2016 mdactx2016 vgactx2016 i.living2016 i.division adl2016".format(
    train_heat_dataset.outcome_original_name
)
stata.run(regress_cmd_2)
stata.run("vif")
out = stata.get_return()

# %%
import numpy as np

bottom10 = np.quantile(all_resilience_index, 0.1)
median = np.quantile(all_resilience_index, 0.5)
top10 = np.quantile(all_resilience_index, 0.9)

jump = abs(top10 - bottom10) / 2
margins_cmd = "margins, at(m_HeatIndex_7d =(30(10)100) vul_index=({:.2f}({:.2f}){:.2f}) ) atmeans".format(
    bottom10, 0.9 * abs(top10 - bottom10) / 2, top10
)
margins_plt_cmd = 'marginsplot, level(83.4) xtitle("Mean Heat Index Lagged 7days") ytitle("{}") title("") legend(order(1 "Least vulnerable (Bottom 10%)" 2 "Median" 3 "Most vulnerable (Top 10%)"))'.format(
    train_heat_dataset.outcome_original_name
)
stata.run(margins_cmd, quietly=True)
stata.run(margins_plt_cmd, quietly=True)
stata.run(
    "graph export {}, replace".format(
        os.path.join(figpath, "ml_vul_{}_interaction.png".format(model_name))
    ),
    quietly=True,
)  # Save the figure

# %%
corr_cmd = "pwcorr vul_index pred3_pe tractdis urban ltactx2016 living2016 m_HeatIndex_7d zPCPhenoAge_acc age2016 female eduy ihs_wealthf2016 dep2016 adl2016 stroke2016 hibpe2016 diabe2016 hearte2016"
stata.run(corr_cmd)

# %%
regress_cmd_2 = "regress {} m_HeatIndex_7d c.m_HeatIndex_7d##c.pred3_pe pmono PNK_pct PBcell_pct PCD8_Plus_pct PCD4_Plus_pct PNCD8_Plus_pct age2016 i.female i.racethn eduy ihs_wealthf2016 i.smoke2016 i.drink2016 bmi2016 tractdis i.urban i.mar_cat2 i.psyche2016 i.stroke2016 i.hibpe2016 i.diabe2016 i.hearte2016 dep2016 ltactx2016 mdactx2016 vgactx2016 i.living2016 i.division adl2016".format(
    train_heat_dataset.outcome_original_name
)
stata.run(regress_cmd_2)
stata.run("vif")

# %%
margins_cmd = "margins, at(m_HeatIndex_7d =(30(10)100) pred3_pe=({:.2f}({:.2f}){:.2f}) ) atmeans".format(
    40, 30, 100
)
margins_plt_cmd = 'marginsplot, level(83.4) xtitle("Census Heat Vulnerability") ytitle("{}") title("") legend(order(1 "Least vulnerable (Bottom 10%)" 2 "Median" 3 "Most vulnerable (Top 10%)"))'.format(
    train_heat_dataset.outcome_original_name
)
stata.run(margins_cmd, quietly=True)
stata.run(margins_plt_cmd, quietly=True)
stata.run(
    "graph export {}, replace".format(
        os.path.join(
            figpath,
            "census_vul_{}_interaction.png".format(
                train_heat_dataset.outcome_original_name
            ),
        )
    ),
    quietly=True,
)  # Save the figure

# %% [markdown]
# ## Model Analysis using Shapley values

# %%
import shap
import matplotlib.pyplot as plt

shap.initjs()

# %%
explainer = shap.DeepExplainer(model_index, all_interaction_vars_tensor)

# %%
shap_values = explainer.shap_values(
    all_interaction_vars_tensor[:1000], check_additivity=False
)

# %%
plt.clf()
shap.summary_plot(
    shap_values[:, :],
    all_interaction_vars_tensor[:1000, :].detach().cpu().numpy(),
    feature_names=interaction_predictors,
    show=False,
)
plt.savefig(
    os.path.join(figpath, "{}_shapley.png".format(model_name)),
    dpi=300,
    bbox_inches="tight",
)
