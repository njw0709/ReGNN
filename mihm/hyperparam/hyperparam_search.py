from .train import train_mihm, TRAIN_DEVICE
from .preprocess import preprocess
from mihm.data.trainutils import train_test_val_split
from ray import train


def train_wrapper(config):
    data_path = "/home/namj/projects/heat_air_epi/HeatResilience.dta"
    # read model and rename cols
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
        "dep2016",
        "adl2016",
        "living2016",
        "division",
    ]

    rename_dict = {
        "zPCPhenoAge_acc": "Pheno Age Accel.",
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
    outcome_col = "Pheno Age Accel."
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
        "psychiatric conditions",
        "stroke",
        "hypertension",
        "diabetes",
        "heart disease",
        "light activity",
        "moderate activity",
        "vigorous activity",
        "depressive symptoms",
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

    # print("Mean and std of standardized predictors: ", heat_dataset.mean_std_dict)
    train_idx, test_idx, val_idx = train_test_val_split(len(heat_dataset))
    train_heat_dataset = heat_dataset.get_subset(train_idx)
    test_heat_dataset = heat_dataset.get_subset(test_idx)
    val_heat_dataset = heat_dataset.get_subset(val_idx)
    all_interaction_predictors = heat_dataset.to_tensor(device=TRAIN_DEVICE)[
        "interaction_predictors"
    ]

    # use val dataset only for hyperparam search
    id = train.get_context().get_trial_name()
    model = train_mihm(
        train_heat_dataset,
        val_heat_dataset,
        hidden_layer_sizes=[config["layer1"], config["layer2"], 1],
        vae=True,
        svd=True,
        k_dims=config["k_dims"],
        epochs=500,
        batch_size=config["batch_size"],
        lr=config["lr"],
        weight_decay=0.1,
        eval=True,
        df_orig=df_orig,
        all_interaction_predictors=all_interaction_predictors,
        id=id,
        save_model=False,
        ray_tune=True,
        use_stata=False,
    )
