import pandas as pd
import numpy as np  # Added for dummy data generation
import os  # Added to handle paths
from regnn.macroutils import (
    MacroConfig,
    ModeratedRegressionConfig,
    generate_stata_command,
    train,
)
from regnn.data import DataFrameReadInConfig
from regnn.model import ReGNNConfig, SVDConfig
from regnn.train import (
    TrainingHyperParams,
    ProbeOptions,
    MSELossConfig,
    RegressionEvalOptions,
    EarlyStoppingConfig,
    WeightDecayConfig,
)
from regnn.macroutils import preprocess  # For data loading and preprocessing


def main():
    # Path to the data file, assuming script is run from project root
    data_path = "/home/namj/projects/ReGNN/notebooks/simulatedv3.dta"

    # Create output directory for models if it doesn't exist
    output_model_dir = "./output_models"
    os.makedirs(output_model_dir, exist_ok=True)

    # Define all column names based on the notebook (these will be moderators)
    cont_cols = [f"x{i+1}" for i in range(15)]
    bin_cols = [f"bin{i+1}" for i in range(10)]
    cat_cols = [f"cat{i+1}" for i in range(5)]
    moderator_cols = cont_cols + bin_cols + cat_cols

    read_cols = moderator_cols + ["T", "Y"]

    # 1. DataFrameReadInConfig
    read_config = DataFrameReadInConfig(
        data_path=data_path,
        read_cols=read_cols,  # This must be set before other column validations
        rename_dict={},
        binary_cols=bin_cols,
        categorical_cols=cat_cols,
        ordinal_cols=[],  # No ordinal columns in this case
        continuous_cols=cont_cols + ["T", "Y"],  # T and Y are continuous
        survey_weight_col=None,  # No survey weights
    )

    # 2. ModeratedRegressionConfig
    regression_config = ModeratedRegressionConfig(
        focal_predictor="T",
        outcome_col="Y",
        controlled_cols=[],  # controlled variables that are not moderators.
        moderators=moderator_cols,
        index_column_name="vul_index",
    )

    # 1. Preprocessing
    # The preprocess function now takes DataFrameReadInConfig and ModeratedRegressionConfig directly.
    all_dataset = preprocess(
        read_config=read_config, regression_config=regression_config
    )

    # 3. ReGNNConfig
    regnn_model_config = ReGNNConfig.create(
        num_moderators=len(
            all_dataset.config.moderators
        ),  # make sure to compute the number of moderators after processing
        num_controlled=len(regression_config.controlled_cols),
        layer_input_sizes=[64, 32],
        dropout=0.1,
        device="cuda",
        batch_norm=True,
        vae=False,
        output_mu_var=False,
        ensemble=False,
        svd=SVDConfig(enabled=False),
        n_ensemble=1,
        include_bias_focal_predictor=True,
        control_moderators=False,
        interaction_direction="positive",
    )

    # 4. TrainingHyperParams
    training_hp = TrainingHyperParams(
        epochs=50,
        batch_size=5000,
        lr=0.001,
        train_test_split_ratio=0.8,
        loss_options=MSELossConfig(
            reduction="mean",
            weight_decay=WeightDecayConfig(
                weight_decay_nn=0.01, weight_decay_regression=0.0
            ),
        ),
        device="cuda",
    )

    # 5. ProbeOptions
    probe_opts = ProbeOptions(
        save_dir=output_model_dir,
        file_id="01",
        save_model_epochs=10,
        model_save_name="sim_regnn_model_T_Y",
        save_intermediate_index=False,  # Default value
        regression_eval_opts=RegressionEvalOptions(
            evaluation_function="stata",
            evaluate=True,
            eval_epochs=5,
            regress_cmd=generate_stata_command(read_config, regression_config),
            index_column_name=regression_config.index_column_name,
        ),
        return_trajectory=True,
        get_testset_results=True,
        get_l2_lengths=False,
    )

    # Create MacroConfig
    macro_config = MacroConfig(
        read_config=read_config,
        regression=regression_config,
        model=regnn_model_config,
        training=training_hp,
        probe=probe_opts,
    )

    with open("/home/namj/projects/ReGNN/scripts/configs.json", "w") as f:
        f.write(macro_config.model_dump_json(indent=4))

    # Run training
    training_output = train(all_dataset, macro_config)

    if probe_opts.return_trajectory:
        model, train_trajectory, test_trajectory = training_output
        print("Training complete. Model and trajectories returned.")
        if train_trajectory:
            print("Last training snapshot:", train_trajectory[-1])
        if test_trajectory:
            print("Last test snapshot:", test_trajectory[-1])
    else:
        model = training_output
        print("Training complete. Model returned.")

    print("Script finished.")


if __name__ == "__main__":
    main()
