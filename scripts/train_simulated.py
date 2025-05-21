import pandas as pd
import numpy as np  # Added for dummy data generation
import os  # Added to handle paths
from regnn.macroutils import MacroConfig, train, ModeratedRegressionConfig
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
        index_col_name="vul_index",
        data_readin_config=read_config,
    )

    # 3. ReGNNConfig
    regnn_model_config = ReGNNConfig.create(
        num_moderators=len(regression_config.moderators),
        num_controlled=len(regression_config.controlled_cols),
        layer_input_sizes=[64, 32],
        dropout=0.1,
        device="cpu",
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
        batch_size=256,
        lr=0.001,
        train_test_split_ratio=0.8,
        loss_options=MSELossConfig(
            reduction="mean",
            weight_decay=WeightDecayConfig(
                weight_decay_nn=0.01, weight_decay_regression=0.0
            ),
        ),
        device="cpu",
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
            regress_cmd=regression_config.generate_stata_command(),
        ),
        return_trajectory=True,
        get_testset_results=True,
        get_l2_lengths=True,
    )

    # Create MacroConfig
    macro_config = MacroConfig(
        read_config=read_config,
        regression=regression_config,
        model=regnn_model_config,
        training=training_hp,
        probe=probe_opts,
    )

    print(macro_config)
    print(macro_config.model_dump_json())

    # # Run training
    # training_output = train(macro_config)

    # if probe_opts.return_trajectory:
    #     model, train_trajectory, test_trajectory = training_output
    #     print("Training complete. Model and trajectories returned.")
    #     if train_trajectory:
    #         print("Last training snapshot:", train_trajectory[-1])
    #     if test_trajectory:
    #         print("Last test snapshot:", test_trajectory[-1])
    # else:
    #     model = training_output
    #     print("Training complete. Model returned.")

    # print("Script finished.")


if __name__ == "__main__":
    main()
