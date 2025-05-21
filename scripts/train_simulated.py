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
)


def main():
    # Path to the data file, assuming script is run from project root
    data_path = "/home/namj/projects/ReGNN/notebooks/simulatedv3.dta"

    # Create output directory for models if it doesn't exist
    output_model_dir = "./output_models"
    os.makedirs(output_model_dir, exist_ok=True)

    df = pd.read_stata(data_path)
    print(df.columns)

    # Define all column names based on the notebook (these will be moderators)
    cont_cols = [f"x{i+1}" for i in range(15)]
    bin_cols = [f"bin{i+1}" for i in range(10)]
    cat_cols = [f"cat{i+1}" for i in range(5)]
    moderator_cols = cont_cols + bin_cols + cat_cols
    print(moderator_cols)

    read_cols = moderator_cols + ["T", "Y"]

    # 1. DataFrameReadInConfig
    read_config = DataFrameReadInConfig(
        df=df,
        read_cols=read_cols,
        df_dtypes={col: "float" for col in cont_cols}
        | {col: "int" for col in bin_cols}
        | {col: "category" for col in cat_cols}
        | {"T": "float"}  # Assuming T is binary/integer
        | {"Y": "float"},  # Assuming Y is continuous
        # survey_weight_col="weights", # Example if you have survey weights
    )

    # # 2. ModeratedRegressionConfig
    # regression_config = ModeratedRegressionConfig(
    #     focal_predictor="T",
    #     outcome_col="Y",
    #     controlled_cols=[],  # All other relevant columns are moderators
    #     moderators=moderator_cols,
    # )

    # # 3. ReGNNConfig
    # nn_config = NNConfig(
    #     input_dim=len(regression_config.moderators),  # Number of moderators
    #     hidden_dims=[64, 32],
    #     output_dim=1,  # Typically 1 for index prediction
    #     activation="relu",
    #     dropout_rate=0.1,
    #     svd=SVDConfig(enabled=False),
    # )
    # regnn_model_config = ReGNNConfig(
    #     nn_config=nn_config,
    #     freeze_focal_interactions=False,
    #     n_ensemble=1,
    # )

    # # 4. TrainingHyperParams
    # training_hp = TrainingHyperParams(
    #     epochs=50,
    #     batch_size=256,
    #     lr=0.001,
    #     train_test_split_ratio=0.8,
    #     loss_options=MSELossConfig(reduction="mean"),
    #     device="cpu",
    #     get_testset_results=True,
    # )

    # # 5. ProbeOptions
    # probe_opts = ProbeOptions(
    #     return_trajectory=True,
    #     save_model=True,
    #     save_model_epochs=10,
    #     model_save_name="sim_regnn_model_T_Y",
    #     save_dir=output_model_dir,
    #     regression_eval_opts=RegressionEvalOptions(
    #         evaluate=True,
    #         eval_epochs=5,
    #         show_focal_col_summary=True,
    #         show_control_col_summary=False,  # No separate controls
    #     ),
    #     get_testset_results=True,
    # )

    # # Create MacroConfig
    # macro_config = MacroConfig(
    #     read_config=read_config,
    #     regression=regression_config,
    #     model=regnn_model_config,
    #     training=training_hp,
    #     probe=probe_opts,
    # )

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
