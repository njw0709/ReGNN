import os
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
    WeightDecayConfig,
    OptimizerConfig,
    LearningRateConfig,
)
from regnn.macroutils import preprocess  # For data loading and preprocessing
from regnn.probe import Trajectory

# Import new Probe Schedule Configs and Enums
from regnn.probe import (
    SaveCheckpointProbeScheduleConfig,
    SaveIntermediateIndexProbeScheduleConfig,
    RegressionEvalProbeScheduleConfig,
    GetObjectiveProbeScheduleConfig,
    PValEarlyStoppingProbeScheduleConfig,
    FrequencyType,
    DataSource,
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
        epochs=100,
        batch_size=1000,
        train_test_split_ratio=0.7,
        optimizer_config=OptimizerConfig(
            weight_decay=WeightDecayConfig(
                weight_decay_nn=0.02, weight_decay_regression=0.00
            ),
            lr=LearningRateConfig(lr_nn=0.001, lr_regression=0.02),
        ),
        loss_options=MSELossConfig(
            reduction="mean",
        ),
        device="cuda",
    )

    # --- Define Probe Schedules ---
    probe_schedules = []

    # 1. Save Checkpoints (matches old save_model=True, save_model_epochs=5)
    probe_schedules.append(
        SaveCheckpointProbeScheduleConfig(
            probe_type="save_checkpoint",
            frequency_type=FrequencyType.EPOCH,
            frequency_value=5,
            data_sources=[DataSource.ALL],
            save_dir=output_model_dir,
            model_save_name="sim_T_Y",
            file_id="01",
        )
    )
    # Also save at the end of training
    probe_schedules.append(
        SaveCheckpointProbeScheduleConfig(
            probe_type="save_checkpoint",
            frequency_type=FrequencyType.POST_TRAINING,
            frequency_value=1,
            data_sources=[DataSource.ALL],
            save_dir=output_model_dir,
            model_save_name="sim_T_Y_final",
            file_id="01",
        )
    )

    # 2. Save Intermediate Index (matches old save_intermediate_index=True)
    # Assuming this runs with the same frequency as checkpointing, or post-training
    probe_schedules.append(
        SaveIntermediateIndexProbeScheduleConfig(
            frequency_type=FrequencyType.EPOCH,
            frequency_value=5,
            data_sources=[DataSource.TEST],
            save_dir=output_model_dir,
            model_save_name="sim_T_Y",
            file_id="01_test_index",
        )
    )
    probe_schedules.append(
        SaveIntermediateIndexProbeScheduleConfig(
            frequency_type=FrequencyType.POST_TRAINING,
            frequency_value=1,
            data_sources=[DataSource.TEST],
            save_dir=output_model_dir,
            model_save_name="sim_T_Y_final",
            file_id="01_test_index",
        )
    )

    # 3. Regression Evaluation (matches old regression_eval_opts)
    stata_cmd = generate_stata_command(read_config, regression_config)
    # Run on TRAIN and TEST data every 5 epochs
    probe_schedules.append(
        RegressionEvalProbeScheduleConfig(
            frequency_type=FrequencyType.EPOCH,
            frequency_value=5,
            data_sources=[DataSource.TRAIN, DataSource.TEST],
            regress_cmd=stata_cmd,
            index_column_name=regression_config.index_column_name,
            evaluation_function="stata",
        )
    )
    # Also run post-training
    probe_schedules.append(
        RegressionEvalProbeScheduleConfig(
            frequency_type=FrequencyType.POST_TRAINING,
            frequency_value=1,
            data_sources=[DataSource.TRAIN, DataSource.TEST, DataSource.ALL],
            regress_cmd=stata_cmd,
            index_column_name=regression_config.index_column_name,
            evaluation_function="stata",
        )
    )

    # 4. Get Objective (Loss) (matches old get_testset_results=True implies test loss)
    # Typically want train and test loss every epoch
    probe_schedules.append(
        GetObjectiveProbeScheduleConfig(
            frequency_type=FrequencyType.EPOCH,
            frequency_value=1,
            data_sources=[DataSource.TRAIN, DataSource.TEST],
        )
    )

    # 5. P-Value Early Stopping (Example - if you want to enable it)
    probe_schedules.append(
        PValEarlyStoppingProbeScheduleConfig(
            frequency_type=FrequencyType.EPOCH,
            frequency_value=1,
            criterion=0.05,
            patience=3,
            n_sequential_epochs_to_pass=2,
            data_sources_to_monitor=[DataSource.TRAIN, DataSource.TEST],
        )
    )

    # 5. ProbeOptions (now just holds the schedules and return_trajectory flag)
    probe_opts = ProbeOptions(schedules=probe_schedules, return_trajectory=True)

    # Create MacroConfig
    macro_config = MacroConfig(
        read_config=read_config,
        regression=regression_config,
        model=regnn_model_config,
        training=training_hp,
        probe=probe_opts,
    )

    with open(os.path.join(output_model_dir, "macro_config_simulated.json"), "w") as f:
        f.write(macro_config.model_dump_json(indent=4))

    # Run training
    training_output = train(all_dataset, macro_config)

    if probe_opts.return_trajectory:
        model, trajectory = training_output

        # Save the entire trajectory
        trajectory_save_path = os.path.join(
            output_model_dir, "full_trajectory_simulated.json"
        )
        with open(trajectory_save_path, "w") as f:
            f.write(trajectory.model_dump_json(indent=4))
        print(
            f"Training complete. Model returned. Full trajectory saved to {trajectory_save_path}"
        )

        if trajectory.data:
            print("Last snapshot in trajectory:", trajectory.data[-1].model_dump())
        else:
            print("No snapshots recorded in the trajectory.")
    else:
        model = training_output
        print(
            "Training complete. Model returned. No trajectory saved as return_trajectory=False."
        )

    print("Script finished.")


if __name__ == "__main__":
    main()
