from mihm.hyperparam.hyperparam_search import train_wrapper
from mihm.hyperparam.stopper import NaNStopper
from ray import tune
import torch

if __name__ == "__main__":
    config = {
        "layer1": tune.randint(3, 60),
        "layer2": tune.randint(2, 30),
        "k_dims": tune.randint(8, 25),
        "batch_size": tune.choice([32, 64, 128, 256, 512, 800]),
        "lr": tune.loguniform(1e-5, 1e-1),
        "weight_decay": tune.loguniform(1e-5, 1e-1),
    }

    scheduler = tune.schedulers.ASHAScheduler(
        metric="composite_metric",
        mode="min",
        max_t=300,
        grace_period=1,
        reduction_factor=2,
    )
    stopper = NaNStopper(
        metric="composite_metric"
    )  # Assuming your metric is named "loss"

    result = tune.run(
        train_wrapper,
        resources_per_trial={"cpu": 1, "gpu": 0.03},
        config=config,
        num_samples=800,
        scheduler=scheduler,
        stop=stopper,
    )

    best_trial = result.get_best_trial("composite_metric", "min", "last")
    print("Best trial config: ", best_trial.config)
    print("Best trial final validation loss: ", best_trial.last_result["test_MSE"])
    print(
        "Best trial final interaction p value: ",
        best_trial.last_result["interaction_pval"],
    )
    print(
        "Best trial final VIF: heat: {}, interaction: {}".format(
            best_trial.last_result["vif_heat"], best_trial.last_result["vif_inter"]
        )
    )
    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()
    best_model_state_dict = best_checkpoint_data["mihm_state_dict"]
    torch.save(best_model_state_dict, "best_model.pt")
