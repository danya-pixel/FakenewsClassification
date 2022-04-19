
import numpy as np
from catboost import CatBoostClassifier
from model import evaluate
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support


def CB_objective(trial, task_type, train_pool, valid_pool, val):
    param = {
        "depth": trial.suggest_int("depth", 1, 14),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Poisson", "MVS"],
        ),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.005, 0.02, 0.05, 0.08, 0.1]),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 0.0001, 1.0, log = True),
        
        "random_seed":42,
        "used_ram_limit": "4gb",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    gbm = CatBoostClassifier(**param, task_type=task_type)

    gbm.fit(train_pool, eval_set=valid_pool, verbose=0, early_stopping_rounds=100)

    preds = gbm.predict(val.drop(['is_fake'], axis=1))
    accuracy = f1_score(val['is_fake'], preds)
    return accuracy


def val_report(device, model, train_dataloader, val_dataloader, loss_fn):
    """Get report for best model

    Args:
        device: Device to evaluation
        model: PyTorch model
        train_dataloader: PyTorch DataLoader
        val_dataloader: PyTorch DataLoader
        loss_fn: Loss function

    Returns:
        dev_correct, dev_predicted
    """
    model.to(device)
    model.eval()

    _, train_correct, train_predicted = evaluate(device,
                                                 model, train_dataloader, loss_fn)
    _, dev_correct, dev_predicted = evaluate(
        device, model, val_dataloader, loss_fn)

    print("Training performance:", precision_recall_fscore_support(
        train_correct, train_predicted, average="micro"))
    print("Development performance:", precision_recall_fscore_support(
        dev_correct, dev_predicted, average="micro"))

    np.mean(dev_predicted == dev_correct)

    print(classification_report(dev_correct, dev_predicted))
    return (dev_correct, dev_predicted)
