# hyperparam_search.py
import itertools
import torch
import random
from train_kan_mnist import train_mnist, Config, load_mnist
from kan import LogActivationLayer, HybridActivationLayer

def run_single_experiment(params):
    cfg = Config()
    cfg.hidden_sizes = params["hidden_sizes"]
    cfg.lr = params["lr"]
    cfg.batch_size = params["batch_size"]
    cfg.patience = params["patience"]
    cfg.debug = False

    cfg.activation_cls = params["activation"]

    print("\n=====================================")
    print("Running experiment with params:")
    for k,v in params.items():
        print(f"  {k}: {v}")
    print("=====================================")

    model = train_mnist(cfg, show_confusion=False)
    
    # Evaluate final accuracy
    _, (x_test, y_test) = load_mnist(cfg)
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        preds = logits.argmax(dim=1)
    acc = (preds == y_test).float().mean().item() * 100
    return acc, model


def grid_search(param_grid, max_trials=None):
    keys, values = zip(*param_grid.items())
    combos = list(itertools.product(*values))

    if max_trials is not None:
        combos = random.sample(combos, max_trials)

    best_acc = -1
    best_params = None
    best_model = None

    for combo in combos:
        params = dict(zip(keys, combo))
        acc, model = run_single_experiment(params)

        print(f"Accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_model = model

    print("\n=====================================")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print("Best Params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("=====================================")

    return best_model, best_params, best_acc
