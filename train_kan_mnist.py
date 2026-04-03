# train_kan_mnist.py
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

from mnist_dataloader import MnistDataloader
from kan import KAN, LogActivationLayer, PosNegActivationLayer


# ============================================================
#                   CONFIGURATION
# ============================================================
class Config:
    input_path = "./input_data"        # MNIST folder
    batch_size = 128
    lr = 1e-3
    hidden_sizes = (128, 64)
    max_epochs = 15
    patience = 3
    debug = False
    normalize = True
    flatten = True


cfg = Config()


# ============================================================
#                   DATA LOADING
# ============================================================
def load_mnist(cfg):
    """Loads MNIST and returns dataloader + test tensors."""
    from os.path import join

    train_img = join(cfg.input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte")
    train_lbl = join(cfg.input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte")
    test_img  = join(cfg.input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
    test_lbl  = join(cfg.input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

    loader = MnistDataloader(train_img, train_lbl, test_img, test_lbl)
    (x_train, y_train), (x_test, y_test) = loader.load_data()

    x_train = torch.tensor(np.array(x_train), dtype=torch.float32)
    x_test  = torch.tensor(np.array(x_test),  dtype=torch.float32)

    if cfg.normalize:
        x_train /= 255.0
        x_test  /= 255.0

    if cfg.flatten:
        x_train = x_train.view(x_train.size(0), -1)
        x_test  = x_test.view(x_test.size(0), -1)

    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    y_test  = torch.tensor(np.array(y_test),  dtype=torch.long)

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    return train_loader, (x_test, y_test)


# ============================================================
#                       METRICS
# ============================================================
def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item() * 100, preds


# ============================================================
#                   TRAINING UTILITIES
# ============================================================
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def validate(model, x_test, y_test, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        val_loss = loss_fn(logits, y_test).item()
        val_acc, preds = accuracy(logits, y_test)
    return val_loss, val_acc, preds


# ============================================================
#                   PLOTTING
# ============================================================
def plot_curves(train_losses, val_losses, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss curves")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(all_labels, all_preds, num_classes=10):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()



# ============================================================
#                   TRAINING WRAPPER
# ============================================================
def train_mnist(cfg, show_confusion=True):
    train_loader, (x_test, y_test) = load_mnist(cfg)

    model = KAN(
        in_count=28 * 28,
        out_count=10,
        hidden_layer_sizes=list(cfg.hidden_sizes),
        debug=cfg.debug,
        activation_cls=PosNegActivationLayer,
    )

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_accs = []

    best_loss = float("inf")
    patience_count = 0
    best_state = None

    for epoch in range(cfg.max_epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_acc, preds = validate(model, x_test, y_test, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{cfg.max_epochs} "
              f"| Train Loss {train_loss:.4f} "
              f"| Val Loss {val_loss:.4f} "
              f"| Val Acc {val_acc:.2f}% "
              f"| Time {time.time() - t0:.2f}s")

        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            patience_count = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print("Early stopping.")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation & confusion matrix
    final_val_loss, final_val_acc, final_preds = validate(model, x_test, y_test, loss_fn)
    print(f"\nFinal Val Loss: {final_val_loss:.4f}, Final Val Acc: {final_val_acc:.2f}%")

    if show_confusion:
        all_preds = final_preds.cpu().numpy()
        all_labels = y_test.cpu().numpy()
        plot_confusion_matrix(all_labels, all_preds, num_classes=10)


    plot_curves(train_losses, val_losses, val_accs)

    return model

def train_mnist_with_stats(cfg, show_confusion=True, sample_index_for_trace: int = 0):
    """
    Same as train_mnist, but:
      - after each epoch, prints layer-wise min/max input/output from stats
      - after each epoch, calls export_single_sample_trace for one test sample
    """
    train_loader, (x_test, y_test) = load_mnist(cfg)

    model = KAN(
        in_count=28 * 28,
        out_count=10,
        hidden_layer_sizes=list(cfg.hidden_sizes),
        debug=cfg.debug,
        activation_cls=PosNegActivationLayer,
    )

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_accs = []

    best_loss = float("inf")
    patience_count = 0
    best_state = None

    for epoch in range(cfg.max_epochs):
        t0 = time.time()

        # -------- Train one epoch --------
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)

        # -------- Standard validation (loss + accuracy) --------
        val_loss, val_acc, preds = validate(model, x_test, y_test, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}/{cfg.max_epochs} "
            f"| Train Loss {train_loss:.4f} "
            f"| Val Loss {val_loss:.4f} "
            f"| Val Acc {val_acc:.2f}% "
            f"| Time {time.time() - t0:.2f}s"
        )

        # -------- Stats pass over test set --------
        model.eval()
        with torch.no_grad():
            _, stats = model(x_test, track_stats=True)

        print("  Layer stats (min/max input & output):")

        def to_scalar(v):
            if v is None:
                return None
            if isinstance(v, torch.Tensor):
                return float(v.item())
            return float(v)

        for s in stats:
            layer_idx = s["layer"]
            min_in = to_scalar(s.get("min_input"))
            max_in = to_scalar(s.get("max_input"))
            min_out = to_scalar(s.get("min_output"))
            max_out = to_scalar(s.get("max_output"))

            print(
                f"    Layer {layer_idx}: "
                f"in [{min_in}, {max_in}], "
                f"out [{min_out}, {max_out}]"
            )

        # -------- Export single-sample trace for this epoch --------
        sample = x_test[sample_index_for_trace : sample_index_for_trace + 1]
        trace_filename = f"activation_trace_epoch_{epoch+1}_sample{sample_index_for_trace}.mat"
        model.export_single_sample_trace(sample, filename=trace_filename)
        print(f"  -> Exported single-sample trace to {trace_filename}")

        # -------- Early stopping logic --------
        if val_loss < best_loss:
            best_loss = val_loss
            patience_count = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print("Early stopping.")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation & confusion matrix
    final_val_loss, final_val_acc, final_preds = validate(model, x_test, y_test, loss_fn)
    print(f"\nFinal Val Loss: {final_val_loss:.4f}, Final Val Acc: {final_val_acc:.2f}%")

    if show_confusion:
        all_preds = final_preds.cpu().numpy()
        all_labels = y_test.cpu().numpy()
        plot_confusion_matrix(all_labels, all_preds, num_classes=10)

    plot_curves(train_losses, val_losses, val_accs)

    return model

## THIS WAS ADDED AND IS STILL NOT WORKING PROPERLY!
def train_function_approx(cfg):
    model = KAN(1, 1, [3, 3], debug=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    def target_fn(x):
        x_abs = (x * 1.5).abs()
        return x + x_abs - x_abs.trunc()

    batch_size = 256
    for step in range(10000):
        x = torch.FloatTensor(batch_size, 1).uniform_(0, 4).double()
        y_true = target_fn(x)

        optimizer.zero_grad()
        y_pred, stats = model(x, return_all = False, track_stats = True)  # calling model(x) runs the forward() method of the KAN class
        #print("y_pred shape:", y_pred.shape)
        #print("y_true shape:", y_true.shape)
        
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
            print(f"y output stats: min={y_pred.min().item()}, max={y_pred.max().item()}, mean={y_pred.mean().item()}")
            print(f"target stats: min={y_true.min().item()}, max={y_true.max().item()}, mean={y_true.mean().item()}")


    plot_response(model, target_fn)

    max_input_overall = max(s['max_input'] for s in stats)
    max_output_overall = max(s['max_output'] for s in stats)
    min_input_overall = min(s['min_input'] for s in stats)
    min_output_overall = min(s['min_output'] for s in stats)

    print(f"Overall max input across all layers: {max_input_overall:.4f}")
    print(f"Overall max output across all layers: {max_output_overall:.4f}")

    print(f"Overall min input across all layers: {min_input_overall:.4f}")
    print(f"Overall min output across all layers: {min_output_overall:.4f}")

    #print(model.get_learnable_params())
    
    #model.plot_neuron_responses(input_domain=(0, 4), resolution=100)
    #model.plot_activation_shapes(input_domain=(0, 8), resolution=100)
    #model.plot_activation_c_comparison(layer_idx=0, neuron_idx=0, c_values=[1.0, 3.0, 5.0, 10.0], input_domain=(-4, 4), resolution=100)
 