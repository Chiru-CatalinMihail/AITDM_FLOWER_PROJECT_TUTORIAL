"""Flower ClientApp: what every hospital / device / user runs on its own data."""

from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from aitdm_fl.task import balanced_class_weights, get_device, load_client_data
from aitdm_fl.task import evaluate as evaluate_fn
from aitdm_fl.task import model_from_test_set, num_classes_of
from aitdm_fl.task import train as train_fn

app = ClientApp()


def get_data_path(context: Context) -> Path:
    """Which partition does this client own?

    * Deployment: each SuperNode can point to its own file with
      --node-config "data-path='/abs/path/client_0.npz'"  (most realistic), or
      just give its partition-id: --node-config "partition-id=0 num-partitions=3"
    * Simulation: Flower sets partition-id automatically (0 ... num-supernodes-1).
    """
    if "data-path" in context.node_config:
        return Path(str(context.node_config["data-path"]))
    partition_id = int(context.node_config["partition-id"])
    return Path(str(context.run_config["partitions-dir"])) / f"client_{partition_id}.npz"


def is_malicious(context: Context) -> bool:
    """Is this client an attacker? (for robustness experiments, see docs/04)

    * Simulation: --run-config "malicious-clients='1,3'"  (partition ids)
    * Deployment: --node-config "data-path='...' malicious=true"
    """
    if context.node_config.get("malicious", False):
        return True
    ids = str(context.run_config["malicious-clients"]).replace(" ", "")
    pid = context.node_config.get("partition-id")
    return pid is not None and ids != "" and str(pid) in ids.split(",")


def poison(global_state: dict, local_state: dict, attack: str) -> dict:
    """Model poisoning: replace the honest update with a harmful one.

    * "sign-flip": push the model 5x in the OPPOSITE direction of learning
    * "noise":     send random Gaussian weights
    """
    poisoned = {}
    for name, w_local in local_state.items():
        w_global = global_state[name]
        if not torch.is_floating_point(w_local):
            poisoned[name] = w_local
        elif attack == "sign-flip":
            poisoned[name] = w_global - 5.0 * (w_local - w_global)
        elif attack == "noise":
            poisoned[name] = torch.randn_like(w_local)
        else:
            raise ValueError(f"Unknown attack '{attack}'. Use 'sign-flip' or 'noise'.")
    return poisoned


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Receive the global model, train it on local data, send it back."""
    partitions_dir = str(context.run_config["partitions-dir"])
    config = msg.content["config"]

    model = model_from_test_set(partitions_dir)
    global_state = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_state)

    batch_size = int(context.run_config["batch-size"])
    train_loader, _ = load_client_data(get_data_path(context), batch_size)

    class_weights = None
    if str(context.run_config["loss-weighting"]) == "balanced":  # rare classes count more
        train_set = train_loader.dataset
        num_classes = num_classes_of(model, tuple(train_set.x.shape[1:]))
        class_weights = balanced_class_weights(train_set.y.numpy(), num_classes)

    train_loss = train_fn(
        model,
        train_loader,
        epochs=int(context.run_config["local-epochs"]),
        lr=float(config["lr"]),
        device=get_device(),
        proximal_mu=float(config.get("proximal-mu", 0.0)),  # set by FedProx
        class_weights=class_weights,
    )

    if is_malicious(context):
        local_state = {k: v.cpu() for k, v in model.state_dict().items()}
        model.load_state_dict(poison(global_state, local_state, str(context.run_config["attack"])))

    metrics = MetricRecord({
        "train_loss": train_loss,
        # "num-examples" is used by the server to weight the aggregation
        "num-examples": len(train_loader.dataset),
        "unit-weight": 1,  # equal weights, used when differential privacy is on
    })
    content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the global model on this client's local validation data."""
    partitions_dir = str(context.run_config["partitions-dir"])
    model = model_from_test_set(partitions_dir)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    _, val_loader = load_client_data(get_data_path(context), int(context.run_config["batch-size"]))
    results = evaluate_fn(model, val_loader, get_device())

    metrics = MetricRecord({
        "eval_loss": results["loss"],
        "eval_acc": results["accuracy"],
        "eval_bal_acc": results["balanced_accuracy"],
        "num-examples": len(val_loader.dataset),
        "unit-weight": 1,
    })
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
