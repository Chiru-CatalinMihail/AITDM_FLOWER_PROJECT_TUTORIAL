# 4. Aggregation strategies

After every round the server receives one model update per client and has to turn them into **one** global model.

That step is the *aggregation strategy*. It is the server's main lever
for the trust dimensions in the project:

| Goal | Problem | Strategies in this repo |
|---|---|---|
| **Accuracy under non-IID data** | Clients pull the model in different directions (*client drift*) | `fedprox`, `fedavgm`, `fedadam`, `fedyogi` |
| **Robustness** | One faulty or malicious client can wreck the average | `fedmedian`, `fedtrimmedavg`, `krum` |
| **Fairness** | The global model serves big or majority clients and neglects small ones | `qfedavg` |
| **Privacy** | The updates themselves can leak training data | DP wrapper (`dp-noise-multiplier > 0`) |

Every strategy is chosen at run time. **No code changes are needed:**

```bash
bash scripts/run_simulation.sh data/partitions/dirichlet "strategy='krum'"
```

The strategies come from `flwr.serverapp.strategy`. They are created in `build_strategy()` in
[`fl_app/aitdm_fl/server_app.py`](../fl_app/aitdm_fl/server_app.py), and their
hyperparameters live in [`fl_app/pyproject.toml`](../fl_app/pyproject.toml).

---

## Notation

$K$ clients participate. In round $t$:

* $w_t$ is the global model,
* client $k$ has $n_k$ samples, with $n = \sum_k n_k$ in total, and returns its locally trained model $w_t^k$,
* its *update* is $\Delta^k = w_t^k - w_t$.

---

## 1. FedAvg: the baseline

$$
w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \, w_t^k
$$

Weighted average of the client models, weighted by dataset size (the `num-examples` our ClientApp reports).

It is simple, communication-efficient (McMahan et al., 2017) and works well on IID data.

**Weak spots:** strong non-IID data (client drift), a single malicious client can move the average arbitrarily far ([see the experiment below](#experiment-1-attacker-out-of-5-clients)), and large clients dominate.

```bash
--run-config "strategy='fedavg'"
```

## 2. Strategies for non-IID data

### FedProx (`fedprox`)

Each client minimises its loss **plus** a penalty for moving away from the global model:

$$
\min_{w} \; L_k(w) + \frac{\mu}{2} \lVert w - w_t \rVert^2
$$

The server side is identical to FedAvg. Flower only sends `proximal-mu` to the clients, and **our `train()` in `task.py` adds the penalty** (look for `proximal_mu`). $\mu = 0$ is FedAvg; a larger $\mu$ means less drift but slower local progress. (Li et al., 2020).

```bash
--run-config "strategy='fedprox' proximal-mu=0.1"
```

### FedAvgM (`fedavgm`)

Applies **momentum on the server**: the averaged update is accumulated over rounds like SGD momentum, which smooths the zig-zag caused by heterogeneous clients. (Hsu et al., 2019)

```bash
--run-config "strategy='fedavgm' server-momentum=0.9"
```

### FedAdam / FedYogi (`fedadam`, `fedyogi`): adaptive federated optimisation

Treats the averaged update as a "pseudo-gradient" and applies **Adam** (or Yogi, a more stable Adam variant) on the server, with its own learning rate $\eta$ = `server-lr`.

This often helps on heterogeneous data, but adds a hyperparameter to tune. (Reddi et al., 2021 - ("FedOpt")).

```bash
--run-config "strategy='fedadam' server-lr=0.01"
```

## 3. Robust (Byzantine-tolerant) aggregation

The mean has a *breakdown point* of 0: one extreme value can move it anywhere. These strategies replace the mean with statistics that ignore outliers.

| Strategy | Rule | Tolerates | Paper |
|---|---|---|---|
| `fedmedian` | Coordinate-wise **median** of the client models | < 50 % attackers | Yin et al., 2018 |
| `fedtrimmedavg` | Per coordinate, drop the `trim-beta` largest and smallest values, average the rest | `trim-beta` fraction on each side | Yin et al., 2018 |
| `krum` | Pick the **single** client model closest to its $K - f - 2$ nearest neighbours | $f$ = `num-malicious`, needs $K > 2f + 2$ | Blanchard et al., 2017 |

Flower also has `MultiKrum` and `Bulyan` (Bulyan needs $K \geq 4f + 3$).

**The price:** robust rules throw information away. With honest but non-IID clients, an "outlier" may just be a hospital with rare patients, so robustness can hurt **fairness**.

This trade-off is one of the key points worth researching in your reports and experiments.

### Simulating attackers

The ClientApp can turn selected clients into attackers (`poison()` in `client_app.py`):

| `attack` | What the malicious client sends |
|---|---|
| `sign-flip` | $w_t - 5 \, \Delta^k$ (pushes the model 5× in the wrong direction) |
| `noise` | random Gaussian weights |

```bash
# Simulation: partition ids of the attackers
bash scripts/run_simulation.sh data/partitions/iid5 "malicious-clients='4' attack='sign-flip' strategy='krum'"

# Deployment: flag a single SuperNode
flower-supernode --insecure --superlink 127.0.0.1:8000 --port 9098 \
    --node-config "data-path='/abs/path/client_4.npz' malicious=true"
```

For *data* poisoning (wrong labels instead of wrong weights), use
`split_data.py --label-noise` ([tutorial 3](03_biased_data_splits.md)).

### Experiment: 1 attacker out of 5 clients

DermaMNIST, IID split over 5 clients (`--num-clients 5 --strategy iid --out data/partitions/iid5`),
5 rounds, client 4 is malicious:

| Attack | Strategy | Accuracy | Balanced acc. | Macro-F1 |
|---|---|---|---|---|
| none | FedAvg | 0.727 | 0.332 | 0.349 |
| sign-flip | FedAvg | **0.025** | 0.144 | 0.019 |
| sign-flip | FedMedian | 0.717 | 0.292 | 0.316 |
| sign-flip | FedTrimmedAvg ($\beta = 0.2$) | 0.717 | 0.312 | 0.344 |
| sign-flip | Krum ($f = 1$) | 0.707 | 0.284 | 0.300 |
| noise | FedAvg | 0.448 | 0.129 | 0.118 |
| noise | Krum ($f = 1$) | 0.691 | 0.438 | 0.426 |

One attacker out of five completely destroys FedAvg.

All three robust rules stay close to the attack-free baseline.

## 4. Fairness: q-FedAvg (`qfedavg`)

Clients with a **higher loss get a larger weight**, proportional to $F_k^q$, where $F_k$ is the loss of client $k$.

With $q = 0$ this is FedAvg; a larger $q$ pushes towards equal performance across clients (a more uniform accuracy distribution) at the cost of some average accuracy. (Li et al., 2020 - ("q-FFL")).

```bash
--run-config "strategy='qfedavg' qfedavg-q=1.0"
```

`qfedavg-client-lr` (default 1.0) sets the step size: Flower estimates the Lipschitz constant as $L = 1 / \text{lr}$, with $\text{lr}$ = `qfedavg-client-lr`. With a too-small value (e.g. our Adam learning rate 0.001) the global model barely moves.

To **measure** fairness, compare per-client metrics: evaluate the final model on every `client_<i>.npz` and report the worst client, or the standard deviation across clients, next to the mean.

The client-side evaluation metrics in `metrics.json` are averaged over clients, so they hide exactly this.

## 5. Privacy: central differential privacy

Setting `dp-noise-multiplier > 0` wraps **any** strategy in
`DifferentialPrivacyServerSideFixedClipping`:

1. each client update $\Delta^k$ is clipped to L2 norm $\lVert \Delta^k \rVert_2 \leq C$, with $C$ = `dp-clipping-norm`,
2. Gaussian noise with standard deviation $\frac{\sigma C}{K}$ is added to the aggregate, with $\sigma$ = `dp-noise-multiplier`,
3. the clients are averaged with **equal** weights instead of by dataset size, as in DP-FedAvg.
   Otherwise one large client could dominate the average and the privacy analysis would not hold.

```bash
--run-config "dp-noise-multiplier=1.0 dp-clipping-norm=1.0"
--run-config "strategy='fedmedian' dp-noise-multiplier=1.0"   # DP + robustness
```

**The privacy budget $\varepsilon$ is computed for you.** For FedAvg-style strategies (`fedavg`, `fedprox`), the server attaches Flower's Rényi-DP accountant (needs the `flwr[dp]` extra, which is already in `requirements.txt`). It prints the cumulative budget after every round and stores it in `metrics.json` under `"privacy"`:

```text
INFO :      Privacy spent: epsilon=12.301691 at delta=1e-05 after 5 releases
```

The guarantee is **client-level**: it protects whether a whole client (hospital) took part, not just a single record. $\delta$ is set with `dp-delta`.

Other strategies still get the noise, but without accounting; a `[DP]` message tells you so.

Same setup as [above](#experiment-1-attacker-out-of-5-clients) (5 IID clients, 5 rounds, $C = 1.0$, $\delta = 10^{-5}$):

| $\sigma$ (noise multiplier) | $\varepsilon$ after 5 rounds | Accuracy | Macro-F1 |
|---|---|---|---|
| 0 (no DP) | $\infty$ | 0.727 | 0.349 |
| 0.1 | 354.9 | 0.689 | 0.318 |
| 0.5 | 30.1 | 0.160 | 0.125 |
| 1.0 | 12.3 | 0.088 | 0.056 |
| 2.0 | 5.4 | 0.116 | 0.052 |


**The privacy–utility trade-off is brutal with few clients.** The only setting that keeps a useful model has $\varepsilon \approx 355$, which promises essentially nothing; $1 \leq \varepsilon \leq 10$ (the range usually called meaningful) destroys the model.

The noise is divided by $K = 5$ only. Real DP federated learning averages over hundreds or thousands of clients per round, and $\varepsilon$ also grows with every extra round.

**Ideas for the report:**
* Plot accuracy against $\varepsilon$
* Vary the clipping norm
* Explain *why* 5 clients cannot reach a useful $\varepsilon$
* Compare with **local DP** (DP-SGD on every client, e.g. with
[Opacus](https://opacus.ai/), which protects individual patients) or Flower's client-side variants (`DifferentialPrivacyClientSideFixedClipping`, adaptive clipping).  See the [Flower DP guide](https://flower.ai/docs/framework/explanation-differential-privacy.html).

---

## Choosing a strategy (cheat sheet)

| Situation | Try first |
|---|---|
| IID or mildly non-IID, honest clients | `fedavg` |
| Strong label skew / pathological split | `fedprox`, `fedavgm`, `fedadam` **(Tunable!)** |
| Suspected faulty or malicious clients, noisy labels | `fedtrimmedavg`, `fedmedian`, `krum` |
| Small clients are neglected | `qfedavg` |
| Must hide which clients took part / protect records | DP wrapper ($\varepsilon$ is reported); local DP-SGD for record-level privacy |

**Mandatory baselines**: Always compare against **FedAvg** and against a **centralised** model trained on the union of all clients' data. Run several seeds, and report balanced accuracy / macro-F1, not only accuracy.

---

## Writing your own strategy

Subclass an existing strategy and override the step you want to change.

This example logs every client's training loss before the normal FedAvg aggregation:

```python
from flwr.serverapp.strategy import FedAvg

class FedAvgWithLogging(FedAvg):
    """FedAvg that prints every client's training loss before aggregating."""

    def aggregate_train(self, server_round, replies):
        replies = list(replies)  # we iterate twice
        for msg in replies:
            if msg.has_content():
                loss = msg.content["metrics"]["train_loss"]
                print(f"round {server_round} | node {msg.metadata.src_node_id} | train_loss={loss:.3f}")
        return super().aggregate_train(server_round, replies)
```

Put it in `server_app.py` and return it from `build_strategy()`. Useful methods to override:

| Method | Runs | Typical use |
|---|---|---|
| `configure_train` | before each round | choose clients, send per-round config (lr schedule, ...) |
| `aggregate_train` | after training | your own aggregation rule, detect outliers, reputation scores |
| `configure_evaluate` / `aggregate_evaluate` | evaluation phase | custom client-side metrics |

Look at the source of the built-in strategies for inspiration
(`python -c "import inspect, flwr.serverapp.strategy as s; print(inspect.getsource(s.FedTrimmedAvg))"`).

## References

* [McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg), AISTATS 2017](https://arxiv.org/abs/1602.05629)
* [Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx), MLSys 2020](https://arxiv.org/abs/1812.06127)
* [Hsu et al., *Measuring the Effects of Non-Identical Data Distribution* (FedAvgM), 2019](https://arxiv.org/abs/1909.06335)
* [Reddi et al., *Adaptive Federated Optimization* (FedAdam/FedYogi), ICLR 2021](https://arxiv.org/abs/2003.00295)
* [Yin et al., *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates* (median, trimmed mean), ICML 2018](https://arxiv.org/abs/1803.01498)
* [Blanchard et al., *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent* (Krum), NeurIPS 2017](https://arxiv.org/abs/1703.02757)
* [Li et al., *Fair Resource Allocation in Federated Learning* (q-FedAvg), ICLR 2020](https://arxiv.org/abs/1905.10497)
* [McMahan et al., *Learning Differentially Private Recurrent Language Models* (DP-FedAvg), ICLR 2018](https://arxiv.org/abs/1710.06963)
