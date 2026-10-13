"""
Training loop shared by every experiment: Adam first, then L-BFGS.

Why two optimizers (the paper does the same, following Raissi et al.):

  Adam      takes cheap, noisy-tolerant steps. Good at getting from a random
            start into the right region, robust to a badly scaled loss.
            But it creeps slowly once the loss is small.

  L-BFGS    a quasi-Newton method: it builds an estimate of the loss
            curvature from recent gradients and takes much larger, better
            aimed steps. Very effective at the end, but it can stall or
            diverge from a random start. So it runs second.

L-BFGS estimates curvature from DIFFERENCES of gradients between steps. Near
the optimum those differences are tiny, and in float32 they drown in rounding
error, so the line search fails and L-BFGS stops early. Your GPU benchmark
showed float64 costs the same as float32 for these small networks, so the
whole project runs in float64 (see pinn.utils.setup_torch).

How a problem plugs in:

    def loss_fn():            # returns each loss term separately
        return {"pde": ..., "ic": ..., "bc": ...}

    def eval_fn():            # optional: accuracy against a reference
        return {"rel_l2": ...}

    trainer = Trainer(params, loss_fn, TrainConfig(...), eval_fn=eval_fn)
    trainer.train()

The total loss is sum(weight[name] * term[name]); a missing weight means 1.
Every term is logged separately, so we can see WHICH part of the physics the
network is failing to satisfy, not just the total.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torch

LossFn = Callable[[], dict[str, torch.Tensor]]
EvalFn = Callable[[], dict[str, float]]
Hook = Callable[["Trainer"], None]


@dataclass
class TrainConfig:
    """All knobs of a training run. Saved with the results."""

    adam_iters: int = 10_000
    adam_lr: float = 1e-3
    lr_decay_rate: float | None = None  # e.g. 0.9 -> multiply lr by 0.9 ...
    lr_decay_every: int = 1_000         # ... every this many Adam steps
    lbfgs_iters: int = 5_000            # upper bound; L-BFGS may stop earlier
    lbfgs_chunk: int = 50               # L-BFGS iterations between log points
    lbfgs_history: int = 50             # past steps L-BFGS remembers
    log_every: int = 100                # Adam steps between log points
    eval_every: int = 1_000             # iterations between eval_fn calls
    print_every: int = 1_000            # iterations between console lines
    weights: dict[str, float] = field(default_factory=dict)
    verbose: bool = True


class Trainer:
    """Adam -> L-BFGS trainer with per-term logging.

    Args:
        params:  everything to optimize: network weights, plus any unknown
                 physical parameters (e.g. damping c and stiffness k in the
                 inverse problem). They are all just tensors to the optimizer.
        loss_fn: returns {term name: scalar loss tensor}.
        config:  TrainConfig.
        eval_fn: optional, returns {metric name: float}. Called every
                 config.eval_every iterations and at the end of each phase.
                 Runs without gradient tracking for speed; a metric that
                 needs derivatives must wrap that part in torch.enable_grad().
        hooks:   optional callables run after every ADAM step, e.g. to
                 resample collocation points (adaptive sampling). Not run during L-BFGS:
                 changing the points would invalidate the curvature history
                 L-BFGS has built up.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        loss_fn: LossFn,
        config: TrainConfig | None = None,
        eval_fn: EvalFn | None = None,
        hooks: Iterable[Hook] = (),
    ) -> None:
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("no trainable parameters were given")
        self.loss_fn = loss_fn
        self.config = config or TrainConfig()
        self.eval_fn = eval_fn
        self.hooks = list(hooks)

        self.iteration = 0              # counts Adam steps + L-BFGS iterations
        self.log: list[dict] = []       # loss terms over training
        self.evals: list[dict] = []     # eval_fn results over training
        self.timings: dict[str, float] = {}
        self._t0 = 0.0
        self._last_eval_iter = -1
        self._last_print_iter = 0
        self._printed_iter = -1

    # ------------------------------------------------------------------ loss
    def compute_loss(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Weighted total loss and the individual terms."""
        terms = self.loss_fn()
        w = self.config.weights
        total = sum(w.get(name, 1.0) * value for name, value in terms.items())
        return total, terms

    # --------------------------------------------------------------- logging
    def _record(self, phase: str, total: torch.Tensor, terms: dict) -> None:
        row = {
            "phase": phase,
            "iter": self.iteration,
            "time": time.perf_counter() - self._t0,
            "loss": total.item(),  # .item() also waits for the GPU to finish
        }
        row.update({name: value.item() for name, value in terms.items()})
        self.log.append(row)
        if not math.isfinite(row["loss"]):
            raise FloatingPointError(
                f"loss became {row['loss']} at iteration {self.iteration} ({phase}). "
                "Try a smaller learning rate or check the loss terms for errors."
            )
        cfg = self.config
        if cfg.verbose and self.iteration - self._last_print_iter >= cfg.print_every:
            self._last_print_iter = self.iteration
            self._print(row)

    def _print(self, row: dict) -> None:
        self._printed_iter = row["iter"]
        parts = [f"{k} {v:.2e}" for k, v in row.items()
                 if k not in ("phase", "iter", "time", "loss")]
        line = (f"  {row['phase']:<6} it {row['iter']:>7}  loss {row['loss']:.3e}"
                f"  [{', '.join(parts)}]  {row['time']:7.1f}s")
        if self.evals and self.evals[-1]["iter"] == row["iter"]:
            metrics = {k: v for k, v in self.evals[-1].items() if k != "iter"}
            line += "  | " + ", ".join(f"{k} {v:.3e}" for k, v in metrics.items())
        print(line)

    def _maybe_eval(self, force: bool = False) -> None:
        if self.eval_fn is None or self.iteration == self._last_eval_iter:
            return
        due = self.iteration - max(self._last_eval_iter, 0) >= self.config.eval_every
        if force or due:
            with torch.no_grad():
                metrics = self.eval_fn()
            self.evals.append({"iter": self.iteration, **{k: float(v) for k, v in metrics.items()}})
            self._last_eval_iter = self.iteration

    # ------------------------------------------------------------------ Adam
    def run_adam(self) -> None:
        cfg = self.config
        if cfg.adam_iters <= 0:
            return
        opt = torch.optim.Adam(self.params, lr=cfg.adam_lr)
        sched = None
        if cfg.lr_decay_rate is not None:
            sched = torch.optim.lr_scheduler.StepLR(
                opt, step_size=cfg.lr_decay_every, gamma=cfg.lr_decay_rate)

        start = time.perf_counter()
        for step in range(1, cfg.adam_iters + 1):
            opt.zero_grad(set_to_none=True)
            total, terms = self.compute_loss()
            total.backward()
            opt.step()
            if sched is not None:
                sched.step()
            self.iteration += 1
            for hook in self.hooks:
                hook(self)
            last = step == cfg.adam_iters
            if step == 1 or step % cfg.log_every == 0 or last:
                self._maybe_eval(force=last)
                self._record("adam", total.detach(), {k: v.detach() for k, v in terms.items()})
        self.timings["adam_seconds"] = time.perf_counter() - start

    # ---------------------------------------------------------------- L-BFGS
    def run_lbfgs(self) -> None:
        cfg = self.config
        if cfg.lbfgs_iters <= 0:
            return
        opt = torch.optim.LBFGS(
            self.params,
            lr=1.0,
            max_iter=cfg.lbfgs_chunk,
            max_eval=int(cfg.lbfgs_chunk * 1.25),
            history_size=cfg.lbfgs_history,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        def closure():
            # L-BFGS calls this several times per iteration (line search),
            # so it must recompute the loss from scratch every time.
            opt.zero_grad(set_to_none=True)
            total, _ = self.compute_loss()
            total.backward()
            return total

        start = time.perf_counter()
        done = 0
        self.timings["lbfgs_stopped_early"] = 0.0
        self.timings["lbfgs_diverged"] = 0.0
        while done < cfg.lbfgs_iters:
            # Safety copy: L-BFGS occasionally blows up. If it does, we roll
            # back instead of throwing away everything Adam achieved.
            backup = [p.detach().clone() for p in self.params]
            # L-BFGS keeps its iteration count in the state of the first parameter.
            before = opt.state[self.params[0]].get("n_iter", 0)
            opt.step(closure)
            ran = opt.state[self.params[0]].get("n_iter", 0) - before

            # Recompute the loss at the parameters L-BFGS actually accepted
            # (the last closure call may have been a rejected trial point).
            total, terms = self.compute_loss()
            if not torch.isfinite(total):
                with torch.no_grad():
                    for p, b in zip(self.params, backup):
                        p.copy_(b)
                self.timings["lbfgs_diverged"] = 1.0
                if cfg.verbose:
                    print("  L-BFGS diverged; restored the last good parameters and stopped.")
                total, terms = self.compute_loss()
                self._record("lbfgs", total.detach(), {k: v.detach() for k, v in terms.items()})
                break

            done += ran
            self.iteration += ran
            self._maybe_eval()
            self._record("lbfgs", total.detach(), {k: v.detach() for k, v in terms.items()})
            if ran < cfg.lbfgs_chunk:
                # Fewer iterations than asked means L-BFGS hit a tolerance
                # (converged) or its line search could not improve further.
                self.timings["lbfgs_stopped_early"] = 1.0
                break
        self._maybe_eval(force=True)
        self.timings["lbfgs_seconds"] = time.perf_counter() - start
        self.timings["lbfgs_iterations"] = float(done)

    # ------------------------------------------------------------ main entry
    def train(self) -> "Trainer":
        self._t0 = time.perf_counter()
        if self.config.verbose:
            n = sum(p.numel() for p in self.params)
            print(f"  training {n} parameters: {self.config.adam_iters} Adam "
                  f"+ up to {self.config.lbfgs_iters} L-BFGS iterations")
        self.run_adam()
        self.run_lbfgs()
        self.timings["total_seconds"] = time.perf_counter() - self._t0
        if self.config.verbose and self.log:
            if self._printed_iter != self.log[-1]["iter"]:  # final line, once
                self._print(self.log[-1])
            print(f"  done in {self.timings['total_seconds']:.1f}s")
        return self

    # --------------------------------------------------------------- results
    def summary(self) -> dict:
        """Final losses, final metrics and timings, for metrics.json."""
        out = {"iterations": self.iteration, **self.timings}
        if self.log:
            out["final_losses"] = {k: v for k, v in self.log[-1].items()
                                   if k not in ("phase", "iter", "time")}
        if self.evals:
            out["final_metrics"] = {k: v for k, v in self.evals[-1].items() if k != "iter"}
        return out

    def save(self, run_dir: str | Path) -> None:
        """Write config, full loss log and eval history to run_dir/history.json."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump({"config": asdict(self.config), "log": self.log,
                       "evals": self.evals, "summary": self.summary()}, f, indent=1)
