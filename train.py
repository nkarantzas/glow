import argparse
import os
import json
import shutil
import random
from itertools import islice

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss

from datasets import get_MNIST
from model import Glow

def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    print("Using seed: {seed}".format(seed=seed))

def check_dataset(num_classes):
    train_dataset, test_dataset = get_MNIST(num_classes)
    return train_dataset, test_dataset

def compute_loss(nll, reduction="mean"):
    if reduction == "mean": 
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none": 
        losses = {"nll": nll}
        
    losses["total_loss"] = losses["nll"]
    return losses

def compute_loss_y(nll, y_logits, y_weight, y, reduction="mean"):
    if reduction == "mean": 
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none": 
        losses = {"nll": nll}

    loss_classes = F.cross_entropy(y_logits, torch.argmax(y, dim=1), reduction=reduction)
    losses["loss_classes"] = loss_classes
    losses["total_loss"] = losses["nll"] + y_weight * loss_classes
    return losses

def main(
    img_shape,
    num_classes,
    batch_size,
    epochs,
    seed,
    hidden_channels,
    K,
    L,
    actnorm_scale,
    flow_permutation,
    flow_coupling,
    LU_decomposed,
    learn_top,
    y_condition,
    y_weight,
    max_grad_clip,
    max_grad_norm,
    lr,
    n_workers,
    n_init_batches,
    output_dir,
    warmup,
):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_manual_seed(seed)
    train_dataset, test_dataset = check_dataset(num_classes)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        drop_last=True,
    )
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
    )

    model = Glow(
        img_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        num_classes,
        learn_top,
        y_condition,
    )

    model = model.to(device)
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=5e-5)

    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup)  # noqa
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch
        x = x.to(device)
        
        if y_condition:
            y = y.to(device)
            z, nll, y_logits = model(x, y)
            losses = compute_loss_y(nll, y_logits, y_weight, y)
            
        else:
            z, nll, y_logits = model(x, None)
            losses = compute_loss(nll)
        
        losses["total_loss"].backward()
        
        if max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_clip)
            
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
        optimizer.step()
        return losses

    def eval_step(engine, batch):
        model.eval()
        x, y = batch
        x = x.to(device)
        
        with torch.no_grad():
            if y_condition:
                y = y.to(device)
                z, nll, y_logits = model(x, y)
                losses = compute_loss_y(nll, y_logits, y_weight, y, reduction="none")
                
            else:
                z, nll, y_logits = model(x, None)
                losses = compute_loss(nll, reduction="none")
                
        return losses

    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(
        output_dir, 
        "glow", 
        n_saved=2, 
        require_empty=False
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, 
        checkpoint_handler, 
        {"model": model, "optimizer": optimizer}
    )

    monitoring_metrics = ["total_loss"]
    RunningAverage(output_transform=lambda x: x["total_loss"]).attach(trainer, "total_loss")
    evaluator = Engine(eval_step)

    # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
    Loss(
        lambda x, y: torch.mean(x),
        output_transform=lambda x: (
            x["total_loss"],
            torch.empty(x["total_loss"].shape[0]),
        ),
    ).attach(evaluator, "total_loss")

    if y_condition:
        monitoring_metrics.extend(["nll"])
        RunningAverage(output_transform=lambda x: x["nll"]).attach(trainer, "nll")

        # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
        Loss(
            lambda x, y: torch.mean(x),
            output_transform=lambda x: (x["nll"], torch.empty(x["nll"].shape[0])),
        ).attach(evaluator, "nll")

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    @trainer.on(Events.STARTED)
    def init(engine):
        model.train()

        init_batches = []
        init_targets = []

        with torch.no_grad():
            for batch, target in islice(train_loader, None, n_init_batches):
                init_batches.append(batch)
                init_targets.append(target)

            init_batches = torch.cat(init_batches).to(device)

            assert init_batches.shape[0] == n_init_batches * batch_size

            if y_condition:
                init_targets = torch.cat(init_targets).to(device)
            else:
                init_targets = None

            model(init_batches, init_targets)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(engine):
        evaluator.run(test_loader)

        scheduler.step()
        metrics = evaluator.state.metrics

        losses = ", ".join([f"{key}: {value:.2f}" for key, value in metrics.items()])

        print(f"Validation Results - Epoch: {engine.state.epoch} {losses}")

    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message(f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]")
        timer.reset()

    trainer.run(train_loader, epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_shape", type=tuple, default=(32, 32, 3), help="Size of input images")
    parser.add_argument("--num_classes", type=int, default=2, choices=[2, 10], help="whether to restrict over 0 & 1 classes")
    
    parser.add_argument("--hidden_channels", type=int, default=512, help="Number of hidden channels")
    parser.add_argument("--K", type=int, default=8, help="Number of layers per block")
    parser.add_argument("--L", type=int, default=4, help="Number of blocks")
    parser.add_argument("--actnorm_scale", type=float, default=1.0, help="Act norm scale")
    parser.add_argument("--flow_permutation", type=str, default="invconv", choices=["invconv", "shuffle", "reverse"], help="Type of flow permutation")
    parser.add_argument("--flow_coupling", type=str, default="affine", choices=["additive", "affine"], help="Type of flow coupling")
    parser.add_argument("--no_LU_decomposed", action="store_false", dest="LU_decomposed", help="Train with LU decomposed 1x1 convs")
    parser.add_argument("--no_learn_top", action="store_false", help="Do not train top layer (prior)", dest="learn_top")
    parser.add_argument("--y_condition", type=bool, default=True, help="Train using class condition")
    parser.add_argument("--y_weight", type=float, default=0.01, help="Weight for class condition loss")
    parser.add_argument("--max_grad_clip", type=float, default=0, help="Max gradient value (clip above - for off)")
    parser.add_argument("--max_grad_norm", type=float, default=0, help="Max norm of gradient (clip above - 0 for off)")
    
    parser.add_argument("--n_workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size used during training")
    parser.add_argument("--epochs", type=int, default=250, help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--warmup", type=float, default=5, help="Use this number of epochs to warmup learning rate linearly from zero to learning rate")
    parser.add_argument("--n_init_batches", type=int, default=8, help="Number of batches to use for Act Norm initialisation")
    
    parser.add_argument("--output_dir", default="output/", help="Directory to output logs and model checkpoints")
    parser.add_argument("--fresh", action="store_true", help="Remove output directory before starting")
    parser.add_argument("--seed", type=int, default=0, help="manual seed")
    args = parser.parse_args()

    try: os.makedirs(args.output_dir)
    except FileExistsError:
        if args.fresh:
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        if (not os.path.isdir(args.output_dir)) or (len(os.listdir(args.output_dir)) > 0):
            raise FileExistsError("Either pass the --fresh flag or a non-existing --output_dir")

    kwargs = vars(args)
    del kwargs["fresh"]

    with open(os.path.join(args.output_dir, "hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    main(**kwargs)
