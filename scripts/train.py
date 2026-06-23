import argparse
import logging
import os
import time

import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

import languagemodel.functional as F
from languagemodel.checkpoint_manager import load_checkpoint, save_checkpoint
from languagemodel.data import data_loading
from languagemodel.model import Transformer
from languagemodel.optimizer import (
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configurações para treinamento de um modelo de linguagem do causal"
    )
    # --- Dados ---
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        help="Path para o dataset de treinamento",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        help="Path para o dataset de validação",
    )

    parser.add_argument(
        "--array_dtype", type=str, default="int32", help="Data type do array"
    )

    # --- Arquitetura do Modelo ---
    parser.add_argument(
        "--vocab_size", type=int, default=10_000, help="Tamanho do vocabulário"
    )
    parser.add_argument(
        "--context_length", type=int, default=256, help="Tamanho do contexto"
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Dimensão do modelo"
    )
    parser.add_argument(
        "--d_ff", type=int, default=1344, help="Dimensão da camada feed-forward"
    )
    parser.add_argument(
        "--num_heads", type=int, default=16, help="Número de cabeças de atenção"
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Número de camadas"
    )
    parser.add_argument(
        "--theta", type=float, default=10_000, help="Parâmetro theta"
    )

    # --- Treinamento ---
    parser.add_argument(
        "--n_steps", type=int, default=20_000, help="Número de épocas"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Tamanho do batch"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed de aleatoriedade"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo (ex: cuda, cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Tipo de dado do PyTorch (ex: float32, float16, bfloat16)",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="../outputs",
        help="Diretório para outputs",
    )

    parser.add_argument(
        "--torch_compile",
        type=argparse.BooleanOptionalAction,
        default=True,
        help="Compila o modelo para acelerar o treinamento",
    )
    parser.add_argument(
        "--gradient_clipping",
        type=argparse.BooleanOptionalAction,
        default=True,
        help="Faz o clipping dos gradientes",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Quantidade de steps para a acumulação dos gradientes",
    )

    parser.add_argument(
        "--l2_norm_max",
        type=float,
        default=1.0,
        help="Norma máxima que os gradientes podem ter",
    )

    # --- Otimizador ---
    parser.add_argument(
        "--betas",
        type=float,
        nargs=2,
        default=(0.90, 0.95),
        help="Parâmetros beta para o otimizador",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=5e-5,
        help="Learning rate mínimo",
    )
    parser.add_argument(
        "--lr_max",
        type=float,
        default=5e-4,
        help="Learning rate máximo",
    )
    parser.add_argument(
        "--t_w",
        type=int,
        default=1_000,
        help="Período aonde o warmup está ativado",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Decaimento de peso"
    )

    # --- Logging (WandB) ---
    parser.add_argument(
        "--wandb_log",
        type=argparse.BooleanOptionalAction,
        default=True,
        help="Usa o wandb como logger",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="language-model-from-scratch",
        help="Nome do projeto para o wandb",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=f"experiment-{time.strftime('%d%m%Y_%H%M%S')}",
        help="Nome para o experimento do wandb",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Intervalo de steps para se realizar um registro de log",
    )

    # --- Avaliação e Checkpointing ---
    parser.add_argument(
        "--val_interval",
        type=int,
        default=50,
        help="Intervalo de steps para se realizar uma avaliação",
    )

    parser.add_argument(
        "--val_steps",
        type=int,
        default=10,
        help="Quantidade de steps a cada avaliação",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path para o checkpointing do modelo",
    )

    args = parser.parse_args()

    if args.dtype == "float32":
        args.dtype = torch.float32
    elif args.dtype == "float16":
        args.dtype = torch.float16
    elif args.dtype == "bfloat16":
        args.dtype = torch.bfloat16
    else:
        raise ValueError(f"Dtype não suportado: {args.dtype}")

    return args


def train():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    best_val_loss = float("inf")
    args = parse_args()
    config = vars(args)

    scaler = (
        torch.amp.GradScaler(device=args.device)
        if args.dtype == torch.float16
        else None
    )

    if scaler is not None:
        logger.info("GrandScaler ativada para treinamento em fp16")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.wandb_log:
        wandb.init(
            project=args.wandb_project, name=args.wandb_name, config=config
        )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    model_args = {
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "theta": args.theta,
        "eps": 1e-6,
        "device": args.device,
        "dtype": args.dtype,
    }

    optimizer_args = {
        "betas": args.betas,
        "lr": args.lr_min,
        "weight_decay": args.weight_decay,
        "eps": 1e-8,
    }

    if args.ckpt_path is not None:
        logger.info(
            f"Treinando o modelo a partir do seguinte checkpointing: {args.ckpt_path}"
        )
        ckpt = load_checkpoint(
            src=args.ckpt_path, model_module=Transformer, optimizer_module=AdamW
        )

        model = ckpt["model"]
        optimizer = ckpt["optimizer"]
        starting_step = ckpt["step"]
    else:
        logger.info("Treinando um novo modelo de zero")

        model = Transformer(**model_args)

        param_dict = {
            pn: p for pn, p in model.named_parameters() if p.requires_grad
        }

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_group = [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(params=optim_group, **optimizer_args)
        optimizer.zero_grad(set_to_none=True)

        starting_step = 0

    if args.torch_compile:
        model = torch.compile(
            model, dynamic=False
        )  # dynamic=False pois o shape das entradas sempre serão o mesmo

    progress_bar = tqdm(
        range(starting_step, args.n_steps), desc="Treinando o modelo"
    )

    total_steps = args.n_steps - starting_step

    logger.info("COMEÇANDO O TREINAMENTO")
    logger.info(f"  Número de Steps = {total_steps}")

    total_loss = 0
    token_positions = (
        torch.arange(0, args.context_length, dtype=torch.long, pin_memory=True)
        .to(args.device, non_blocking=True)
        .unsqueeze(0)
    )  # A posição dos tokens é pré-calculada pois todas as entradas terão a mesma quantidade de tokens
    model.train()
    for step in progress_bar:
        # -----------------------------------------------------------
        # etapa de avaliação por val_steps

        if args.val_interval > 0 and (
            step % args.val_interval == 0 or step == total_steps
        ):
            with torch.no_grad():
                model.eval()
                val_losses = torch.empty(args.val_steps)
                for i in range(args.val_steps):
                    x_batch, y_batch = data_loading(
                        filename=args.val_dataset_path,
                        array_dtype=args.array_dtype,
                        batch_size=args.batch_size,
                        context_length=args.context_length,
                    )

                    x_batch, y_batch = (
                        x_batch.to(args.device, non_blocking=True),
                        y_batch.to(args.device, non_blocking=True),
                    )

                    x_batch_token_positions = token_positions.expand_as(x_batch)

                    with torch.amp.autocast(
                        device_type=args.device, dtype=args.dtype
                    ):
                        logits = model(
                            x=x_batch, token_positions=x_batch_token_positions
                        )
                        val_batch_loss = F.cross_entropy_loss(
                            logits=logits, target=y_batch
                        )

                    val_losses[i] = val_batch_loss.item()
                model.train()

                val_loss = val_losses.mean()
                val_perplexity = F.perplexity(val_losses)

                logger.info(
                    f"Step {step}: val/loss: {val_loss}, val/perplexity: {val_perplexity}"
                )

                if args.wandb_log:
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/perplexity": val_perplexity,
                        },
                        step=step,
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    save_checkpoint(
                        model=model._orig_mod
                        if hasattr(model, "_orig_mod")
                        else model,
                        optimizer=optimizer,
                        model_args=model_args,
                        optimizer_args=optimizer_args,
                        loss=best_val_loss,
                        step=step,
                        out=os.path.join(
                            args.out_dir,
                            f"ckpt_{time.strftime('%d%m%Y_%H')}.pt",
                        ),
                    )

        # -----------------------------------------------------------
        # single step de treinamento

        loss_accumulation = 0
        for _micro_step in range(args.gradient_accumulation_steps):
            x_batch, y_batch = data_loading(
                filename=args.train_dataset_path,
                array_dtype=args.array_dtype,
                batch_size=args.batch_size,
                context_length=args.context_length,
            )

            x_batch, y_batch = (
                x_batch.to(args.device, non_blocking=True),
                y_batch.to(args.device, non_blocking=True),
            )

            x_batch_token_positions = token_positions.expand_as(x_batch)

            with torch.amp.autocast(device_type=args.device, dtype=args.dtype):
                logits = model(
                    x=x_batch, token_positions=x_batch_token_positions
                )
                loss = F.cross_entropy_loss(logits=logits, target=y_batch)

            loss /= args.gradient_accumulation_steps
            loss_accumulation += loss.detach()

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        lr = get_lr_cosine_schedule(
            lr_min=args.lr_min,
            lr_max=args.lr_max,
            t=step,
            t_w=args.t_w,
            t_c=args.n_steps,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        total_loss += loss_accumulation
        progress_bar.set_postfix(loss=loss_accumulation.item(), lr=lr)

        if scaler is not None:
            if args.gradient_clipping:
                scaler.unscale_(optimizer)
                gradient_clipping(list(model.parameters()), args.l2_norm_max)

            scaler.step(optimizer)
            scaler.update()
        else:
            if args.gradient_clipping:
                gradient_clipping(list(model.parameters()), args.l2_norm_max)

            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == total_steps:
            train_loss = (
                total_loss / args.log_interval if step != 0 else total_loss
            )
            train_perplexity = F.perplexity(train_loss)
            total_loss = 0

            if args.wandb_log:
                wandb.log(
                    {
                        "train/loss": train_loss.item(),
                        "train/perplexity": train_perplexity.item(),
                        "lr": lr,
                    },
                    step=step,
                )

    save_checkpoint(
        model=model._orig_mod if hasattr(model, "_orig_mod") else model,
        optimizer=optimizer,
        model_args=model_args,
        optimizer_args=optimizer_args,
        loss=loss_accumulation,
        step=step,
        out=os.path.join(
            args.out_dir,
            f"final_ckpt_{time.strftime('%d%m%Y_%H%M%S')}.pt",
        ),
    )

    wandb.finish()


if __name__ == "__main__":
    train()
