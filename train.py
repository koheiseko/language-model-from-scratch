import argparse
import logging
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from tokenizers import (
    Tokenizer,
)
from tqdm.auto import tqdm
from tqdm.contrib.logging import (
    logging_redirect_tqdm,
)

import languagemodel.functional as F
import wandb
from languagemodel.checkpoint_manager import (
    load_checkpoint,
    save_checkpoint,
)
from languagemodel.dataloader import (
    data_loading,
)
from languagemodel.generation import (
    generate,
)
from languagemodel.model import (
    Transformer,
)
from languagemodel.optimizer import (
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
)

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    # Reduz mensagens internas pouco importantes
    logging.getLogger("wandb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def positive_int(value: str) -> int:
    number = int(value)

    if number <= 0:
        raise argparse.ArgumentTypeError("O valor deve ser um inteiro positivo")

    return number


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configurações para treinamento de um modelo de linguagem do causal"
    )
    # --- Dados ---
    parser.add_argument(
        "--train_dataset_path",
        default="data/corpuscarolina/processed/train.bin",
        type=str,
        help="Path para o dataset de treinamento",
    )
    parser.add_argument(
        "--val_dataset_path",
        default="data/corpuscarolina/processed/val.bin",
        type=str,
        help="Path para o dataset de validação",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Diretório para outputs",
    )

    parser.add_argument(
        "--array_dtype",
        type=str,
        default="uint16",
        help="Data type do dataset",
    )

    # --- Arquitetura do Modelo ---
    parser.add_argument(
        "--vocab_size",
        type=positive_int,
        default=32_000,
        help="Tamanho do vocabulário",
    )
    parser.add_argument(
        "--context_length",
        type=positive_int,
        default=256,
        help="Tamanho do contexto",
    )
    parser.add_argument(
        "--d_model",
        type=positive_int,
        default=512,
        help="Dimensão do modelo",
    )
    parser.add_argument(
        "--d_ff",
        type=positive_int,
        default=1344,
        help="Dimensão da camada feed-forward",
    )
    parser.add_argument(
        "--num_heads",
        type=positive_int,
        default=8,
        help="Número de cabeças de atenção",
    )
    parser.add_argument(
        "--num_layers",
        type=positive_int,
        default=6,
        help="Número de camadas",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=10_000,
        help="Parâmetro theta do RoPE",
    )

    # --- Treinamento ---
    parser.add_argument(
        "--n_steps",
        type=positive_int,
        default=61_036,
        help="Número de steps",
    )
    parser.add_argument(
        "--batch_size",
        type=positive_int,
        default=64,
        help="Tamanho do batch",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo (cuda ou cpu)",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="float16",
        help="Precisão usada pelo autocast (float32, float16 ou bfloat16)",
    )

    parser.add_argument(
        "--torch_compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compila o modelo para acelerar o treinamento",
    )
    parser.add_argument(
        "--gradient_clipping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Faz o clipping dos gradientes",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=positive_int,
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
        "--min_learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate mínimo",
    )
    parser.add_argument(
        "--max_learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate máximo",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=1_500,
        help="Período aonde o warmup está ativado",
    )
    parser.add_argument(
        "--cosine_cycle_iters",
        type=int,
        default=61_035,
        help="Período aonde o cosine cycle está ativado",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Decaimento de peso",
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
        default=500,
        help="Intervalo de steps para se realizar uma avaliação",
    )

    parser.add_argument(
        "--val_steps",
        type=positive_int,
        default=10,
        help="Quantidade de steps feitos em cada avaliação",
    )

    parser.add_argument(
        "--sample_interval",
        type=int,
        default=250,
        help="Quantidade de steps para cada sample",
    )

    parser.add_argument(
        "--save_interval",
        type=int,
        default=10_000,
        help="Quantidade de steps para cada save",
    )

    parser.add_argument(
        "--start_step_save_best_model",
        type=int,
        default=5000,
        help="Dita a partir de qual step o melhor modelo vai ser continuamente salvo. "
        "Quando o valor é < 0 o salvamento por melhor modelo é desativado",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path para o checkpointing do modelo",
    )

    args = parser.parse_args()

    if args.amp_dtype == "float16":
        args.amp_dtype = torch.float16
    elif args.amp_dtype == "bfloat16":
        args.amp_dtype = torch.bfloat16
    else:
        raise ValueError(f"Dtype não suportado: {args.amp_dtype}")

    return args


def train():
    configure_logging()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    config = vars(args)

    if args.sample_interval > 0:
        tokenizer = Tokenizer.from_pretrained("nicholasKluge/TeenyTinyLlama-160m")

    amp_enabled = args.amp_dtype in (torch.float16, torch.bfloat16)
    use_grad_scaler = amp_enabled and args.amp_dtype == torch.float16

    if args.device == "cuda" and args.amp_dtype == torch.bfloat16:
        with torch.cuda.device(args.device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("A GPU selecionada não suporta BF16")

    scaler = torch.amp.GradScaler(
        args.device,
        enabled=use_grad_scaler,
    )

    if use_grad_scaler:
        logger.info("GrandScaler ativada para treinamento em fp16")

    def autocast_context():
        if not amp_enabled:
            return nullcontext()

        return torch.autocast(
            device_type=args.device,
            dtype=args.amp_dtype,
        )

    if args.wandb_log:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config,
        )

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    if args.ckpt_path is not None:
        logger.info(
            f"Treinando o modelo a partir do seguinte checkpointing: {args.ckpt_path}"
        )
        ckpt = load_checkpoint(
            src=args.ckpt_path,
            model_module=Transformer,
            optimizer_module=AdamW,
            device=args.device,
        )

        model_args = ckpt["model_args"]
        optimizer_args = ckpt["optimizer_args"]

        model = ckpt["model"]
        optimizer = ckpt["optimizer"]

        best_val_loss = ckpt["loss"]
        starting_step = ckpt["step"] + 1
    else:
        logger.info("Treinando um novo modelo de zero")

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
            "dtype": torch.float32,
        }

        optimizer_args = {
            "betas": args.betas,
            "lr": args.min_learning_rate,
            "weight_decay": args.weight_decay,
            "eps": 1e-8,
        }

        model = Transformer(**model_args)

        param_dict = {
            pn: p for pn, p in model.named_parameters() if p.requires_grad
        }

        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_group = [
            {
                "params": decay_params,
                "weight_decay": args.weight_decay,
            },
            {
                "params": nodecay_params,
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            params=optim_group,
            **optimizer_args,
        )
        optimizer.zero_grad(set_to_none=True)

        best_val_loss = float("inf")

        starting_step = 0

    total_parameters = sum([p.numel() for p in model.parameters()])

    logger.info("=" * 72)
    logger.info("INICIANDO TREINAMENTO")
    logger.info(
        "MODEL | parameters=%.2fM | layers=%d | d_model=%d | heads=%d",
        total_parameters / 1_000_000,
        model_args["num_layers"],
        model_args["d_model"],
        model_args["num_heads"],
    )
    logger.info(
        "DATA  | batch=%d | context=%d | grad_accum=%d | effective_batch=%d",
        args.batch_size,
        args.context_length,
        args.gradient_accumulation_steps,
        args.batch_size * args.gradient_accumulation_steps,
    )
    logger.info(
        "ENV   | device=%s | amp_dtype=%s | compiled=%s",
        args.device,
        str(args.amp_dtype).removeprefix("torch."),
        args.torch_compile,
    )
    logger.info(
        "STEPS | start=%d | end=%d | remaining=%d",
        starting_step,
        args.n_steps,
        args.n_steps - starting_step,
    )
    logger.info(
        "SAVE | save_interval=%d | start_step_save_best_model=%d",
        args.save_interval,
        args.start_step_save_best_model,
    )
    logger.info(
        "VAL | val_interval=%d | val_steps=%d",
        args.val_interval,
        args.val_steps,
    )
    logger.info("SAMPLE | sample_interval=%d", args.sample_interval)
    logger.info("=" * 72)

    if args.torch_compile:
        model = torch.compile(
            model,
            dynamic=False,
        )  # dynamic=False pois o shape das entradas sempre serão o mesmo

    progress_bar = tqdm(
        range(
            starting_step,
            args.n_steps,
        ),
        initial=starting_step,
        total=args.n_steps,
        desc="Treinando o modelo",
        unit="step",
        dynamic_ncols=True,
        leave=True,
    )

    train_dataset = np.memmap(
        filename=args.train_dataset_path,
        mode="r",
        dtype=args.array_dtype,
    )
    val_dataset = np.memmap(
        filename=args.val_dataset_path,
        mode="r",
        dtype=args.array_dtype,
    )

    total_loss = 0
    step_loss = 0
    model.train()

    with logging_redirect_tqdm():
        for step in progress_bar:
            # -----------------------------------------------------------
            # single step de treinamento

            loss_accumulation = 0
            for _micro_step in range(args.gradient_accumulation_steps):
                (
                    x_batch,
                    y_batch,
                ) = data_loading(
                    dataset=train_dataset,
                    array_dtype=args.array_dtype,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                )

                (
                    x_batch,
                    y_batch,
                ) = (
                    x_batch.to(
                        args.device,
                        non_blocking=True,
                    ),
                    y_batch.to(
                        args.device,
                        non_blocking=True,
                    ),
                )

                with autocast_context():
                    logits = model(x=x_batch)
                    loss = F.cross_entropy(
                        inputs=logits,
                        targets=y_batch,
                    )

                    loss = loss / args.gradient_accumulation_steps

                loss_accumulation += loss.detach()

                scaler.scale(loss).backward()

            lr = get_lr_cosine_schedule(
                it=step,
                min_learning_rate=args.min_learning_rate,
                max_learning_rate=args.max_learning_rate,
                warmup_iters=args.warmup_iters,
                cosine_cycle_iters=args.cosine_cycle_iters,
            )

            for group in optimizer.param_groups:
                group["lr"] = lr

            total_loss += loss_accumulation
            step_loss += 1

            progress_bar.set_postfix(
                {
                    "loss": f"{loss_accumulation.item():.4f}",
                    "lr": f"{lr:.2e}",
                },
                refresh=False,
            )

            if args.gradient_clipping:
                scaler.unscale_(optimizer)
                gradient_clipping(
                    list(model.parameters()),
                    args.l2_norm_max,
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # -----------------------------------------------------------
            # etapa de avaliação por val_steps

            last_step = step == args.n_steps - 1

            if last_step or (
                args.val_interval > 0 and step % args.val_interval == 0
            ):
                model.eval()
                with torch.no_grad():
                    total_val_loss = 0
                    for _ in range(args.val_steps):
                        (
                            x_batch,
                            y_batch,
                        ) = data_loading(
                            dataset=val_dataset,
                            array_dtype=args.array_dtype,
                            batch_size=args.batch_size,
                            context_length=args.context_length,
                        )

                        (
                            x_batch,
                            y_batch,
                        ) = (
                            x_batch.to(
                                args.device,
                                non_blocking=True,
                            ),
                            y_batch.to(
                                args.device,
                                non_blocking=True,
                            ),
                        )

                        with autocast_context():
                            logits = model(x=x_batch)
                            val_batch_loss = F.cross_entropy(
                                inputs=logits,
                                targets=y_batch,
                            )

                        total_val_loss += val_batch_loss.detach()

                    model.train()

                    val_loss = total_val_loss / args.val_steps
                    val_perplexity = val_loss.exp()

                    logger.info(
                        "VAL   | step=%6d | loss=%7.4f | ppl=%8.2f",
                        step,
                        val_loss.item(),
                        val_perplexity.item(),
                    )

                    if args.wandb_log:
                        wandb.log(
                            {
                                "val/loss": val_loss,
                                "val/perplexity": val_perplexity,
                            },
                            step=step,
                        )

                    if (
                        args.start_step_save_best_model >= 0
                        and step >= args.start_step_save_best_model
                    ):
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss

                            checkpoint_path = os.path.join(
                                args.out_dir,
                                f"ckpt_{time.strftime('%d%m%Y_%H%M%S')}.pt",
                            )

                            save_checkpoint(
                                model=model._orig_mod
                                if hasattr(
                                    model,
                                    "_orig_mod",
                                )
                                else model,
                                optimizer=optimizer,
                                model_args=model_args,
                                optimizer_args=optimizer_args,
                                loss=best_val_loss,
                                step=step,
                                out=checkpoint_path,
                            )

                            logger.info(
                                "CKPT  | step=%6d | best_loss=%.4f | path=%s",
                                step,
                                best_val_loss.item()
                                if isinstance(
                                    best_val_loss,
                                    torch.Tensor,
                                )
                                else best_val_loss,
                                checkpoint_path,
                            )

            # -----------------------------------------------------------
            # etapa de geração de samples

            if args.sample_interval > 0 and (
                last_step or (step + 1) % args.sample_interval == 0
            ):
                model.eval()

                prompts = [
                    "O homem é",
                    "A capital do Brasil é",
                    "Era uma vez",
                ]

                sample_outputs = []

                for prompt in prompts:
                    with torch.inference_mode(), autocast_context():
                        generated_tokens = generate(
                            model=model,
                            max_new_tokens=5,
                            eos_id=tokenizer.token_to_id("</s>"),
                            prompt_tokens=tokenizer.encode(prompt).ids,
                            device=args.device,
                            temperature=0.7,
                        )

                    response = tokenizer.decode(generated_tokens.tolist())

                    sample_outputs.append(f"  {prompt!r:<20} -> {response!r}")

                logger.info(
                    "SAMPLES | step=%d\n%s",
                    step,
                    "\n".join(sample_outputs),
                )

                model.train()

            # -----------------------------------------------------------
            # checkpoint a cada save_interval e no último step

            if last_step or (
                args.save_interval > 0 and step % args.save_interval == 0
            ):
                checkpoint_path = os.path.join(
                    args.out_dir,
                    f"ckpt_{time.strftime('%d%m%Y_%H%M%S')}.pt"
                    if not last_step
                    else f"ckpt_last_{time.strftime('%d%m%Y_%H%M%S')}.pt",
                )

                save_checkpoint(
                    model=model._orig_mod
                    if hasattr(
                        model,
                        "_orig_mod",
                    )
                    else model,
                    optimizer=optimizer,
                    model_args=model_args,
                    optimizer_args=optimizer_args,
                    loss=loss_accumulation,
                    step=step,
                    out=checkpoint_path,
                )

                logger.info(
                    "CKPT  | step=%6d | best_loss=%.4f | path=%s",
                    step,
                    best_val_loss.item()
                    if isinstance(
                        best_val_loss,
                        torch.Tensor,
                    )
                    else best_val_loss,
                    checkpoint_path,
                )

            # -----------------------------------------------------------
            # etapa de logging do w&b

            if args.log_interval > 0 and (
                last_step or step % args.log_interval == 0
            ):
                train_loss = total_loss / step_loss if step != 0 else total_loss
                train_perplexity = train_loss.exp()
                total_loss = 0
                step_loss = 0

                logger.info(
                    "TRAIN | step=%6d | loss=%7.4f | ppl=%8.2f | lr=%.3e",
                    step,
                    train_loss.item(),
                    train_perplexity.item(),
                    lr,
                )

                if args.wandb_log:
                    wandb.log(
                        {
                            "train/loss": train_loss.item(),
                            "train/perplexity": train_perplexity.item(),
                            "lr": lr,
                        },
                        step=step,
                    )

    wandb.finish()


if __name__ == "__main__":
    train()
