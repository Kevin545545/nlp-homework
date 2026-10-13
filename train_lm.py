#!/usr/bin/env python3
"""
Trains a smoothed trigram model over a given vocabulary.
Depending on the smoother, you need to supply hyperparameters and additional files.
"""
import argparse
import logging
import sys
from pathlib import Path
import torch

from probs import read_vocab, UniformLanguageModel, AddLambdaLanguageModel, \
    BackoffAddLambdaLanguageModel, EmbeddingLogLinearLanguageModel, ImprovedLogLinearLanguageModel

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

UNIFORM   = "uniform"
ADDLAMBDA = "add_lambda"
BACKOFF   = "add_lambda_backoff"
LOGLINEAR = "log_linear"
IMPROVED  = "log_linear_improved"
SMOOTHERS = (UNIFORM, ADDLAMBDA, BACKOFF, LOGLINEAR, IMPROVED)


def get_model_filename(args: argparse.Namespace) -> Path:
    prefix = f"corpus={args.train_file.name}~vocab={args.vocab_file.name}~smoother={args.smoother}"
    if args.smoother in (UNIFORM,):
        return Path(f"{prefix}.model")
    if args.smoother in (ADDLAMBDA, BACKOFF):
        return Path(f"{prefix}~lambda={args.lambda_}.model")
    elif args.smoother in (LOGLINEAR, IMPROVED):
        if args.smoother == IMPROVED:
            prefix += (f"~optimizer={args.optimizer}~lr={args.learning_rate}"
                       f"~batch={args.batch_size}~shuffle={args.shuffle}~oov={args.oov_feature}")
        return Path(f"{prefix}~lexicon={args.lexicon.name}~l2={args.l2_regularization}~epochs={args.epochs}.model")
    else:   
        raise NotImplementedError(f"Don't know how to construct filename for smoother {args.smoother}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Required arguments
    parser.add_argument(
        "vocab_file",
        type=Path,
        help="Vocabulary file",
    )
    parser.add_argument(
        "smoother",
        type=str,
        help=f"Smoothing method",
        choices=SMOOTHERS
    )
    parser.add_argument(
        "train_file",
        type=Path,
        help="Training corpus (as a single file)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the model (if not specified, will construct a filename with `get_model_filename`)"
    )

    # for add-lambda smoothers
    parser.add_argument(
        "--lambda",
        dest="lambda_",  # store in the lambda_ attribute because lambda is a Python keyword 
        type=float,
        default=0.0,
        help="Strength of smoothing for add_lambda and add_lambda_backoff smoothers (default 0)",
    )

    # # for log-linear smoothers
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=None,
        help="File of word embeddings (needed for our log-linear models)",
    )
    parser.add_argument(
        "--l2_regularization",
        type=float,
        default=0.0,
        help="Strength of L2 regularization in log-linear models (default 0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs for log-linear models (default 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )

    # Optional settings apply only to log_linear_improved.
    parser.add_argument("--batch_size", type=int, default=1, help="Improved model: mini-batch size (default 1)")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd", help="Improved model optimizer")
    parser.add_argument("--learning_rate", type=float, default=None, help="Improved model rate (default SGD 1e-5, Adam 1e-3)")
    parser.add_argument("--shuffle", action="store_true", help="Improved model: shuffle examples each epoch")
    parser.add_argument("--dev_files", type=Path, nargs="+", default=(), help="Improved model: held-out dev files for early stopping")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience in epochs")
    parser.add_argument("--min_delta", type=float, default=0.0001, help="Minimum dev cross-entropy improvement")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed")
    parser.add_argument("--oov_feature", action="store_true", help="Improved model: add J.1 learned OOV bias")
    parser.add_argument("--history_csv", type=Path, help="Improved model: save epoch objectives as CSV")

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    if args.smoother != IMPROVED and (args.batch_size != 1 or args.optimizer != "sgd"
            or args.learning_rate is not None or args.shuffle or args.dev_files
            or args.patience != 3 or args.min_delta != 0.0001 or args.seed != 0
            or args.oov_feature or args.history_csv is not None):
        parser.error("Optional improvement settings require log_linear_improved")
    return args

def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device
    # (e.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration).
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)
    
    # Now construct a language model, giving the appropriate arguments to the constructor.
    # Some checking of the arguments is done first.

    vocab = read_vocab(args.vocab_file)
    parser = argparse.ArgumentParser(description=__doc__)
    if args.smoother == UNIFORM:
        log.warning(f"Uniform model will ignore the training file {args.train_file}")
        lm = UniformLanguageModel(vocab)
    elif args.smoother == ADDLAMBDA:
        if args.lambda_ == 0.0:
            log.warning("You're training an add-0 (unsmoothed) model")
        lm = AddLambdaLanguageModel(vocab, args.lambda_)
    elif args.smoother == BACKOFF:
        lm = BackoffAddLambdaLanguageModel(vocab, args.lambda_)
    elif args.smoother == LOGLINEAR:
        if args.lexicon is None:
            parser.error(f"{args.smoother} requires a lexicon")
        lm = EmbeddingLogLinearLanguageModel(vocab, args.lexicon, args.l2_regularization, args.epochs)
    elif args.smoother == IMPROVED:
        if args.lexicon is None:
            parser.error(f"{args.smoother} requires a lexicon")
        lm = ImprovedLogLinearLanguageModel(
            vocab, args.lexicon, args.l2_regularization, args.epochs,
            batch_size=args.batch_size, optimizer_name=args.optimizer,
            learning_rate=args.learning_rate, shuffle=args.shuffle,
            dev_files=args.dev_files, patience=args.patience,
            min_delta=args.min_delta, seed=args.seed,
            use_oov_feature=args.oov_feature)
    else:
        log.critical(f"Initialization code for smoother {args.smoother} is missing")
        sys.exit(1)

    log.info("Training...")
    lm.train(args.train_file)

    # Save the model to a file.
    
    if args.output is None:
        model_path = get_model_filename(args)
    else:
        model_path = args.output
    lm.save(model_path)
    if args.history_csv is not None:
        import csv
        args.history_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.history_csv.open("w", newline="") as f:
            fields = ["epoch", "F", "beta"]
            if args.dev_files:
                fields.append("dev_bits_per_token")
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(lm.training_history)
        log.info(f"Saved training history to {args.history_csv}")

if __name__ == "__main__":
    main()
