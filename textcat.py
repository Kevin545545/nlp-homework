#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path
import torch

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_1",
        type=Path,
        help="path to the gen model",
    )
    parser.add_argument(
        "model_2",
        type=Path,
        help="path to the spam model",
    )
    parser.add_argument(
        "prior_prob",
        type=float,
        help="prior probability of the first model",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


@torch.no_grad()
def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0

    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        # If the factor p(z | xy) = 0, then it will drive our cumulative file 
        # probability to 0 and our cumulative log_prob to -infinity.  In 
        # this case we can stop early, since the file probability will stay 
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break 

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)

    return log_prob


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
        
    log.info("Testing...")
    lm_1 = LanguageModel.load(args.model_1, device=args.device)
    lm_2 = LanguageModel.load(args.model_2, device=args.device)
    
    assert lm_1.vocab == lm_2.vocab, f"The vocalulary for {args.model_1} and {args.model_2} do not match !"
    if not (0.0 < args.prior_prob < 1.0):
        raise ValueError("prior_prob must be between 0 and 1")
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.

    log.info("Per-file log-probabilities:")
    model_1_count = 0
    model_2_count = 0
    error = 0
    for file in args.test_files:
        log_prob_model_1: float = file_log_prob(file, lm_1)
        log_prob_model_2: float = file_log_prob(file, lm_2)
        # Use bayes' theory the log version to compute the probability
        # of p(gen | foo.txt)
        if (log_prob_model_1 + math.log(args.prior_prob) > 
            log_prob_model_2 + math.log(1 - args.prior_prob)):
            print(f"{args.model_1}\t{file}")
            model_1_count += 1
            if (args.model_1.stem.split('_')[0] != file.parent.name):
                error += 1
        else:
            print(f"{args.model_2}\t{file}")
            model_2_count += 1
            if (args.model_2.stem.split('_')[0] != file.parent.name):
                error += 1
    
    total = model_1_count + model_2_count
    print(f"{model_1_count} files were more probably from {args.model_1} ({model_1_count * 100 / total:.2f}%)")
    print(f"{model_2_count} files were more probably from {args.model_2} ({model_2_count * 100/ total:.2f}%)")
    # print(f"Error rate:{error * 100 / total:.2f}%")



if __name__ == "__main__":
    main()

