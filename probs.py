#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

# Refactored by Arya McCarthy <xkcd@jhu.edu> because inheritance is cool
# and so is separating business logic from other stuff.  (9/19/2019)

# Patched by Arya McCarthy <arya@jhu.edu> to fix a counting issue that
# evidently was known pre-2016 but then stopped being handled?

# Further refactoring by Jason Eisner <jason@cs.jhu.edu> 
# and Brian Lu <zlu39@jhu.edu>.  (9/26/2021)

from __future__ import annotations

import logging
import math
import pickle
import sys

from pathlib import Path

import torch
from torch import nn
from torch import optim
from jaxtyping import Float
from typeguard import typechecked
from typing import Counter, Collection
from collections import Counter

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

##### TYPE DEFINITIONS (USED FOR TYPE ANNOTATIONS)
from typing import Iterable, List, Optional, Set, Tuple, Union

Wordtype = str  # if you decide to integerize the word types, then change this to int
Vocab    = Collection[Wordtype]   # and change this to Integerizer[str]
Zerogram = Tuple[()]
Unigram  = Tuple[Wordtype]
Bigram   = Tuple[Wordtype, Wordtype]
Trigram  = Tuple[Wordtype, Wordtype, Wordtype]
Ngram    = Union[Zerogram, Unigram, Bigram, Trigram]
Vector   = List[float]
TorchScalar = Float[torch.Tensor, ""] # a torch.Tensor with no dimensions, i.e., a scalar


##### CONSTANTS
BOS: Wordtype = "BOS"  # special word type for context at Beginning Of Sequence
EOS: Wordtype = "EOS"  # special word type for observed token at End Of Sequence
OOV: Wordtype = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL: Wordtype = "OOL"  # special word type whose embedding is used for OOV and all other Out-Of-Lexicon words


##### UTILITY FUNCTIONS FOR CORPUS TOKENIZATION

def read_tokens(file: Path, vocab: Optional[Vocab] = None) -> Iterable[Wordtype]:
    """Iterator over the tokens in file.  Tokens are whitespace-delimited.
    If vocab is given, then tokens that are not in vocab are replaced with OOV."""

    # OPTIONAL SPEEDUP: You may want to modify this to integerize the
    # tokens, using integerizer.py as in previous homeworks.
    # In that case, redefine `Wordtype` from `str` to `int`.

    # PYTHON NOTE: This function uses `yield` to return the tokens one at
    # a time, rather than constructing the whole sequence and using
    # `return` to return it.
    #
    # A function that uses `yield` is called a "generator."  As with other
    # iterators, it computes new values only as needed.  The sequence is
    # never fully constructed as an single object in memory.
    #
    # You can iterate over the yielded sequence, for example, like this:
    #      for token in read_tokens(my_file, vocab):
    #          process(token)
    # Whenever the `for` loop needs another token, read_tokens magically picks up 
    # where it left off and continues running until the next `yield` statement.

    with open(file) as f:
        for line in f:
            for token in line.split():
                if vocab is None or token in vocab:
                    yield token
                else:
                    yield OOV  # replace this out-of-vocabulary word with OOV
            yield EOS  # Every line in the file implicitly ends with EOS.


def num_tokens(file: Path) -> int:
    """Give the number of tokens in file, including EOS."""
    return sum(1 for _ in read_tokens(file))


def read_trigrams(file: Path, vocab: Vocab) -> Iterable[Trigram]:
    """Iterator over the trigrams in file.  Each triple (x,y,z) is a token z
    (possibly EOS) with a left context (x,y)."""
    x, y = BOS, BOS
    for z in read_tokens(file, vocab):
        yield (x, y, z)
        if z == EOS:
            x, y = BOS, BOS  # reset for the next sequence in the file (if any)
        else:
            x, y = y, z  # shift over by one position.


def draw_trigrams_forever(file: Path, 
                          vocab: Vocab, 
                          randomize: bool = False) -> Iterable[Trigram]:
    """Infinite iterator over trigrams drawn from file.  We iterate over
    all the trigrams, then do it again ad infinitum.  This is useful for 
    SGD training.  
    
    If randomize is True, then randomize the order of the trigrams each time.  
    This is more in the spirit of SGD, but the randomness makes the code harder to debug, 
    and forces us to keep all the trigrams in memory at once.
    """
    trigrams = read_trigrams(file, vocab)
    if not randomize:
        import itertools
        return itertools.cycle(trigrams)  # repeat forever
    else:
        import random
        pool = tuple(trigrams)   
        while True:
            for trigram in random.sample(pool, len(pool)):
                yield trigram

##### READ IN A VOCABULARY (e.g., from a file created by build_vocab.py)

def read_vocab(vocab_file: Path) -> Vocab:
    vocab: Vocab = set()
    with open(vocab_file, "rt") as f:
        for line in f:
            word = line.strip()
            vocab.add(word)
    log.info(f"Read vocab of size {len(vocab)} from {vocab_file}")
    # Convert from an unordered Set to an ordered List.  This ensures that iterating
    # over the vocab will always hit the words in the same order, so that you can 
    # safely store a list or tensor of embeddings in that order, for example.
    return sorted(vocab)   
    # Alternatively, you could choose to represent a Vocab as an Integerizer (see above).
    # Then you won't need to sort, since Integerizers already have a stable iteration order.

##### LANGUAGE MODEL PARENT CLASS

class LanguageModel:

    def __init__(self, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.progress = 0   # To print progress.

        self.event_count:   Counter[Ngram] = Counter()  # numerator c(...) function.
        self.context_count: Counter[Ngram] = Counter()  # denominator c(...) function.
        # In this program, the argument to the counter should be an Ngram, 
        # which is always a tuple of Wordtypes, never a single Wordtype:
        # Zerogram: context_count[()]
        # Bigram:   context_count[(x,y)]   or equivalently context_count[x,y]
        # Unigram:  context_count[(y,)]    or equivalently context_count[y,]
        # but not:  context_count[(y)]     or equivalently context_count[y]  
        #             which incorrectly looks up a Wordtype instead of a 1-tuple

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    # We need to collect two kinds of n-gram counts.
    # To compute p(z | xy) for a trigram xyz, we need c(xy) for the 
    # denominator and c(yz) for the backed-off numerator.  Both of these 
    # look like bigram counts ... but they are not quite the same thing!
    #
    # For a sentence of length N, we are iterating over trigrams xyz where
    # the position of z falls in 1 ... N+1 (so z can be EOS but not BOS),
    # and therefore
    # the position of y falls in 0 ... N   (so y can be BOS but not EOS).
    # 
    # When we write c(yz), we are counting *events z* with *context* y:
    #         c(yz) = |{i in [1, N]: w[i-1] w[i] = yz}|
    # We keep these "event counts" in `event_count` and use them in the numerator.
    # Notice that z=BOS is not possible (BOS is not a possible event).
    # 
    # When we write c(xy), we are counting *all events* with *context* xy:
    #         c(xy) = |{i in [1, N]: w[i-2] w[i-1] = xy}|
    # We keep these "context counts" in `context_count` and use them in the denominator.
    # Notice that y=EOS is not possible (EOS cannot appear in the context).
    #
    # In short, c(xy) and c(yz) count the training bigrams slightly differently.  
    # Likewise, c(y) and c(z) count the training unigrams slightly differently.
    #
    # Note: For bigrams and unigrams that don't include BOS or EOS -- which
    # is most of them! -- `event_count` and `context_count` will give the
    # same value.  So you could save about half the memory if you were
    # careful to store those cases only once.  (How?)  That would make the
    # code slightly more complicated, but would be worth it in a real system.

    def count_trigram_events(self, trigram: Trigram) -> None:
        """Record one token of the trigram and also of its suffixes (for backoff)."""
        (x, y, z) = trigram
        self.event_count[(x, y, z )] += 1
        self.event_count[   (y, z )] += 1
        self.event_count[      (z,)] += 1  # the comma is necessary to make this a tuple
        self.event_count[        ()] += 1

    def count_trigram_contexts(self, trigram: Trigram) -> None:
        """Record one token of the trigram's CONTEXT portion, 
        and also the suffixes of that context (for backoff)."""
        (x, y, _) = trigram    # we don't care about z
        self.context_count[(x, y )] += 1
        self.context_count[   (y,)] += 1
        self.context_count[     ()] += 1

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes an estimate of the trigram log probability log p(z | x,y)
        according to the language model.  The log_prob is what we need to compute
        cross-entropy and to train the model.  It is also unlikely to underflow,
        in contrast to prob.  In many models, we can compute the log_prob directly, 
        rather than first computing the prob and then calling math.log."""
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling log_prob on an instance of LanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.log_prob is not implemented yet (you should override LanguageModel.log_prob)"
        )      
        
    def sample(self, max_length: int, x: Wordtype = BOS, y: Wordtype = BOS) -> list[Wordtype]:
        """Samples tokens from the probability distribution
        """
        sentence: list[Wordtype] = []
        vocab_list = list(self.vocab)
        for _ in range(max_length):
            # Create the probability distribution
            prob_list = []
            for z in self.vocab:
                logprob = self.log_prob(x, y, z)
                prob_list.append(logprob)
            # Install it into tensor
            weights = torch.tensor(prob_list)
            probabilities = torch.softmax(weights, dim = 0)

            # Use torch.multinomial to sample
            z_index = int(torch.multinomial(probabilities, num_samples = 1)[0].item())
            z = vocab_list[z_index]
            if (z == EOS):
                return sentence
            sentence.append(z)
            x, y = y, z
        return sentence
    
    def save(self, model_path: Path) -> None:
        log.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            # torch.save is similar to pickle.dump but handles tensors too
        log.info(f"Saved model to {model_path}")

    @classmethod
    def load(cls, model_path: Path, device: str = 'cpu') -> "LanguageModel":
        log.info(f"Loading model from {model_path}")
        model = torch.load(model_path, weights_only = False, map_location=device)
            # torch.load is similar to pickle.load but handles tensors too
            # map_location allows loading tensors on different device than saved
        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls} but got {type(model)} from file {model_path}")
        log.info(f"Loaded model from {model_path}")
        return model

    def train(self, file: Path) -> None:
        """Create vocabulary and store n-gram counts.  In subclasses, we might
        override this with a method that computes parameters instead of counts."""

        log.info(f"Training from corpus {file}")

        # Clear out any previous training.
        self.event_count   = Counter()
        self.context_count = Counter()

        for trigram in read_trigrams(file, self.vocab):
            self.count_trigram_events(trigram)
            self.count_trigram_contexts(trigram)
            self.show_progress()

        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.event_count[()]} tokens")

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")


##### SPECIFIC FAMILIES OF LANGUAGE MODELS

class CountBasedLanguageModel(LanguageModel):

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # For count-based language models, it is usually convenient
        # to compute the probability first (by dividing counts) and
        # then taking the log.
        prob = self.prob(x, y, z)
        if prob == 0.0:
            return -math.inf
        return math.log(prob)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == CountBasedLanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling prob on an instance of CountBasedLanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.prob is not implemented yet (you should override CountBasedLanguageModel.prob)"
        )

        
            

class UniformLanguageModel(CountBasedLanguageModel):
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return 1 / self.vocab_size


class AddLambdaLanguageModel(CountBasedLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab)
        if lambda_ < 0.0:
            raise ValueError(f"Negative lambda argument of {lambda_} could result in negative smoothed probs")
        self.lambda_ = lambda_

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        assert self.event_count[x, y, z] <= self.context_count[x, y]
        return ((self.event_count[x, y, z] + self.lambda_) /
                (self.context_count[x, y] + self.lambda_ * self.vocab_size))

        # Notice that summing the numerator over all values of typeZ
        # will give the denominator.  Therefore, summing up the quotient
        # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
        # as is required for any probability function.


class BackoffAddLambdaLanguageModel(AddLambdaLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab, lambda_)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # TODO: Reimplement me so that I do backoff
        # Don't forget the difference between the Wordtype z and the
        # 1-element tuple (z,). If you're looking up counts,
        # these will have very different counts!
        assert self.event_count[x, y, z] <= self.context_count[x, y]
        assert self.event_count[y, z] <= self.context_count[y,]
        assert self.event_count[z,] <= self.context_count[()]

        lm_v = self.lambda_ * self.vocab_size
        p_z = (self.event_count[z,] + self.lambda_)/ (self.context_count[()] + lm_v)
        p_zy = (self.event_count[y, z] + lm_v * p_z) / (self.context_count[y,] + lm_v)
        p_zxy = (self.event_count[x, y, z] + lm_v * p_zy) / (self.context_count[x, y] + lm_v)
        
        return p_zxy


class EmbeddingLogLinearLanguageModel(LanguageModel, nn.Module):
    # Note the use of multiple inheritance: we are both a LanguageModel and a torch.nn.Module.
    
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int) -> None:
        super().__init__(vocab)
        if l2 < 0:
            raise ValueError("Negative regularization strength {l2}")
        self.l2: float = l2
        self.epochs = epochs

        # TODO: ADD CODE TO READ THE LEXICON OF WORD VECTORS AND STORE IT IN A USEFUL FORMAT.
        self.lexicon = {}
        self.dim = 0
        with open(lexicon_file, "r") as file:
            for i, line in enumerate(file):
                sentence = line.strip().split()
                if i == 0:
                    self.dim: int = int(sentence[1])
                    continue
                embedding = [float(x) for x in sentence[1:]]
                embedding_torch = torch.tensor(embedding, dtype = torch.float32)
                self.lexicon[sentence[0]] = embedding_torch

        vocab_embeddings = [
            self.lexicon[OOL if word == OOV or word not in self.lexicon else word]
            for word in self.vocab
        ]
        self.register_buffer("vocab_embeddings", torch.stack(vocab_embeddings))

        # We wrap the following matrices in nn.Parameter objects.
        # This lets PyTorch know that these are parameters of the model
        # that should be listed in self.parameters() and will be
        # updated during training.
        #
        # We can also store other tensors in the model class,
        # like constant coefficients that shouldn't be altered by
        # training, but those wouldn't use nn.Parameter.
        self.X = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
        self.Y = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Return log p(z | xy) according to this language model."""
        # https://pytorch.org/docs/stable/generated/torch.Tensor.item.html
        return self.log_prob_tensor(x, y, z).item()

    @typechecked
    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> TorchScalar:
        """Return the same value as log_prob, but stored as a tensor."""
        
        # As noted below, it's important to use a tensor for training.
        # Most of your intermediate quantities, like logits below, will
        # also be stored as tensors.  (That is normal in PyTorch, so it
        # would be weird to append `_tensor` to their names.  We only
        # appended `_tensor` to the name of this method to distinguish
        # it from the class's general `log_prob` method.)

        # TODO: IMPLEMENT ME!
        # This method should call the logits helper method.
        # You are free to define other helper methods too.
        #
        # Be sure to use vectorization over the vocabulary to
        # compute the normalization constant Z, or this method
        # will be very slow. Some useful functions of pytorch that could
        # be useful are torch.logsumexp and torch.log_softmax.
        #
        # The return type, TorchScalar, represents a torch.Tensor scalar.
        # See Question 7 in INSTRUCTIONS.md for more info about fine-grained 
        # type annotations for Tensors.
        if x not in self.lexicon or x == OOV:
            x = OOL
        if y not in self.lexicon or y == OOV:
            y = OOL
        if z not in self.lexicon or z == OOV:
            z = OOL
        
        x_emb = self.lexicon[x]
        y_emb = self.lexicon[y]
        z_emb = self.lexicon[z]
        p_xyz = x_emb @ (self.X @ z_emb) + y_emb @ (self.Y @ z_emb)
        Z = torch.logsumexp(self.logits(x, y), dim = 0)
        return p_xyz - Z 

    def logits(self, x: Wordtype, y: Wordtype) -> Float[torch.Tensor,"vocab"]:
        """Return a vector of the logs of the unnormalized probabilities f(xyz) * θ 
        for the various types z in the vocabulary.
        These are commonly known as "logits" or "log-odds": the values that you 
        exponentiate and renormalize in order to get a probability distribution."""
        # TODO: IMPLEMENT ME!
        # Don't forget that you can create additional methods
        # that you think are useful, if you'd like.
        # It's cleaner than making this function massive.
        #
        # The operator `@` is a nice way to write matrix multiplication:
        # you can write J @ K as shorthand for torch.mul(J, K).
        # J @ K looks more like the usual math notation.
        # 
        # This function's return type is declared (using the jaxtyping module)
        # to be a torch.Tensor whose elements are Floats, and which has one
        # dimension of length "vocab".  This can be multiplied in a type-safe
        # way by a matrix of type Float[torch.Tensor,"vocab","embedding"]
        # because the two strings "vocab" are identical, representing matched
        # dimensions.  At runtime, "vocab" will be replaced by size of the
        # vocabulary, and "embedding" will be replaced by the embedding
        # dimensionality as given by the lexicon.  See
        # https://www.cs.jhu.edu/~jason/465/hw-lm/code/INSTRUCTIONS.html#a-note-on-type-annotations
        if x not in self.lexicon or x == OOV:
            x = OOL
        if y not in self.lexicon or y == OOV:
            y = OOL
        x_emb = self.lexicon[x]
        y_emb = self.lexicon[y]
        context = x_emb @ self.X + y_emb @ self.Y
        return context @ self.vocab_embeddings.T

    def train(self, file: Path):    # type: ignore
        
        ### Technically this method shouldn't be called `train`,
        ### because this means it overrides not only `LanguageModel.train` (as desired)
        ### but also `nn.Module.train` (which has a different type). 
        ### However, we won't be trying to use the latter method.
        ### The `type: ignore` comment above tells the type checker to ignore this inconsistency.
        
        # Initialize the parameter matrices to be full of zeros.
        nn.init.zeros_(self.X)   # type: ignore
        nn.init.zeros_(self.Y)   # type: ignore

        N = num_tokens(file)
        log.info("Start optimizing on {N} training tokens...")

        # Optimization hyperparameters.
        eta0 = 1e-5  # initial learning rate

        # This is why we needed the nn.Parameter above.
        # The optimizer needs to know the list of parameters
        # it should be trying to update.
        optimizer = optim.SGD(self.parameters(), lr=eta0)

        # Optionally, your training loop can call scheduler.step()
        # right after each optimizer.step().  This changes the
        # stepsize for the next step.  But you'll need to define the
        # scheduler first.  So try uncommenting the lines below.  That
        # creates a scheduler whose step() method reduces the SGD
        # learning rate just like Algorithm 1, so that SGD will
        # converge.  Basically the step() method will execute line 8
        # and then line 6.
        #
        # lr_lambda =  2 * self.l2 / N   # should be 2*C/N.  (What is C and where does this class store it?)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1 / (1 + eta0 * lr_lambda * t))

        #####################
        # TODO: Implement your SGD here by taking gradient steps on a sequence
        # of training examples.  Here's how to use PyTorch to make it easy:
        #
        # To get the training examples, you can use the `read_trigrams` function
        # we provided, which will iterate over all N trigrams in the training
        # corpus.  (Its use is illustrated in fileprob.py.)
        #
        # For each successive training example i, compute the stochastic
        # objective F_i(θ).  This is called the "forward" computation. Don't
        # forget to include the regularization term. Part of F_i(θ) will be the
        # log probability of training example i, which the helper method
        # log_prob_tensor computes.  It is important to use log_prob_tensor
        # (as opposed to log_prob which returns float) because torch.Tensor
        # is an object with additional bookkeeping that tracks e.g. the gradient
        # function for backpropagation as well as accumulated gradient values
        # from backpropagation.
        #
        # To get the gradient of this objective (∇F_i(θ)), call the `backward()`
        # method on the number F_i(θ) that you computed at the previous step.
        # This invokes back-propagation to get the gradient of this number with
        # respect to the parameters θ.  This should be easier than implementing
        # the gradient method from the handout.
        #
        # Before you call `backward`, you should call `optimizer.zero_grad()` to
        # initialize the gradient to 0.  This clears out any previous gradient so
        # that `backward()` doesn't add to it.
        #
        # Finally, update the parameters in the direction of the gradient, as
        # shown in Algorithm 1 in the reading handout.  You can do this `+=`
        # yourself, or you can call the `step` method of the `optimizer` object
        # we created above.  See the reading handout for more details on this.
        #
        # For the EmbeddingLogLinearLanguageModel, you should run SGD
        # optimization for the given number of epochs and then stop.  You might 
        # want to print progress dots using the `show_progress` method defined above.  
        # Even better, you could show a graphical progress bar using the tqdm module --
        # simply iterate over
        #     tqdm.tqdm(read_trigrams(file), total=N*epochs)
        # instead of iterating over
        #     read_trigrams(file)
        #####################

        from tqdm import tqdm
        
        for epoch in range(self.epochs):
            # Use tqdm to track the process
            process = tqdm(
                read_trigrams(file, self.vocab),
                total = N,
                desc = f"Epoch {epoch + 1}/{self.epochs}"
            )
            # Optimize the theta
            for (x, y, z) in process:
                optimizer.zero_grad()
                regularization = self.l2 / N * (torch.sum(self.X ** 2)
                                    + torch.sum(self.Y ** 2))
                f_i = self.log_prob_tensor(x, y, z) - regularization
                (-f_i).backward()
                optimizer.step()

            # Calculate the final objective function
            with torch.no_grad():
                total_log_prob = 0.0

                for x, y, z in read_trigrams(file, self.vocab):
                    total_log_prob += self.log_prob_tensor(x, y, z).item()

                regularization = self.l2 * (
                    torch.sum(self.X ** 2) +
                    torch.sum(self.Y ** 2)
                )

                F = (total_log_prob - regularization.item()) / N

            print(f"epoch {epoch + 1}: F = {F}")

        log.info(f"Finished training on {N} tokens")
        log.info("done optimizing.")

        # So how does the `backward` method work?
        #
        # As Python sees it, your parameters and the values that you compute
        # from them are not actually numbers.  They are `torch.Tensor` objects.
        # A Tensor may represent a numeric scalar, vector, matrix, etc.
        #
        # Every Tensor knows how it was computed.  For example, if you write `a
        # = b + exp(c)`, PyTorch not only computes `a` but also stores
        # backpointers in `a` that remember how the numeric value of `a` depends
        # on the numeric values of `b` and `c`.  In turn, `b` and `c` have their
        # own backpointers that remember what they depend on, and so on, all the
        # way back to the parameters.  This is just like the backpointers in
        # parsing!
        #
        # Every Tensor has a `backward` method that computes the gradient of its
        # numeric value with respect to the parameters, using "back-propagation"
        # through this computation graph.  In particular, once you've computed
        # the forward quantity F_i(θ) as a tensor, you can trace backwards to
        # get its gradient -- i.e., to find out how rapidly it would change if
        # each parameter were changed slightly.


class ImprovedLogLinearLanguageModel(EmbeddingLogLinearLanguageModel):
    """J.2 unigram feature, optional J.1 OOV bias, and optional SGD improvements."""

    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int,
                 *, batch_size: int = 1, optimizer_name: str = "sgd",
                 learning_rate: float | None = None, shuffle: bool = False,
                 dev_files: Collection[Path] = (), patience: int = 3,
                 min_delta: float = 0.0001, seed: int = 0,
                 use_oov_feature: bool = False) -> None:
        super().__init__(vocab, lexicon_file, l2, epochs)
        if batch_size < 1 or epochs < 0 or patience < 1:
            raise ValueError("batch_size and patience must be positive; epochs must be nonnegative")
        if optimizer_name not in ("sgd", "adam"):
            raise ValueError("optimizer_name must be sgd or adam")
        if learning_rate is None:
            learning_rate = 0.001 if optimizer_name == "adam" else 0.00001
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be nonnegative and finite")

        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.shuffle = shuffle
        self.dev_files = tuple(Path(p) for p in dev_files)
        self.patience = patience
        self.min_delta = min_delta
        self.seed = seed
        self.word_to_index = {w: i for i, w in enumerate(self.vocab)}
        self.bos_index = len(self.vocab)

        self.beta = nn.Parameter(self.X.new_zeros(()))
        self.register_buffer("unigram_features", self.X.new_zeros(len(self.vocab)))
        mask = self.X.new_zeros(len(self.vocab))
        mask[self.word_to_index[OOV]] = 1
        self.register_buffer("oov_mask", mask)
        self.oov_bias = nn.Parameter(self.X.new_zeros(())) if use_oov_feature else None

        # Contexts can contain BOS, although BOS is never a predicted word.
        bos_embedding = self.lexicon.get(BOS, self.lexicon[OOL])
        self.register_buffer("context_embeddings", torch.cat(
            (self.vocab_embeddings, bos_embedding.unsqueeze(0)), dim=0))

    def _extra_scores(self) -> torch.Tensor:
        result = self.beta * self.unigram_features
        if self.oov_bias is not None:
            result = result + self.oov_bias * self.oov_mask
        return result

    def logits(self, x: Wordtype, y: Wordtype) -> torch.Tensor:
        return super().logits(x, y) + self._extra_scores()

    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> torch.Tensor:
        # Both numerator and denominator include the added features.
        scores = self.logits(x, y)
        index = self.word_to_index.get(z, self.word_to_index[OOV])
        return scores[index] - torch.logsumexp(scores, dim=0)

    def _encode(self, paths: Collection[Path]) -> torch.Tensor:
        def context_index(word: Wordtype) -> int:
            if word == BOS:
                return self.bos_index
            return self.word_to_index.get(word, self.word_to_index[OOV])

        rows = [(context_index(x), context_index(y), self.word_to_index[z])
                for path in paths for x, y, z in read_trigrams(path, self.vocab)]
        # Keep examples on CPU; transfer only the current batch to the model device.
        return torch.tensor(rows, dtype=torch.long, device="cpu").reshape(-1, 3)

    def _batch_log_probs(self, examples: torch.Tensor) -> torch.Tensor:
        examples = examples.to(self.X.device)
        x, y, z = examples.unbind(dim=1)
        contexts = self.context_embeddings[x] @ self.X + self.context_embeddings[y] @ self.Y
        scores = contexts @ self.vocab_embeddings.T + self._extra_scores()
        observed = scores.gather(1, z[:, None]).squeeze(1)
        return observed - torch.logsumexp(scores, dim=1)

    def _squared_parameters(self) -> torch.Tensor:
        # Include the learned feature weights, not just X and Y.
        return sum(parameter.square().sum() for parameter in self.parameters())

    def _total_log_prob(self, examples: torch.Tensor) -> float:
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(examples), 128):
                total += self._batch_log_probs(examples[start:start + 128]).sum().item()
        return total

    def train(self, file: Path):  # type: ignore
        training = self._encode((file,))
        N = len(training)
        if N == 0:
            raise ValueError("Training corpus is empty")
        for parameter in self.parameters():
            nn.init.zeros_(parameter)

        # J.2: log(c(z)+1). Counts come exclusively from the training corpus.
        counts = torch.bincount(training[:, 2], minlength=len(self.vocab))
        self.unigram_features.copy_(torch.log(counts.to(
            device=self.X.device, dtype=self.X.dtype) + 1))

        development = self._encode(self.dev_files) if self.dev_files else None
        if development is not None and len(development) == 0:
            raise ValueError("Development corpus is empty")
        if development is not None:
            training_path = Path(file).resolve()
            if any(p.resolve() == training_path for p in self.dev_files):
                raise ValueError("Use held-out development files, not the training file")

        optimizer_class = optim.Adam if self.optimizer_name == "adam" else optim.SGD
        optimizer = optimizer_class(self.parameters(), lr=self.learning_rate)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        best_dev = math.inf
        best_state = None
        bad_epochs = 0
        self.training_history = []
        self.best_epoch = None
        log.info(f"Training on {N} tokens: {self.optimizer_name}, "
                 f"learning rate={self.learning_rate:g}, batch size={self.batch_size}")

        for epoch in range(self.epochs):
            ordered = training[torch.randperm(N, generator=generator, device="cpu")] if self.shuffle else training
            for start in range(0, N, self.batch_size):
                batch = ordered[start:start + self.batch_size]
                optimizer.zero_grad()
                # Mean of F_i over the batch: the penalty coefficient remains C/N.
                loss = -self._batch_log_probs(batch).mean() + (self.l2 / N) * self._squared_parameters()
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite training loss; try a smaller learning rate")
                loss.backward()
                optimizer.step()

            total_log_prob = self._total_log_prob(training)
            with torch.no_grad():
                penalty = self.l2 * self._squared_parameters().item()
            F = (total_log_prob - penalty) / N
            if not math.isfinite(F):
                raise RuntimeError("Non-finite epoch objective; try a smaller learning rate")
            row = {"epoch": epoch + 1, "F": F, "beta": self.beta.item()}
            suffix = ""
            if development is not None:
                dev_entropy = -self._total_log_prob(development) / (len(development) * math.log(2))
                if not math.isfinite(dev_entropy):
                    raise RuntimeError("Non-finite development cross-entropy")
                row["dev_bits_per_token"] = dev_entropy
                suffix = f", dev cross-entropy = {dev_entropy:.5f} bits/token"
                if dev_entropy < best_dev - self.min_delta:
                    best_dev = dev_entropy
                    best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                    self.best_epoch = epoch + 1
                    bad_epochs = 0
                else:
                    bad_epochs += 1
            self.training_history.append(row)
            print(f"epoch {epoch + 1}: F = {F}, beta = {self.beta.item():.5f}{suffix}", flush=True)
            if development is not None and bad_epochs >= self.patience:
                log.info(f"Early stopping after epoch {epoch + 1}")
                break

        if best_state is not None:
            self.load_state_dict(best_state)
            log.info(f"Restored epoch {self.best_epoch}, dev cross-entropy {best_dev:.5f}")
        log.info(f"Finished training on {N} tokens; beta={self.beta.item():.5f}")
