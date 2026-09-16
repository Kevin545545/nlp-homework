#!/usr/bin/env python3
"""
601.465/665 — Natural Language Processing
Assignment 1: Designing Context-Free Grammars

Assignment written by Jason Eisner
Modified by Kevin Duh
Re-modified by Alexandra DeLucia

Code template written by Alexandra DeLucia,
based on the submitted assignment with Keith Harrigian
and Carlos Aguirre Fall 2019
"""
import argparse
import os
import random
import subprocess
import sys

# Want to know what command-line arguments a program allows?
# Commonly you can ask by passing it the --help option, like this:
#     python randsent.py --help
# This is possible for any program that processes its command-line
# arguments using the argparse module, as we do below.
#
# NOTE: When you use the Python argparse module, parse_args() is the
# traditional name for the function that you create to analyze the
# command line.  Parsing the command line is different from parsing a
# natural-language sentence.  It's easier.  But in both cases,
# "parsing" a string means identifying the elements of the string and
# the roles they play.

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args (an argparse.Namespace): Stores command-line attributes
    """
    # Initialize parser
    parser = argparse.ArgumentParser(description="Generate random sentences from a PCFG")
    # Grammar file (required argument)
    parser.add_argument(
        "-g",
        "--grammar",
        type=str, required=True,
        help="Path to grammar file",
    )
    # Start symbol of the grammar
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )
    # Number of sentences
    parser.add_argument(
        "-n",
        "--num_sentences",
        type=int,
        help="Number of sentences to generate (default is 1)",
        default=1,
    )
    # Max number of nonterminals to expand when generating a sentence
    parser.add_argument(
        "-M",
        "--max_expansions",
        type=int,
        help="Max number of nonterminals to expand when generating a sentence",
        default=450,
    )
    # Print the derivation tree for each generated sentence
    parser.add_argument(
        "-t",
        "--tree",
        action="store_true",
        help="Print the derivation tree for each generated sentence",
        default=False,
    )
    return parser.parse_args()


class Grammar:
    def __init__(self, grammar_file):
        """
        Context-Free Grammar (CFG) Sentence Generator

        Args:
            grammar_file (str): Path to a .gr grammar file
        
        Returns:
            self
        """
        # Parse the input grammar file
        self.rules = {}
        self._load_rules_from_file(grammar_file)

    def _load_rules_from_file(self, grammar_file):
        """
        Read grammar file and store its rules in self.rules

        Args:
            grammar_file (str): Path to the raw grammar file 
        """
        
        with open(grammar_file, 'r', encoding='utf-8') as file:
            for line in file:
                # Handle the inline and start hash comment, and empty sentence
                line = line.split('#', maxsplit = 1)[0].strip()
                if not line: 
                    continue 
                line_list = line.split()

                # Initialize the overal data structure
                # {"NP": [(1, [Det, Noun]), (1, [NP, PP])]}
                if (self.rules.get(line_list[1])):
                    self.rules[line_list[1]].append((float(line_list[0]), line_list[2:]))
                else:
                    self.rules[line_list[1]] = [(float(line_list[0]), line_list[2:])]

                


    def sample(self, derivation_tree, max_expansions, start_symbol):
        """
        Sample a random sentence from this grammar

        Args:
            derivation_tree (bool): if true, the returned string will represent 
                the tree (using bracket notation) that records how the sentence 
                was derived
                               
            max_expansions (int): max number of nonterminal expansions we allow

            start_symbol (str): start symbol to generate from

        Returns:
            str: the random sentence or its derivation tree
        """
        sentence = []
        # A mutable counter
        max_num = [max_expansions]
        self._sampleRec(derivation_tree, 
                   max_num, start_symbol, sentence)
        return " ".join(sentence)


    def _sampleRec(self, derivation_tree, 
                   max_expansions, start_symbol, sentence):
        if max_expansions[0] == 0:
            sentence.append("...")
            return
        max_expansions[0] -= 1

        rhs = [item[1] for item in self.rules[start_symbol]]
        weights = [item[0] for item in self.rules[start_symbol]]
        choice = random.choices(rhs, weights)[0]

        # Each time open a non-terminal
        if (derivation_tree):
            sentence.append('(' + start_symbol)
            
        for symbol in choice:
            if symbol not in self.rules:
                sentence.append(symbol)
                
            else:
                self._sampleRec(derivation_tree,
                                max_expansions,
                                symbol, sentence)
        # After process all its expansions, close the nonterminal
        if (derivation_tree):
            sentence.append(')')
                       
            


####################
### Main Program
####################
def main():
    # Parse command-line options
    args = parse_args()

    # Initialize Grammar object
    grammar = Grammar(args.grammar)

    # Generate sentences
    for i in range(args.num_sentences):
        # Use Grammar object to generate sentence
        sentence = grammar.sample(
            derivation_tree=args.tree,
            max_expansions=args.max_expansions,
            start_symbol=args.start_symbol
        )

        # Print the sentence with the specified format.
        # If it's a tree, we'll pipe the output through the prettyprint script.
        if args.tree:
            prettyprint_path = os.path.join(os.getcwd(), 'prettyprint')
            subprocess.run(
                ['perl', prettyprint_path],
                input=sentence,
                text=True
            )
        else:
            print(sentence)


if __name__ == "__main__":
    main()
