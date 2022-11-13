""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 6 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import scipy.io
from nltk.stem import PorterStemmer
import re


def read_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the data from the path and returns the data in the form of a tuple of two matrices.
    The first matrix is a matrix with m rows and n columns, where m is the number of emails and n is the number of features.
    The second matrix is a matrix with m rows and 1 column, where m is the number of emails and 1 is the number of labels.

    Args:
        path: Path to the data.
    
    Returns:
        A tuple of two matrices.
    """

    raw_data = scipy.io.loadmat(str(path))
    x, y = raw_data["X"], raw_data["y"]
    return x, y


def read_test_data(path: Path) -> tuple:
    """
    Reads the test data from the path and returns the data in the form of a tuple of two matrices.
    The first matrix is a matrix with m rows and n columns, where m is the number of emails and n is the number of features.
    The second matrix is a matrix with m rows and 1 column, where m is the number of emails and 1 is the number of labels.

    Args:
        path: Path to the data.
    
    Returns:
        A tuple of two matrices.
    """

    raw_data = scipy.io.loadmat(str(path))
    x, y = raw_data["Xtest"], raw_data["ytest"]
    return x, y


def read_tokens(path: Path, vocabulary: list) -> list:
    """
    Reads the tokens from the path and returns the tokens in the form of a list.
    The tokens are stemmed and the stop words are removed. 
    The tokens are then filtered by the vocabulary.
    
    Args:
        path: Path to the tokens.
        vocabulary: Vocabulary to filter the tokens.
    
    Returns:
        A list of tokens.
    """

    def _tokenize_email(content: str, _vocabulary: list) -> list:
        """
        Tokenize the email content.
        
        Args:
            content: Email content.
            _vocabulary: Vocabulary to filter the tokens.
        
        Returns:
            A list of tokens.
        """
        tokens = []
        words = re.split(
            r"\s|@|\$|/|#|\.|-|:|&|\*|\+|=|\[|\?|!|\(|\)|}|,|'|\"|>|_|<|;|%", content
        )

        stemmer = PorterStemmer()

        for word in words:
            # Remove non-alphanumeric characters
            alphanumeric = re.compile(r"[^a-zA-Z0-9]")
            word = alphanumeric.sub("", word)

            # Stem the word
            word = stemmer.stem(word)

            # Get index if it exists
            if word in _vocabulary:
                tokens.append(_vocabulary.index(word))

        return tokens

    def _process_email(content: str, _vocabulary: list) -> list:
        """
        Process the email content.

        Args:
            content: Email content.
            _vocabulary: Vocabulary to filter the tokens.

        Returns:
            A list of tokens.
        """
        return _tokenize_email(pre_processing(content), _vocabulary)

    with open(path, "r") as email:
        return _process_email(email.read(), vocabulary)


def read_vocabulary(path: Path) -> list:
    """
    Reads the vocabulary from the path and returns the vocabulary in the form of a list.

    Args:
        path: Path to the vocabulary.
    
    Returns:
        A list of tokens.
    """
    with open(path, "r") as vocab:
        return [line[:-1].split("\t")[-1] for line in vocab.readlines()]


def pre_processing(content: str) -> str:
    """
    Pre-process the email content.

    Args:
        content: Email content.

    Returns:
        The pre-processed email content. 
    """
    content = content.lower()

    def replace(old: str, new: str) -> str:
        """
        Replace the old string with the new string.

        Args:
            old: Old string.
            new: New string.
        
        Returns:
            The replaced string.
        """
        return re.sub(old, new, content)

    # strip html tags
    content = replace("<[^<>]+>", " ")

    # replace digits with a word 'number'
    content = replace("[0-9]+", "number")

    # replace dollar signs with a word 'dollar'
    content = replace("[$]+", "dollar")

    # strings starting with http are replaced with _http
    content = replace("(http|https)://[^\s]*", "httpaddr")

    # strings containing @ sign are replaced with _addr
    content = replace("[^\s]+@[^\s]+", "contentaddr")

    return content
