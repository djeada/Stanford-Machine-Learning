""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 6 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

from pathlib import Path
import scipy.io
from nltk.stem import PorterStemmer
import re


def read_data(path: Path) -> tuple:
    """
    x is a matrix with m rows and n columns
    y is a matrix with m rows and 1 column
    """

    raw_data = scipy.io.loadmat(str(path))
    x, y = raw_data["X"], raw_data["y"]
    return x, y


def read_test_data(path: Path) -> tuple:
    """
    x is a matrix with m rows and n columns
    y is a matrix with m rows and 1 column
    """

    raw_data = scipy.io.loadmat(str(path))
    x, y = raw_data["Xtest"], raw_data["ytest"]
    return x, y


def read_tokens(path: Path, vocabulary: list) -> list:
    def _tokenize_email(content, _vocabulary):
        tokens = []
        words = re.split(
            r"\s|@|\$|/|#|\.|-|:|&|\*|\+|=|\[|\?|!|\(|\)|}|,|'|\"|>|_|<|;|%",
            content,
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

    def _process_email(content, _vocabulary):
        return _tokenize_email(pre_processing(content), _vocabulary)

    with open(path, "r") as email:
        return _process_email(email.read(), vocabulary)


def read_vocabulary(path: Path) -> list:
    with open(path, "r") as vocab:
        return [line[:-1].split("\t")[-1] for line in vocab.readlines()]


def pre_processing(content: str) -> str:
    content = content.lower()

    def replace(old, new):
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
