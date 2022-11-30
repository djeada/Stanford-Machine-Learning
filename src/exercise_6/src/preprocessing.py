""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 6 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import scipy.io
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


def read_data(
    path: Path, column_names: Tuple[str, ...] = ("X", "y")
) -> Tuple[np.ndarray, ...]:
    """
    Reads the data from a CSV file and returns the data as a numpy array.
    :param path: The path to the CSV file.
    :param column_names: The names of the columns to read.
    :return: The data as a tuple of numpy arrays.
    """

    raw_data = scipy.io.loadmat(str(path))
    return tuple(raw_data[column_name] for column_name in column_names)


def include_intercept(x: np.ndarray) -> np.ndarray:
    """
    Prepends a column of ones to the input array.
    :param x: The input array.
    :return: The input array with a column of ones prepended.
    """

    if x.ndim == 1:
        x = x[:, np.newaxis]

    return np.insert(x, 0, 1, axis=1)


@dataclass
class ClassifiedData:
    """
    Data separated into positive and negative classes.
    """

    positive: np.ndarray
    negative: np.ndarray

    @classmethod
    def from_data(cls, x: np.ndarray, y: np.ndarray) -> "Data":
        """
        Creates a Data object from the input and output arrays.
        :param x: The input array.
        :param y: The output array.
        :return: A Data object.
        """

        positive = x[(y == 1).flatten()]
        negative = x[(y == 0).flatten()]

        return cls(positive, negative)


class WordCleanerBase(ABC):
    @abstractmethod
    def clean(self, word: str) -> str:
        """
        Clean the word.
        :param word: The word to clean.
        :return: The cleaned word.
        """
        pass


class HtmlTagsStripper(WordCleanerBase):
    def clean(self, word: str) -> str:
        """
        Strip the HTML tags from the word.
        :param word: The word to strip.
        :return: The stripped word.
        """
        return re.sub("<[^<>]+>", " ", word)


class DigitsReplacer(WordCleanerBase):
    def clean(self, word: str) -> str:
        """
        Replace the digits in the word with the word 'number'.
        :param word: The word to replace the digits in.
        :return: The word with the digits replaced.
        """
        return re.sub("[0-9]+", "number", word)


class DollarSignsReplacer(WordCleanerBase):
    def clean(self, word: str) -> str:
        """
        Replace the dollar signs in the word with the word 'dollar'.
        :param word: The word to replace the dollar signs in.
        :return: The word with the dollar signs replaced.
        """
        return re.sub("[$]+", "dollar", word)


class UrlsReplacer(WordCleanerBase):
    def clean(self, word: str) -> str:
        """
        Replace the URLs in the word with the word 'httpaddr'.
        :param word: The word to replace the URLs in.
        :return: The word with the URLs replaced.
        """
        return re.sub("(http|https)://[^\s]*", "httpaddr", word)


class EmailAddressesReplacer(WordCleanerBase):
    def clean(self, word: str) -> str:
        """
        Replace the email addresses in the word with the word 'contentaddr'.
        :param word: The word to replace the email addresses in.
        :return: The word with the email addresses replaced.
        """
        return re.sub("[^\s]+@[^\s]+", "contentaddr", word)


class WordCleaner:
    def __init__(
        self,
        cleaners: list = [
            HtmlTagsStripper(),
            DigitsReplacer(),
            DollarSignsReplacer(),
            UrlsReplacer(),
            EmailAddressesReplacer(),
        ],
    ):
        """
        Initialize the word cleaner.
        :param cleaners: The cleaners to clean the words.
        """
        self.cleaners = cleaners

    def clean(self, word: str) -> str:
        """
        Clean the word.
        :param word: The word to clean.
        :return: The cleaned word.
        """
        word = word.lower()
        for cleaner in self.cleaners:
            word = cleaner.clean(word)
        return word


def read_text_file(path: Path) -> list:
    """
    Reads the content from the file specified by the path.

    :param path: The path to the text file.
    :return: The content of the text file.
    """
    path = Path(path)
    content = path.read_text()
    content = content.splitlines()
    content = [line.split("\t")[-1] for line in content]
    return content


class Tokenizer:
    """
    Tokenizes the words in the input text.
    """

    def __init__(self, vocabulary: list):
        """
        Initialize the tokenizer.
        :param vocabulary: The vocabulary to filter the tokens.
        """
        self.vocabulary = vocabulary

    def tokenize(self, content: str) -> list:
        """
        Tokenizes the content.
        :param content: The content to tokenize.
        :return: The tokens.
        """
        tokens = []
        words = re.split(
            r"\s|@|\$|/|#|\.|-|:|&|\*|\+|=|\[|\?|!|\(|\)|}|,|'|\"|>|_|<|;|%", content
        )

        for word in words:
            alphanumeric = re.compile(r"[^a-zA-Z0-9]")
            word = alphanumeric.sub("", word)  # remove non-alphanumeric characters

            # check if the word is in the vocabulary
            for i, vocab_word in enumerate(self.vocabulary):
                if vocab_word in word:
                    tokens.append(i)
                    break

        return tokens


def extract_features(tokens: list, num_words: int) -> np.ndarray:
    """
    Extracts the features from the tokens.
    :param tokens: The tokens to extract the features from.
    :param num_words: The number of words in the vocabulary.
    :return: The features.
    """
    features = np.zeros(num_words)
    features[tokens] = 1
    return features
