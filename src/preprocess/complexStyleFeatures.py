from collections import Counter
from typing import Tuple

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from scipy.stats import entropy

from src.config.Constants import EPSILON, TEXT_COLUMN_NAME
from src.preprocess.contentFeatures import PENN2WN, bag_of_words, stem_words


def complex_style_features_extraction(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame):
    """
    @return: a representation of the data using complex
    stylistic features: lexicographic diversity and readability
    """

    def complex_style_features_extraction_helper(
        x: pd.DataFrame,
    ) -> pd.DataFrame:
        x = pd.concat(
            [
                honore_measure(x),
                hapax_disLegemena(x),
                Yules_characteristic(x),
                simpsons_measure(x),
                brunets_measure(x),
                avg_word_frequency(x),
                entropy_over_words_frequencies(x),
                flesch_reading_ease_flesch_kincaid_grade_level(x),
                gunning_fog_index(x),
            ],
            axis=1,
        )
        return x

    return (
        complex_style_features_extraction_helper(x_train),
        complex_style_features_extraction_helper(x_test),
    )


def characters_n_grams(
    x_train: pd.DataFrame, x_test: pd.DataFrame, n: int, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return bag_of_words(
        x_train, x_test, analyzer="char", ngram_range=(n, n), **kwargs
    )


def pos_tagger_helper(text: str) -> str:
    return "".join(
        [
            PENN2WN.get(pos, wn.NOUN)
            for _, pos in nltk.pos_tag(nltk.word_tokenize(text))
        ]
    )


def pos_n_grams(
    x_train: pd.DataFrame, x_test: pd.DataFrame, n: int, **kwargs
) -> (pd.DataFrame, pd.DataFrame):
    return bag_of_words(
        x_train,
        x_test,
        preprocessor=pos_tagger_helper,
        ngram_range=(1, n),
        analyzer="char",
        **kwargs
    )


def honore_measure(x: pd.DataFrame) -> pd.DataFrame:
    """
    Honore Measure (R) for lexicographic diversity : 100*log(N) / (1 - v1/V)
    N = number of words in the text
    v1 = number of words that appears once
    V = number of unique words
    """

    def get_honore_measure(text: str) -> float:
        words = nltk.word_tokenize(text)
        N = len(words)
        V = len(set(words))
        words_count = Counter(words)
        v1 = len([_ for _, count in words_count.items() if count == 1])
        if v1 == V:
            return 10 ** 5  # big number, to prevent division by 0
        return 100 * np.log(N) / (1 - (v1 / V))

    return pd.DataFrame(
        x[TEXT_COLUMN_NAME]
        .astype(str)
        .swifter.apply(get_honore_measure)
        .rename("honore_measure")
    )


def hapax_disLegemena(x: pd.DataFrame) -> pd.DataFrame:
    """
    hapax disLegemena Measure for lexicographic diversity: v2/N
    v2 = number of words that appears twice
    N = number of words or number of unique words
    """

    def get_hapax_disLegemena(text: str) -> (float, float):
        words = nltk.word_tokenize(text)
        words_count = Counter(words)
        v2 = len([_ for _, count in words_count.items() if count == 2])
        return pd.Series(
            (v2 / (len(words) + EPSILON), v2 / (len(set(words)) + EPSILON))
        )

    return pd.DataFrame(
        x[TEXT_COLUMN_NAME].astype(str).swifter.apply(get_hapax_disLegemena)
    ).set_axis(["hapax_disLegemena(H)", "hapax_disLegemena(S)"], axis=1)


def Yules_characteristic(x: pd.DataFrame) -> pd.DataFrame:
    """
    Yules Characteristic (K) for lexicographic diversity:
    10,000 * (M - N) / N**2, M = sum_i i**2 * vi
    N = number of words
    i =  word frequency
    vi = number of words that appears i times
    """

    def get_Yules_characteristic(text: str) -> float:
        words = nltk.word_tokenize(text)
        words_count = Counter(words)
        freq_count = Counter(words_count.values())
        M = np.sum([(i * i) * freq_count[i] for i in words_count.values()])
        N = len(words) + EPSILON
        return 10000 * (M - N) / N ** 2

    return pd.DataFrame(
        x[TEXT_COLUMN_NAME]
        .astype(str)
        .swifter.apply(get_Yules_characteristic)
        .rename("Yules_characteristic")
    )


def simpsons_measure(x: pd.DataFrame) -> pd.DataFrame:
    """
    Simpson’s measure (D) for lexicographic diversity: 1 - (sum_n n(n-1))/N(N-1)
    N = number of words
    n = frequency of word
    """

    def get_simpsons_measure(text: str) -> float:
        words = nltk.word_tokenize(text)
        words_count = Counter(words)
        N = len(words)
        return 1 - np.sum([n * (n - 1) for n in words_count.values()]) / N * (
            N - 1
        )

    return pd.DataFrame(
        x[TEXT_COLUMN_NAME]
        .astype(str)
        .swifter.apply(get_simpsons_measure)
        .rename("simpsons_measure")
    )


def brunets_measure(x: pd.DataFrame) -> pd.DataFrame:
    """
    Brunet's Measure (W) for lexicographic diversity: (V - a) / log(N)
    N = number of words
    V = number of unique words
    a = constant - 0.17
    """

    def get_brunets_measure(text: str) -> float:
        words = nltk.word_tokenize(text)
        a = 0.17
        V = len(set(words))
        N = len(words) + EPSILON
        return (V - a) / (np.log(N))

    return pd.DataFrame(
        x[TEXT_COLUMN_NAME]
        .astype(str)
        .swifter.apply(get_brunets_measure)
        .rename("brunets_measure")
    )


def avg_word_frequency(x: pd.DataFrame) -> pd.DataFrame:
    """
    the average frequency of a word, based on zipf's law: mean(ceil(log(f(w*)/f(w))))
    f(w*) = number of appearances of most common word
    f(w) = number of appearances of word w
    """

    def get_avg_word_frequency(text: str) -> float:
        words = nltk.word_tokenize(text)
        words_count = Counter(words)
        maximum = np.max(list(words_count.values()))
        return np.average(
            [np.ceil(np.log(maximum / words_count[word])) for word in words]
        )

    return pd.DataFrame(
        x[TEXT_COLUMN_NAME]
        .astype(str)
        .swifter.apply(get_avg_word_frequency)
        .rename("avg_word_frequency")
    )


def entropy_over_words_frequencies(x: pd.DataFrame) -> pd.DataFrame:
    """
    the entropy over words frequencies: -sum(p1*log(pi))
    """

    def get_entropy(text: str) -> float:
        words = nltk.word_tokenize(text)
        words_count = Counter(words)
        distribution = np.array(list(words_count.values())) / (
            len(words) + EPSILON
        )
        return entropy(distribution, base=2)

    return pd.DataFrame(
        x[TEXT_COLUMN_NAME]
        .astype(str)
        .swifter.apply(get_entropy)
        .rename("entropy")
    )


def __syllable_count(word: str) -> int:
    """
    count the number of syllable in a word.
    function from
    https://github.com/Hassaan-Elahi/Writing-Styles-Classification-Using-Stylometric-Analysis/blob/master/Code/main.py
    """
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
    if count == 0:
        count += 1
    return count


def flesch_reading_ease_flesch_kincaid_grade_level(
    x: pd.DataFrame,
) -> pd.DataFrame:
    """
    readability score using Flesch Reading Ease and the Flesch Kincaid Grade Level.
    """

    def get_flesch_reading_ease_flesch_kincaid_grade_level(
        text: str,
    ) -> (float, float):
        words = nltk.word_tokenize(text)
        sentences = nltk.sent_tokenize(text)
        syllable_count = np.sum([__syllable_count(word) for word in words])
        flesch = (
            206.835
            - 1.015 * (len(words) / (len(sentences) + EPSILON))
            - 84.6 * (syllable_count / (len(words) + EPSILON))
        )
        flesch_kincaid = (
            0.39 * (len(words) / (len(sentences) + EPSILON))
            + 11.8 * (syllable_count / (len(words) + EPSILON))
            - 15.59
        )
        return pd.Series((flesch, flesch_kincaid))

    return pd.DataFrame(
        x[TEXT_COLUMN_NAME]
        .astype(str)
        .swifter.apply(get_flesch_reading_ease_flesch_kincaid_grade_level)
    ).set_axis(["flesch", "flesch_kincaid"], axis=1)


def gunning_fog_index(x: pd.DataFrame) -> pd.DataFrame:
    """
    the Gunning Fog index for text readability
    """

    def get_gunning_fog_index(text: str) -> float:
        words = stem_words(text)
        sentences = nltk.sent_tokenize(text)
        complex_words = [word for word in words if __syllable_count(word) >= 3]
        return 0.4 * (
            (len(words) / (len(sentences) + EPSILON))
            + 100 * (len(complex_words) / (len(words) + EPSILON))
        )

    return pd.DataFrame(
        x[TEXT_COLUMN_NAME]
        .astype(str)
        .swifter.apply(get_gunning_fog_index)
        .rename("gunning_fog_index")
    )
