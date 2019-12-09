import pandas as pd
from html import unescape
from re import compile, sub
from string import punctuation
from os import path
from itertools import groupby
import nltk
from nltk.corpus import stopwords as nltk_stopwords


ws_pattern = compile(r'\s+')
camelcase_split_pattern = compile(r'([a-zA-Z0-9#]+)([A-Z]+[a-z]+)')
web_url_pattern = compile(r"https?:\/\/[^ ^\r^\n]*")
first_single_end_pattern = compile(r'([!|?|.|;|:|,|-|_|*]{2,})$')
second_single_end_pattern = compile(r'(\s?[!|?|.|;|:|,|-|_|*]{2,})')
clean_end_pattern = compile(r'([!|?|.|;|:|,|-|_] [!|?|.|;|:|,|-|_])+')
rt_pattern = compile(r'\bRT\b')
at_pattern = compile(r'@\w+')
match_punctuation = compile(r'([.,/#!$%^&*;:{}=-_`~()])*\1')
end_pat = compile(r'.*[!|?|.|;]$')


def escape_html(text, unescape=unescape):
    try:
        return unescape(text)
    except:
        return text


def split_camelcase(text, camelcase_split_pattern=camelcase_split_pattern):
    return camelcase_split_pattern.sub(r'\1 \2', text)


def reduce_consecutive_letters(text, groupby=groupby):
    return ''.join(''.join(s)[:2] for _, s in groupby(text))


def delete_urls(text, web_url_pattern=web_url_pattern):
    return web_url_pattern.sub('', text)


def all_up_to_cap(text):
    for token in text.split(' '):
        if token.isupper():
            yield token.capitalize()
        else:
            yield token


def split_underscore(text):
    return text.replace('_', ' ')


def split_apostrophe(text):
    return text.replace("'", ' ')


def remove_multi_whitespace(text, sub=sub, ws_pattern=ws_pattern):
    return sub(ws_pattern, ' ', text).strip()


def replace_multi_punctuation(text, match_punctuation=match_punctuation):
    return match_punctuation.sub(r'\1', text)


def clean_end_signs(text, clean_end_pattern=clean_end_pattern):
    def end_replace(match):
        return match.groups()[0][-1]
    return clean_end_pattern.sub(end_replace, text)


def replace_RT_at_hash(text, rt_pattern=rt_pattern, at_pattern=at_pattern):
    text = rt_pattern.sub('', text)
    text = at_pattern.sub('', text)
    text = text.replace('#', '')
    return text


def clean_text(text, lower=True):
    text = replace_RT_at_hash(text)
    text = delete_urls(text)
    text = escape_html(text)
    text = split_camelcase(text)
    text = split_underscore(text)
    text = split_apostrophe(text)
    # text = reduce_consecutive_letters(text)
    # text = ' '.join(all_up_to_cap(text))
    text = remove_multi_whitespace(text)
    if lower:
        return text.lower()
    else:
        return text


def discard_ngrams_with_digits(ngrams):
    return [ngram for ngram in ngrams if not any(char.isdigit() for char in ngram)]

# def discard_ngrams_with_punctuation(ngrams):
#     return [ngram for ngram in ngrams if not any(char)]


def tokenize(text, remove_punctuation, stopwords=False, to_strip=punctuation + ' '):
    tokens = [token.strip(punctuation) for token in text.split(' ')]
    tokens = [token for token in tokens if token]
    if stopwords:
        if isinstance(stopwords, str):
            while True:
                try:
                    stopwords = nltk_stopwords.words(stopwords)
                    break
                except LookupError:
                    nltk.download("stopwords")
        tokens = [token for token in tokens if token.lower() not in stopwords]
    if remove_punctuation:
        tokens = [token for token in tokens if token not in punctuation]
    return tokens


def gramify(tokens, minimum, maximum, punctuation=punctuation, regex=compile(f"(^[{punctuation} ]*|[{punctuation} ]*$)"), remove_tokens_with_punctuation=True):
    if minimum == 1:
        grams = set(gram for gram in tokens if not any(c in punctuation for c in gram))
    else:
        grams = set()
    for n in range(max(minimum, 2), maximum + 1):
        for i in range(len(tokens) - n + 1):
            gram_tokens = tokens[i:i + n]
            gram = ' '.join(gram_tokens).strip()
            gram = regex.sub("", gram)
            if not remove_tokens_with_punctuation or not any(c in punctuation for c in gram):
                grams.add(gram)
    return grams


if __name__ == '__main__':
    # text = 'RT @JoeyClipstar 釋內文之英文單字均可再點入查詢'
    text = 'RT @JoeyClipstar: Bow_Woooow_Signs RT to #BadBoyRecords  - The Breakfast Club http://t.co/3w58p6Sbx2 RT http://t.co/LbQU2brfpf !!!!??'
    print(text)
    clean_text = clean_text(text)
    print(clean_text)
    tokens = tokenize(clean_text, 'zh')
    print(tokens)
    grams = gramify(tokens, 2, 3)
    print(grams)
