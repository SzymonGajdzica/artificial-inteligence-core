from collections import defaultdict
from random import shuffle

import emoji
import tqdm
from chardet import detect

from src.file_manager import create_file_with_data, read_data, create_file_with_stats


def parse_comment(comment):
    return emoji.get_emoji_regexp().sub(u'', comment) \
        .replace(r'\n', ' ') \
        .replace(r"\'", "'") \
        .replace(r'\u200d', '') \
        .replace(r'\u200b', '') \
        .replace(r'\u0000', '') \
        .replace(r'\u0026', '&') \
        .replace(r'\u003e', '>') \
        .replace(r'\u003c', '<') \
        .replace(r'\u003d', '=') \
        .replace(r'\t', ' ') \
        .replace('\t', ' ') \
        .replace('️', ' ') \
        .replace('  ', ' ') \
        .strip()


def validate_rating(rating):
    try:
        return 5 >= int(rating) >= 1
    except ValueError:
        return False


def validate_comment(comment):
    return 10 <= len(comment) <= 600 and 'ascii' in str(detect(comment.encode("utf-8"))) and '\\' not in comment


def process_and_filter_data(raw_data_set, down_sample, equal_sets):
    filtered_data_set = set()
    data_dictionary_sets = defaultdict(set)
    with tqdm.trange(len(raw_data_set)) as bar:
        for rating, comment in raw_data_set:
            bar.update(1)
            parsed_comment = parse_comment(comment)
            if validate_rating(rating) and validate_comment(parsed_comment):
                data_dictionary_sets[rating].add((rating, parsed_comment))
            else:
                filtered_data_set.add((rating, parsed_comment))
    if equal_sets:
        minimal_size = min((len(data_set) for data_set in data_dictionary_sets.values()))
        for key in data_dictionary_sets:
            data_dictionary_sets[key] = set(list(data_dictionary_sets[key])[:minimal_size])
    for key in data_dictionary_sets:
        data_dictionary_sets[key] = set(
            list(data_dictionary_sets[key])[:int(len(data_dictionary_sets[key]) * down_sample)])
    data_list = [
        single_data_set_element
        for single_data_set in data_dictionary_sets.values()
        for single_data_set_element in single_data_set
    ]
    shuffle(data_list)
    return data_list, list(filtered_data_set)


def pre_process(down_sample=1.0, equal_sets=False):
    print('Pre-processing data with parameters: down_sample={0}, equal_sets={1}'.format(down_sample, equal_sets))
    raw_data_set = read_data('input/result.tsv')

    data_list, filtered_data_list = process_and_filter_data(raw_data_set, down_sample, equal_sets)

    create_file_with_stats('data/valid_stats.tsv', data_list)
    create_file_with_stats('data/filtered_stats.tsv', filtered_data_list)

    create_file_with_data('data/valid.tsv', data_list)
    create_file_with_data('data/filtered.tsv', filtered_data_list)

    create_file_with_data('data/dev.tsv', data_list[:int(len(data_list) * 0.1)])
    create_file_with_data('data/test.tsv', data_list[int(len(data_list) * 0.1) + 1:int(len(data_list) * 0.2) - 1])
    create_file_with_data('data/train.tsv', data_list[int(len(data_list) * 0.2):])
