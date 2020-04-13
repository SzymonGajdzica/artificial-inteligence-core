import csv
from collections import defaultdict

from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from os import path
from chardet import detect
import emoji


def parse_comment(comment):
    return emoji.get_emoji_regexp().sub(u'', comment)\
        .replace(r'\n', ' ')\
        .replace(r'\u0026', '&')\
        .replace(r'\u200d', '')\
        .replace(r'\u003d', '=')\
        .replace(r'\t', ' ')\
        .replace('\t', ' ')\
        .replace('★', '')\
        .replace('♡', '')\
        .replace('️', ' ')\
        .replace('☆', '')\
        .replace('  ', ' ')\
        .strip()


def validate_rating(rating):
    try:
        return 5 >= int(rating) >= 1
    except ValueError:
        return False


def validate_comment(comment):
    return 3 <= len(comment) <= 600 and 'ascii' in str(detect(comment.encode("utf-8")))


def create_file_with_data(file_path, data_set, number_of_elements):
    with open(file_path, mode='w', encoding="utf-8") as file:
        for x in range(number_of_elements):
            rating, comment = data_set.pop()
            file.write('{0}\t{1}\n'.format(rating, comment))


def pre_process():
    data_set = set()
    filtered_data_set = set()
    labels_counter = defaultdict(lambda: 0)
    with open('input/result.tsv', mode='r', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter='\t', quotechar='"')
        for row in reader:
            if len(row) == 2:
                rating = row[0]
                comment = parse_comment(row[1])
                labels_counter[rating] = labels_counter[rating] + 1
                if validate_rating(rating) and validate_comment(comment):
                    data_set.add((rating, comment))
                else:
                    filtered_data_set.add((rating, comment))

    with open('data/stats.tsv', mode='w', encoding="utf-8") as file:
        file.write('{0}\t{1}\t{2}%\n'.format('label', 'number_of_elements', 'percent_of_all_elements'))
        for key, value in sorted(labels_counter.items(), key=lambda item: item[0]):
            file.write('{0}\t{1}\t{2}%\n'.format(key, value, int((value / float(len(data_set)) * 100.0))))

    create_file_with_data('data/valid.tsv', set(data_set), len(data_set))
    create_file_with_data('data/filtered.tsv', filtered_data_set, len(filtered_data_set))

    data_size = len(data_set)
    create_file_with_data('data/dev.tsv', data_set, int(data_size * 0.1))
    create_file_with_data('data/test.tsv', data_set, int(data_size * 0.1))
    create_file_with_data('data/train.tsv', data_set, len(data_set))


if __name__ == '__main__':
    if not path.isfile('data/dev.tsv') or not path.isfile('data/test.tsv') or not path.isfile('data/train.tsv'):
        pre_process()

    corpus = CSVClassificationCorpus(data_folder='data',
                                     column_name_map={0: "label", 1: "text"},
                                     delimiter='\t',
                                     test_file='test.tsv',
                                     dev_file='dev.tsv',
                                     train_file='train.tsv'
                                     ).downsample(0.1)

    if path.isfile('results/checkpoint.pt'):
        print("Starting from checkpoint")
        trainer = ModelTrainer.load_checkpoint('results/checkpoint.pt', corpus)
    else:
        word_embeddings = [WordEmbeddings('glove'),
                           FlairEmbeddings('news-forward'),
                           FlairEmbeddings('news-backward')
                           ]
        document_embeddings = DocumentRNNEmbeddings(embeddings=word_embeddings)
        classifier = TextClassifier(document_embeddings=document_embeddings,
                                    label_dictionary=corpus.make_label_dictionary()
                                    )
        trainer = ModelTrainer(classifier, corpus)

    trainer.train(base_path='results',
                  learning_rate=0.7,
                  mini_batch_size=16,
                  max_epochs=20,
                  checkpoint=True
                  )
