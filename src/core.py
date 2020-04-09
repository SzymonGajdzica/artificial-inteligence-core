import csv

from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from os import path
from chardet import detect


def parse_comment(comment):
    return comment.replace(r'\n', ' ')\
        .replace(r'\u0026', '&')\
        .replace(r'\u200d', ' ')\
        .replace(r'\u003d', '=')\
        .replace('👍', '')


def validate_rating(rating):
    try:
        return 5 >= int(rating) >= 1
    except ValueError:
        return False


def validate_comment(comment):
    return 'ascii' in str(detect(comment.encode("utf-8")))


def create_file_with_data(file_path, data_set, number_of_elements):
    with open(file_path, mode='w', encoding="utf-8") as file:
        for x in range(number_of_elements):
            rating, comment = data_set.pop()
            file.write('{0}\t{1}\n'.format(rating, comment))


def pre_process():
    data_set = set()
    filtered_data_set = set()
    with open('input/result.csv', mode='r', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter='\t', quotechar='"')
        for row in reader:
            if len(row) == 2:
                rating = parse_comment(row[0])
                comment = row[1]
                if validate_rating(rating) and validate_comment(comment):
                    data_set.add((rating, comment))
                else:
                    filtered_data_set.add((rating, comment))
    data_size = len(data_set)
    create_file_with_data('data/dev.csv', data_set, int(data_size * 0.1))
    create_file_with_data('data/test.csv', data_set, int(data_size * 0.1))
    create_file_with_data('data/train.csv', data_set, len(data_set))
    create_file_with_data('data/filtered.csv', filtered_data_set, len(filtered_data_set))


if __name__ == '__main__':
    if not path.isfile('data/dev.csv') or not path.isfile('data/test.csv') or not path.isfile('data/train.csv'):
        pre_process()

    corpus = CSVClassificationCorpus('data',
                                     {0: "label", 1: "text"},
                                     skip_header=False,
                                     delimiter='\t',
                                     test_file='test.csv',
                                     dev_file='dev.csv',
                                     train_file='train.csv'
                                     ).downsample(0.1)

    if path.isfile('results/checkpoint.pt'):
        print("Starting from checkpoint")
        trainer = ModelTrainer.load_checkpoint('results/checkpoint.pt', corpus)
    else:
        word_embeddings = [WordEmbeddings('glove'),
                           FlairEmbeddings('news-forward'),
                           FlairEmbeddings('news-backward')
                           ]
        document_embeddings = DocumentLSTMEmbeddings(word_embeddings,
                                                     hidden_size=512,
                                                     reproject_words=True,
                                                     reproject_words_dimension=256,
                                                     )
        classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary())
        trainer = ModelTrainer(classifier, corpus)

    trainer.train('results',
                  learning_rate=0.7,
                  mini_batch_size=32,
                  anneal_factor=0.5,
                  patience=3,
                  max_epochs=10,
                  checkpoint=True,
                  embeddings_storage_mode='gpu'
                  )
