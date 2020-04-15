import csv
from os import path

from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

from src.pre_process import pre_process


def load_weights():
    weights = dict()
    if not path.isfile('data/valid_stats.tsv'):
        return weights
    with open('data/valid_stats.tsv', mode='r', encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            weights[row['label']] = 1.0 / float(row['part_of_all_elements'])
    print('weights', weights)
    return weights


if __name__ == '__main__':
    if not path.isfile('data/dev.tsv') or not path.isfile('data/test.tsv') or not path.isfile('data/train.tsv'):
        pre_process(down_sample=0.2,
                    equal_sets=True
                    )

    corpus = CSVClassificationCorpus(data_folder='data',
                                     column_name_map={0: "label", 1: "text"},
                                     delimiter='\t',
                                     skip_header=True,
                                     test_file='test.tsv',
                                     dev_file='dev.tsv',
                                     train_file='train.tsv'
                                     )

    if path.isfile('results/checkpoint.pt'):
        print("Starting from checkpoint")
        trainer = ModelTrainer.load_checkpoint('results/checkpoint.pt', corpus)
    else:
        word_embeddings = [WordEmbeddings('glove'),
                           FlairEmbeddings('news-forward'),
                           FlairEmbeddings('news-backward')
                           ]
        document_embeddings = DocumentRNNEmbeddings(embeddings=word_embeddings,
                                                    hidden_size=512,
                                                    reproject_words=True,
                                                    reproject_words_dimension=256
                                                    )
        weights = load_weights()
        classifier = TextClassifier(document_embeddings=document_embeddings,
                                    label_dictionary=corpus.make_label_dictionary(),
                                    loss_weights=weights
                                    )
        trainer = ModelTrainer(classifier, corpus)

    trainer.train(base_path='results',
                  learning_rate=0.7,
                  mini_batch_size=32,
                  anneal_factor=0.5,
                  patience=3,
                  max_epochs=30,
                  checkpoint=True
                  )
