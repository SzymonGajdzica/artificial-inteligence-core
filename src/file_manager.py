import csv


def create_file_with_data(file_path, data_set):
    with open(file_path, mode='w', encoding="utf-8") as file:
        file.write('%s\t%s\n' % ('rating', 'comment'))
        for rating, comment in data_set:
            file.write('{0}\t{1}\n'.format(rating, comment))


def create_file_with_stats(file_path, data_set):
    with open(file_path, mode='w', encoding="utf-8") as file:
        file.write('{0}\t{1}\t{2}\n'.format('label', 'number_of_elements', 'part_of_all_elements'))
        for rating in ['1', '2', '3', '4', '5']:
            value = len(list(filter(lambda m_value: m_value[0] == rating, data_set)))
            file.write('{0}\t{1}\t{2}\n'.format(rating, value, value / float(len(data_set))))


def read_data(file_path):
    raw_data_set = set()
    with open(file_path, mode='r', encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter='\t', quotechar='"')
        for row in reader:
            raw_data_set.add((row['rating'], row['comment']))
    return raw_data_set
