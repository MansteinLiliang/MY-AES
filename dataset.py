# -*- coding: utf-8 -*-
import random
import codecs
import sys
import nltk
import logging
import re
import numpy as np
import pickle as pk
from keras.preprocessing import sequence
from sklearn.utils import shuffle
logger = logging.getLogger(__name__)
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'

asap_ranges = {
    0: (0, 60),
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60)
}


def get_ref_dtype():
    return ref_scores_dtype


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    new_tokens = []
    for index, token in enumerate(tokens):
        if token == '@' and (index + 1) < len(tokens):
            tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
            # tokens.pop(index)
        else:
            new_tokens.append(token)
    return new_tokens


def get_score_range(prompt_id):
    return asap_ranges[prompt_id]


def get_model_friendly_scores(scores_array, prompt_id_array):
    '''

    :param scores_array:
    :param prompt_id_array: int or np.ndarray
    :return:
    '''
    arg_type = type(prompt_id_array)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = asap_ranges[prompt_id_array]
        scores_array = (scores_array - low) / (high - low)
    else:
        assert scores_array.shape[0] == prompt_id_array.shape[0]
        dim = scores_array.shape[0]
        low = np.zeros(dim)
        high = np.zeros(dim)
        for ii in range(dim):
            low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
        scores_array = (scores_array - low) / (high - low)
    assert np.all(scores_array >= 0) and np.all(scores_array <= 1)
    return scores_array


def convert_to_dataset_friendly_scores(scores_array, prompt_id_array):
    arg_type = type(prompt_id_array)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = asap_ranges[prompt_id_array]
        scores_array = scores_array * (high - low) + low
        assert np.all(scores_array >= low) and np.all(scores_array <= high)
    else:
        assert scores_array.shape[0] == prompt_id_array.shape[0]
        dim = scores_array.shape[0]
        low = np.zeros(dim)
        high = np.zeros(dim)
        for ii in range(dim):
            low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
        scores_array = scores_array * (high - low) + low
    return scores_array


def is_number(token):
    return bool(num_regex.match(token))


def load_vocab(vocab_path):
    logger.info('Loading vocabulary from: ' + vocab_path)
    with open(vocab_path, 'rb') as vocab_file:
        vocab = pk.load(vocab_file)
    return vocab


def create_vocab(file_path, prompt_id, tokenize_text, to_lower, maxlen=0, vocab_size=0):
    logger.info('Creating vocabulary from: ' + file_path)
    if maxlen > 0:
        logger.info('  Removing sequences with more than ' + maxlen + ' words')
    total_words, unique_words = 0, 0
    word_freqs = {}
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[6])
            if essay_set == prompt_id or prompt_id <= 0:
                if to_lower:
                    content = content.lower()
                if tokenize_text:
                    content = tokenize(content)
                else:
                    content = content.split()
                if maxlen > 0 and len(content) > maxlen:
                    continue
                for word in content:
                    try:
                        word_freqs[word] += 1
                    except KeyError:
                        unique_words += 1
                        word_freqs[word] = 1
                    total_words += 1
    logger.info('  %i total words, %i unique words' % (total_words, unique_words))
    import operator
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
    if vocab_size <= 0:
        # Choose vocab size automatically by removing all singletons
        vocab_size = 0
        for word, freq in sorted_word_freqs:
            if freq > 1:
                vocab_size += 1
    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    vcb_len = len(vocab)
    index = vcb_len
    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        vocab[word] = index
        index += 1
    return vocab


def read_essays(file_path, prompt_id):
    logger.info('Reading tsv from: ' + file_path)
    essays_list = []
    essays_ids = []
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            if int(tokens[1]) == prompt_id or prompt_id <= 0:
                essays_list.append(tokens[2].strip())
                essays_ids.append(int(tokens[0]))
    return essays_list, essays_ids


def get_sents(string, vocab, doc_len=50, sent_len=50):
    """
    :param string:
    :return: ndarray:shape=(doc_len, sent_len), mask: shape=(doc_len, sent_len), num_hit, unk_hit, total
    """
    num_hit = 0
    unk_hit = 0
    total = 0
    sents = nltk.sent_tokenize(string)
    doc_matrix = np.zeros(shape=(doc_len, sent_len))
    mask = np.zeros(shape=(doc_len, sent_len))
    for i, sent in enumerate(sents):
        indices = []
        sent_mask = []
        if i>= doc_len:
            break
        tokens = nltk.word_tokenize(sent)
        for word in tokens:
            sent_mask.append(1)
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1
        # If maxlen is provided, any sequence longer
        doc_matrix[i,:], mask[i,:]  = sequence.pad_sequences([indices,sent_mask], sent_len, padding='post')
    # transpose the matrix set it shape=(sent , doc)
    return doc_matrix.transpose((1,0)), mask.transpose((1,0)), num_hit, unk_hit, total


def read_dataset(file_path, prompt_id, vocab, to_lower, score_index=6, char_level=False, doc_len=100, sent_len=50):
    '''
    :param file_path:
    :param prompt_id:
    :param vocab:
    :param to_lower:
    :param score_index:
    :param char_level:
    :return: data_x, mask_x, data_y, prompt_ids, maxlen_x
    '''
    logger.info('Reading dataset from: ' + file_path)
    logger.info('Removing sequences with more than ' + str(doc_len*sent_len) + ' words')
    data_x, data_y, mask_x, prompt_ids = [], [], [], []
    num_hit, unk_hit, total = 0., 0., 0.
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[score_index])
            if essay_set == prompt_id or prompt_id <= 0:
                if to_lower:
                    content = content.lower()

                # indices = np.zeros([doc_len, sent_len])
                indices, mask, n_hit, u_hit, tt = get_sents(content, vocab, doc_len, sent_len)
                num_hit += n_hit
                unk_hit +=u_hit
                total += tt
                mask_x.append(mask)
                data_x.append(indices)
                data_y.append(score)
    logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
    logger.info('max_sent_len of prompt%d is %d' % (prompt_id, sent_len))
    return data_x, mask_x, data_y, sent_len


def train_batch_generator(X, masks, y, doc_num=1):
    """
    :param X:
    :param masks:
    :param y:
    :param doc_num:if it's 1, the tensor_shape=(sent_len, sent_num).Or it's shape=(sent_len, sent_num*doc_len)
    :return: X_new, masks_new, y_new
    """
    epoch = 1
    while(True):
        X_new, masks_new, y_new = shuffle(X, masks, y)
        print "epoch: "+str(epoch)+" begin......"
        for i in xrange(doc_num,len(X_new),doc_num):
            yield epoch, np.hstack(X_new[i-doc_num:i]), np.hstack(masks_new[i-doc_num:i]), np.hstack(y_new[i-doc_num:i])
        epoch+=1


def dev_test_batch_generator(X, masks, y, doc_num=1):
    """
    按照doc_num的长度来生成batch，如果最后不足batch则是动态variable的长度
    :param X:
    :param masks:
    :param y:
    :param doc_num:
    :return:

    """
    for i in xrange(0,len(X),doc_num):
        #   I utilize the indexing trick, out of range index will automated detected
        yield np.hstack(X[i:i+doc_num]), np.hstack(masks[i:i+doc_num]), np.hstack(y[i:i+doc_num])


def get_data(paths, prompt_id, vocab_size, doc_len, sent_len, tokenize_text=True, to_lower=True, sort_by_len=False,
             vocab_path=None, score_index=6):
    train_path, dev_path, test_path = paths[0], paths[1], paths[2]

    if not vocab_path:
        vocab = create_vocab(train_path, prompt_id, tokenize_text, to_lower, vocab_size=vocab_size)
        if len(vocab) < vocab_size:
            logger.warning('The vocabualry includes only %i words (less than %i)' % (len(vocab), vocab_size))
        else:
            assert vocab_size == 0 or len(vocab) == vocab_size
    else:
        vocab = load_vocab(vocab_path)
        if len(vocab) != vocab_size:
            logger.warning(
                'The vocabualry includes %i words which is different from given: %i' % (len(vocab), vocab_size))
    logger.info('  Vocab size: %i' % (len(vocab)))

    train_x, train_masks, train_y, train_maxlen = read_dataset(train_path, prompt_id, vocab, to_lower, doc_len=doc_len, sent_len=sent_len)
    dev_x, dev_masks, dev_y, dev_maxlen = read_dataset(dev_path, prompt_id, vocab, to_lower, doc_len=doc_len, sent_len=sent_len)
    test_x, test_masks, test_y, test_maxlen = read_dataset(test_path, prompt_id, vocab, to_lower, doc_len=doc_len, sent_len=sent_len)

    return ((train_x, train_masks, np.array(train_y)), (dev_x, dev_masks, np.array(dev_y)),
            (test_x, test_masks, np.array(test_y)), vocab, len(vocab))
