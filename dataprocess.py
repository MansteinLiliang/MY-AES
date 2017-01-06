from __future__ import absolute_import
import codecs
file_path='./data/training_set_rel3.tsv'
import os
file_makes = set(['train','dev','test'])
import gensim
import numpy as np
import cPickle as pkl
import logging
emb_path = '/home/yhw/liliangpython/GoogleNews-vectors-negative300.bin'
total_data_path = './data/training_set_rel3.tsv'
prompt_vocab_path = './data/fold_0/train.tsv'
prompt = 1
word_embedding_size = 300
logger = logging.getLogger(__name__)
from dataset import tokenize

def vocab_init(file_path, prompt_id, tokenize_text, to_lower):
    logger.info('Creating vocabulary from: ' + file_path)
    total_words, unique_words = 0, 0
    word_freqs = {}
    with codecs.open(file_path, mode='r', encoding='UTF8', errors='ignore') as input_file:
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

    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    index = len(vocab)
    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
    return vocab

def w2v_process():
    total_vocab = vocab_init(total_data_path, 0, True, True)
    print 'total_data vocab shape = ' + str(len(total_vocab))
    model = gensim.models.Word2Vec.load_word2vec_format(emb_path, binary=True)
    model_vocab = model.vocab
    print 'model_vocab shape = '+ str(len(model_vocab))
    print total_vocab.keys()[:20]
    print model_vocab.keys()[:20]
    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    vcb_len = len(vocab)
    index = vcb_len

    word_embedding = {}
    word_embedding['<pad>'] = np.zeros((300,), dtype='float32')
    word_embedding['<unk>'] = np.random.uniform(-1.0, 1.0, (300,))
    word_embedding['<num>'] = np.random.uniform(-1.0, 1.0, (300,))
    for word in total_vocab.keys():
        if word in model_vocab:
            vocab[word] = index
            word_embedding[word] = model[word]
            index += 1
    pkl.dump(word_embedding, open('./data/word_vectors.pk', 'w'))
    pkl.dump(vocab, open('./data/total_vocab.pk', 'w'))
    print len(vocab)
    return

def init_all_files():
    for root, dirs, files in os.walk('./data'):
        for dir in dirs:
            for file_make in file_makes:
                file = codecs.open('./'+dir+'/'+file_make+'.tsv','w',encoding='UTF8')
                ids = codecs.open('./'+dir+'/'+file_make+'_ids.txt','r',encoding='UTF8')
                ids_dic = set()
                for id in ids:
                    ids_dic.add(int(id))
                ids = ids_dic
                count = 0
                with codecs.open(file_path, mode='r', encoding='UTF8', errors='ignore') as input_file:
                    file.write(input_file.next())
                    # input_file.next()
                    for line in input_file:
                        tokens = line.strip().split('\t')
                        essay_id = int(tokens[0])
                        assert essay_id != None
                        # print essay_id
                        if essay_id in ids:
                            count += 1
                            file.write(line)
                assert count == len(ids)
                file.close()
    return

w2v_process()