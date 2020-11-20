import re
import os
from tqdm import tqdm
import logging
import pandas as pd
from preprocess.preprocess import load_stopwords, seg_sentence
from gensim import corpora, models, similarities
from w2v_wmd.dataProcess import seg_sentence, load_stopwords
from utils import read_pkl_file, save_pkl_file, eval_MPP

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

data_dir = "../data/"
subwayqq_path = data_dir + "documents.csv"
tfidf_file_dir = data_dir + "tfidf/"
stopwords_file = data_dir + 'punct.txt'


class TfIdf_Model:
    def __init__(self, docs,  tfidf_file_dir, stop_words_file=None):
        logger.info('TfIdf_Model is initializing...')
        self.docs = docs
        self.tfidf_file_dir = tfidf_file_dir
        self.stop_words_file = stop_words_file

        # if not os.path.exists(self.seg_file_dir):
        #     os.mkdir(self.seg_file_dir)
        if self.stop_words_file:
            self.stopwords = load_stopwords(stop_words_file)
        if not os.path.exists(tfidf_file_dir):
            os.mkdir(tfidf_file_dir)

    # compute tfidf and save to file
    def compute_tfidf(self, tfidf_file, tfidf_dictionary_file, docs_tfidf_file):

        texts = []
        for question in self.docs:
            texts.append(self.preprocess_data(question))
        # texts = []
        # for cur_data in self.data:
        #     texts.append(cur_data["query_seg"])

        dictionary = corpora.Dictionary(texts)
        feature_cnt = len(dictionary.token2id)
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(corpus)
        docs_tfidf = tfidf[corpus]

        save_pkl_file(tfidf, tfidf_file)
        save_pkl_file(dictionary, tfidf_dictionary_file)
        save_pkl_file(docs_tfidf, docs_tfidf_file)

        return tfidf, docs_tfidf, dictionary

    # init search engine
    def init_search_engine(self):
        tfidf_file = os.path.join(self.tfidf_file_dir, "tfidf_model.pkl")
        tfidf_dictionary_file = os.path.join(self.tfidf_file_dir, "tfidf_dictionary.pkl")
        docs_tfidf_file = os.path.join(self.tfidf_file_dir, "docs_tfidf.pkl")
        if os.path.exists(tfidf_file):
            self.tfidf_model = read_pkl_file(tfidf_file)
            self.dictionary = read_pkl_file(tfidf_dictionary_file)
            self.docs_tfidf = read_pkl_file(docs_tfidf_file)
        else:
            self.tfidf_model, self.docs_tfidf, self.dictionary = self.compute_tfidf(tfidf_file, tfidf_dictionary_file, docs_tfidf_file)

        self.cosine_similar = similarities.MatrixSimilarity(self.docs_tfidf, num_features=len(self.dictionary.token2id))
        # self.cosine_similar = similarities.SparseMatrixSimilarity(self.docs_tfidf, num_features=len(self.dictionary.token2id))


    # preprocess query
    def preprocess_data(self, sentence):
        if self.stop_words_file:
            sentence_words = seg_sentence(sentence, self.stopwords)
        else:
            sentence_words = seg_sentence(sentence)

        return sentence_words

    # search releated top n file, you can try to use min-heap to implement it.
    # but here we will use limited insertion
    def search_related_files(self, query, top_k):
        query_words = self.preprocess_data(query)
        top_docs = []
        query_bow = self.dictionary.doc2bow(query_words)
        query_tfidf = self.tfidf_model[query_bow]
        sims = self.cosine_similar[query_tfidf]
        similars = []
        for index, sim in enumerate(sims.tolist()):
            similars.append((index, sim))
        similars = sorted(similars, key=lambda x: x[1], reverse=True)
        for item in similars[:top_k]:
            index = item[0]
            sim = item[1]
            top_docs.append((index, self.docs[index], sim))
        return top_docs


def eval(queries, tfIdf_model, topk):
    labels = []
    pred = []
    with tqdm(total=len(queries)) as pbar:
        for query, qid in queries:
            pre = []
            labels.append(qid)
            top_docs = tfIdf_model.search_related_files(query, topk)
            for index, question, sim in top_docs:
                pre.append(index)
            pred.append(pre)
            pbar.update(1)
    MRR = eval_MPP(labels, pred)
    # print("MRR:{}".format(MRR))
    logger.info("MRR:{}".format(MRR))

if __name__ == '__main__':
    all_data = pd.read_csv(subwayqq_path)
    # with open(subwayqq_path, "rb") as fin:
    #     all_data = pickle.load(fin)
    queries = []
    docs = all_data["docs"].drop_duplicates().reset_index(drop=True).values.tolist()
    for index, row in all_data.iterrows():
        query = row["similar_query"]
        question = row["standard_question"]
        q_id = docs.index(question)
        queries.append((query, q_id))
    # initialize document manager
    tfIdf_model = TfIdf_Model(docs, tfidf_file_dir, stopwords_file)
    # # initialzie search engine
    tfIdf_model.init_search_engine()
    eval(queries, tfIdf_model, topk=10)


    # 单个查询
    # query = queries[0]
    # # # search query and get top documents with weight
    # top_docs = tfIdf_model.search_related_files(query, 10)
    # print('query is: ', query)
    # print('result is: ')
    # print(top_docs)