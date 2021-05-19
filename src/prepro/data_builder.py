import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin

import torch
# 用fairseq的时候就用multiprocessing
from multiprocessing import Pool

from others.logging import logger
# from others.tokenization import BertTokenizer
# fairseq环境去掉这句话。
from transformers import BertTokenizer, BasicTokenizer, AutoTokenizer

# from pytorch_transformers import BertTokenizer
from pytorch_transformers import XLNetTokenizer

from others.utils import clean
from others.rouge_not_a_wrapper import avg_rouge
from prepro.utils import _get_word_ngrams

import xml.etree.ElementTree as ET

nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]


def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)



def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            tgt.append([])
            continue
        if (flag):
            tgt[-1].extend(tokens)
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt



def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter('hedline'))
    if (len(title_node) > 0):
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter('hl1'))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter('byline'))
    byline_node = [n for n in byline_node if n.attrib['class'] == 'normalized_byline']
    if (len(byline_node) > 0):
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter('abstract'))
    if (len(abs_node) > 0):
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter('p'))][0]
        except:
            print(p)

    else:
        return None, None
    abs = ' '.join(abs).split(';')
    abs[-1] = abs[-1].replace('(m)', '')
    abs[-1] = abs[-1].replace('(s)', '')

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace('(' + ww + ')', '')
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter('block'):
        att = doc_node.get('class')
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if (att == 'full_text'):
            paras = [p.text.lower().split() for p in list(doc_node.iter('p'))]
            break
    if (len(paras) > 0):
        if (len(byline) > 0):
            paras = [title + ['[unused4]'] + byline + ['[unused5]']] + paras
        else:
            paras = [title + ['[unused4]']] + paras

        return paras, abs
    else:
        return None, None


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    # java edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit -ssplit.newlineIsSentenceBreak always -filelist mapping_for_corenlp.txt -outputFormat json -outputDirectory ./output_dir -props StanfordCoreNLP-chinese.properties -cp "*" -Xmx2g
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, args):
        self.args = args
        if args.bart:
            self.tokenizer = AutoTokenizer.from_pretrained('/home/ybai/downloads/bart', do_lower_case=True,
                                                      cache_dir='../tmp/', local_files_only=False)


            self.sep_token = '</s>'
            self.cls_token = '<s>'
            self.pad_token = '<pad>'
            self.tgt_bos = 'madeupword0000'
            self.tgt_eos = 'madeupword0001'
            self.tgt_sent_split = 'madeupword0002'
            self.sep_vid = self.tokenizer.encoder[self.sep_token]
            self.cls_vid = self.tokenizer.encoder[self.cls_token]
            self.pad_vid = self.tokenizer.encoder[self.pad_token]
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True, cache_dir='../tmp/', local_files_only=True, tokenize_chinese_chars=False)

            self.eng_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir='../tmp/', local_files_only=True)
            self.chinese_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=True, cache_dir='../temp/', local_files_only=True, tokenize_chinese_chars=False)
            self.basic_word_tokenizer = BasicTokenizer( do_lower_case=True, never_split=None, tokenize_chinese_chars=False)
            self.whole_word_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking", do_lower_case=True, cache_dir='../tmp/', local_files_only=True, tokenize_chinese_chars=False)
            # self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True, cache_dir='../temp/')

            self.sep_token = '[SEP]'
            self.cls_token = '[CLS]'
            self.pad_token = '[PAD]'
            self.tgt_bos = '[unused1]'
            self.tgt_eos = '[unused2]'
            self.tgt_sent_split = '[unused3]'
            self.sep_vid = self.tokenizer.vocab[self.sep_token]
            self.cls_vid = self.tokenizer.vocab[self.cls_token]
            self.pad_vid = self.tokenizer.vocab[self.pad_token]

    # 英中版本
    def preprocess(self, src, tgt, tgt_eng, sent_labels, use_bert_basic_tokenizer=False, is_test=False):
        '''
    # 正常版本
    # def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False, language=None):
        # 正常
        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        # 分句符号的位置
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]

        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        if language == None:
            tgt_subtokens_str = '[unused1] ' + ' [unused3] '.join(
                [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused2]'
            # 为gigaword数据加入padn
        elif language == 'en':
            # tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            #     [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for
            #      tt in tgt])
            tgt_subtokens_str = '[unused1] ' + ' [unused3] '.join(
                [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for
                 tt in tgt])
            len0 = len(tgt_subtokens_str.split()) - 1
            tgt_subtokens_str += ' [unused5]' + ' [unused10]' * len0
            tgt_subtokens_str += ' [unused2]'


            # print('tgt_subtokens_str = ', tgt_subtokens_str)
            # exit()

        else:
            # 在末尾去掉unused1，补充n个unused10,再加上unused1
            tgt_subtokens_str = ''
            content_str = ' [unused3] '.join(
                [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for
                 tt in tgt]) + ' [unused2]'
            len0 = len(content_str.split())
            tgt_subtokens_str += '[unused1] '
            tgt_subtokens_str += ' [unused10]' * (len0 - 1)
            tgt_subtokens_str += ' [unused5] '
            tgt_subtokens_str += content_str



            # tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            #     [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for
            #      tt in tgt])
            # tgt_subtokens_str += ' [unused1]'


            # print('tgt_subtokens_str = ', tgt_subtokens_str)
            # exit()


        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]

        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
        # print("indexs =  ", tgt_subtoken_idxs)
        # exit()


        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt
'''

# 英中版本
        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        # 如果是中英的话,这里不要sent_labels
        # for l in sent_labels:
        #     _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]

        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        # 改成了英文
        src_subtokens = self.tokenizer.tokenize(text)
        # print("src_subtokens ", src_subtokens)
        #
        # 这一步是为了处理整个词而做的代码
        # src_subtokens_new = []
        # for i, each in enumerate(src_subtokens):
        #     if each == 'cls' or each == 'sep':
        #         each = '[' + each.upper() + ']'
        #         src_subtokens_new.append(each)
        #     elif each == '[' and (src_subtokens[i + 1] == 'cls' or  src_subtokens[i + 1] == 'sep'):
        #         continue
        #     elif each == ']' and i > 0 and (src_subtokens[i - 1] == 'cls' or src_subtokens[i - 1] == 'sep'):
        #         continue
        #     else:
        #         src_subtokens_new.append(each)
        #
        # src_subtokens = src_subtokens_new

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        # print("src = ", text)
        # print("src_tokens = ", src_subtokens)
        # print("src_tokens idxs= ", src_subtoken_idxs)
        # exit()
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        # print("src = ", src)
        # print("_segs = ", _segs)
        # print("segs = ", segs)
        # exit()
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        # print("src = ", len(src_subtoken_idxs))
        # print(src_subtoken_idxs)
        # print("_segs = ", _segs)
        # print("segs = ", segs)
        # print("sef_ids = ", len(segments_ids))
        # print(segments_ids)

        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]



        # 这里处理full data的时候脑子抽了，用的中文的目标词典，不过eng_chn也需要中文目标词典
        if self.args.bart:
            tgt_subtokens_str = 'madeupword0000 ' + ' madeupword0002 '.join(
                [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for
                 tt in tgt]) + ' madeupword0001'
            # 上边是正常代码，下边我想把segment删掉试试rouge-l会不会变大
            # tgt_subtokens_str = '[unused1] ' + ' '.join(
            #     [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused2]'

            # 加入拼接目标的代码，仅拼接id，要放在convert_tokens_to_ids前边，并且要先英文后中文
            tgt_subtokens_str = ('madeupword0000 ' + ' madeupword0002 '.join(
                [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for
                 tt
                 in tgt_eng]) + ' <mask> ') + tgt_subtokens_str
        else:
            tgt_subtokens_str = '[unused1] ' + ' [unused3] '.join(
                [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused2]'
        # 上边是正常代码，下边我想把segment删掉试试rouge-l会不会变大
        # tgt_subtokens_str = '[unused1] ' + ' '.join(
        #     [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused2]'


        # 加入拼接目标的代码，仅拼接id，要放在convert_tokens_to_ids前边，并且要先英文后中文
            tgt_subtokens_str = ('[unused1] ' + ' [unused3] '.join(
                [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt
                 in tgt_eng]) + ' [unused5] ') + tgt_subtokens_str

        # 拼接但前边的英文部分为pad的部分。
        # tgt_subtokens_str = ('[unused1] ' + ' [unused3] '.join(
        #     [' '.join([' [PAD] ' for word in tt]) for tt
        #      in tgt_eng]) + ' [unused5] ') + tgt_subtokens_str

        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]

        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken) # + 30522
        # print("tgt_subtokens_str = ", tgt_subtokens_str)
        # print("tgt subtoken_idxs  = ", tgt_subtoken_idxs)
        # exit()


        # 加tgt_embedding
        _segs = [-1] + [i for i, t in enumerate(tgt_subtoken_idxs) if t == 5] + [len(tgt_subtoken_idxs) - 1]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        tgt_segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                tgt_segments_ids += s * [0]
            else:
                tgt_segments_ids += s * [1]
        # print("tgt = ", len(tgt_subtoken_idxs))
        # print(tgt_subtoken_idxs)
        # print("_segs = ", _segs)
        # print("segs = ", segs)
        # print("sef_ids = ", len(tgt_segments_ids))
        # print(tgt_segments_ids)
        # exit()


        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]


        # tgt_txt +=

        # code for tgt_eng
        if self.args.bart:
            tgt_subtokens_str_eng = 'madeupword0000 ' + ' madeupword0002 '.join(
                [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt_eng]) + ' madeupword0001'
        else:
            tgt_subtokens_str_eng = '[unused1] ' + ' [unused3] '.join(
                [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt_eng]) + ' [unused2]'
        tgt_subtoken_eng = tgt_subtokens_str_eng.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken_eng) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs_eng = self.tokenizer.convert_tokens_to_ids(tgt_subtoken_eng)

        tgt_txt_eng = '<q>'.join([' '.join(tt) for tt in tgt_eng])


        # print("tgt_subtoken_idxs = ", tgt_subtoken_idxs)
        # print("tgt_txt = ", tgt_txt)
        # exit()




        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, tgt_subtoken_idxs_eng, tgt_txt_eng, tgt_segments_ids

        # enen版本需要把tgt和tgteng调过来
        # print("tgteng-------------------------------------------------------", end='\r')
        # return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs_eng, segments_ids, cls_ids, src_txt, tgt_txt_eng, tgt_subtoken_idxs, tgt_txt, tgt_segments_ids


def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_bert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)
    # print('json file = ', json_file)
    # exit()

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))

    datasets = []
    for d in jobs:
        # 英中
        # source, tgt, tgt_eng = d['src'], d['tgt'], d['tgt_eng']
        # sent_labels = greedy_selection(source[:args.max_src_nsents], tgt_eng, 3)
        # if (args.lower):
        #     source = [' '.join(s).lower().split() for s in source]
        #     tgt = [' '.join(s).lower().split() for s in tgt]
        #     tgt_eng = [' '.join(s).lower().split() for s in tgt_eng]
        # b_data = bert.preprocess(source, tgt, tgt_eng, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        # 英德，当时搞数据集的时候两个目标搞反了，尴尬
        source, tgt_eng, tgt = d['src'], d['tgt'], d['tgt_eng']
        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt_eng, 3)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
            tgt_eng = [' '.join(s).lower().split() for s in tgt_eng]
        b_data = bert.preprocess(source, tgt, tgt_eng, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        # 中英
        # source, tgt, tgt_chn = d['src'], d['tgt'], d['tgt_chn']
        # sent_labels=None
        # if (args.lower):
        #     source = [' '.join(s).lower().split() for s in source]
        #     tgt = [' '.join(s).lower().split() for s in tgt]
        #     tgt_chn = [' '.join(s).lower().split() for s in tgt_chn]
        # b_data = bert.preprocess(source, tgt, tgt_chn, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)



        #正常版本
        # source, tgt = d['src'], d['tgt']
        #
        # sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        # if (args.lower):
        #     source = [' '.join(s).lower().split() for s in source]
        #     tgt = [' '.join(s).lower().split() for s in tgt]
        # if 'en' in json_file:
        #     b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
        #                              is_test=is_test, language='en')
        # elif 'fr' in json_file:
        #     b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
        #                              is_test=is_test, language = 'fr')
        # elif 'zh' in json_file:
        #     b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
        #                              is_test=is_test, language = 'zh')
        if (b_data is None):
            continue

        # 普通版本
        # src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        # b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
        #                "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
        #                'src_txt': src_txt, "tgt_txt": tgt_txt}

        # 英中 这里要注意事实上如果是中英的话，所有的语言已经有所变化了
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, tgt_subtoken_idxs_eng, tgt_txt_eng, tgt_seg_ids = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs, "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt, "tgt_eng": tgt_subtoken_idxs_eng, "tgt_txt_eng": tgt_txt_eng, 'tgt_segs':tgt_seg_ids}
        # 英英
        # src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, tgt_subtoken_idxs_eng, tgt_txt_eng, tgt_seg_ids = b_data
        # b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs, "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
        #                'src_txt': src_txt, "tgt_txt": tgt_txt, "tgt_eng": tgt_subtoken_idxs_eng, "tgt_txt_eng": tgt_txt_eng, 'tgt_segs':tgt_seg_ids}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
    corpus_mapping = {}
    for corpus_type in ['valid', 'test', 'train']:
        temp = []
        for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        real_name = f.split('/')[-1].split('.')[0]
        if (real_name in corpus_mapping['valid']):
            valid_files.append(f)
        elif (real_name in corpus_mapping['test']):
            test_files.append(f)
        elif (real_name in corpus_mapping['train']):
            train_files.append(f)
        # else:
        #     train_files.append(f)

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset, ensure_ascii=False))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset, ensure_ascii=False))
                p_ct += 1
                dataset = []


def _format_to_lines(params):
    f, args = params
    # print(f)
    source, tgt = load_json(f, args.lower)
    return {'src': source, 'tgt': tgt}

def format_to_lines_xgiga(args):
    json_dir_init = os.path.abspath(args.raw_path)
    dataset_split = ['test' ,'valid', 'train']
    train_files, valid_files, test_files = [],[],[]
    # 把各自的文件保存好
    for data_sp in dataset_split:
        if data_sp == 'train':
            train_files = os.listdir(os.path.join(json_dir_init, data_sp))
        elif data_sp == 'test':
            test_files = os.listdir(os.path.join(json_dir_init, data_sp))
        elif data_sp == 'valid':
            valid_files = os.listdir(os.path.join(json_dir_init, data_sp))

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        # if corpus_type != "valid":
        a_lst = [(os.path.join(json_dir_init, corpus_type, f),args) for f, args in a_lst]
        # else:
        # a_lst = [(os.path.join(json_dir_init, 'dev', f),args) for f, args in a_lst]

        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset, ensure_ascii=False))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset, ensure_ascii=False))
                p_ct += 1
                dataset = []

def format_to_lines_new(args):
    json_dir_init = os.path.abspath(args.raw_path)
    dataset_split = ['test' ,'valid', 'train']
    lang_split = ['eng', 'chn']
    train_files, valid_files, test_files = {},{},{}
    # 把各自的文件保存好
    for data_sp in dataset_split:
        for lan_sp in lang_split:
            if data_sp == 'train':
                train_files[lan_sp] = os.listdir(os.path.join(json_dir_init, data_sp, lan_sp))
            elif data_sp == 'test':
                test_files[lan_sp] = os.listdir(os.path.join(json_dir_init, data_sp, lan_sp))
            elif data_sp == 'valid':
                valid_files[lan_sp] = os.listdir(os.path.join(json_dir_init, data_sp, lan_sp))

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}

    for corpus_type in ['train', 'valid', 'test']:
        # alist是需要作用的参数，第一个是f(所有文件名列表)，第二个是args
        # a_lst = [(f_eng, corpora[corpus_type]['chn'][i], args) for i, f_eng in enumerate(corpora[corpus_type]['eng'])]
        a_lst = [(f_eng, ''.join(f_eng.split('.')[:-2]) + '.chnref.json', args) for f_eng in corpora[corpus_type]['eng']]


        # exit()
        # for each in a_lst:
        #     print(each[0],end='\r')
        #     # print(each[1], end='\r')
        #     assert each[0].split('.')[0] == each[1].split('.')[0]
        #     assert each[1] in corpora[corpus_type]['chn']
    # exit()

        a_lst = [(os.path.join(json_dir_init, corpus_type, 'eng', f_eng) , os.path.join(json_dir_init, corpus_type, 'chn', f_chn) ,args) for f_eng, f_chn, args in a_lst]


        # a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines_new, a_lst):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    print("saving to file ", pt_file)
                    save.write(json.dumps(dataset, ensure_ascii=False))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                print("saving to file ", pt_file)
                save.write(json.dumps(dataset, ensure_ascii=False))
                p_ct += 1
                dataset = []

    # bert = BertData(args)
    # print("mode format to lines new")
    # print(args.test_file)
    # print("_format_to_lines_new的结果是：")
    # print(_format_to_lines_new((args.test_file, args)))
    # json0 = _format_to_lines_new((args.test_file, args))
    # tgt =json0['tgt']
    # tgt = [' '.join(s).lower().split() for s in tgt]
    #
    # print("bert tokenize的结果")
    # tgt_subtokens_str = '[unused1] ' + ' [unused3] '.join(
    #     [' '.join(bert.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=False)) for tt in
    #      tgt]) + ' [unused2]'
    # print('tgt_subtokens_str: ', tgt_subtokens_str)
    # tgt_subtoken = tgt_subtokens_str.split()[:bert.args.max_tgt_ntokens]
    # print('tgt_subtoken: ', tgt_subtoken)
    #
    # tgt_subtoken_idxs = bert.tokenizer.convert_tokens_to_ids(tgt_subtoken)
    # tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
    # print('tgt_subtoken_idxs: ', tgt_subtoken_idxs)
    return


def _format_to_lines_new(params):
    f_eng, f_chn, args = params
    # 这里如果是中文的话，其实是一样的，把中英反过来较好了
    # 英中
    # source, tgt_eng = load_json(f_eng, args.lower)
    # _, tgt = load_json(f_chn, args.lower)
    # return {'src': source, 'tgt': tgt, 'tgt_eng': tgt_eng}
    # 中英
    source, tgt_chn = load_json(f_chn, args.lower)
    if len(tgt_chn) > 1:
        tgt = [tgt_chn[0]]
        tgt_chn = tgt_chn[1:]
    else:
        _, tgt = load_json(f_eng, args.lower)


    return {'src': source, 'tgt': tgt, 'tgt_chn': tgt_chn}

# 用的时候直接指定giga的所有文件夹
def tokenize_xgiga(args):
    stories_dir_init = os.path.abspath(args.raw_path)
    dataset_split = ['test', 'train', 'dev']
    # lang_split = ['fr', 'zh']
    lang_split = ['en', 'fr', 'zh']
    # 处理新数据把里边的每一个文件变成文件里的每一行即可
    print(os.listdir(args.raw_path))
    for data_sp in dataset_split:
        for lan_sp in lang_split:
            assert os.path.isdir(os.path.join(stories_dir_init, lan_sp, data_sp))
    # exit()
            # assert os.path.isdir(os.path.join(stories_dir_init, data_sp, lan_sp))
    # absolute dir
    tokenized_stories_dir_init = os.path.abspath(args.save_path)
    if not os.path.isdir(tokenized_stories_dir_init):
        os.mkdir(tokenized_stories_dir_init)
    # 建文件夹
    for lan_sp in lang_split:
        tmp_path0 = os.path.join(tokenized_stories_dir_init, lan_sp)
        if not os.path.isdir(tmp_path0):
            os.mkdir(tmp_path0)
        for data_sp in dataset_split:
            tmp_path = os.path.join(tokenized_stories_dir_init, lan_sp, data_sp)
            if not os.path.isdir(tmp_path):
                os.mkdir(tmp_path)


    for lan_sp in lang_split:
        for data_sp in dataset_split:
            stories_dir = os.path.join(stories_dir_init, lan_sp, data_sp)
            tokenized_stories_dir = os.path.join(tokenized_stories_dir_init, lan_sp, data_sp)
            stories = os.listdir(stories_dir)

            # make IO list file
            print("Making list of files to tokenize...")
            with open("mapping_for_corenlp.txt", "w") as f:
                for s in stories:
                    if (not s.endswith('story') and not s.endswith('chnref')):
                        continue
                    f.write("%s\n" % (os.path.join(stories_dir, s)))

            if lan_sp == 'en':
                print("Preparing to tokenize english %s to %s..." % (stories_dir, tokenized_stories_dir))
                command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                           '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
                           'json', '-outputDirectory', tokenized_stories_dir]
            elif lan_sp == 'fr':
                print("Preparing to tokenize french %s to %s..." % (stories_dir, tokenized_stories_dir))
                command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                           '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt',
                           '-outputFormat',
                           'json', '-outputDirectory', tokenized_stories_dir, '-props',
                           'StanfordCoreNLP-french.properties', '-cp', '\"*\"', '-Xmx2g']
            else:
                print("Preparing to tokenize chinese %s to %s..." % (stories_dir, tokenized_stories_dir))
                command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                           '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt',
                           '-outputFormat',
                           'json', '-outputDirectory', tokenized_stories_dir, '-props',
                           'StanfordCoreNLP-chinese.properties', '-cp', '\"*\"', '-Xmx2g']

                # java edu.stanford.nlp.pipeline.StanfordCoreNLP
                # -annotators tokenize,ssplit -ssplit.newlineIsSentenceBreak always
                # -filelist mapping_for_corenlp.txt -outputFormat json
                # -outputDirectory ./output_dir -props StanfordCoreNLP-chinese.properties -cp "*" -Xmx2g



            print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
            subprocess.call(command)
            print("Stanford CoreNLP Tokenizer has finished.")
            os.remove("mapping_for_corenlp.txt")

            # Check that the tokenized stories directory contains the same number of files as the original directory
            num_orig = len(os.listdir(stories_dir))
            num_tokenized = len(os.listdir(tokenized_stories_dir))
            if num_orig != num_tokenized:
                raise Exception(
                    "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                        tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
            print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

def tokenize_new(args):
    stories_dir_init = os.path.abspath(args.raw_path)
    dataset_split = ['test', 'train', 'valid']
    lang_split = ['eng', 'chn']
    # 处理新数据把里边的每一个文件变成文件里的每一行即可
    for data_sp in dataset_split:
        for lan_sp in lang_split:
            assert os.path.isdir(os.path.join(stories_dir_init, data_sp, lan_sp))
    # absolute dir
    tokenized_stories_dir_init = os.path.abspath(args.save_path)
    if not os.path.isdir(tokenized_stories_dir_init):
        os.mkdir(tokenized_stories_dir_init)
    for data_sp in dataset_split:
        tmp_path0 = os.path.join(tokenized_stories_dir_init, data_sp)
        if not os.path.isdir(tmp_path0):
            os.mkdir(tmp_path0)
        for lan_sp in lang_split:
            tmp_path = os.path.join(tokenized_stories_dir_init, data_sp, lan_sp)
            if not os.path.isdir(tmp_path):
                os.mkdir(tmp_path)
    for data_sp in dataset_split:
        for lan_sp in lang_split:
            stories_dir = os.path.join(stories_dir_init, data_sp, lan_sp)
            tokenized_stories_dir = os.path.join(tokenized_stories_dir_init, data_sp, lan_sp)
            stories = os.listdir(stories_dir)

            # make IO list file
            print("Making list of files to tokenize...")
            with open("mapping_for_corenlp.txt", "w") as f:
                for s in stories:
                    if (not s.endswith('story') and not s.endswith('chnref')):
                        continue
                    f.write("%s\n" % (os.path.join(stories_dir, s)))

            if lan_sp == 'eng':
                print("Preparing to tokenize english %s to %s..." % (stories_dir, tokenized_stories_dir))
                command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                           '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
                           'json', '-outputDirectory', tokenized_stories_dir]
            else:
                print("Preparing to tokenize chinese %s to %s..." % (stories_dir, tokenized_stories_dir))
                command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                           '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt',
                           '-outputFormat',
                           'json', '-outputDirectory', tokenized_stories_dir, '-props',
                           'StanfordCoreNLP-chinese.properties', '-cp', '\"*\"', '-Xmx2g']

                # java edu.stanford.nlp.pipeline.StanfordCoreNLP
                # -annotators tokenize,ssplit -ssplit.newlineIsSentenceBreak always
                # -filelist mapping_for_corenlp.txt -outputFormat json
                # -outputDirectory ./output_dir -props StanfordCoreNLP-chinese.properties -cp "*" -Xmx2g



            print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
            subprocess.call(command)
            print("Stanford CoreNLP Tokenizer has finished.")
            os.remove("mapping_for_corenlp.txt")

            # Check that the tokenized stories directory contains the same number of files as the original directory
            num_orig = len(os.listdir(stories_dir))
            num_tokenized = len(os.listdir(tokenized_stories_dir))
            if num_orig != num_tokenized:
                raise Exception(
                    "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                        tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
            print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def format_xsum_to_lines(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'valid']

    corpus_mapping = json.load(open(pjoin(args.raw_path, 'XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json')))

    for corpus_type in datasets:
        mapped_fnames = corpus_mapping[corpus_type]
        root_src = pjoin(args.raw_path, 'restbody')
        root_tgt = pjoin(args.raw_path, 'firstsentence')
        # realnames = [fname.split('.')[0] for fname in os.listdir(root_src)]
        realnames = mapped_fnames

        a_lst = [(root_src, root_tgt, n) for n in realnames]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_xsum_to_lines, a_lst):
            if (d is None):
                continue
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_xsum_to_lines(params):
    src_path, root_tgt, name = params
    f_src = pjoin(src_path, name + '.restbody')
    f_tgt = pjoin(root_tgt, name + '.fs')
    if (os.path.exists(f_src) and os.path.exists(f_tgt)):
        print(name)
        source = []
        for sent in open(f_src):
            source.append(sent.split())
        tgt = []
        for sent in open(f_tgt):
            tgt.append(sent.split())
        return {'src': source, 'tgt': tgt}
    return None




def translation_xgiga(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']

    from fairseq.models.transformer import TransformerModel
    en2de = TransformerModel.from_pretrained(
        '/home/ybai/downloads/wmt19.en-de.joined-dict.ensemble',
        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
        bpe='fastbpe',
        tokenizer='moses'
        # data_name_or_path='data-bin/wmt17_zh_en_full',
        # bpe_codes='data-bin/wmt17_zh_en_full/zh.code'
    )
    de2en = TransformerModel.from_pretrained(
        '/home/ybai/downloads/wmt19.de-en.joined-dict.ensemble',
        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
        bpe='fastbpe',
        tokenizer='moses'
        # data_name_or_path='data-bin/wmt17_zh_en_full',
        # bpe_codes='data-bin/wmt17_zh_en_full/zh.code'
    )
    en2de.cuda()
    de2en.cuda()
    en2de.eval()
    de2en.eval()
    for corpus_type in datasets:
        # 这里把*改成了en，只做英文的
        for json_f in glob.glob(pjoin(args.raw_path, 'xgiga.en.' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            # a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
            _translation_xgiga((corpus_type, json_f, args, pjoin(args.save_path, real_name),en2de, de2en))
        # print(a_lst)
        # pool = Pool(args.n_cpus)
        # for d in pool.imap(_translation_xgiga, a_lst):
        #     pass

        # pool.close()
        # pool.join()

def _translation_xgiga(params):
    corpus_type, json_file, args, save_file, en2de, de2en = params

    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return
    # bert = BertData(args)
    jobs = json.load(open(json_file))

    datasets = []
    sources = []
    tgts = []
    pos = 0
    for d in jobs:
        pos += 1
        print("processing", end='')
        for i in range(pos % 5):
            print('.',end='')
        print("                        ",end='\r')


        # 英中
        source, tgt  = d['src'], d['tgt']
        sent_labels = None
        if (args.lower):
            # print("1")
            # print("source before = ", source)
            # print("tgt before = ", tgt)
            source = [' '.join(s) for s in source]
            tgt = [' '.join(s) for s in tgt]

        sources.append(source)
        tgts.append(tgt[0])
    # print(tgts)
    print("translating tgts...")
    eng_tgts = en2de.translate(tgts)
    print("translating eng_tgts...")
    back_tgts = de2en.translate(eng_tgts)
    # print(en2de.translate(tgts))
    print("calculating rouge and construct the dataset...")
    for i, each in enumerate(sources):
        # if i > 9:
        #     break
        # print("tgts[i] = ", tgts[i].lower())
        # print("back_tgts[i] = ", back_tgts[i].lower())
        rouge = avg_rouge([tgts[i].lower()], [back_tgts[i].lower()])

        # print(rouge)
        if rouge[0][0] >= 0.6 and rouge[1][0] >= 0.2:
            tgt_eng = [eng_tgts[i].lower().split()]
            tgt = [tgts[i].lower().split()]
            # sources之前就是个list，所以不用再包装

            source = [s.lower().split() for s in sources[i]]
            tmp_json = {
                'src': source,
                'tgt': tgt,
                'tgt_eng': tgt_eng,
            }
            datasets.append(tmp_json)
    # print(datasets)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    with open(save_file, 'w') as save:
        save.write(json.dumps(datasets, ensure_ascii=False))
    return



'''
        # print("source = ", source)
        # print("tgt = ", tgt)
        # 1,2,L f p r

        # en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
        #                        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
        #                        tokenizer='moses', bpe='fastbpe')
        print(en2de.translate(tgt + tgt + tgt))
        tgt_eng = [en2de.translate(each) for each in tgt]
        # print(tgt_eng)
        back_tgt = [de2en.translate(each).lower() for each in tgt_eng]
        # print(back_tgt)
        # print("tgt = ", tgt)
        rouge = avg_rouge(tgt, back_tgt)
        # print(rouge)
        if rouge[0][0] >= 0.6 and rouge[1][0] >= 2:
            tgt_eng = [' '.join(s).lower().split() for s in tgt_eng]
            tgt = [' '.join(s).lower().split() for s in tgt]
            source = [' '.join(s).lower().split() for s in source]
            tmp_json = {
                'src':source,
                'tgt':tgt,
                'tgt_eng':tgt_eng,
            }
            datasets.append(tmp_json)



            # b_data = bert.preprocess(source, tgt, tgt_chn, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

    # exit()
        # 先连接成句子，
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        # 然后翻译过去，
        # 然后翻译回来，然后看rouge
        # 然后大于多少就放过去。生成新的

        # 英中 这里要注意事实上如果是中英的话，所有的语言已经有所变化了
        # src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, tgt_subtoken_idxs_eng, tgt_txt_eng, tgt_seg_ids = b_data
        # b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs, "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
        #                'src_txt': src_txt, "tgt_txt": tgt_txt, "tgt_eng": tgt_subtoken_idxs_eng, "tgt_txt_eng": tgt_txt_eng, 'tgt_segs':tgt_seg_ids}
        # # 英英
        # # src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, tgt_subtoken_idxs_eng, tgt_txt_eng, tgt_seg_ids = b_data
        # # b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs, "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
        # #                'src_txt': src_txt, "tgt_txt": tgt_txt, "tgt_eng": tgt_subtoken_idxs_eng, "tgt_txt_eng": tgt_txt_eng, 'tgt_segs':tgt_seg_ids}
        # datasets.append(b_data_dict)
        '''


        # save.write('\n'.join(dataset))
        # p_ct += 1
    # datasets = []
    # gc.collect()

