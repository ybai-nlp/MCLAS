#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math

import torch
import json
from tensorboardX import SummaryWriter

from others.utils import rouge_results_to_str, test_rouge, tile
from translate.beam import GNMTGlobalScorer
import time
import numpy as np


def _get_attn_subsequent_mask(size):
    """
    Get an attention mask to avoid using the subsequent info.

    Args:
        size: int

    Returns:
        (`LongTensor`):

        * subsequent_mask `[1 x size x size]`
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask

def build_predictor(args, tokenizer, symbols, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')

    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = args.model_path

        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens


    def split_2language_ids(self, ids):
        first_language_ids = []
        second_language_ids = []
        flag = 0
        if self.args.bart:
            language_sep = 50264
        else:
            language_sep = 5
        for each in ids:
            if each == language_sep:
                flag = 1
                continue
            elif (each > language_sep or each == 3 and each != 2):
                if flag == 0:
                    first_language_ids.append(each)
                else:
                    second_language_ids.append(each)
        # 除了unused3之外都得删掉
        # print(second_language_ids)
        return first_language_ids, second_language_ids


    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, gold_score, tgt_str, src =  translation_batch["predictions"],translation_batch["scores"],translation_batch["gold_score"],batch.tgt_str, batch.src
        # preds, pred_score, gold_score, tgt_str, src =  translation_batch["predictions"],translation_batch["scores"],translation_batch["gold_score"],batch.tgt_eng_str, batch.src
        tgt_eng_str = batch.tgt_eng_str
        if self.args.predict_first_language and self.args.multi_task:
            tgt = batch.tgt_eng
        else:
            tgt = batch.tgt

        translations = []
        # 之前的predict_2language 和predict_chinese混在一起了。
        # 现在想办法把逻辑分开。
        # 两个都要有一个pred_str和gold_str来记录拼接的结果，这是一定的。
        # 我觉得要先分predict_2language,这一步把两个str，source_lang和target_lang分开，然后再是否predict chinese，chinese只对target_lang做就行
        for b in range(batch_size):
            # preds还全是id状态
            pred_ids = [int(n) for n in preds[b][0]]
            tgt_ids = [int(n) for n in tgt[b]]
            pred_sents_str = self.vocab.convert_ids_to_tokens(pred_ids)
            pred_sents_str = ' '.join(pred_sents_str).replace(' ##','')
            gold_sents_str = self.vocab.convert_ids_to_tokens(tgt_ids)
            gold_sents_str = ' '.join(gold_sents_str).replace(' ##','')

            if self.args.predict_2language:
                # 这里是选择预测第一个language还是第二个language
                if self.args.predict_first_language:
                    pred_ids, _ = self.split_2language_ids(pred_ids)
                    # print("tgtids = ", tgt_ids)
                    tgt_ids, _ = self.split_2language_ids(tgt_ids)
                else:
                    _, pred_ids = self.split_2language_ids(pred_ids)
                    _, tgt_ids = self.split_2language_ids(tgt_ids)


            if self.args.predict_chinese:
                # 得变成id再变回来
                pred_str = ' '.join(self.vocab.convert_ids_to_tokens(pred_ids)).replace(' ##', '').replace('[unused2]', '').replace('[unused1]', '')
                tgt_str = ' '.join(self.vocab.convert_ids_to_tokens(tgt_ids)).replace(' ##', '').replace('[unused2]', '').replace('[unused1]', '')
                pred_tokens = self.vocab.tokenize(pred_str, tokenize_chinese_chars=False)
                tgt_tokens = self.vocab.tokenize(tgt_str, tokenize_chinese_chars=False)
                pred_ids = self.vocab.convert_tokens_to_ids(pred_tokens)
                tgt_ids = self.vocab.convert_tokens_to_ids(tgt_tokens)
                pred_sents = ' '.join(str(n) for n in pred_ids)
                gold_sent = ' '.join(str(n) for n in tgt_ids)

            else:
                pred_sents = self.vocab.convert_ids_to_tokens(pred_ids)
                pred_sents = ' '.join(pred_sents).replace(' ##', '')
                gold_sent = self.vocab.convert_ids_to_tokens(tgt_ids)
                gold_sent = ' '.join(gold_sent).replace(' ##', '')
                # gold_sent = ' '.join(tgt_str[b].split())
                # gold_sent = ' '.join(tgt_str[b].split())







            # pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
            # pred_sents = ' '.join(pred_sents).replace(' ##','')
            # gold_sent = ' '.join(tgt_str[b].split())
            # print("gold_sent = ", gold_sent)

            gold_eng_sent = ' '.join(tgt_eng_str[b].split())

            # translation = Translation(fname[b],src[:, b] if src is not None else None,
            #                           src_raw, pred_sents,
            #                           attn[b], pred_score[b], gold_sent,
            #                           gold_score[b])
            # src = self.spm.DecodeIds([int(t) for t in translation_batch['batch'].src[0][5] if int(t) != len(self.spm)])


            if self.args.bart:
                raw_src = [self.vocab.decoder[int(t)] for t in src[b]][:500]
            else:
                raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
            raw_src = ' '.join(raw_src)
            # translation = (pred_sents, gold_sent, raw_src, pred_sents_str, gold_sents_str)
            translation = (pred_sents, gold_sent, raw_src, pred_sents_str, gold_sents_str, gold_eng_sent)
            # translation = (pred_sents[0], gold_sent)
            translations.append(translation)

        return translations

    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        gold_str_path = self.args.result_path + '.%d.goldstr' % step
        can_str_path = self.args.result_path + '.%d.canstr' % step

        eng_gold_path = self.args.result_path + '.%d.gold_eng' % step
        self.gold_eng_out_file = codecs.open(eng_gold_path, 'w', 'utf-8')

        # self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        # self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        # raw_gold_path = self.args.result_path + '.%d.raw_gold' % step
        # raw_can_path = self.args.result_path + '.%d.raw_candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        self.gold_str_out_file = codecs.open(gold_str_path, 'w', 'utf-8')
        self.can_str_out_file = codecs.open(can_str_path, 'w', 'utf-8')


        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        # pred_results, gold_results = [], []
        ct = 0

        with torch.no_grad():
            for batch in data_iter:
                if(self.args.recall_eval):
                    gold_tgt_len = batch.tgt.size(1)
                    self.min_length = gold_tgt_len + 20
                    self.max_length = gold_tgt_len + 60
                batch_data = self.translate_batch(batch)
                translations = self.from_batch(batch_data)

                for trans in translations:
                    print(time.asctime(time.localtime(time.time())), "----- now is the test sample: ", ct, end='\r')

                    # pred, gold, src, pred_strrr, gold_strrr,  = trans
                    pred, gold, src, pred_strrr, gold_strrr, gold_eng = trans
                    if self.args.bart:
                        pred_str = pred.replace('madeupword0000', '').replace('madeupword0001', '').replace('<pad>', '').replace('<unk>', '').replace(r' +', ' ').replace(' madeupword0002 ', '<q>').replace('madeupword0002', '').strip()

                        # 这里由于target也是从id转过来的了，所以做了如下变换。
                        gold_str = gold.replace('[unused1]', '').replace('[unused4]', '').replace('[PAD]', '').replace('[unused2]', '').replace(r' +', ' ').replace(' [unused3] ', '<q>').replace('[unused3]', '').strip()
                    else:
                        pred_str = pred.replace('[unused1]', '').replace('[unused4]', '').replace('[PAD]', '').replace('[unused2]', '').replace(r' +', ' ').replace(' [unused3] ', '<q>').replace('[unused3]', '').strip()

                        # 这里由于target也是从id转过来的了，所以做了如下变换。
                        gold_str = gold.replace('[unused1]', '').replace('[unused4]', '').replace('[PAD]', '').replace('[unused2]', '').replace(r' +', ' ').replace(' [unused3] ', '<q>').replace('[unused3]', '').strip()
                    # gold_str = gold.strip()
                    if(self.args.recall_eval):
                        _pred_str = ''
                        gap = 1e3
                        for sent in pred_str.split('<q>'):
                            can_pred_str = _pred_str+ '<q>'+sent.strip()
                            can_gap = math.fabs(len(_pred_str.split())-len(gold_str.split()))
                            # if(can_gap>=gap):
                            if(len(can_pred_str.split())>=len(gold_str.split())+10):
                                pred_str = _pred_str
                                break
                            else:
                                gap = can_gap
                                _pred_str = can_pred_str



                        # pred_str = ' '.join(pred_str.split()[:len(gold_str.split())])
                    # self.raw_can_out_file.write(' '.join(pred).strip() + '\n')
                    # self.raw_gold_out_file.write(' '.join(gold).strip() + '\n')
                    self.gold_eng_out_file.write(gold_eng + '\n')
                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    self.src_out_file.write(src.strip() + '\n')

                    # 下边是加的
                    self.can_str_out_file.write(pred_strrr + '\n')
                    self.gold_str_out_file.write(gold_strrr + '\n')
                    # print("pred_strrr = ", pred_strrr)
                    # print("gold_strrr = ", gold_strrr)
                    ct += 1
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()
                self.can_str_out_file.flush()
                self.gold_str_out_file.flush()

        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()
        self.can_str_out_file.close()
        self.gold_str_out_file.close()

        if (step != -1):
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)

    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

    def translate_batch(self, batch, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                batch,
                self.max_length,
                min_length=self.min_length)

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src
        # print("mask_tgt = ", batch.mask_tgt.size())
        # print(batch.mask_tgt)

        # print("tgt_segs = ", batch.tgt_segs.size())
        # print(batch.tgt_segs)
        # print("tgt = ", batch.tgt.size())
        # print(batch.tgt)
        # exit()


        if self.args.bart:
            src_features = self.model.bert.model.encoder(input_ids=src, attention_mask=mask_src)[0]
            # print("output = ", self.model。)
            # past_key_values =
            # print("src_features = ", src_features.size())
            # print(src_features)
        else:
            src_features = self.model.bert(src, segs, mask_src)
        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        device = src_features.device

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = tile(src_features, beam_size, dim=0)

        if self.args.bart:
            # print("src = ", src.size())
            # print(src)
            # print("mask_src0 = ", mask_src.size())
            # print(mask_src)
            mask_src = tile(mask_src, beam_size, dim=0).byte()
            bart_dec_cache = None

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)

        if self.args.language_limit:
            language_limit = torch.Tensor(json.load(open(self.args.tgt_mask))).long().cuda()

        # language_seg
        language_segs = torch.full(
            [batch_size * beam_size, 1],
            0,
            dtype=torch.long,
            device=device)


        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch


        for step in range(max_length):
            # print("alive_seq = ", alive_seq.size())
            # print(alive_seq)
            # print("language_segs = ", language_segs.size())
            # print(language_segs)
            decoder_input = alive_seq[:, -1].view(1, -1)


            # language_seg和alive_seq始终保持一致，alive_seq ，language也选择。
            # 选择完之后，看这次的token是否为5，做一个01矩阵，新的language直接等于1的位置取反，那就取反
            # 先给取反的位置赋0，（算取反向量，乘以01矩阵）加到原有的上边即可。
            # 最后concat到一起
            decoder_seg_input = language_segs[:, -1].view(1,-1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0,1)


            decoder_seg_input = decoder_seg_input.transpose(0,1)


            if self.args.bart:
                tgt_mask = torch.zeros(decoder_input.size()).byte().cuda()

                # causal_mask = (1 - _get_attn_subsequent_mask(tgt_mask.size(1)).float().cuda()) # * float("-inf")).cuda()
                causal_mask = torch.triu(torch.zeros(tgt_mask.size(1), tgt_mask.size(1)).float().fill_(float("-inf")).float(), 1).cuda()
                # print(src_features.size())
                # print(mask_src.size())
                # print(mask_src)
                # model_inputs = self.bert.model.prepare_inputs_for_generation(
                #     decoder_input, past=src_features, attention_mask=tgt_mask, use_cache=True
                # )
                dec_output = self.model.bert.model.decoder(input_ids=alive_seq, encoder_hidden_states=src_features, encoder_padding_mask=mask_src, decoder_padding_mask=tgt_mask, decoder_causal_mask=causal_mask, decoder_cached_states=bart_dec_cache, use_cache=True)
                dec_out = dec_output[0]
                bart_dec_cache = dec_output[1][1]
                # print(bart_dec_cache)
                # print('dec_out = ')
                # print(dec_out[0])
                # exit()
            elif self.args.predict_first_language and self.args.multi_task:
                dec_out, dec_states = self.model.decoder_monolingual(decoder_input, src_features, dec_states,
                                                         step=step, tgt_segs=decoder_seg_input)
            else:
                dec_out, dec_states = self.model.decoder(decoder_input, src_features, dec_states,
                                                         step=step, tgt_segs=decoder_seg_input)


            # Generator forward.
            log_probs = self.generator.forward(dec_out.transpose(0,1).squeeze(0))
            vocab_size = log_probs.size(-1)

            if self.args.language_limit:


                mask_language_limit = torch.zeros(log_probs.size()).cuda()
                mask_language_limit.index_fill_(1, language_limit, 1)
                # 如果两个语言拼接，那么生成第二语言才限制 如果是直接跨语言的话，那从最开始就要限制
                if self.args.predict_2language:
                    mask_language_limit = mask_language_limit.long() * decoder_seg_input
                    mask_language_limit = mask_language_limit + (1 - decoder_seg_input)
                else:
                    mask_language_limit = mask_language_limit.long() * torch.ones(decoder_seg_input.size()).long().cuda()

                # 这里，把除了备选位置的都赋为负无穷
                log_probs.masked_fill_((1 - mask_language_limit).byte(), -1e20)


            if step < min_length:
                log_probs[:, self.end_token] = -1e20


            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if(self.args.block_trigram):
                cur_len = alive_seq.size(1)
                if(cur_len>3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        if self.args.bart:
                            words = [self.vocab.decoder[w] for w in words]
                        else:
                            words = [self.vocab.ids_to_tokens[w] for w in words]
                        words = ' '.join(words).replace(' ##','').split()
                        if(len(words)<=3):
                            continue
                        trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)


            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            # print("topk_ids = ", topk_ids)
            # print("end_token = ", self.end_token)
            if self.args.bart:
                is_finished = topk_ids.eq(2)
            else:
                is_finished = topk_ids.eq(self.end_token)

            # 是否是第二个语言
            # 先把最后一部分取出来，然后把新加入5的填上1，再拼接起来。
            is_languaged = topk_ids.eq(5)
            # is_languaged = alive_seq[:, -2].unsqueeze(0).eq(5)
            language_segs = language_segs.index_select(0, select_indices)

            last_segs = language_segs[:, -1]
            tmp_seg = last_segs.masked_fill(is_languaged, 1)
            language_segs = torch.cat([language_segs, tmp_seg.view(-1,1)], -1)



            # print("is_finished = ", is_finished)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
