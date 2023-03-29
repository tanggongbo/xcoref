import torch
import torch.nn as nn
from transformers import AutoModel
import util
import logging
from collections import Iterable
import numpy as np
import torch.nn.init as init
import higher_order as ho

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class CorefModel(nn.Module):
    def __init__(self, config, device, num_genres=None):
        super().__init__()
        self.config = config
        self.device = device

        self.num_genres = num_genres if num_genres else len(config['genres'])
        self.max_seg_len = config['max_segment_len']
        self.max_span_width = config['max_span_width']
        assert config['loss_type'] in ['marginalized', 'hinge']
        if config['coref_depth'] > 1 or config['higher_order'] == 'cluster_merging':
            assert config['fine_grained']  # Higher-order is in slow fine-grained scoring

        # Model
        self.dropout = nn.Dropout(p=config['dropout_rate'])
        self.bert = AutoModel.from_pretrained(config['bert_pretrained_name_or_path_src'])
        self.bert_trg = AutoModel.from_pretrained(config['bert_pretrained_name_or_path_trg'])
        # self.bert = BertModel.from_pretrained(config['bert_pretrained_name_or_path'])

        self.bert_emb_size = self.bert.config.hidden_size
        self.span_emb_size = self.bert_emb_size * 3
        if config['use_features']:
            self.span_emb_size += config['feature_emb_size']
        self.pair_emb_size = self.span_emb_size * 3
        if config['use_metadata']:
            self.pair_emb_size += 2 * config['feature_emb_size']
        if config['use_features']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_segment_distance']:
            self.pair_emb_size += config['feature_emb_size']

        self.emb_span_width = self.make_embedding(self.max_span_width) if config['use_features'] else None
        self.emb_span_width_prior = self.make_embedding(self.max_span_width) if config['use_width_prior'] else None
        self.emb_antecedent_distance_prior = self.make_embedding(10) if config['use_distance_prior'] else None
        self.emb_genre = self.make_embedding(self.num_genres)
        self.emb_same_speaker = self.make_embedding(2) if config['use_metadata'] else None
        self.emb_segment_distance = self.make_embedding(config['max_training_sentences']) if config[
            'use_segment_distance'] else None
        self.emb_top_antecedent_distance = self.make_embedding(10)
        self.emb_cluster_size = self.make_embedding(10) if config['higher_order'] == 'cluster_merging' else None

        self.mention_token_attn = self.make_ffnn(self.bert_emb_size, 0, output_size=1) if config[
            'model_heads'] else None
        self.mention_token_attn_trg_adapter = self.make_ffnn(1, [config['adapter_size']], output_size=1) if config[
            'model_heads'] else None
        self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'],
                                                  output_size=1)
        self.span_emb_score_ffnn_trg_adapter = self.make_ffnn(1, [config['adapter_size']], output_size=1)

        self.span_width_score_ffnn = self.make_ffnn(config['feature_emb_size'],
                                                    [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if \
            config['use_width_prior'] else None
        self.span_width_score_ffnn_trg_adapter = self.make_ffnn(1, [config['adapter_size']], output_size=1) if \
            config['use_width_prior'] else None
        self.coarse_bilinear = self.make_ffnn(self.span_emb_size, 0, output_size=self.span_emb_size)
        self.antecedent_distance_score_ffnn = self.make_ffnn(config['feature_emb_size'], 0, output_size=1) if \
            config['use_distance_prior'] else None
        self.coref_score_ffnn = self.make_ffnn(self.pair_emb_size, [config['ffnn_size']] * config['ffnn_depth'],
                                               output_size=1) if config['fine_grained'] else None
        self.coref_score_ffnn_trg_adapter = self.make_ffnn(1, [config['adapter_size']], output_size=1) \
            if config['fine_grained'] else None

        # self.transform_trg2src = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'],
        #                                         output_size=self.span_emb_size)
        self.gate_ffnn = self.make_ffnn(2 * self.span_emb_size, 0, output_size=self.span_emb_size) if config[
                                                                                                          'coref_depth'] > 1 else None
        self.span_attn_ffnn = self.make_ffnn(self.span_emb_size, 0, output_size=1) if config[
                                                                                          'higher_order'] == 'span_clustering' else None
        self.cluster_score_ffnn = self.make_ffnn(3 * self.span_emb_size + config['feature_emb_size'],
                                                 [config['cluster_ffnn_size']] * config['ffnn_depth'], output_size=1) if \
            config['higher_order'] == 'cluster_merging' else None

        self.update_steps = 0  # Internal use for debug
        self.debug = False  # True

    def make_embedding(self, dict_size, std=0.02):
        emb = nn.Embedding(dict_size, self.config['feature_emb_size'])
        init.normal_(emb.weight, std=std)
        return emb

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    def forward(self, *input):
        return self.get_predictions_and_loss(*input)

    def get_span_rep(self, bert, side, is_training, input_ids, input_mask, speaker_ids, sentence_map, gold_starts,
                     gold_ends, gold_mention_cluster_map):

        device = self.device
        conf = self.config
        # Get token emb
        mention_doc = bert(input_ids, attention_mask=input_mask)[0]  # [num seg, num max tokens, emb size]
        input_mask = input_mask.to(torch.bool)
        mention_doc = mention_doc[input_mask]
        speaker_ids = speaker_ids[input_mask]
        num_words = mention_doc.shape[0]

        # Get candidate span
        sentence_indices = sentence_map  # [num tokens]
        candidate_starts = torch.unsqueeze(torch.arange(0, num_words, device=device), 1).repeat(1, self.max_span_width)
        candidate_ends = candidate_starts + torch.arange(0, self.max_span_width, device=device)
        candidate_start_sent_idx = sentence_indices[candidate_starts]
        candidate_end_sent_idx = sentence_indices[torch.min(candidate_ends, torch.tensor(num_words - 1, device=device))]
        candidate_mask = (candidate_ends < num_words) & (candidate_start_sent_idx == candidate_end_sent_idx)
        candidate_starts, candidate_ends = candidate_starts[candidate_mask], candidate_ends[
            candidate_mask]  # [num valid candidates]
        num_candidates = candidate_starts.shape[0]

        # Get candidate labels
        if is_training and side == "source":
            same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(candidate_starts, 0))
            same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(candidate_ends, 0))
            same_span = (same_start & same_end).to(torch.long)
            candidate_labels = torch.matmul(torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float),
                                            same_span.to(torch.float))
            candidate_labels = torch.squeeze(candidate_labels.to(torch.long),
                                             0)  # [num candidates]; non-gold span has label 0

        # Get span embedding
        span_start_emb, span_end_emb = mention_doc[candidate_starts], mention_doc[candidate_ends]
        candidate_emb_list = [span_start_emb, span_end_emb]
        if conf['use_features']:
            candidate_width_idx = candidate_ends - candidate_starts
            candidate_width_emb = self.emb_span_width(candidate_width_idx)
            candidate_width_emb = self.dropout(candidate_width_emb)
            candidate_emb_list.append(candidate_width_emb)
        # Use attended head or avg token
        candidate_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(num_candidates, 1)
        candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts, 1)) & (
                candidate_tokens <= torch.unsqueeze(candidate_ends, 1))
        if conf['model_heads']:
            token_attn = self.mention_token_attn(mention_doc)
            if side == "source":
                token_attn = torch.squeeze(token_attn, 1)
            else:
                # adapter
                token_attn = torch.squeeze(self.mention_token_attn_trg_adapter(token_attn), 1)
        else:
            token_attn = torch.ones(num_words, dtype=torch.float, device=device)  # Use avg if no attention
        candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn, 0)
        candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)
        head_attn_emb = torch.matmul(candidate_tokens_attn, mention_doc)
        candidate_emb_list.append(head_attn_emb)
        candidate_span_emb = torch.cat(candidate_emb_list, dim=1)  # [num candidates, new emb size]

        # Get span score
        candidate_mention_scores = self.span_emb_score_ffnn(candidate_span_emb)
        if side == "source":
            candidate_mention_scores = torch.squeeze(candidate_mention_scores, 1)
        else:
            # adapter
            candidate_mention_scores = torch.squeeze(self.span_emb_score_ffnn_trg_adapter(candidate_mention_scores), 1)
        if conf['use_width_prior']:
            width_score = self.span_width_score_ffnn(self.emb_span_width_prior.weight)
            if side == "source":
                width_score = torch.squeeze(width_score, 1)
            else:
                # adapter
                width_score = torch.squeeze(self.span_width_score_ffnn_trg_adapter(width_score), 1)
            candidate_width_score = width_score[candidate_width_idx]
            candidate_mention_scores += candidate_width_score

        # Extract top spans
        candidate_idx_sorted_by_score = torch.argsort(candidate_mention_scores, descending=True).tolist()
        candidate_starts_cpu, candidate_ends_cpu = candidate_starts.tolist(), candidate_ends.tolist()
        num_top_spans = int(min(conf['max_num_extracted_spans'], conf['top_span_ratio'] * num_words))
        selected_idx_cpu = self._extract_top_spans(candidate_idx_sorted_by_score, candidate_starts_cpu,
                                                   candidate_ends_cpu, num_top_spans)
        assert len(selected_idx_cpu) == num_top_spans
        selected_idx = torch.tensor(selected_idx_cpu, device=device)
        top_span_starts, top_span_ends = candidate_starts[selected_idx], candidate_ends[selected_idx]
        # print("selected_idx, top_span_starts:", selected_idx.shape, top_span_starts.shape)
        top_span_emb = candidate_span_emb[selected_idx]
        top_span_cluster_ids = candidate_labels[selected_idx] if (is_training and side == "source") else None
        top_span_mention_scores = candidate_mention_scores[selected_idx]

        # if side == "target":
        #     #  add another FFNN to transform trg-side to src-side
        #     top_span_emb = self.transform_trg2src(top_span_emb)

        return top_span_emb, top_span_starts, top_span_ends, num_top_spans, top_span_cluster_ids, top_span_mention_scores, speaker_ids

    def do_scoring(self, top_span_emb, top_span_starts, top_span_ends, num_top_spans,
                   top_span_cluster_ids, top_span_mention_scores, do_loss, input_ids, input_mask,
                   speaker_ids, genre):

        device = self.device
        conf = self.config
        input_mask = input_mask.to(torch.bool)

        # Coarse pruning on each mention's antecedents
        max_top_antecedents = min(num_top_spans, conf['max_top_antecedents'])
        top_span_range = torch.arange(0, num_top_spans, device=device)
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0)
        antecedent_mask = (antecedent_offsets >= 1)
        pairwise_mention_score_sum = torch.unsqueeze(top_span_mention_scores, 1) + torch.unsqueeze(
            top_span_mention_scores, 0)
        source_span_emb = self.dropout(self.coarse_bilinear(top_span_emb))
        target_span_emb = self.dropout(torch.transpose(top_span_emb, 0, 1))
        pairwise_coref_scores = torch.matmul(source_span_emb, target_span_emb)
        pairwise_fast_scores = pairwise_mention_score_sum + pairwise_coref_scores
        pairwise_fast_scores += torch.log(antecedent_mask.to(torch.float))
        if conf['use_distance_prior']:
            distance_score = torch.squeeze(
                self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), 1)
            bucketed_distance = util.bucket_distance(antecedent_offsets)
            antecedent_distance_score = distance_score[bucketed_distance]
            pairwise_fast_scores += antecedent_distance_score
        top_pairwise_fast_scores, top_antecedent_idx = torch.topk(pairwise_fast_scores, k=max_top_antecedents)
        top_antecedent_mask = util.batch_select(antecedent_mask, top_antecedent_idx,
                                                device)  # [num top spans, max top antecedents]
        top_antecedent_offsets = util.batch_select(antecedent_offsets, top_antecedent_idx, device)

        # Slow mention ranking
        if conf['fine_grained']:
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb = None, None, None, None
            if conf['use_metadata']:
                top_span_speaker_ids = speaker_ids[top_span_starts]
                top_antecedent_speaker_id = top_span_speaker_ids[top_antecedent_idx]
                same_speaker = torch.unsqueeze(top_span_speaker_ids, 1) == top_antecedent_speaker_id
                same_speaker_emb = self.emb_same_speaker(same_speaker.to(torch.long))
                genre_emb = self.emb_genre(genre)
                genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat(num_top_spans, max_top_antecedents,
                                                                                     1)
            if conf['use_segment_distance']:
                num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
                token_seg_ids = torch.arange(0, num_segs, device=device).unsqueeze(1).repeat(1, seg_len)
                token_seg_ids = token_seg_ids[input_mask]
                top_span_seg_ids = token_seg_ids[top_span_starts]
                top_antecedent_seg_ids = token_seg_ids[top_span_starts[top_antecedent_idx]]
                top_antecedent_seg_distance = torch.unsqueeze(top_span_seg_ids, 1) - top_antecedent_seg_ids
                top_antecedent_seg_distance = torch.clamp(top_antecedent_seg_distance, 0,
                                                          self.config['max_training_sentences'] - 1)
                seg_distance_emb = self.emb_segment_distance(top_antecedent_seg_distance)
            if conf['use_features']:  # Antecedent distance
                top_antecedent_distance = util.bucket_distance(top_antecedent_offsets)
                top_antecedent_distance_emb = self.emb_top_antecedent_distance(top_antecedent_distance)

            for depth in range(conf['coref_depth']):
                top_antecedent_emb = top_span_emb[top_antecedent_idx]  # [num top spans, max top antecedents, emb size]
                feature_list = []
                if conf['use_metadata']:  # speaker, genre
                    feature_list.append(same_speaker_emb)
                    feature_list.append(genre_emb)
                if conf['use_segment_distance']:
                    feature_list.append(seg_distance_emb)
                if conf['use_features']:  # Antecedent distance
                    feature_list.append(top_antecedent_distance_emb)
                feature_emb = torch.cat(feature_list, dim=2)
                feature_emb = self.dropout(feature_emb)
                target_emb = torch.unsqueeze(top_span_emb, 1).repeat(1, max_top_antecedents, 1)
                similarity_emb = target_emb * top_antecedent_emb
                pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)
                top_pairwise_slow_scores = torch.squeeze(self.coref_score_ffnn(pair_emb), 2)
                top_pairwise_scores = top_pairwise_slow_scores + top_pairwise_fast_scores
                if conf['higher_order'] == 'cluster_merging':
                    cluster_merging_scores = ho.cluster_merging(top_span_emb, top_antecedent_idx, top_pairwise_scores,
                                                                self.emb_cluster_size, self.cluster_score_ffnn, None,
                                                                self.dropout,
                                                                device=device, reduce=conf['cluster_reduce'],
                                                                easy_cluster_first=conf['easy_cluster_first'])
                    break
                elif depth != conf['coref_depth'] - 1:
                    if conf['higher_order'] == 'attended_antecedent':
                        refined_span_emb = ho.attended_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores,
                                                                  device)
                    elif conf['higher_order'] == 'max_antecedent':
                        refined_span_emb = ho.max_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores,
                                                             device)
                    elif conf['higher_order'] == 'entity_equalization':
                        refined_span_emb = ho.entity_equalization(top_span_emb, top_antecedent_emb, top_antecedent_idx,
                                                                  top_pairwise_scores, device)
                    elif conf['higher_order'] == 'span_clustering':
                        refined_span_emb = ho.span_clustering(top_span_emb, top_antecedent_idx, top_pairwise_scores,
                                                              self.span_attn_ffnn, device)

                    gate = self.gate_ffnn(torch.cat([top_span_emb, refined_span_emb], dim=1))
                    gate = torch.sigmoid(gate)
                    top_span_emb = gate * refined_span_emb + (1 - gate) * top_span_emb  # [num top spans, span emb size]
        else:
            top_pairwise_scores = top_pairwise_fast_scores  # [num top spans, max top antecedents]

        if not do_loss:
            if conf['fine_grained'] and conf['higher_order'] == 'cluster_merging':
                top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores],
                                              dim=1)  # [num top spans, max top antecedents + 1]
            return top_antecedent_idx, top_antecedent_scores

        # Get gold labels
        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedent_idx]
        top_antecedent_cluster_ids += (top_antecedent_mask.to(
            torch.long) - 1) * 100000  # Mask id on invalid antecedents
        same_gold_cluster_indicator = (top_antecedent_cluster_ids == torch.unsqueeze(top_span_cluster_ids, 1))
        non_dummy_indicator = torch.unsqueeze(top_span_cluster_ids > 0, 1)
        pairwise_labels = same_gold_cluster_indicator & non_dummy_indicator
        dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))
        top_antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1)

        # Get loss
        top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)
        if conf['loss_type'] == 'marginalized':
            log_marginalized_antecedent_scores = torch.logsumexp(
                top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
            log_norm = torch.logsumexp(top_antecedent_scores, dim=1)
            loss = torch.sum(log_norm - log_marginalized_antecedent_scores)
        elif conf['loss_type'] == 'hinge':
            top_antecedent_mask = torch.cat(
                [torch.ones(num_top_spans, 1, dtype=torch.bool, device=device), top_antecedent_mask], dim=1)
            top_antecedent_scores += torch.log(top_antecedent_mask.to(torch.float))
            highest_antecedent_scores, highest_antecedent_idx = torch.max(top_antecedent_scores, dim=1)
            gold_antecedent_scores = top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float))
            highest_gold_antecedent_scores, highest_gold_antecedent_idx = torch.max(gold_antecedent_scores, dim=1)
            slack_hinge = 1 + highest_antecedent_scores - highest_gold_antecedent_scores
            # Calculate delta
            highest_antecedent_is_gold = (highest_antecedent_idx == highest_gold_antecedent_idx)
            mistake_false_new = (highest_antecedent_idx == 0) & torch.logical_not(dummy_antecedent_labels.squeeze())
            delta = ((3 - conf['false_new_delta']) / 2) * torch.ones(num_top_spans, dtype=torch.float, device=device)
            delta -= (1 - conf['false_new_delta']) * mistake_false_new.to(torch.float)
            delta *= torch.logical_not(highest_antecedent_is_gold).to(torch.float)
            loss = torch.sum(slack_hinge * delta)

        # Add mention loss
        if conf['mention_loss_coef']:
            gold_mention_scores = top_span_mention_scores[top_span_cluster_ids > 0]
            non_gold_mention_scores = top_span_mention_scores[top_span_cluster_ids == 0]
            loss_mention = -torch.sum(torch.log(torch.sigmoid(gold_mention_scores))) * conf['mention_loss_coef']
            loss_mention += -torch.sum(torch.log(1 - torch.sigmoid(non_gold_mention_scores))) * conf[
                'mention_loss_coef']
            loss += loss_mention

        if conf['higher_order'] == 'cluster_merging':
            top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores],
                                              dim=1)
            log_marginalized_antecedent_scores2 = torch.logsumexp(
                top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
            log_norm2 = torch.logsumexp(top_antecedent_scores, dim=1)  # [num top spans]
            loss_cm = torch.sum(log_norm2 - log_marginalized_antecedent_scores2)
            if conf['cluster_dloss']:
                loss += loss_cm
            else:
                loss = loss_cm

        # Debug
        if self.debug:
            if self.update_steps % 20 == 0:
                logger.info('---------debug step: %d---------' % self.update_steps)
                # logger.info('candidates: %d; antecedents: %d' % (num_candidates, max_top_antecedents))
                logger.info('spans/gold: %d/%d; ratio: %.2f' % (
                    num_top_spans, (top_span_cluster_ids > 0).sum(), (top_span_cluster_ids > 0).sum() / num_top_spans))
                if conf['mention_loss_coef']:
                    logger.info('mention loss: %.4f' % loss_mention)
                if conf['loss_type'] == 'marginalized':
                    logger.info(
                        'norm/gold: %.4f/%.4f' % (torch.sum(log_norm), torch.sum(log_marginalized_antecedent_scores)))
                else:
                    logger.info('loss: %.4f' % loss)
        self.update_steps += 1

        return loss
    def do_scoring_cross(self, top_span_emb_src, top_span_starts_src, top_span_ends_src, num_top_spans_src,
                         input_ids_src, input_mask_src, speaker_ids_src, genre_src,
                         top_span_emb_trg, top_span_starts_trg, top_span_ends_trg, num_top_spans_trg,
                         input_ids_trg, input_mask_trg, speaker_ids_trg):

        device = self.device
        conf = self.config
        input_mask_src = input_mask_src.to(torch.bool)
        input_mask_trg = input_mask_trg.to(torch.bool)

        # Coarse pruning on each mention's antecedents
        max_top_antecedents = min(num_top_spans_src, num_top_spans_trg, conf['max_top_x_antecedents'])
        # different from monolingual antecedent offset, we need to compute the x-lingual mention distance
        # based on: 1. start distance; 2: end distance; 3. width gap
        distance_start = top_span_starts_src.unsqueeze(0).transpose(0, 1).repeat(1, top_span_starts_trg.shape[0]) - \
                         top_span_starts_trg.repeat(top_span_starts_src.shape[0], 1)

        antecedent_mask_x = (distance_start > -conf['x-lingual_antecedent_offset_threshold'])
        source_span_emb = self.dropout(self.coarse_bilinear(top_span_emb_src))
        target_span_emb = self.dropout(torch.transpose(top_span_emb_trg, 0, 1))
        pairwise_coref_scores = torch.matmul(source_span_emb, target_span_emb)
        pairwise_fast_scores = pairwise_coref_scores
        # print(f'pairwise_fast_scores.shape: {pairwise_fast_scores.shape}')
        # print(f'antecedent_mask_x.shape: {antecedent_mask_x.shape}')
        pairwise_fast_scores += torch.log(antecedent_mask_x.to(torch.float))
        if conf['use_distance_prior']:
            distance_score = torch.squeeze(
                self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), 1)
            bucketed_distance = util.bucket_x_distance_v2(distance_start)
            antecedent_distance_score = distance_score[bucketed_distance]
            pairwise_fast_scores += antecedent_distance_score
        top_pairwise_fast_scores, top_antecedent_idx = torch.topk(pairwise_fast_scores, k=max_top_antecedents)
        top_pairwise_fast_scores = torch.nan_to_num(top_pairwise_fast_scores)
        top_antecedent_mask = util.batch_select(antecedent_mask_x, top_antecedent_idx,
                                                device)  # [num top spans, max top antecedents]
        top_antecedent_distance = util.batch_select(distance_start, top_antecedent_idx, device)

        # Slow mention ranking
        if conf['fine_grained']:
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb = None, None, None, None
            if conf['use_metadata']:
                top_span_speaker_ids_src = speaker_ids_src[top_span_starts_src]
                top_span_speaker_ids_trg = speaker_ids_trg[top_span_starts_trg]
                top_antecedent_speaker_id = top_span_speaker_ids_trg[top_antecedent_idx]
                same_speaker = torch.unsqueeze(top_span_speaker_ids_src, 1) == top_antecedent_speaker_id
                same_speaker_emb = self.emb_same_speaker(same_speaker.to(torch.long))
                genre_emb_src = self.emb_genre(genre_src)
                # shape (num_top_spans_src, num_top_spans_trg, 1)
                genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb_src, 0), 0).repeat(num_top_spans_src,
                                                                                         max_top_antecedents, 1)
            if conf['use_segment_distance']:
                num_segs_src, seg_len_src = input_ids_src.shape[0], input_ids_src.shape[1]
                token_seg_ids_src = torch.arange(0, num_segs_src, device=device).unsqueeze(1).repeat(1, seg_len_src)
                token_seg_ids_src = token_seg_ids_src[input_mask_src]
                top_span_seg_ids_src = token_seg_ids_src[top_span_starts_src]

                num_segs_trg, seg_len_trg = input_ids_trg.shape[0], input_ids_trg.shape[1]
                token_seg_ids_trg = torch.arange(0, num_segs_trg, device=device).unsqueeze(1).repeat(1, seg_len_trg)
                token_seg_ids_trg = token_seg_ids_trg[input_mask_trg]

                top_antecedent_seg_ids = token_seg_ids_trg[top_span_starts_trg[top_antecedent_idx]]
                top_antecedent_seg_distance = torch.unsqueeze(top_span_seg_ids_src, 1) - top_antecedent_seg_ids
                top_antecedent_seg_distance = torch.clamp(top_antecedent_seg_distance, 0,
                                                          self.config['max_training_sentences'] - 1)
                seg_distance_emb = self.emb_segment_distance(top_antecedent_seg_distance)
            if conf['use_features']:  # Antecedent distance
                top_antecedent_distance = util.bucket_x_distance_v2(top_antecedent_distance)
                top_antecedent_distance_emb = self.emb_top_antecedent_distance(top_antecedent_distance)

            for depth in range(conf['coref_depth']):
                # [num top spans, max top antecedents, emb size]
                top_antecedent_emb = top_span_emb_trg[top_antecedent_idx]
                feature_list = []
                if conf['use_metadata']:  # speaker, genre
                    feature_list.append(same_speaker_emb)
                    feature_list.append(genre_emb)
                    # print(f'same_speaker_emb: {same_speaker_emb.shape}')
                    # print(f'genre_emb: {genre_emb.shape}')
                if conf['use_segment_distance']:
                    feature_list.append(seg_distance_emb)
                    # print(f'seg_distance_emb: {seg_distance_emb.shape}')
                if conf['use_features']:  # Antecedent distance
                    feature_list.append(top_antecedent_distance_emb)

                feature_emb = torch.cat(feature_list, dim=2)
                feature_emb = self.dropout(feature_emb)
                target_emb = torch.unsqueeze(top_span_emb_src, 1).repeat(1, max_top_antecedents, 1)
                similarity_emb = target_emb * top_antecedent_emb
                pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)
                top_pairwise_slow_scores = self.coref_score_ffnn(pair_emb)
                if not self.config['share_module']:
                    # adapter
                    top_pairwise_slow_scores = torch.squeeze(self.coref_score_ffnn_trg_adapter(
                        top_pairwise_slow_scores), 2)
                else:
                    top_pairwise_slow_scores = torch.squeeze(top_pairwise_slow_scores, 2)

            # print(f'slow scores: {top_pairwise_slow_scores}')
            # print(f'fast scores: {top_pairwise_fast_scores}')
            top_pairwise_scores = top_pairwise_slow_scores + top_pairwise_fast_scores


        else:
            top_pairwise_scores = top_pairwise_fast_scores  # [num top spans, max top antecedents]

        # # only consider scores higher than a threshold
        # score_mask = (top_pairwise_scores > conf['threshold_x_score']).to(torch.int)
        # top_pairwise_scores = top_pairwise_scores * score_mask
        # # print(f'top scores: {top_pairwise_scores}')
        # # print(torch.log(torch.clamp(top_pairwise_scores, max=100000) + 1))
        # loss = float(1.0) / (torch.sum(torch.sum(
        #     torch.log(torch.clamp(top_pairwise_scores, max=100000) + 1), dim=1)) + 0.01)  # 0.01 for smoothing
        # only consider the maximum score
        # score_mask = (top_pairwise_scores > conf['threshold_x_score']).to(torch.int)
        max_pairwise_scores = torch.max(top_pairwise_scores, dim=-1)[0]
        # max_pairwise_scores = max_pairwise_scores - torch.max(max_pairwise_scores, dim=0)[0]
        # print(f'top scores: {max_pairwise_scores}')
        # print(torch.log(torch.clamp(top_pairwise_scores, max=100000) + 1))
        # loss = float(1.0) / (torch.max(max_pairwise_scores, dim=0)[0] + torch.logsumexp(max_pairwise_scores, dim=0))
        loss = float(1.0) / torch.logsumexp(max_pairwise_scores, dim=0)

        return loss, top_pairwise_scores, top_antecedent_idx

    def get_predictions_and_loss(self, input_ids_src, input_mask_src, speaker_ids_src, sentence_len_src, genre_src,
                                 sentence_map_src, is_training_src, gold_starts_src, gold_ends_src,
                                 gold_mention_cluster_map_src, input_ids_trg, input_mask_trg, speaker_ids_trg,
                                 sentence_len_trg, genre_trg, sentence_map_trg, is_training_trg,
                                 gold_starts_trg, gold_ends_trg, gold_mention_cluster_map_trg):
        """ Model and input are already on the device """
        device = self.device
        conf = self.config

        do_loss = False
        if is_training_src:
            do_loss = True

        if do_loss:
            top_span_emb_src, top_span_starts_src, top_span_ends_src, num_top_spans_src, top_span_cluster_ids_src, \
            top_span_mention_scores_src, speaker_ids_src = self.get_span_rep(self.bert, "source", is_training_src,
                                                            input_ids_src, input_mask_src, speaker_ids_src,
                                                            sentence_map_src,
                                                            gold_starts_src, gold_ends_src,
                                                            gold_mention_cluster_map_src)


            loss_coref = self.do_scoring(top_span_emb_src, top_span_starts_src, top_span_ends_src, num_top_spans_src,
                                       top_span_cluster_ids_src, top_span_mention_scores_src, do_loss,
                                       input_ids_src, input_mask_src, speaker_ids_src, genre_src)

            if conf["coref_loss_both"] or conf["with_agreement"]:
                top_span_emb_trg, top_span_starts_trg, top_span_ends_trg, num_top_spans_trg, top_span_cluster_ids_trg, \
                top_span_mention_scores_trg, speaker_ids_trg = self.get_span_rep(self.bert_trg, "target",
                                                                                 is_training_trg,
                                                                                 input_ids_trg, input_mask_trg,
                                                                                 speaker_ids_trg,
                                                                                 sentence_map_trg,
                                                                                 gold_starts_trg, gold_ends_trg,
                                                                                 gold_mention_cluster_map_trg)

            if conf["coref_loss_both"]:
                loss_coref += self.do_scoring(top_span_emb_trg, top_span_starts_trg, top_span_ends_trg, num_top_spans_trg,
                                           top_span_cluster_ids_trg, top_span_mention_scores_trg, do_loss,
                                           input_ids_trg, input_mask_trg, speaker_ids_trg, genre_trg)

            if conf["with_agreement"]:
                loss_agreement, top_pairwise_scores_cross, top_antecedent_idx_cross = \
                    self.do_scoring_cross(top_span_emb_src, top_span_starts_src, top_span_ends_src,
                                          num_top_spans_src, input_ids_src, input_mask_src,
                                          speaker_ids_src, genre_src, top_span_emb_trg,
                                          top_span_starts_trg, top_span_ends_trg, num_top_spans_trg, input_ids_trg,
                                          input_mask_trg, speaker_ids_trg)
                # print(f'debugging loss_agreement:{loss_agreement}')
                # print(f'debugging loss_coref:{loss_coref}')
                return loss_coref + loss_agreement
                #
                # # compute the coref scores between src- and trg-side mentions with the same scorer
                # # in the supervised coref-task
                # # then select the pair with highest score for each trg-side mention,
                # # then maximize these scores of all target mentions
                # # need to add the non-antecedent scores (0) as in the coref-scorer module
                # num_spans_src = top_span_emb_src.shape[0]
                # num_spans_trg = top_span_emb_trg.shape[0]
                # scores = torch.zeros(1, num_spans_trg, device=device)
                # #  compute the scores in a loop, to save memory, but will take more time, sigh
                # for src_span in top_span_emb_src:
                #     scores_src = []
                #     src_span = torch.unsqueeze(src_span, 0).repeat(num_spans_trg, 1)
                #     similarity_emb = src_span * top_span_emb_trg
                #     pair_emb = torch.cat([src_span, top_span_emb_trg, similarity_emb], -1)
                #     # need to pad tp pair_embed, so that has the same shape with  that in original coref-scorer func
                #     # i.e., self.span_emb_size * 3  --> self.pair_emb_size
                #     pair_emb = torch.nn.functional.pad(pair_emb, (0, (self.pair_emb_size - self.span_emb_size * 3)))
                #     pairwise_scores_src = self.coref_score_ffnn(pair_emb)
                #     # adapter
                #     pairwise_scores_src = torch.squeeze(self.coref_score_ffnn_trg_adapter(pairwise_scores_src), -1).reshape(1, -1)
                #     scores = torch.cat([scores, pairwise_scores_src], dim=0)
                #
                # scores = torch.cat([scores, torch.zeros(num_spans_src + 1, 1, device=device)], dim=-1)
                #
                # loss_agreement_mention = 0
                # # from source-side to target-side
                # idx_map = torch.argmax(scores[1:], dim=-1).tolist()
                # for row, col in enumerate(idx_map):
                #     loss_agreement_mention += torch.exp(- scores[1:][row][col])
                #
                # # from trg-side to src-side
                # idx_map = torch.argmax(scores[:, :-1], dim=0).tolist()
                # for col, row in enumerate(idx_map):
                #     loss_agreement_mention += torch.exp(- scores[:, :-1][row][col])
                # loss_agreement_mention /= (num_spans_src + num_spans_trg)
                #
                # '''parallel computation, rather than using a loop'''
                # # span_emb = torch.cat((top_span_emb_src, top_span_emb_trg), dim=0)
                # # num_spans = span_emb.shape[0]
                # # num_spans_src = top_span_emb_src.shape[0]
                # # num_spans_trg = top_span_emb_trg.shape[0]
                # # target_emb = torch.unsqueeze(span_emb, 1).repeat(1, num_spans, 1)
                # # antecedent_emb = span_emb.repeat(num_spans, 1, 1)
                # # similarity_emb = target_emb * antecedent_emb
                # # pair_emb = torch.cat([target_emb, antecedent_emb, similarity_emb], 2)
                # # # need to pad tp pair_embed, so that has the same shape with  that in original coref-scorer func
                # # # i.e., self.span_emb_size * 3  --> self.pair_emb_size
                # # pair_emb = torch.nn.functional.pad(pair_emb, (0, (self.pair_emb_size - self.span_emb_size * 3)))
                # # pairwise_scores = torch.squeeze(self.coref_score_ffnn(pair_emb), 2)
                # # antecedent_scores = torch.cat([pairwise_scores, torch.zeros(num_spans, 1, device=device)],
                # #                               dim=1)  # [num top spans, max top antecedents + 1]
                # # loss_agreement_mention = 0
                # # # from source-side to target-side
                # # src2trg_scores = torch.narrow(antecedent_scores, 0, 0, num_spans_src)
                # # src2trg_scores = torch.narrow(src2trg_scores, 1, num_spans_src, num_spans_trg + 1)
                # # # src2trg_scores = antecedent_scores[:num_spans_src][num_spans_src:]  # [num_span_src, num_span_trg+1]
                # # idx_map = torch.argmax(src2trg_scores, dim=-1).tolist()
                # # for row, col in enumerate(idx_map):
                # #     loss_agreement_mention += torch.exp(- src2trg_scores[row][col])
                # #
                # # # loss_agreement_mention /= num_spans_src
                # # # from trg-side to src-side
                # # trg2src_scores = torch.narrow(antecedent_scores, 0, num_spans_src, num_spans_trg)
                # # trg2src_scores = torch.narrow(trg2src_scores, 1, 0, num_spans_src)
                # # # trg2src_scores = pairwise_scores[num_spans_src:][:num_spans_src]  # [num_span_trg, num_span_src]
                # # trg2src_scores = torch.cat([trg2src_scores, torch.zeros(num_spans_trg, 1, device=device)], dim=1)
                # # idx_map = torch.argmax(trg2src_scores, dim=-1).tolist()
                # # for row, col in enumerate(idx_map):
                # #     loss_agreement_mention += torch.exp(- trg2src_scores[row][col])
                # #
                # # # loss_agreement_mention /= num_spans_trg
                # # loss_agreement_mention /= num_spans
                #
                # # # document-level agreement
                # # doc_src = torch.mean(top_span_emb_src, 0, keepdim=True)
                # # doc_trg = torch.mean(top_span_emb_trg, 0, keepdim=True)
                # # similarity_emb = doc_src * doc_trg
                # # pair_emb = torch.cat([doc_src, doc_trg, similarity_emb], -1)
                # # pairwise_scores_doc = self.coref_score_ffnn(pair_emb)[0][0]
                # # loss_agreement_document = torch.exp(- pairwise_scores_doc)
                #
                # # return loss_src + loss_trg + loss_agreement_mention + loss_agreement_document
                # return loss_coref + loss_agreement_mention
            else:
                return loss_coref
        else:
            # only deal with the source-side
            bert, side, is_training, input_ids, input_mask, speaker_ids, sentence_map, genre, \
            gold_starts, gold_ends, gold_mention_cluster_map = self.bert, "source", is_training_src, \
                                                               input_ids_src, input_mask_src, speaker_ids_src, \
                                                               sentence_map_src, genre_src, gold_starts_src, \
                                                               gold_ends_src, gold_mention_cluster_map_src

            top_span_emb, top_span_starts, top_span_ends, num_top_spans, top_span_cluster_ids, \
                top_span_mention_scores, speaker_ids = self.get_span_rep(bert, side, is_training, input_ids,
                                                                         input_mask, speaker_ids, sentence_map,
                                                                         gold_starts, gold_ends,
                                                                         gold_mention_cluster_map)

            top_antecedent_idx, top_antecedent_scores = self.do_scoring(top_span_emb, top_span_starts,
                                                                        top_span_ends, num_top_spans,
                                                                        top_span_cluster_ids,
                                                                        top_span_mention_scores, do_loss,
                                                                        input_ids, input_mask,
                                                                        speaker_ids, genre)

            return top_span_starts, top_span_ends, top_antecedent_idx, top_antecedent_scores

    def _extract_top_spans(self, candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans):
        """ Keep top non-cross-overlapping candidates ordered by scores; compute on CPU because of loop """
        selected_candidate_idx = []
        start_to_max_end, end_to_min_start = {}, {}
        for candidate_idx in candidate_idx_sorted:
            if len(selected_candidate_idx) >= num_top_spans:
                break
            # Perform overlapping check
            span_start_idx = candidate_starts[candidate_idx]
            span_end_idx = candidate_ends[candidate_idx]
            cross_overlap = False
            for token_idx in range(span_start_idx, span_end_idx + 1):
                max_end = start_to_max_end.get(token_idx, -1)
                if token_idx > span_start_idx and max_end > span_end_idx:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(token_idx, -1)
                if token_idx < span_end_idx and 0 <= min_start < span_start_idx:
                    cross_overlap = True
                    break
            if not cross_overlap:
                # Pass check; select idx and update dict stats
                selected_candidate_idx.append(candidate_idx)
                max_end = start_to_max_end.get(span_start_idx, -1)
                if span_end_idx > max_end:
                    start_to_max_end[span_start_idx] = span_end_idx
                min_start = end_to_min_start.get(span_end_idx, -1)
                if min_start == -1 or span_start_idx < min_start:
                    end_to_min_start[span_end_idx] = span_start_idx
        # Sort selected candidates by span idx
        selected_candidate_idx = sorted(selected_candidate_idx,
                                        key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))
        if len(selected_candidate_idx) < num_top_spans:  # Padding
            selected_candidate_idx += ([selected_candidate_idx[0]] * (num_top_spans - len(selected_candidate_idx)))
        return selected_candidate_idx

    def get_predicted_antecedents(self, antecedent_idx, antecedent_scores):
        """ CPU list input """
        predicted_antecedents = []
        for i, idx in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if idx < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents

    def get_predicted_clusters(self, span_starts, span_ends, antecedent_idx, antecedent_scores):
        """ CPU list input """
        # Get predicted antecedents
        predicted_antecedents = self.get_predicted_antecedents(antecedent_idx, antecedent_scores)

        # Get predicted clusters
        mention_to_cluster_id = {}
        predicted_clusters = []
        for i, predicted_idx in enumerate(predicted_antecedents):
            if predicted_idx < 0:
                continue
            assert i > predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx}'
            # Check antecedent's cluster
            antecedent = (int(span_starts[predicted_idx]), int(span_ends[predicted_idx]))
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1)
            if antecedent_cluster_id == -1:
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent])
                mention_to_cluster_id[antecedent] = antecedent_cluster_id
            # Add mention to cluster
            mention = (int(span_starts[i]), int(span_ends[i]))
            predicted_clusters[antecedent_cluster_id].append(mention)
            mention_to_cluster_id[mention] = antecedent_cluster_id

        predicted_clusters = [tuple(c) for c in predicted_clusters]
        return predicted_clusters, mention_to_cluster_id, predicted_antecedents

    def update_evaluator(self, span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator):
        predicted_clusters, mention_to_cluster_id, _ = self.get_predicted_clusters(span_starts, span_ends,
                                                                                   antecedent_idx, antecedent_scores)
        mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters
