import glob
import logging
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from torch.optim import Adam
import util
import time
from os.path import join
from metrics import CorefEvaluator
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
import conll
import sys
from tensorize import CorefDataProcessor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class Runner:
    def __init__(self, config_name, gpu_id=0, seed=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed

        # Set up config
        self.config = util.initialize_config(config_name)

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)

        # Set up seed
        if seed:
            util.set_seed(seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        # Set up data
        # self.data = CorefDataProcessor(self.config)
        self.data_src = CorefDataProcessor(self.config, "source", language=self.config['lan_src'])
        self.data_trg = CorefDataProcessor(self.config, "target", language=self.config['lan_trg'])

    def initialize_model(self, saved_model=None, checkpoint=None):
        if 'xlm-roberta' in self.config['bert_tokenizer_name_src']:
            if self.config['gold_mention']:
                if self.config['synthetic']:
                    from model_x_single_gold import CorefModel
                else:
                    from model_x_gold import CorefModel
            else:
                if self.config['synthetic'] or self.config['transfer_zero']:
                    from model_x_single import CorefModel
                else:
                    from model_x import CorefModel
        else:
            from model import CorefModel
        model = CorefModel(self.config, self.device)
        if saved_model:
            self.load_model_pretrained(model, saved_model, self.config['freezing'])
        if self.config['eval_only_src'] or self.config['eval_only_trg'] or self.config['transfer_zero']:
            if not self.config['no_training']:
                self.load_model_checkpoint(model, checkpoint)
        return model

    def train(self, model):
        conf = self.config
        logger.info(conf)
        epochs, grad_accum = conf['num_epochs'], conf['gradient_accumulation_steps']

        model.to(self.device)
        logger.info('Model parameters:')
        for name, param in model.named_parameters():
            logger.info('%s: %s' % (name, tuple(param.shape)))

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info('Tensorboard summary path: %s' % tb_path)

        # Set up data
        # examples_train, examples_dev, examples_test = self.data.get_tensor_examples()
        examples_train_src, examples_dev_src, examples_test_src = self.data_src.get_tensor_examples()
        if conf['synthetic']:
            if not conf['dev_on_srctrg']:
                examples_train_trg = self.data_trg.get_tensor_examples()
                examples_dev_trg, examples_test_trg = examples_train_trg, examples_train_trg
            else:
                examples_train_trg, examples_dev_trg, examples_test_trg = self.data_trg.get_tensor_examples()
        else:
            examples_train_trg, examples_dev_trg, examples_test_trg = self.data_trg.get_tensor_examples()

        stored_info = self.data_src.get_stored_info()
        if conf['dev_on_srctrg']:
            stored_info_trg = self.data_trg.get_stored_info()
        assert len(examples_train_src) == len(examples_train_trg)
        if self.config['gold_mention']:
            # filter doc without gold clusters only for zh en
            for i, (src, trg) in enumerate(zip(examples_train_src, examples_train_trg)):
                (_, data_src), (_, data_trg) = src, trg
                if _ in ["nw/xinhua/00/chtb_0052_0", "nw/xinhua/00/chtb_0071_0", "nw/xinhua/02/chtb_0254_0"]:
                    # error in the data, out of range because of the gold labels (limited sent len during modeling)
                    examples_train_src.remove(src)
                    examples_train_trg.remove(trg)
                if data_src[-1].shape[0] == 0 or data_trg[-1].shape[0] == 0:
                    examples_train_src.remove(src)
                    examples_train_trg.remove(trg)

            for i, (src, trg) in enumerate(zip(examples_dev_src, examples_dev_trg)):
                (_, data_src), (_, data_trg) = src, trg
                if data_src[-1].shape[0] == 0 or data_trg[-1].shape[0] == 0:
                    examples_dev_src.remove(src)
                    examples_dev_trg.remove(trg)

        # examples_train = list(zip(examples_train_src, examples_train_trg))
        examples_train = list(zip(examples_train_src, examples_train_trg))
        # examples_dev = list(zip(examples_dev_src, examples_dev_trg))
        # examples_test = list(zip(examples_test_src, examples_test_trg))

        # Set up optimizer and scheduler
        total_update_steps = len(examples_train) * epochs // grad_accum
        optimizers = self.get_optimizer(model)
        schedulers = self.get_scheduler(optimizers, total_update_steps)

        # Get model parameters for grad clipping
        bert_param, task_param = model.get_params()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(examples_train))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1 = 0
        start_time = time.time()
        model.zero_grad()
        for epo in range(epochs):
            random.shuffle(examples_train)  # Shuffle training set
            for (doc_key, example_src), (doc_key, example_trg) in examples_train:
                # Forward pass
                model.train()
                # example_gpu = [d.to(self.device) for d in example]
                # _, loss = model(*example_gpu)

                example_gpu_src = [d.to(self.device) for d in example_src]
                example_gpu_trg = [d.to(self.device) for d in example_trg]
                # print("training:", doc_key)
                if 'xlm-roberta' in self.config['bert_tokenizer_name_src']:
                    loss = model("source", *example_gpu_src, *example_gpu_trg)
                else:
                    loss = model(*example_gpu_src, *example_gpu_trg)

                # Backward; accumulate gradients and clip by grad norm
                if grad_accum > 1:
                    loss /= grad_accum
                loss.backward()

                if conf['max_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(bert_param, conf['max_grad_norm'])
                    torch.nn.utils.clip_grad_norm_(task_param, conf['max_grad_norm'])
                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    for optimizer in optimizers:
                        optimizer.step()
                    model.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % conf['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                        start_time = end_time

                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', schedulers[0].get_last_lr()[0], len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Task', schedulers[1].get_last_lr()[-1], len(loss_history))

                    # Evaluate
                    if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                        side = "source"
                        f1, _ = self.evaluate(model, examples_dev_src, examples_dev_trg, stored_info, side,
                                              len(loss_history),
                                              official=False, conll_path=self.config['conll_eval_path'],
                                              tb_writer=tb_writer)
                        if f1 > max_f1:
                            max_f1 = f1
                            self.save_model_checkpoint(model, len(loss_history))
                        logger.info('Eval max f1: %.2f' % max_f1)
                        if conf['dev_on_srctrg']:
                            side = "target"
                            f1, _ = self.evaluate(model, examples_dev_src, examples_dev_trg, stored_info_trg, side,
                                                  len(loss_history),
                                                  official=True, conll_path=self.config['conll_eval_path_zh'],
                                                  tb_writer=tb_writer)

                        start_time = time.time()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Wrap up
        tb_writer.close()

        # evaluate based on the last checkpoint
        checkpoint = glob.glob(self.config['log_dir'] + "/model_Apr*bin")[-1]
        model.eval()
        self.load_model_checkpoint(model, checkpoint)
        logger.info('Now evaluating on the source side: test')
        runner.evaluate(model, examples_test_src, examples_test_trg, stored_info, "source", 0, official=True,
                        conll_path=runner.config['conll_test_path'], tb_writer=None)

        return loss_history

    def evaluate(self, model, tensor_examples_src, tensor_examples_trg, stored_info, side,
                 step, official=False, conll_path=None, tb_writer=None):
        if self.config['dev_on_srctrg']:
            official = False
        tensor_examples = tensor_examples_src if side == "source" else tensor_examples_trg
        tensor_examples = list(zip(tensor_examples, tensor_examples))
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction = {}

        span_starts_src_list = []
        # path_matrix = join(self.config['log_dir'], f'matrix_src.pt')

        model.eval()
        for i, ((doc_key, tensor_example_src), (doc_key, tensor_example_trg)) in enumerate(tensor_examples):
            # if self.config['eval_in_domain']:
            #     if doc_key[:2] not in ["nw", "mz", "wb"]:
            #         # out domain examples
            #         continue
            # print("evaluating: ", doc_key)
            gold_clusters = stored_info['gold'][doc_key]
            example_gpu = [d.to(self.device) for d in tensor_example_src]
            # for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            #     gold_clusters = stored_info['gold'][doc_key]
            #     tensor_example = tensor_example[:7]  # Strip out gold
            #     example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                if 'xlm-roberta' in self.config['bert_tokenizer_name_src']:
                    span_starts, span_ends, antecedent_idx, antecedent_scores = model(side, *example_gpu, *example_gpu)
                else:
                    span_starts, span_ends, antecedent_idx, antecedent_scores = model(*example_gpu, *example_gpu)
            # span_starts_src_list.append(span_starts)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            # print("span_starts:{}, antecedent_idx:{}, antecedent_scores:{}".
            #       format(len(span_starts), len(antecedent_idx), len(antecedent_scores)))
            predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores,
                                                        gold_clusters, evaluator)
            doc_to_prediction[doc_key] = predicted_clusters

        p, r, f = evaluator.get_prf()
        metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        # torch.save({"span_starts_src": span_starts_src_list}, path_matrix)


        if official:
            conll_results = conll.evaluate_conll(conll_path, doc_to_prediction, stored_info['subtoken_maps'])
            official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            logger.info('Official avg F1: %.4f' % official_f1)

        if self.config['transfer_hybrid']:
            return doc_to_prediction

        return f * 100, metrics

    def evaluate_hybrid(self, model, tensor_examples_src, tensor_examples_trg, stored_info_trg,
                          step, direct_res=None, official=False, conll_path=None, tb_writer=None):

        tensor_examples = list(zip(tensor_examples_src, tensor_examples_trg))
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction_trg = {}

        # save the generated black-box alignment score matrix and span idx,
        # for manually check what are aligned
        path_matrix = join(self.config['log_dir'], f'matrix4align.pt')
        span_starts_src_list, span_ends_src_list, span_starts_trg_list, span_ends_trg_list, \
        matrix_list, antecedent_idx, src_mentions_predicted = [], [], [], [], [], [], []

        model.eval()
        for i, ((doc_key, tensor_example_src), (doc_key, tensor_example_trg)) in enumerate(tensor_examples):
            gold_clusters_trg = stored_info_trg['gold'][doc_key]
            example_gpu_src = [d.to(self.device) for d in tensor_example_src]
            example_gpu_trg = [d.to(self.device) for d in tensor_example_trg]
            with torch.no_grad():
                span_starts_src, span_ends_src, antecedent_idx_src, antecedent_scores_src, \
                span_starts_trg, span_ends_trg, antecedent_ids_cross, scores_cross_coref = \
                    model("target", *example_gpu_src, *example_gpu_trg)

            # store the tensors of each doc
            span_starts_src_list.append(span_starts_src)
            span_ends_src_list.append(span_ends_src)
            span_starts_trg_list.append(span_starts_trg)
            span_ends_trg_list.append(span_ends_trg)
            matrix_list.append(scores_cross_coref)
            antecedent_idx.append(antecedent_ids_cross)

            # # get the predicted clusters from direct transfer when it is not none
            if direct_res is not None:
                cluster_from_direct_transfer = [list(c) for c in direct_res[doc_key]]
                mentions_in_direct_transfer = [mention for c in direct_res[doc_key] for mention in c]
            else:
                cluster_from_direct_transfer = []
                mentions_in_direct_transfer = []

            # #
            # # # 1. process the src-side clusters; 2. map trg-side spans to src-side spans;
            # # # 3. generate target-side clusters; 4. compare with the gold cluster labels
            span_starts_src, span_ends_src = span_starts_src.tolist(), span_ends_src.tolist()
            span_starts_trg, span_ends_trg = span_starts_trg.tolist(), span_ends_trg.tolist()
            antecedent_idx_src, antecedent_scores_src = antecedent_idx_src.tolist(), antecedent_scores_src.tolist()
            predicted_clusters_src, mention_to_cluster_id_src, _ = model.get_predicted_clusters(span_starts_src,
                                                                                                span_ends_src,
                                                                                                antecedent_idx_src,
                                                                                                antecedent_scores_src)
            mentions_in_src = [mention for c in predicted_clusters_src for mention in c]
            src_mentions_predicted.append(mentions_in_src)

            mention_idx_trg2src = torch.argmax(scores_cross_coref, dim=1).tolist()  # assume a mapping
            # # print("num of mapping trg2src {}, should be {}".format(len(mention_idx_trg2src), len(span_starts_trg)))
            # # # print(mention_idx_trg2src)
            # # # print(set(mention_idx_trg2src))
            span_trg2src = {}
            for idx_src, idx_trg in enumerate(mention_idx_trg2src):
                if (span_starts_trg[idx_trg], span_ends_trg[idx_trg]) not in mentions_in_direct_transfer:
                    # only consider trg mentions that are not in the predicted clusters
                    if (span_starts_src[idx_src], span_ends_src[idx_src]) in mentions_in_src:
                        # and only consider the pair of mentions that the src mention is in the final prediction
                        span_trg2src[(span_starts_trg[idx_trg], span_ends_trg[idx_trg])] = (
                            span_starts_src[idx_src], span_ends_src[idx_src])
            print("before and after filtering {} --> {}".format(len(mention_idx_trg2src), len(span_trg2src)))
            mention_to_cluster_id_trg = {}
            for trg_span, src_span in span_trg2src.items():
                if src_span in mention_to_cluster_id_src:
                    # this src mention is in a final cluster
                    mention_to_cluster_id_trg[trg_span] = mention_to_cluster_id_src[src_span]
            print("num of trg mentions having a cluster {}".format(len(mention_to_cluster_id_trg)))
            predicted_clusters_trg = {}
            for mention_trg, cluster_id in mention_to_cluster_id_trg.items():
                if cluster_id not in predicted_clusters_trg:
                    predicted_clusters_trg[cluster_id] = []
                predicted_clusters_trg[cluster_id].append(mention_trg)

            if self.config['transfer_hybrid']:
                existing_ids = [ids for ids, cluster in predicted_clusters_trg.items()]
                existing_ids.sort()
                # now consider hybrid method, consider both projection and direct transfer
                new_cluster_id = existing_ids[-1] if len(existing_ids) != 0 else 0
                for cluster in cluster_from_direct_transfer:
                    new_cluster_id += 1
                    predicted_clusters_trg[new_cluster_id] = []
                    for trg_span in cluster:
                        mention_to_cluster_id_trg[trg_span] = new_cluster_id
                        predicted_clusters_trg[new_cluster_id].append(trg_span)

            model.update_evaluator_transfer(predicted_clusters_trg, mention_to_cluster_id_trg,
                                            gold_clusters_trg, evaluator)
            num_mentions_with_antecedent_trg = len([m for id_c, c in predicted_clusters_trg.items() for m in c])
            print("num_mentions_with_antecedent_trg: {}".format(num_mentions_with_antecedent_trg))
            # # change the type for official evaluation
            predicted_clusters_trg = [tuple(c) for cluster_id, c in predicted_clusters_trg.items()]
            doc_to_prediction_trg[doc_key] = predicted_clusters_trg

        # # save the x-lingual coreference score matrix and spans info to a tensor file
        # torch.save({"span_starts_src": span_starts_src_list, "span_ends_src": span_ends_src_list,
        #             "span_starts_trg": span_starts_trg_list, "span_ends_trg": span_ends_trg_list,
        #             "src_mentions_predicted": src_mentions_predicted, "antecedent_idx": antecedent_idx,
        #             "matrix": matrix_list}, path_matrix)

        p, r, f = evaluator.get_prf()
        metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        if official:
            conll_results = conll.evaluate_conll(conll_path, doc_to_prediction_trg, stored_info_trg['subtoken_maps'])
            official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            logger.info('Official avg F1: %.4f' % official_f1)

        return f * 100, metrics
        # metrics = {'Eval_Avg_Precision': 1 * 100, 'Eval_Avg_Recall': 1 * 100, 'Eval_Avg_F1': 1 * 100}
        # return 1, metrics


    def evaluate_transfer(self, model, tensor_examples_src, tensor_examples_trg, stored_info_trg,
                          step, direct_res=None, official=False, conll_path=None, tb_writer=None):

        tensor_examples = list(zip(tensor_examples_src, tensor_examples_trg))
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction_trg = {}

        # save the generated black-box alignment score matrix and span idx,
        # for manually check what are aligned
        path_matrix = join(self.config['log_dir'], f'matrix4align.pt')
        span_starts_src_list, span_ends_src_list, span_starts_trg_list, span_ends_trg_list, matrix_list = [], [], [], [], []

        model.eval()
        for i, ((doc_key, tensor_example_src), (doc_key, tensor_example_trg)) in enumerate(tensor_examples):
            gold_clusters_trg = stored_info_trg['gold'][doc_key]
            example_gpu_src = [d.to(self.device) for d in tensor_example_src]
            example_gpu_trg = [d.to(self.device) for d in tensor_example_trg]
            with torch.no_grad():
                span_starts_src, span_ends_src, antecedent_idx_src, antecedent_scores_src, \
                span_starts_trg, span_ends_trg, antecedent_ids_cross, scores_cross_coref = \
                    model("source", *example_gpu_src, *example_gpu_trg)

            # store the tensors of each doc
            span_starts_src_list.append(span_starts_src)
            span_ends_src_list.append(span_ends_src)
            span_starts_trg_list.append(span_starts_trg[antecedent_ids_cross])
            span_ends_trg_list.append(span_ends_trg[antecedent_ids_cross])
            matrix_list.append(scores_cross_coref)

            # # get the predicted clusters from direct transfer when it is not none
            if direct_res is not None:
                cluster_from_direct_transfer = [list(c) for c in direct_res[doc_key]]
                mentions_in_direct_transfer = [mention for c in direct_res[doc_key] for mention in c]
            else:
                cluster_from_direct_transfer = []
                mentions_in_direct_transfer = []
            #
            # # 1. process the src-side clusters; 2. map trg-side spans to src-side spans;
            # # 3. generate target-side clusters; 4. compare with the gold cluster labels
            span_starts_src, span_ends_src = span_starts_src.tolist(), span_ends_src.tolist()
            span_starts_trg, span_ends_trg = span_starts_trg.tolist(), span_ends_trg.tolist()
            antecedent_idx_src, antecedent_scores_src = antecedent_idx_src.tolist(), antecedent_scores_src.tolist()
            predicted_clusters_src, mention_to_cluster_id_src, _ = model.get_predicted_clusters(span_starts_src,
                                                                                                span_ends_src,
                                                                                                antecedent_idx_src,
                                                                                                antecedent_scores_src)
            mentions_in_src = [mention for c in predicted_clusters_src for mention in c]

            mention_idx_trg2src = torch.argmax(scores_cross_coref, dim=1).tolist()  # assume a mapping
            # # print("num of mapping trg2src {}, should be {}".format(len(mention_idx_trg2src), len(span_starts_trg)))
            # # print(mention_idx_trg2src)
            # # print(set(mention_idx_trg2src))
            span_trg2src = {}
            for idx_src, idx_trg in enumerate(mention_idx_trg2src):
                if (span_starts_trg[idx_trg], span_ends_trg[idx_trg]) not in mentions_in_direct_transfer:
                    # only consider trg mentions that are not in the predicted clusters
                    if (span_starts_src[idx_src], span_ends_src[idx_src]) in mentions_in_src:
                        # and only consider the pair of mentions that the src mention is in the final prediction
                        span_trg2src[(span_starts_trg[idx_trg], span_ends_trg[idx_trg])] = (
                            span_starts_src[idx_src], span_ends_src[idx_src])
            print("before and after filtering {} --> {}".format(len(mention_idx_trg2src), len(span_trg2src)))
            mention_to_cluster_id_trg = {}
            for trg_span, src_span in span_trg2src.items():
                if src_span in mention_to_cluster_id_src:
                    # this src mention is in a final cluster
                    mention_to_cluster_id_trg[trg_span] = mention_to_cluster_id_src[src_span]
            print("num of trg mentions having a cluster {}".format(len(mention_to_cluster_id_trg)))
            predicted_clusters_trg = {}
            for mention_trg, cluster_id in mention_to_cluster_id_trg.items():
                if cluster_id not in predicted_clusters_trg:
                    predicted_clusters_trg[cluster_id] = []
                predicted_clusters_trg[cluster_id].append(mention_trg)

            if self.config['transfer_hybrid']:
                existing_ids = [ids for ids, cluster in predicted_clusters_trg.items()]
                existing_ids.sort()
                # now consider hybrid method, consider both projection and direct transfer
                new_cluster_id = existing_ids[-1] if len(existing_ids) != 0 else 0
                for cluster in cluster_from_direct_transfer:
                    new_cluster_id += 1
                    predicted_clusters_trg[new_cluster_id] = []
                    for trg_span in cluster:
                        mention_to_cluster_id_trg[trg_span] = new_cluster_id
                        predicted_clusters_trg[new_cluster_id].append(trg_span)

            model.update_evaluator_transfer(predicted_clusters_trg, mention_to_cluster_id_trg,
                                            gold_clusters_trg, evaluator)
            num_mentions_with_antecedent_trg = len([m for id_c, c in predicted_clusters_trg.items() for m in c])
            print("num_mentions_with_antecedent_trg: {}".format(num_mentions_with_antecedent_trg))
            # change the type for official evaluation
            predicted_clusters_trg = [tuple(c) for cluster_id, c in predicted_clusters_trg.items()]
            doc_to_prediction_trg[doc_key] = predicted_clusters_trg

        # save the x-lingual coreference score matrix and spans info to a tensor file
        torch.save({"span_starts_src": span_starts_src_list, "span_ends_src": span_ends_src_list,
                    "span_starts_trg": span_starts_trg_list, "span_ends_trg": span_ends_trg_list,
                    "matrix": matrix_list}, path_matrix)

        p, r, f = evaluator.get_prf()
        metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        if official:
            conll_results = conll.evaluate_conll(conll_path, doc_to_prediction_trg, stored_info_trg['subtoken_maps'])
            official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            logger.info('Official avg F1: %.4f' % official_f1)

        return f * 100, metrics
        # metrics = {'Eval_Avg_Precision': 1 * 100, 'Eval_Avg_Recall': 1 * 100, 'Eval_Avg_F1': 1 * 100}
        # return 1, metrics


    def evaluate_transfer_simple_version(self, model, tensor_examples_src, tensor_examples_trg, stored_info_trg,
                          step, res_src=None, official=False, conll_path=None, tb_writer=None):
        # TODO CHANGE THE DATA OF ENGLISH-SIDE (should be translated from ZH)
        tensor_examples = list(zip(tensor_examples_src, tensor_examples_trg))
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction_trg = {}

        # save the generated black-box alignment score matrix and span idx,
        # for manually check what are aligned
        path_matrix = join(self.config['log_dir'], f'matrix4align.pt')
        span_starts_src_list, span_ends_src_list, span_starts_trg_list, span_ends_trg_list, matrix_list, \
            src_mentions_predicted= [],[],[],[],[],[]

        model.eval()
        for i, ((doc_key, tensor_example_src), (doc_key, tensor_example_trg)) in enumerate(tensor_examples):
            # print("evaluating: ", doc_key)
            gold_clusters_trg = stored_info_trg['gold'][doc_key]
            example_gpu_src = [d.to(self.device) for d in tensor_example_src]
            example_gpu_trg = [d.to(self.device) for d in tensor_example_trg]
            # for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            #     gold_clusters = stored_info['gold'][doc_key]
            #     tensor_example = tensor_example[:7]  # Strip out gold
            #     example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                span_starts_src, span_ends_src, antecedent_idx_src, antecedent_scores_src, \
                span_starts_trg, span_ends_trg, scores_cross_coref = \
                    model("source", *example_gpu_src, *example_gpu_trg)

            # store the tensors of each doc
            span_starts_src_list.append(span_starts_src)
            span_ends_src_list.append(span_ends_src)
            span_starts_trg_list.append(span_starts_trg)
            span_ends_trg_list.append(span_ends_trg)
            matrix_list.append(scores_cross_coref)

            # # get the predicted clusters from direct transfer when it is not none
            if res_src is not None:
                cluster_from_src = [list(c) for c in res_src[doc_key]]
                mentions_in_src = [mention for c in res_src[doc_key] for mention in c]
            else:
                cluster_from_src = []
                mentions_in_src = []
            src_mentions_predicted.append(mentions_in_src)

            # 1. process the src-side clusters; 2. map trg-side spans to src-side spans;
            # 3. generate target-side clusters; 4. compare with the gold cluster labels
            span_starts_src, span_ends_src = span_starts_src.tolist(), span_ends_src.tolist()
            span_starts_trg, span_ends_trg = span_starts_trg.tolist(), span_ends_trg.tolist()
            antecedent_idx_src, antecedent_scores_src = antecedent_idx_src.tolist(), antecedent_scores_src.tolist()
            # print("span_starts_src:{}, span_starts_trg:{}, antecedent_idx:{}, antecedent_scores:{}".
            #       format(len(span_starts_src), len(span_starts_trg),
            #              len(antecedent_idx_src), len(antecedent_scores_src)))
            predicted_clusters_src, mention_to_cluster_id_src, _ = model.get_predicted_clusters(span_starts_src,
                                                                                                span_ends_src,
                                                                                                antecedent_idx_src,
                                                                                                antecedent_scores_src)

        # save the x-lingual coreference score matrix and spans info to a tensor file
        torch.save({"span_starts_src": span_starts_src_list, "span_ends_src": span_ends_src_list,
                    "span_starts_trg": span_starts_trg_list, "span_ends_trg": span_ends_trg_list,
                    "src_mentions_predicted": src_mentions_predicted, "matrix": matrix_list}, path_matrix)

        # p, r, f = evaluator.get_prf()
        # metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
        # for name, score in metrics.items():
        #     logger.info('%s: %.2f' % (name, score))
        #     if tb_writer:
        #         tb_writer.add_scalar(name, score, step)
        #
        # if official:
        #     conll_results = conll.evaluate_conll(conll_path, doc_to_prediction_trg, stored_info_trg['subtoken_maps'])
        #     official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        #     logger.info('Official avg F1: %.4f' % official_f1)
        #
        # return f * 100, metrics
        metrics = {'Eval_Avg_Precision': 1 * 100, 'Eval_Avg_Recall': 1 * 100, 'Eval_Avg_F1': 1 * 100}
        return 1, metrics


    def evaluate_transfer_v1_simple_version(self, model, tensor_examples_src, tensor_examples_trg, stored_info_trg,
                          step, direct_res=None, official=False, conll_path=None, tb_writer=None):
        # TODO CHANGE THE DATA OF ENGLISH-SIDE (should be translated from ZH)
        tensor_examples = list(zip(tensor_examples_src, tensor_examples_trg))
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction_trg = {}

        # save the generated black-box alignment score matrix and span idx,
        # for manually check what are aligned
        path_matrix = join(self.config['log_dir'], f'matrix4align.pt')
        span_starts_src_list, span_ends_src_list, span_starts_trg_list, span_ends_trg_list, matrix_list = [],[],[],[],[]

        model.eval()
        for i, ((doc_key, tensor_example_src), (doc_key, tensor_example_trg)) in enumerate(tensor_examples):
            # print("evaluating: ", doc_key)
            gold_clusters_trg = stored_info_trg['gold'][doc_key]
            example_gpu_src = [d.to(self.device) for d in tensor_example_src]
            example_gpu_trg = [d.to(self.device) for d in tensor_example_trg]
            # for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            #     gold_clusters = stored_info['gold'][doc_key]
            #     tensor_example = tensor_example[:7]  # Strip out gold
            #     example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                span_starts_src, span_ends_src, antecedent_idx_src, antecedent_scores_src, \
                span_starts_trg, span_ends_trg, scores_cross_coref = \
                    model("source", *example_gpu_src, *example_gpu_trg)

            # store the tensors of each doc
            span_starts_src_list.append(span_starts_src)
            span_ends_src_list.append(span_ends_src)
            span_starts_trg_list.append(span_starts_trg)
            span_ends_trg_list.append(span_ends_trg)
            matrix_list.append(scores_cross_coref)

            # get the predicted clusters from direct transfer when it is not none
            if direct_res is not None:
                cluster_from_direct_transfer = [list(c) for c in direct_res[doc_key]]
                mentions_in_direct_transfer = [mention for c in direct_res[doc_key] for mention in c]
            else:
                cluster_from_direct_transfer = []
                mentions_in_direct_transfer = []

            # 1. process the src-side clusters; 2. map trg-side spans to src-side spans;
            # 3. generate target-side clusters; 4. compare with the gold cluster labels
            span_starts_src, span_ends_src = span_starts_src.tolist(), span_ends_src.tolist()
            span_starts_trg, span_ends_trg = span_starts_trg.tolist(), span_ends_trg.tolist()
            antecedent_idx_src, antecedent_scores_src = antecedent_idx_src.tolist(), antecedent_scores_src.tolist()
            # print("span_starts_src:{}, span_starts_trg:{}, antecedent_idx:{}, antecedent_scores:{}".
            #       format(len(span_starts_src), len(span_starts_trg),
            #              len(antecedent_idx_src), len(antecedent_scores_src)))
            predicted_clusters_src, mention_to_cluster_id_src, _ = model.get_predicted_clusters(span_starts_src,
                                                                                                span_ends_src,
                                                                                                antecedent_idx_src,
                                                                                                antecedent_scores_src)
            # num_mentions_with_antecedent_src = len([m for c in predicted_clusters_src for m in c])
            # print("num_mentions_with_antecedent_src: {}".format(num_mentions_with_antecedent_src))
            # predicted_clusters_src: a list of tuple(tuple(span_start, span_end))
            # map src-side mention to trg-side mentions, (remove the 1st one which is a 0 vector for non-antecedent
            # mention_idx_src2trg = torch.argmax(scores_cross_coref, dim=1).tolist()[1:]

            # # src to trg mapping
            # mention_idx_src2trg = torch.argmax(scores_cross_coref[1:, :-1], dim=1).tolist()  # assume a mapping
            # # print("num of mapping src2trg {}, should be {}".format(len(mention_idx_src2trg), len(span_starts_src)))
            # # print(mention_idx_src2trg)
            # # print(set(mention_idx_src2trg))
            # span_trg2src = {}
            # for j, idx_trg in enumerate(mention_idx_src2trg):
            #     if idx_trg == len(span_starts_trg):
            #         # no antecedent, and skip it
            #         print("skipping, should not happen")
            #         continue
            #     if (span_starts_trg[idx_trg], span_ends_trg[idx_trg]) not in mentions_in_direct_transfer:
            #         # only consider trg mentions that are not in the predicted clusters
            #         span_trg2src[(span_starts_trg[idx_trg], span_ends_trg[idx_trg])] = (
            #             span_starts_src[j], span_ends_src[j])
            # print("before and after filtering {} --> {}".format(len(mention_idx_src2trg), len(span_trg2src)))

            # trg to src mapping
            mention_idx_trg2src = torch.argmax(scores_cross_coref[1:, :-1], dim=0).tolist()  # assume a mapping
            # print("num of mapping trg2src {}, should be {}".format(len(mention_idx_trg2src), len(span_starts_trg)))
            # print(mention_idx_trg2src)
            # print(set(mention_idx_trg2src))
            span_trg2src = {}
            for idx_trg, idx_src in enumerate(mention_idx_trg2src):
                # if idx_src == 0:
                #     # no antecedent in the src-side, so skip
                #     continue
                # else:
                #     idx_src -= 1  # need a shift
                if (span_starts_trg[idx_trg], span_ends_trg[idx_trg]) not in mentions_in_direct_transfer:
                    # only consider trg mentions that are not in the predicted clusters
                    span_trg2src[(span_starts_trg[idx_trg], span_ends_trg[idx_trg])] = (
                        span_starts_src[idx_src], span_ends_src[idx_src])
            print("before and after filtering {} --> {}".format(len(mention_idx_trg2src), len(span_trg2src)))
            mention_to_cluster_id_trg = {}
            for trg_span, src_span in span_trg2src.items():
                if src_span in mention_to_cluster_id_src:
                    # this src mention is in a final cluster
                    mention_to_cluster_id_trg[trg_span] = mention_to_cluster_id_src[src_span]
            print("num of trg mentions having a cluster {}".format(len(mention_to_cluster_id_trg)))
            predicted_clusters_trg = {}
            for mention_trg, cluster_id in mention_to_cluster_id_trg.items():
                if cluster_id not in predicted_clusters_trg:
                    predicted_clusters_trg[cluster_id] = []
                predicted_clusters_trg[cluster_id].append(mention_trg)

            if self.config['transfer_hybrid']:
                existing_ids = [ids for ids, cluster in predicted_clusters_trg.items()]
                existing_ids.sort()
                # now consider hybrid method, consider both projection and direct transfer
                new_cluster_id = existing_ids[-1] if len(existing_ids) != 0 else 0
                for cluster in cluster_from_direct_transfer:
                    new_cluster_id += 1
                    predicted_clusters_trg[new_cluster_id] = []
                    for trg_span in cluster:
                        mention_to_cluster_id_trg[trg_span] = new_cluster_id
                        predicted_clusters_trg[new_cluster_id].append(trg_span)

            model.update_evaluator_transfer(predicted_clusters_trg, mention_to_cluster_id_trg,
                                            gold_clusters_trg, evaluator)
            num_mentions_with_antecedent_trg = len([m for id_c, c in predicted_clusters_trg.items() for m in c])
            print("num_mentions_with_antecedent_trg: {}".format(num_mentions_with_antecedent_trg))
            # change the type for official evaluation
            predicted_clusters_trg = [tuple(c) for cluster_id, c in predicted_clusters_trg.items()]
            doc_to_prediction_trg[doc_key] = predicted_clusters_trg

        # save the x-lingual coreference score matrix and spans info to a tensor file
        torch.save({"span_starts_src": span_starts_src_list, "span_ends_src": span_ends_src_list,
                    "span_starts_trg": span_starts_trg_list, "span_ends_trg": span_ends_trg_list,
                    "matrix": matrix_list}, path_matrix)

        p, r, f = evaluator.get_prf()
        metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        if official:
            conll_results = conll.evaluate_conll(conll_path, doc_to_prediction_trg, stored_info_trg['subtoken_maps'])
            official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            logger.info('Official avg F1: %.4f' % official_f1)

        return f * 100, metrics

    def predict(self, model, tensor_examples):
        logger.info('Predicting %d samples...' % len(tensor_examples))
        model.to(self.device)
        predicted_spans, predicted_antecedents, predicted_clusters = [], [], []

        for i, tensor_example in enumerate(tensor_examples):
            tensor_example = tensor_example[:7]
            example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*example_gpu)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            clusters, mention_to_cluster_id, antecedents = model.get_predicted_clusters(span_starts, span_ends,
                                                                                        antecedent_idx,
                                                                                        antecedent_scores)

            spans = [(span_start, span_end) for span_start, span_end in zip(span_starts, span_ends)]
            predicted_spans.append(spans)
            predicted_antecedents.append(antecedents)
            predicted_clusters.append(clusters)

        return predicted_clusters, predicted_spans, predicted_antecedents

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        bert_param, task_param = model.get_params(named=True)
        grouped_bert_param = [
            {
                'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
                'lr': float(self.config['bert_learning_rate']),
                'weight_decay': float(self.config['adam_weight_decay'])
            }, {
                'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
                'lr': float(self.config['bert_learning_rate']),
                'weight_decay': 0.0
            }
        ]
        optimizers = [
            AdamW(grouped_bert_param, lr=float(self.config['bert_learning_rate']),
                  eps=float(self.config['adam_eps'])),
            Adam(model.get_params()[1], lr=float(self.config['task_learning_rate']),
                 eps=float(self.config['adam_eps']), weight_decay=0)
        ]
        return optimizers
        # grouped_parameters = [
        #     {
        #         'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': 0.0
        #     }, {
        #         'params': [p for n, p in task_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in task_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': 0.0
        #     }
        # ]
        # optimizer = AdamW(grouped_parameters, lr=self.config['task_learning_rate'], eps=self.config['adam_eps'])
        # return optimizer

    def get_scheduler(self, optimizers, total_update_steps):
        # Only warm up bert lr
        warmup_steps = int(total_update_steps * float(self.config['warmup_ratio']))

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
            )

        def lr_lambda_task(current_step):
            return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task)
        ]
        return schedulers
        # return LambdaLR(optimizer, [lr_lambda_bert, lr_lambda_bert, lr_lambda_task, lr_lambda_task])

    def save_model_checkpoint(self, model, step):
        if step < self.config['save_from_steps']:
            return  # Debug
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_{step}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, model, checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % checkpoint)

    def load_model_pretrained(self, model, model_saved, freezing):
        dict_params = torch.load(model_saved, map_location=torch.device('cpu'))
        del dict_params['emb_segment_distance.weight']  # not for this parameter, because its shape varies
        model.load_state_dict(dict_params, strict=False)
        logger.info('Loaded model from %s' % model_saved)
        if freezing:
            # freeze these pretrained parameters,
            for name, param in model.named_parameters():
                if name in dict_params and "coref_score_ffnn" not in name:
                    # exclude the "coref_score_ffnn"
                    param.requires_grad = False


if __name__ == '__main__':
    config_name, gpu_id, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    runner = Runner(config_name, gpu_id, seed=seed)
    model = runner.initialize_model(runner.config['saved_model'], runner.config['checkpoint'])
    if runner.config['eval_only_src'] or runner.config['eval_only_trg'] or runner.config['transfer_zero']:
        _, examples_dev_src, examples_test_src = runner.data_src.get_tensor_examples()
        if runner.config['synthetic']:
            examples_dev_trg, examples_test_trg = examples_dev_src, examples_test_src
        else:
            _, examples_dev_trg, examples_test_trg = runner.data_trg.get_tensor_examples()
        # examples = list(zip(examples_test_src, examples_test_trg)) \
        #     if runner.config['testing'] else list(zip(examples_dev_src, examples_dev_trg))
        stored_info_src = runner.data_src.get_stored_info()
        stored_info_trg = runner.data_trg.get_stored_info()
        if runner.config['eval_only_src']:
            # logger.info('Now evaluating on the source side: dev')
            # runner.evaluate(model, examples_dev_src, examples_dev_trg, stored_info_src, "source", 0, official=True,
            #                 conll_path=runner.config['conll_eval_path'], tb_writer=None)
            logger.info('Now evaluating on the source side: test')
            runner.evaluate(model, examples_test_src, examples_test_trg, stored_info_src, "source", 0, official=True,
                            conll_path=runner.config['conll_test_path'], tb_writer=None)
        if runner.config['eval_only_trg'] and not runner.config['synthetic']:
            # logger.info('Now evaluating on the target side: dev')
            # runner.evaluate(model, examples_dev_src, examples_dev_trg, stored_info_trg, "target", 0, official=True,
            #                 conll_path=runner.config['conll_eval_path'], tb_writer=None)
            logger.info('Now evaluating on the target side: test')
            runner.evaluate(model, examples_test_src, examples_test_trg, stored_info_trg, "target", 0, official=True,
                            conll_path=runner.config['conll_test_path'], tb_writer=None)
        if runner.config['transfer_zero']:
            # logger.info('Now evaluating on transfer-zero: dev')
            # runner.evaluate_transfer(model, examples_dev_src, examples_dev_trg, stored_info_trg, 0,
            #                          official=True, conll_path=runner.config['conll_eval_path'], tb_writer=None)
            # if runner.config['transfer_hybrid']:
            #     # new for hybrid method
            #     # res_src = runner.evaluate(model, examples_test_src, examples_test_trg, stored_info_src, "source", 0,
            #     #                              official=False,
            #     #                              conll_path=runner.config['conll_test_path'], tb_writer=None)
            #     res_src = runner.evaluate(model, examples_test_src, examples_test_trg, stored_info_src, "source", 0,
            #                               official=False,
            #                               conll_path=runner.config['conll_test_path'], tb_writer=None)
            #
            #     runner.config['direct_transfer'] = False
            #     runner.config['with_adapters'] = True
            #     # runner.evaluate_transfer_simple_version(model, examples_test_src, examples_test_trg, stored_info_trg, 0,
            #     #                        res_src=res_src,
            #     #                        official=True, conll_path=runner.config['conll_test_path'], tb_writer=None)
            #     runner.evaluate_hybrid(model, examples_test_src, examples_test_trg, stored_info_trg, 0,
            #                              direct_res=res_src,
            #                              official=True, conll_path=runner.config['conll_test_path'], tb_writer=None)
            # el
            if runner.config['direct_transfer']:
                direct_res = runner.evaluate(model, examples_test_src, examples_test_trg, stored_info_trg, "target", 0,
                                             official=True,
                                             conll_path=runner.config['conll_test_path'], tb_writer=None)
                if runner.config['transfer_hybrid']:
                    runner.config['direct_transfer'] = False
                    runner.config['with_adapters'] = True
                    runner.evaluate_transfer(model, examples_test_src, examples_test_trg, stored_info_trg, 0,
                                             direct_res=direct_res,
                                             official=True, conll_path=runner.config['conll_test_path'], tb_writer=None)
            else:
                logger.info('Now evaluating on transfer-zero: test')
                runner.evaluate_transfer(model, examples_test_src, examples_test_trg, stored_info_trg, 0,
                                         direct_res=None,
                                         official=True, conll_path=runner.config['conll_test_path'], tb_writer=None)
    else:
        runner.train(model)
