from torch.utils.data import Dataset
from collections import Counter
import random
import torch


class MetaTaskVocab(Dataset):

    def __init__(self, examples, num_tasks, num_support, num_query, tokenizer, seed, max_len=510, vocab=None,
                 vocab_out=None, label=None):
        self.queries = None
        self.supports = None
        self.examples = examples

        mention_domain = [e['domain'] for e in examples]
        self.domain_counter = dict(Counter(mention_domain))
        self.domains = list(self.domain_counter)

        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.examples)

        self.num_tasks = num_tasks
        self.num_support = num_support
        self.num_query = num_query
        self.tokenizer = tokenizer
        self.max_seq_len = max_len

        self.batch_num = 1
        self.vocab = " " + vocab + " "
        self.vocab_out = " " + vocab_out + " "
        self.label = label

        self.create_batch(self.num_tasks)

    def create_batch(self, num_tasks):
        self.supports = []
        self.queries = []

        text_examples = [e for e in self.examples if
                         self.vocab in e['text'].lower() and self.vocab_out not in e['text'].lower()]
        text_examples_opp = [e for e in self.examples if
                             self.vocab_out in e['text'].lower() and self.vocab not in e['text'].lower()]

        #         print('text examples length is', len(text_examples))
        #         print('text examples_opp is', len(text_examples_opp))
        for task in range(num_tasks):
            #             random.seed(self.seed)
            selected_examples = random.sample(text_examples, self.num_support // 2 + self.num_query // 2)
            selected_examples_opp = random.sample(text_examples_opp, self.num_support + self.num_query - (
                    self.num_support // 2 + self.num_query // 2))
            example_train = selected_examples[:(self.num_support // 2)] + selected_examples_opp[
                                                                          :(self.num_support - self.num_support // 2)]
            example_test = selected_examples[(self.num_support // 2):] + selected_examples_opp[
                                                                         (self.num_support - self.num_support // 2):]

            self.supports.append(example_train)
            self.queries.append(example_test)

    def create_feature_set(self, examples):
        all_input_ids = torch.empty(len(examples), self.max_seq_len, dtype=torch.long)
        all_attention_mask = torch.empty(len(examples), self.max_seq_len, dtype=torch.long)
        all_segment_ids = torch.empty(len(examples), self.max_seq_len, dtype=torch.long)
        all_label_ids = torch.empty(len(examples), dtype=torch.long)

        for idx, example in enumerate(examples):
            if self.vocab is not None:
                if self.vocab in example['text']:
                    label_id = self.label
                else:
                    label_id = 1 - self.label
                input_ids = self.tokenizer.encode(
                    example['text'].replace(self.vocab, " [MASK] ").replace(self.vocab_out, ' [MASK] '))
            else:
                input_ids = self.tokenizer.encode(example['text'])
            attention_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            while len(input_ids) < self.max_seq_len:
                input_ids.append(0)
                attention_mask.append(0)
                segment_ids.append(0)

            all_input_ids[idx] = torch.Tensor(input_ids).to(torch.long)
            all_attention_mask[idx] = torch.Tensor(attention_mask).to(torch.long)
            all_segment_ids[idx] = torch.Tensor(segment_ids).to(torch.long)
            all_label_ids[idx] = torch.Tensor([label_id]).to(torch.long)

        return all_input_ids, all_attention_mask, all_segment_ids, all_label_ids

    def __getitem__(self, index):
        # random.seed(self.batch_num)
        support_set = self.create_feature_set(self.supports[index])
        query_set = self.create_feature_set(self.queries[index])
        self.batch_num += 1

        return support_set[:-1], support_set[-1], query_set[:-1], query_set[-1]

    def __len__(self):
        return self.num_tasks
