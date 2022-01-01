from torch.utils.data import Dataset
from collections import Counter
import random
import torch


class MetaTaskCategory(Dataset):

    def __init__(self, examples, num_tasks, num_support, num_query, tokenizer, seed, max_len=510, label=None):
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

        self.create_batch(self.num_tasks)

    def create_batch(self, num_tasks):
        domains = []
        start_domains = self.domains[::]
        for t in range(num_tasks):
            pair = random.sample(start_domains, 2)
            domains.append(pair)
            start_domains.remove(pair[0])
            start_domains.remove(pair[1])

        self.supports = []
        self.queries = []

        for task in range(num_tasks):
            a, b = domains[task]
            domain_examples_a = [dict(e, **{'label': 0}) for e in self.examples if e['domain'] == a]
            domain_examples_b = [dict(e, **{'label': 1}) for e in self.examples if e['domain'] == b]

            examples_a = random.sample(domain_examples_a, self.num_support // 2 + self.num_query // 2)
            try:
                examples_b = random.sample(domain_examples_b, self.num_support // 2 + self.num_query // 2)
            except:
                pass

            example_train = examples_a[:self.num_support // 2] + examples_b[:self.num_support // 2]
            random.shuffle(example_train)
            example_test = examples_a[self.num_support // 2:] + examples_b[self.num_support // 2:]
            random.shuffle(example_test)

            self.supports.append(example_train)
            self.queries.append(example_test)

    def create_feature_set(self, examples):
        all_input_ids = torch.empty(len(examples), self.max_seq_len, dtype=torch.long)
        all_attention_mask = torch.empty(len(examples), self.max_seq_len, dtype=torch.long)
        all_segment_ids = torch.empty(len(examples), self.max_seq_len, dtype=torch.long)
        all_label_ids = torch.empty(len(examples), dtype=torch.long)

        for idx, example in enumerate(examples):
            input_ids = self.tokenizer.encode(example['text'])
            attention_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            while len(input_ids) < self.max_seq_len:
                input_ids.append(0)
                attention_mask.append(0)
                segment_ids.append(0)

            label_id = example['label']
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
