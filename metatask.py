from torch.utils.data import Dataset
from collections import Counter
import random
import torch
from utils import LABEL_MAP


class MetaTask(Dataset):
    """
    Each metatask has num_tasks batches, where each batch contains num_support support samples
    and num_query query samples.
    """

    def __init__(self, examples, num_tasks, num_support, num_query, tokenizer, seed, max_len=510):
        """
        examples: list of samples
        num_tasks: number of training tasks
        num_support: number of support sample per task
        num_query: number of query sample per task
        """
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
        self.supports = []
        self.queries = []

        task_domains = random.sample(self.domains, num_tasks)

        for task in range(num_tasks):
            domain = task_domains[task]
            domain_examples = [e for e in self.examples if e['domain'] == domain]

            random.seed(self.seed)
            selected_examples = random.sample(domain_examples, self.num_support + self.num_query)
            example_train = selected_examples[:self.num_support]
            example_test = selected_examples[self.num_support:]

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

            label_id = LABEL_MAP[example['label']]
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
