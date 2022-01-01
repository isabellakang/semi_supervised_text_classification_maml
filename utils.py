import random
import json
import torch
import numpy as np
from collections import Counter
from transformers import BertModel, BertTokenizer

LABEL_MAP = {'positive': 1, 'negative': 0, 'neutral': 2, 0: 0, 1: 1, 2: 2}
bert_input_size = 1000
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def random_seed(value):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)


def create_batch_of_tasks(taskset, is_shuffle=False, batch_size=4):
    idxs = list(range(0, len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size, len(taskset)))]


def split_examples(json_path, lower_threshold=30, upper_threshold=100):
    if type(json_path) == str:
        reviews = json.load(open('maml_data/' + json_path))
    else:
        reviews = json_path
    mention_domain = [r['domain'] for r in reviews]
    counts = Counter(mention_domain)
    s = sorted([(key, val) for key, val in counts.items()], key=lambda x: x[1])
    print(s)
    low_resource_domains = [domain for domain in counts if
                            (counts[domain] <= upper_threshold) and (counts[domain] >= lower_threshold)]
    high_resource_domains = [domain for domain in counts if counts[domain] > upper_threshold]
    train_examples = [r for r in reviews if r['domain'] in high_resource_domains]
    test_examples = [r for r in reviews if r['domain'] in low_resource_domains]
    print("low resource:", low_resource_domains)
    print("high resource:", high_resource_domains)
    print("lengths of low and high resource domains:", len(low_resource_domains), len(high_resource_domains))
    print("length of train and test examples are", len(train_examples), len(test_examples))
    return train_examples, test_examples


class TrainingArgs:
    def __init__(self, labels=2, meta_epochs=10,
                 bert_model=BertModel, neurons=100, epochs=10, lr=0.01, update_steps=2, a1=0.1, a2=0.1, a3=0.1):
        self.epochs = epochs
        self.num_labels = labels
        self.meta_epoch = meta_epochs
        self.outer_batch_size = 2
        self.inner_batch_size = 12
        self.meta_lr = 5e-3
        self.update_lr = lr
        self.update_step = update_steps
        self.update_step_test = update_steps
        self.neurons = neurons
        self.bert_model = bert_model.from_pretrained('bert-base-uncased')
        self.alpha1 = None  # a1
        self.alpha2 = None  # a2
        self.alpha3 = 0.05  # a3
        self.xi = 1
        self.eps = 1
