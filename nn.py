import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
import torch.nn as nn

from metatask import MetaTask
from utils import random_seed


class Net(nn.Module):
    def __init__(self, num_input, num_hidden, num_output=2):
        super(Net, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.inner_batch_size = 12
        self.inner_update_lr = 5e-3

        self.classifier = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            nn.BatchNorm1d(num_hidden),  # batch norm
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            #           nn.ReLU(),
            #           # nn.Dropout(0.25),
            #           nn.Linear(num_hidden, num_output),
        )

    def forward(self, x, bn_training=True):
        if not bn_training:
            with torch.no_grad():
                return self.classifier(x)
        else:
            return self.classifier(x)

    def pretrain(self, num_epochs, seed, train_examples, num_tasks, num_support, num_query):
        random_seed(seed)
        train_sets = set()

        self.to(self.device)

        for epoch in range(num_epochs):
            train_set = MetaTask(train_examples, num_tasks=num_tasks, num_support=num_support, num_query=num_query,
                                 tokenizer=self.tokenizer, seed=epoch, max_len=128)
            train_sets |= {elt for elt in train_set}

            train_list = list(train_sets)

            self.train()
            for epoch in range(num_epochs):
                for task_id, task in enumerate(train_list):
                    cross_entropy = nn.CrossEntropyLoss()
                    support, query = task

                    support_dataloader = DataLoader(support, shuffle=True, batch_size=self.inner_batch_size)
                    inner_optimizer = Adam(self.parameters(), lr=self.inner_update_lr)

                    for i_batch, examples_batch in enumerate(support_dataloader):
                        input_ids, attention_mask, segment_ids, label_id = (elt.to(self.device) for elt in
                                                                            examples_batch)

                        with torch.no_grad():
                            x = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)[1]
                        outputs = self.forward(x)

                        loss = cross_entropy(outputs, label_id)

                        loss.backward()

                        inner_optimizer.step()
                        inner_optimizer.zero_grad()
