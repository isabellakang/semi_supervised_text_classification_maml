import csv
import torch
import numpy as np

from maml import MAML
from metatask import MetaTask
from utils import TrainingArgs, random_seed, split_examples, bert_tokenizer


def train(neurons=50, epochs=10, lr=0.01, update_steps=2, train_tasks=19, test_tasks=3, num_support=5,
          num_query=5, vat_flag1=False, vat_flag2=False, finetune_flag3=False,
          qry_num=5, print_flag=True, filename=None, seed_value=123):

    args = TrainingArgs(neurons=neurons, epochs=epochs, lr=lr, update_steps=update_steps)
    random_seed(seed_value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maml = MAML(args).to(device)
    maml_vat = MAML(args).to(device)

    train_set, test_set = split_examples("data/amazon_reviews.json")
    trainset = MetaTask(train_set, train_tasks, num_support, num_query, bert_tokenizer, seed_value, max_len=128)
    testset = MetaTask(test_set, test_tasks, num_support, 100 - qry_num - num_support, bert_tokenizer, seed_value,
                       max_len=128)

    test_accuracies = []

    if not filename:
        filename = "results/{}_neurons_{},{},{}.csv".format(neurons, num_support, qry_num, num_query)

    with open(filename, mode='w') as f:
        w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(['epoch', 'maml_acc', 'vat_acc', 'maml_vatloss', 'vat_vatloss'])

        for epoch in range(args.epochs):
            print('Epoch:', epoch)
            # fetch meta_batchsz num of episode each time
            for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(trainset):

                accs = maml(x_spt, y_spt, x_qry, y_qry, vat_flag1, vat_flag2, vat_flag3=False, qry_num=qry_num)
                accs_vat = maml_vat(x_spt, y_spt, x_qry, y_qry, vat_flag1, vat_flag2, vat_flag3=True, qry_num=qry_num)

                if step == 0 and print_flag:
                    print('step:', step, '\ttraining acc:', accs[-1], '\tVAT training acc:', accs_vat[-1])

                if step % 6 == 0:  # evaluation
                    accs_all_test = []
                    accs_all_test_vat = []

                    for _, (x_spt, y_spt, x_qry, y_qry) in enumerate(testset):
                        accs, loss = maml.finetuning(x_spt, y_spt, x_qry, y_qry, vat_flag2=vat_flag2, vat_flag3=False)
                        accs_vat, loss_vat = maml_vat.finetuning(x_spt, y_spt, x_qry, y_qry, vat_flag2=vat_flag2,
                                                                 vat_flag3=finetune_flag3)

                        accs_all_test.append(accs)
                        accs_all_test_vat.append(accs_vat)

                    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                    accs_vat = np.array(accs_all_test_vat).mean(axis=0).astype(np.float16)
                    print('Test acc:', accs[-1], '\tTest acc VAT:', accs_vat[-1])
                    print('Test vat loss:', loss.item(), '\tTest VAT vat loss:', loss_vat.item())
                    test_accuracies.append((accs[-1], accs_vat[-1]))
                    w.writerow([epoch, accs[-1], accs_vat[-1], loss.item(), loss_vat.item()])

                    print("test accuracies are:", test_accuracies)
    x = [elt[0] for elt in test_accuracies][10:]
    y = [elt[1] for elt in test_accuracies][10:]
    print("accuracies of MAML and of +VAT are", np.mean(x), np.mean(y))

    return test_accuracies
