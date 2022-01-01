# semi_supervised_text_classication_maml

maml.py contains the MAML meta learner.

vat.py contains the virtual adversarial training method from https://github.com/lyakaap/VAT-pytorch/blob/master/vat.py as described by Miyato et. al.

Each of the MetaTask classes create the different types of meta tasks (sentiment, category, cloze-style).
You can create your own tasks modeled after the MetaTask class for custom classes.

training.py contains train() which runs the meta learning script on data/amazon_reviews.json.