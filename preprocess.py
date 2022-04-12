from nltk.corpus import stopwords
import re
from multiprocessing import Pool
from tqdm.notebook import tqdm
from string import punctuation
from pymystem3 import Mystem
from functools import partial
import torch
from torch.utils.data import TensorDataset, DataLoader


def base_preprocessing(text, analyzer):
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(f'|'.join(["»", "«", "—"]), '', text)
    text = re.sub(' +', ' ', text)
    text = re.sub('[{}]'.format(punctuation), '', text)
    text = analyzer.lemmatize(text)

    return ' '.join([word for word in text if word not in stopwords.words('russian')+[' ', '\n', " "]])


def get_lemmas_from_text(text_series):
    mystem_analyzer = Mystem()
    with Pool(8) as pool:
        lemmas = list(
            tqdm(pool.map(partial(base_preprocessing, analyzer=mystem_analyzer), text_series), total=len(text_series)))
    return lemmas


class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_inputs(example_texts, example_labels, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    input_items = []
    examples = zip(example_texts, example_labels)
    for (_, (text, label)) in enumerate(examples):

        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label))

    return input_items


def get_data_loader(features, batch_size, shuffle=True):
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask,
                         all_segment_ids, all_label_ids)

    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader
