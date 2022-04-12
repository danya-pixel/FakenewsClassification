from nltk.corpus import stopwords
import re
from multiprocessing import Pool
from tqdm.notebook import tqdm
from string import punctuation
from pymystem3 import Mystem


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
            tqdm(pool.starmap(base_preprocessing, zip(text_series, mystem_analyzer)), total=len(text_series)))
    return lemmas
