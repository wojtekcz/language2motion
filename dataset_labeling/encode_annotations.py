from pathlib import Path
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import numpy as np

from bert_serving.client import BertClient
bc = BertClient()

# get annotations - sentences

dataset_path = Path('../data/2017-06-22/')

ann_paths = sorted([x for x in dataset_path.iterdir() if '_annotations.json' in x.name]); len(ann_paths)

def get_annotations(x: Path):    
    annotations = eval(x.read_text(encoding='utf-8'))
    sample_id = int(x.name[:5])
    dicts = [{'sample_id': sample_id, 'annotation': a} for a in annotations]    
    return dicts

flatten = lambda l: [item for sublist in l for item in sublist]

ann_dicts = flatten([get_annotations(x) for x in tqdm(ann_paths[:])])

annotations_df = pd.DataFrame(ann_dicts)
print(len(annotations_df))
annotations_df.head()

# create sentence encodings with bert

encodings = bc.encode(annotations_df.annotation.to_list()[:])
encodings.shape

# save, then load dataset

annotations_df2 = annotations_df

# annotations_df2['encoding'] = 
encodings_list = [encodings[x] for x in range(encodings.shape[0])]
annotations_df2['encoding'] = encodings_list
annotations_df2.head()

dataset_path2 = Path('../data')

annotations_df2.to_pickle(str(dataset_path2/'annotations.pkl'))
annotations_df2.to_csv(str(dataset_path2/'annotations.csv'))
