import os
import torch
import tqdm
import logging
import pandas as pd
import random
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
import copy
from typing import List, Optional
from dataclasses import dataclass
from filelock import FileLock
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class HaaforDatasetInputExample:
    
    example_id: str
    before_headline: str
    before_body: str
    after_headline: str
    after_body: str
    label: Optional[int]
        

@dataclass
class HaaforDatasetInputFeature:
    
    input_ids: List[int]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    doc_ids: Optional[List[List[int]]]
    labels: Optional[int]
            
            
class NSPDataset(Dataset):

    features: List[HaaforDatasetInputFeature]

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: str = "train",
        dynamic_doc_masking: bool=False
    ):
        self.dynamic_doc_masking = dynamic_doc_masking
        self.mode = mode
        self.max_seq_length = max_seq_length
        processor = processors[task]()

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(mode, tokenizer.__class__.__name__, str(max_seq_length), task,),
        )
        
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")

                if mode == "dev":
                    examples = processor.get_dev_examples(data_dir)
                elif mode == "test":
                    examples = processor.get_test_examples(data_dir)
                else:
                    examples = processor.get_train_examples(data_dir)

                logger.info("Training examples: {}".format(len(examples)))
                self.features = convert_examples_to_features(examples, max_seq_length, tokenizer,)
                logger.info("Saving features into cached file {}".format(cached_features_file))
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> HaaforDatasetInputFeature:
        # doc id dynamic masking
        if self.mode == "train" and self.dynamic_doc_masking and random.random() < 0.15:
            rd = random.random()
            first_index = self.features[i].token_type_ids.index(1)
            second_index = self.features[i].token_type_ids.count(1)
            dynamic_features = copy.deepcopy(self.features[i])
            if 0 <= rd < 0.4:
                dynamic_features.doc_ids = [0] * first_index + dynamic_features.doc_ids[first_index:]
            elif 0.45 <= rd < 0.8:
                dynamic_features.doc_ids = dynamic_features.doc_ids[:first_index] + [0] * second_index + [1] * (self.max_seq_length - first_index - second_index)
            else:
                dynamic_features.doc_ids = [0] * (first_index + second_index) + dynamic_features.doc_ids[first_index + second_index:]

            return dynamic_features

        return self.features[i]

    
class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()
    
    
class HaaforProcessor(DataProcessor):
    
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        train = self._read_csv(data_dir, "train")                
                        
        return self._create_examples(train, "train")

    
    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        dev = self._read_csv(data_dir, "dev")
        
        return self._create_examples(dev, "dev")

    
    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        test = self._read_csv(data_dir, "test")
        
        return self._create_examples(test, "test")

    
    def _read_csv(self, input_dir, set_type):
        df = pd.read_csv(input_dir + "/{}.csv".format(set_type))
        data = df.to_dict(orient='index')
        
        return data

    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        dic_examples = []
        for i, data_raw in corpus.items():
            examples.append(
                HaaforDatasetInputExample(
                    example_id = data_raw["ORIGINAL_INDEX"],
                    before_headline = data_raw["BEFORE_HEADLINE"],
                    before_body = data_raw["BEFORE_BODY"],
                    after_headline = data_raw["AFTER_HEADLINE"],  
                    after_body = data_raw["AFTER_BODY"],
                    label = data_raw["LABEL"] if "LABEL" in data_raw else None,
                )
            )
            
            if set_type == "train":
                dic_examples.append(
                str(data_raw["BEFORE_HEADLINE"]) + " <HBS> " +
                str(data_raw["BEFORE_BODY"])
                )
                dic_examples.append(
                str(data_raw["AFTER_HEADLINE"]) + " <HBS> " +
                str(data_raw["AFTER_BODY"])
                )
        
        if set_type == "train":
            set_examples = set(dic_examples)
            dic_examples = defaultdict(int, enumerate(d for d in set_examples))
            inv_dic = defaultdict(int, {v: k+2 for k, v in dic_examples.items()})
            inv_dic["[UNK]"] = 0
            inv_dic["[PAD]"] = 1

            torch.save(inv_dic, "data_in/cached_doc_id_dict")
                
        return examples


        
def convert_examples_to_features(
    examples: List[HaaforDatasetInputExample], max_length: int, tokenizer: PreTrainedTokenizer
) -> List[HaaforDatasetInputFeature]:
    """
    Loads a data file into a list of `HaaforDatasetInputExample`
    """
    
    doc_id_dic = torch.load("data_in/cached_doc_id_dict")
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("\nWriting example {} of {}".format(ex_index, len(examples)))

        text_a = example.before_headline + " <HBS> " + example.before_body
        text_b = example.after_headline + " <HBS> " + example.after_body
        
        text_a_doc_id = doc_id_dic[text_a]
        text_b_doc_id = doc_id_dic[text_b]
        
        inputs = tokenizer(
            text_a,
            text_b,
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        
        
        first_ids = inputs.token_type_ids.index(1)
        second_ids = inputs.token_type_ids.count(1)
        
        doc_ids = [text_a_doc_id] * first_ids + [text_b_doc_id] * second_ids + [1] * (max_length - first_ids - second_ids)
        
        features.append(
            HaaforDatasetInputFeature(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask if "attention_mask" in inputs else None,
                token_type_ids=inputs.token_type_ids if "token_type_ids" in inputs else None,
                doc_ids=doc_ids,
                labels=example.label
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: {}".format(f))

    return features

processors = {"haafor": HaaforProcessor}