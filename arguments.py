from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments as OriginalTrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
        
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=".cache", metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(default="haafor", metadata={"help": "The name of the task"})
    data_dir: str = field(default="data_in", metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    dynamic_doc_masking: bool = field(default=False, metadata={"help": "Dynamic doc id masking"})

        

@dataclass
class TrainingArguments(OriginalTrainingArguments):
    
    output_dir: str = field(
        default="data_out",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    
    logging_dir: Optional[str] = field(default="data_out", metadata={"help": "Tensorboard log dir."})
    
    ensemble: bool = field(default=False, metadata={"help": "Ensemble result"})