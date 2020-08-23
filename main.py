import logging
import os
from transformers import (
    AutoConfig, 
    AutoTokenizer,
    HfArgumentParser,
    set_seed
)
from modeling import AlbertForSequenceClassificationV2
from datasets import NSPDataset
from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from trainer import SentenceOrderPredictorTrainer, prediction

def main():
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(model_args)
    print(data_args)
    print(training_args)
    
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    
    if not os.path.exists(training_args.logging_dir):
        os.mkdir(training_args.logging_dir)
    
    # Setup logging
    logger = logging.getLogger(__name__)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.FileHandler(training_args.logging_dir + "/logging.log", 'w', encoding='utf-8'), logging.StreamHandler()]
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)
        
    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                                        cache_dir=model_args.cache_dir, num_labels=2)
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                                              cache_dir=model_args.cache_dir, additional_special_tokens=["<HBS>"])

    train = NSPDataset(data_args.data_dir, tokenizer, data_args.task_name, data_args.max_seq_length, data_args.overwrite_cache, "train", data_args.dynamic_doc_masking)
    dev = NSPDataset(data_args.data_dir, tokenizer, data_args.task_name, data_args.max_seq_length, data_args.overwrite_cache, "dev")
    test = NSPDataset(data_args.data_dir, tokenizer, data_args.task_name, data_args.max_seq_length, data_args.overwrite_cache, "test")

    model = AlbertForSequenceClassificationV2.from_pretrained(model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir)

    model.resize_token_embeddings(len(tokenizer))
    
    trainer = SentenceOrderPredictorTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev
    )

    # Training
    if training_args.do_train:

        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)    
 
    
    # Evaluation
    
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                results.update(result)

    logger.info("Validation set result : {}".format(results))


    if training_args.do_predict:
        logger.info("*** Test ***")

        predictions = trainer.predict(test_dataset=test)

        output_test_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_master():
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results *****")
                logger.info("{}".format(predictions))
                writer.write("prediction : \n{}\n\n".format(prediction(predictions.predictions).tolist()))
                if predictions.label_ids is not None:
                    writer.write("ground truth : \n{}\n\n".format(predictions.label_ids.tolist()))
                    writer.write("metrics : \n{}\n\n".format(predictions.metrics))
            
            if training_args.ensemble:
                import torch
                output_test_file = os.path.join(training_args.output_dir, "ensemble_candidate.txt")
                with open(output_test_file, "a") as writer:
                    soft_prediction = torch.softmax(torch.tensor(predictions.predictions), axis=1)
                    writer.write("{}\n".format(soft_prediction.tolist()))

    return results

    
    
if __name__ == "__main__":
    main()
