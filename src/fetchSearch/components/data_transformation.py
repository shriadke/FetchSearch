import os
from fetchSearch.logging import logger
from fetchSearch.entity import DataTransformationConfig
from sentence_transformers import InputExample, evaluation
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import DataLoader

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def get_dataloader(self, data, split):
        examples = []
        data = data[split]
        n_examples = data.num_rows

        for i in range(n_examples):
            example = data[i]
            if split in ["val", "test"] and i > 3:
                break
            examples.append(InputExample(texts=[example['offer_ext'], example['search_query']], label=float(example["score"])))
        logger.info(f"in {split}, We have a {type(examples)} of length {len(examples)} containing {type(examples[0])}'s.")
        dataloader = DataLoader(examples, shuffle=True, batch_size=16)
        return examples, dataloader

    def convert(self):
        dataset = load_from_disk(self.config.data_path)

        
        train_examples, train_dataloader = self.get_dataloader(dataset, "train")
        torch.save(train_dataloader, os.path.join(self.config.root_dir,"train.pth"))

        val_examples, val_dataloader = self.get_dataloader(dataset, "val")
        val_evaluator = evaluation.EmbeddingSimilarityEvaluator([],[],[]).from_input_examples(examples=val_examples)
        torch.save(val_dataloader, os.path.join(self.config.root_dir,"val.pth"))
        torch.save(val_evaluator, os.path.join(self.config.root_dir,"val_eval.pth"))

        test_examples, test_dataloader = self.get_dataloader(dataset, "test")
        test_evaluator = evaluation.EmbeddingSimilarityEvaluator([],[],[]).from_input_examples(examples=test_examples)
        torch.save(test_dataloader, os.path.join(self.config.root_dir,"test.pth"))
        torch.save(test_evaluator, os.path.join(self.config.root_dir,"test_eval.pth"))
