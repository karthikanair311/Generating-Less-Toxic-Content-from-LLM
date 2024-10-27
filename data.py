import logging
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    """
    Preprocesses a dataset for a PPO-based dialogue summarization approach. This includes
    filtering dialogues by length, loading a tokenizer, tokenizing samples, and preparing dataset splits.

    Attributes:
        model_name (str): Name of the pre-trained tokenizer model.
        dataset_name (str): Name of the dataset to load.
        input_min_length (int): Minimum length of dialogues to include.
        input_max_length (int): Maximum length of dialogues to include.
    """
    
    def __init__(self, model_name: str, dataset_name: str, input_min_length: int, input_max_length: int):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.input_min_length = input_min_length
        self.input_max_length = input_max_length
        self.tokenizer = None
        self.dataset = None

    def load_and_filter_data(self):
        """
        Loads the dataset using the 'train' split and filters dialogues based on specified length constraints.
        """
        try:
            self.dataset = load_dataset(self.dataset_name, split="train")
            self.dataset = self.dataset.filter(
                lambda x: self.input_min_length < len(x["dialogue"]) <= self.input_max_length,
                batched=False
            )
            logger.info(f"Dataset loaded and filtered with dialogues between {self.input_min_length} and {self.input_max_length} characters.")
        except Exception as e:
            logger.error("Failed to load or filter dataset: %s", e)
            raise

    def prepare_tokenizer(self):
        """
        Loads the tokenizer from the specified model name and configures it for the device.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error("Failed to load tokenizer: %s", e)
            raise

    def tokenize_sample(self, sample: Dict) -> Dict:
        """
        Tokenizes a sample dialogue with a summarization prompt and renames the key for PPO compatibility.

        Args:
            sample (Dict): A single sample from the dataset.

        Returns:
            Dict: The modified sample with tokenized data.
        """
        prompt = f"Summarize the following conversation:\n\n{sample['dialogue']}\n\nSummary:"
        tokenized_input = self.tokenizer(prompt, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        sample["input_ids"] = tokenized_input["input_ids"].squeeze()
        sample["query"] = self.tokenizer.decode(sample["input_ids"])
        return sample

    def preprocess_dataset(self):
        """
        Executes the entire preprocessing pipeline including loading, filtering, tokenizing, and formatting.
        """
        self.load_and_filter_data()
        self.prepare_tokenizer()
        self.dataset = self.dataset.map(self.tokenize_sample, batched=False)
        self.dataset.set_format(type="torch")
        logger.info("Dataset preprocessed and formatted for PyTorch.")

    def get_dataset_splits(self):
        """
        Splits the preprocessed dataset into training and testing parts, based on a 20% test size.

        Returns:
            DatasetDict: Contains 'train' and 'test' splits of the preprocessed dataset.
        """
        if self.dataset is None:
            self.preprocess_dataset()
        train_test_splits = self.dataset.train_test_split(test_size=0.2, seed=42)
        logger.info("Dataset split into train and test parts.")
        return train_test_splits