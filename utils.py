import logging
from trl.core import LengthSampler
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizer, AutoTokenizer
from transformers import set_seed
from typing import Optional, Any
from torch.nn import Module

# Setting up the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparison:
    def __init__(self, tokenizer, ref_model, ppo_model, sentiment_pipe, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.ppo_model = ppo_model
        self.sentiment_pipe = sentiment_pipe
        self.device = device

    def validate_and_truncate(self, inputs, max_len=512):
        """
        Truncates text inputs to a specified maximum length while maintaining validity.

        Args:
            inputs (list): List of text strings.
            max_len (int, optional): Maximum length for the truncated text. Defaults to 512.

        Returns:
            list: List of truncated text strings.
        """


        tokens = self.tokenizer.tokenize(inputs)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]  # Truncate the tokens
        new_text = self.tokenizer.convert_tokens_to_string(tokens)

        return new_text

    def compare_models(self, dataset, batch_size=20, output_min_length= 100, output_max_length= 200):
        """
        Compares the performance of the reference and PPO models on sentiment.

        Args:
            dataset (dict): Dictionary containing test data (queries).
            batch_size (int, optional): Batch size for processing queries. Defaults to 20.

        Returns:
            dict: Dictionary containing comparison results:
                - query (list): List of original queries.
                - response_before (list): List of responses generated by the reference model.
                - response_after (list): List of responses generated by the PPO model.
                - reward_before (list): List of sentiment scores before generation.
                - reward_after (list): List of sentiment scores after generation.
        """

        compare_results = {"query": [], "response_before": [], "response_after": [], "reward_before": [], "reward_after": []}
        df_batch = dataset["test"][:batch_size]  # Select a batch of queries

        gen_len = LengthSampler(output_min_length, output_max_length)

        for i in tqdm(range(batch_size)):
            # Generate output length and prepare input tensors
            prompt_tensor = torch.as_tensor(df_batch["input_ids"][i]).unsqueeze(dim=0).to(self.device)

            max_new_tokens = gen_len()

            # Generate responses and decode
            generation_kwargs = {"max_new_tokens": max_new_tokens, "min_length": 5, "top_k": 0.0, "top_p": 1.0, "do_sample": True}
            summary_ref = self.ref_model.generate(input_ids=prompt_tensor, **generation_kwargs).squeeze()[-max_new_tokens:]
            summary_ppo = self.ppo_model.generate(input_ids=prompt_tensor, **generation_kwargs).squeeze()[-max_new_tokens:]
            response_before = self.tokenizer.decode(summary_ref)
            response_after = self.tokenizer.decode(summary_ppo)

            # Sentiment analysis of query-response pairs
            query = df_batch["query"][i]
            text_before = query + response_before
            text_after = query + response_after
            text_before = self.validate_and_truncate(text_before)
            reward_before = self.analyze_sentiment(text_before)
            text_after = self.validate_and_truncate(text_after)
            reward_after = self.analyze_sentiment(text_after)


            compare_results["query"].append(query)
            compare_results["response_before"].append(response_before)
            compare_results["response_after"].append(response_after)
            compare_results["reward_before"].append(reward_before[0]["score"])
            compare_results["reward_after"].append(reward_after[0]["score"])

        return compare_results

    def analyze_sentiment(self, texts):
        """
        Performs sentiment analysis on a list of texts using the sentiment_pipe.

        Args:
            texts (list): List of text strings.

        Returns:
            list: List of sentiment analysis results.
        """

        reward_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}
        return self.sentiment_pipe(texts, **reward_kwargs)

    
class ZeroShotSummarizer:
    def __init__(self, model: Optional[AutoModelForSeq2SeqLM] = None, model_name: str = "t5-base", seed: int = 42):
        """
        Initializes the ZeroShotSummarizer class.

        Args:
            model (Optional[AutoModelForSeq2SeqLM]): Pre-trained model for summarization. Defaults to None.
            model_name (str): Name of the pre-trained model to use if model is not provided. Defaults to "t5-base".
            seed (int): Random seed for model reproducibility. Defaults to 42.
        """
        logger.info("Initializing the ZeroShotSummarizer...")
        self.model = model or AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = self.model.to('cpu')
        self.seed = seed
        logger.info(f"Model loaded: {model_name}")

    def summarize(self, prompt: str, tokenizer: PreTrainedTokenizer, max_length: int = 100) -> str:
        """
        Summarizes a given conversation using the zero-shot summarization approach.

        Args:
            prompt (str): The conversation text to be summarized.
            tokenizer (PreTrainedTokenizer): The tokenizer object to use for encoding.
            max_length (int): Maximum length of the generated summary. Defaults to 100.

        Returns:
            str: The model-generated summary of the conversation.
        """
        logger.info("Starting summarization process...")
        set_seed(self.seed)  # Set random seed for consistent output
        formatted_prompt = f"Instruct: Summarize the following conversation.\n{prompt}\nOutput:\n"
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
        logger.debug("Input formatted and encoded.")

        outputs = self.model.generate(input_ids, max_length=max_length)
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_summary = output.split('Output:\n')[1] if 'Output:\n' in output else output
        
        logger.info("Summarization complete.")
        return output_summary
    

class ModelSummary:
    """
    Provides a summary of a PyTorch model by calculating and reporting the number
    of trainable and total parameters. This is useful for understanding the capacity
    and complexity of a model.

    Attributes:
        model (Module): A PyTorch model.
        trainable_model_params (int): Total number of trainable parameters in the model.
        all_model_params (int): Total number of parameters in the model, whether trainable or not.
    """

    def __init__(self, model: Module) -> None:
        """
        Initializes the ModelSummary class with a specific model.

        Args:
            model (Module): The PyTorch model to be summarized.
        """
        self.model = model
        self.trainable_model_params = 0
        self.all_model_params = 0

    def calculate_parameters(self) -> None:
        """
        Calculates the total and trainable parameters of the model. This method updates the
        instance variables for storing the total count of parameters.
        """
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            self.all_model_params += param_count
            if param.requires_grad:
                self.trainable_model_params += param_count

        logger.info("Calculated model parameters: %s trainable, %s total", self.trainable_model_params, self.all_model_params)

    def print_summary(self) -> None:
        """
        Prints a summary of the model's parameters. The summary includes total parameters,
        trainable parameters, and the percentage of parameters that are trainable.
        """
        # Ensure parameters are calculated
        if self.trainable_model_params == 0 or self.all_model_params == 0:
            self.calculate_parameters()

        print(f"\nTrainable model parameters: {self.trainable_model_params}\n"
              f"All model parameters: {self.all_model_params}\n"
              f"Percentage of trainable model parameters: {100 * self.trainable_model_params / self.all_model_params:.2f}%")

    
if __name__ == "__main__":

    # Example usage
    summarizer = ZeroShotSummarizer()
    # Load your tokenizer here (replace with your tokenizer loading logic)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    conversation = "What is the weather like today?"
    response = "It's a beautiful sunny day!"
    summary = summarizer.summarize(f"{conversation}\n{response}", tokenizer)
    print(f"Conversation Summary: {summary}")

    # Example usage
    model = "your model instance"
    summary = ModelSummary(model)
    summary.print_number_of_trainable_model_parameters()