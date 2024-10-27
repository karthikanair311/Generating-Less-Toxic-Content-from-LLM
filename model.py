import torch
import logging
from typing import Dict, Optional, Tuple, Union
import transformers
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
from tqdm import tqdm
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PEFTFineTuner:
    """
    This class encapsulates the functionality to fine-tune language models using
    Parameter Efficient Fine-tuning (PEFT) techniques with LoRA (Low-Rank Adaptation).
    It is specifically designed for dialogue summarization tasks and provides methods to
    prepare the model, set up training configurations, and execute the training process.

    Attributes:
        opt (Dict): Configuration options for the model and training process.
        tokenizer (PreTrainedTokenizer): Tokenizer associated with the model being fine-tuned.
        model (Optional[AutoModelForSeq2SeqLM]): The transformer model to be fine-tuned. If not provided, it must be loaded separately.
        lora_config (LoraConfig): Configuration settings for applying LoRA to the model.
    """

    def __init__(self, opt: Dict, tokenizer: transformers.PreTrainedTokenizer, model: Optional[AutoModelForSeq2SeqLM] = None) -> None:
        self.opt = opt
        self.model_name = opt['model_name']
        self.seed = opt.get('seed', 42)
        self.model = model
        self.tokenizer = tokenizer
        self.lora_config = LoraConfig(
            r=opt["LoraConfig"]["rank"],
            lora_alpha=opt["LoraConfig"]["lora_alpha"],
            target_modules=opt["LoraConfig"]["target_modules"],
            bias=opt["LoraConfig"]["bias"],
            lora_dropout=opt["LoraConfig"]["lora_dropout"],
            task_type=opt["LoraConfig"]["task_type"],
        )
        logger.info("PEFTFineTuner initialized with model: %s", self.model_name)

    def prepare_model(self) -> None:
        """
        Prepares the model for PEFT by applying LoRA configurations and enabling
        gradient checkpointing for efficient memory usage. Raises an exception if no model is loaded.
        """
        if self.model is not None:
            try:
                self.model = prepare_model_for_kbit_training(self.model)
                self.model.gradient_checkpointing_enable()
                self.model = get_peft_model(self.model, self.lora_config)
                logger.info("Model prepared with LoRA and gradient checkpointing enabled.")
            except Exception as e:
                logger.error("Failed to apply LoRA configuration or enable gradient checkpointing: %s", e)
                raise RuntimeError("Model preparation failed.") from e
        else:
            logger.error("No model is loaded to prepare.")
            raise ValueError("No model provided for preparation.")

    def fine_tune(self, output_dir: str, train_dataset, eval_dataset) -> Tuple[AutoModelForSeq2SeqLM, Trainer]:
        """
        Fine-tunes the PEFT-enabled model using specified datasets. Configures training parameters, initializes a Trainer,
        and manages the training process. Catches and logs errors during the training setup and execution.

        Parameters:
            output_dir (str): Directory to save outputs and model checkpoints.
            train_dataset: Dataset for training the model.
            eval_dataset: Dataset for evaluating the model performance during training.

        Returns:
            Tuple[AutoModelForSeq2SeqLM, Trainer]: The fine-tuned model and the trainer instance.

        Raises:
            Exception: General exceptions related to training setup or execution are logged and re-raised.
        """
        try:
            self.prepare_model()
            self.model.config.use_cache = False
            training_args = TrainingArguments(
                output_dir=output_dir,
                warmup_steps=self.opt["peft_training_args"]['warmup_steps'],
                per_device_train_batch_size=self.opt["peft_training_args"]['per_device_train_batch_size'],
                gradient_accumulation_steps=self.opt["peft_training_args"]['gradient_accumulation_steps'],
                max_steps=self.opt["peft_training_args"]['max_steps'],
                learning_rate=self.opt["peft_training_args"]['learning_rate'],
                optim=self.opt["peft_training_args"]['optim'],
                logging_steps=self.opt["peft_training_args"]['logging_steps'],
                logging_dir=self.opt["peft_training_args"]['logging_dir'],
                save_strategy=self.opt["peft_training_args"]['save_strategy'],
                save_steps=self.opt["peft_training_args"]['save_steps'],
                evaluation_strategy=self.opt["peft_training_args"]['evaluation_strategy'],
                eval_steps=self.opt["peft_training_args"]['eval_steps'],
                do_eval=self.opt["peft_training_args"]['do_eval'],
                gradient_checkpointing=self.opt["peft_training_args"]['gradient_checkpointing'],
                report_to=self.opt["peft_training_args"]['report_to'],
                group_by_length=self.opt["peft_training_args"]['group_by_length'],
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            logger.info("Training started.")
            return self.model, trainer
        except Exception as e:
            logger.error("An error occurred during fine-tuning: %s", e)
            raise


class ToxicityEvaluator:
    """
    Class for evaluating the toxicity level of generated text. Utilizes a specified machine learning model
    to predict the toxicity of text samples and provide statistical summaries of these predictions.

    Attributes:
        tokenizer (PreTrainedTokenizer): Tokenizer associated with the toxicity model.
        toxicity_model_name (str): Name or path to the pre-trained toxicity model.
        toxicity_model (PreTrainedModel): The machine learning model used for toxicity prediction.
        device (torch.device): The computation device (CPU or GPU).
        evaluator (Any): An evaluation object loaded from the 'evaluate' library specific for toxicity.
        toxic_label (str): The label used to denote toxic content in model predictions.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, toxicity_model_name: str, toxicity_model: PreTrainedModel, device: Union[str, torch.device] = "cpu") -> None:
        self.tokenizer = tokenizer
        self.toxicity_model_name = toxicity_model_name
        self.toxicity_model = toxicity_model
        self.toxicity_model.to(torch.device(device))
        self.device = torch.device(device)
        self.toxic_label = self.get_toxic_label()
        self.evaluator = evaluate.load("toxicity", model=self.toxicity_model_name, module_type="measurement", toxic_label=self.toxic_label)

    def get_evaluator(self):
        """
        Retrieves the evaluator object used for measuring toxicity.

        Returns:
            The evaluator object loaded with the toxicity measurement configuration.
        """
        return self.evaluator


    def get_toxic_label(self) -> str:
        """
        Retrieves the first non-neutral label suitable for toxicity detection from the model's configuration.
        
        Returns:
            str: The first non-neutral label found in the model's vocabulary suitable for toxicity detection.
        
        Raises:
            ValueError: If no non-neutral labels are found in the model's vocabulary.
        """
        available_labels = list(self.toxicity_model.config.id2label.values())
        non_neutral_labels = [label for label in available_labels if label.lower() not in ['neutral', 'other', 'not offensive']]

        if non_neutral_labels:
            toxic_label = non_neutral_labels[0]
            return toxic_label
        else:
            raise ValueError("No non-neutral labels found in the model's vocabulary. Consider using a different model or manually defining a toxicity threshold.")

    def evaluate_toxicity(self, model: PreTrainedModel, dataset: Dict, GenerationConfig: Dict = None, num_samples: int = 100) -> Tuple[float, float]:
        """
        Evaluates the toxicity level of generated text on a dataset.

        Args:
            model (PreTrainedModel): The model to generate text for toxicity evaluation.
            dataset (Dict): Dataset containing text samples, keyed by 'dialogue'.
            num_samples (int): Number of samples to evaluate, defaults to 100.

        Returns:
            Tuple[float, float]: Mean and standard deviation of toxicity scores across the evaluated samples.
        
        Raises:
            RuntimeError: If any issues occur during model inference.
        """
        if not GenerationConfig:
            self.model = model
            self.model.to(self.device)
            self.model.eval()
        else:
            max_new_tokens=100

        toxicities = []

        for  i, sample in tqdm(enumerate(dataset), total=num_samples):
            if i >= num_samples:
                break

            if not GenerationConfig:
                input_text = sample["dialogue"]
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)

                try:
                    with torch.no_grad():
                        output_ids = self.model.generate(input_ids.to(self.device), max_length=512)
                    generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    result = self.evaluator.compute(predictions=[generated_text])
                    toxic_score = result["toxicity"]
                    toxicities.append(toxic_score)
                except Exception as e:
                    raise RuntimeError(f"Error during model inference: {e}")
            else:
                input_text = sample["query"]
                
                input_ids = self.tokenizer(input_text, return_tensors="pt", padding=True).input_ids.to(self.device)

                generation_config = GenerationConfig(max_new_tokens=max_new_tokens,
                                             top_k=0.0,
                                             top_p=1.0,
                                             do_sample=True)
                
                response_token_ids = model.generate(input_ids=input_ids,
                                            generation_config=generation_config)
                
                generated_text = self.tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
                
                toxicity_score = self.evaluator.compute(predictions=[(input_text + " " + generated_text)])
                
                toxicities.extend(toxicity_score["toxicity"])

        mean_toxicity = np.mean(toxicities)
        std_toxicity = np.std(toxicities)

        return mean_toxicity, std_toxicity