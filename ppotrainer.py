import torch
from trl import PPOConfig, AutoModelForSeq2SeqLMWithValueHead, PPOTrainer
from transformers import pipeline
from trl.core import LengthSampler
import torch
from tqdm import tqdm

class MyPPOTrainer(PPOTrainer):
  def __init__(self, config, model, ref_model, tokenizer, dataset, data_collator):
    super().__init__(config, model, ref_model, tokenizer, dataset, data_collator)

  def compute_rewards(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `response_length`)

        Returns:
            `torch.FloatTensor`: Per token rewards, shape (`batch_size`, `response_length`)
            `torch.FloatTensor`: Non score rewards, shape (`batch_size`, `response_length`)
            `torch.FloatTensor`: KL penalty, shape (`batch_size`, `response_length`)
        """
        rewards, non_score_rewards, kls = [], [], []
        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
            # compute KL penalty (from difference in logprobs)
            kl = self._kl_penalty(logprob, ref_logprob)
            kls.append(kl)
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            last_non_masked_index = mask.nonzero()[-1]

            # reward is preference model score + KL penalty
            reward[last_non_masked_index] += score
            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards), torch.stack(kls)
  

class PPO_DialogueTrainer:
    def __init__(self, model_name,toxicity_model_name:str,model, ref_model=None, tokenizer=None, dataset=None, data_collator=None,
                 learning_rate=1.41e-5, max_ppo_epochs=1, mini_batch_size=4, batch_size=16
                 ):
        self.model_name = model_name
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.data_collator = data_collator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sentiment_pipe = pipeline("sentiment-analysis", model=toxicity_model_name, device=self.device)
        self.configure_trainer(learning_rate, max_ppo_epochs, mini_batch_size, batch_size)

    def get_sentiment_pipeline(self):
        return self.sentiment_pipe

    def configure_trainer(self, learning_rate, max_ppo_epochs, mini_batch_size, batch_size):
        self.config = PPOConfig(
            model_name=self.model_name,
            learning_rate=learning_rate,
            ppo_epochs=max_ppo_epochs,
            mini_batch_size=mini_batch_size,
            batch_size=batch_size
        )

        self.ppo_trainer = PPOTrainer(config=self.config,
                                       model=self.model,
                                       ref_model=self.ref_model,
                                       tokenizer=self.tokenizer,
                                       dataset=self.dataset["train"],
                                       data_collator=self.data_collator)
    
    def generate_output_length(self, output_min_length, output_max_length):
        """
        Samples a random output length within a predefined range.
        """

        output_length_sampler = LengthSampler(output_min_length, output_max_length)
        return output_length_sampler()
    
    def validate_and_truncate(self, inputs, max_len=512):
        """
        Truncates query-response pairs if exceeding a limit.
        """

        new_inputs = []
        for text in inputs:
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]  # Truncate the tokens
            new_text = self.tokenizer.convert_tokens_to_string(tokens)
            new_inputs.append(new_text)
        return new_inputs
        
    def train(self, output_min_length, output_max_length, max_ppo_steps=10):
        """
        Trains the dialogue generation model using PPO with sentiment as reward.

        Args:
            max_ppo_steps (int, optional): Maximum number of training steps. Defaults to 10.
        """

        for step, batch in enumerate(tqdm(self.ppo_trainer.dataloader, total=max_ppo_steps)):
            if step >= max_ppo_steps:
                break

            prompt_tensors = batch["input_ids"]
            summary_tensors = []

            for prompt_tensor in prompt_tensors:
                max_new_tokens = self.generate_output_length(output_min_length, output_max_length)
                generation_kwargs = {"max_new_tokens": max_new_tokens, "min_length": 5, "top_k": 0.0, "top_p": 1.0, "do_sample": True}
                summary = self.ppo_trainer.generate(prompt_tensor, **generation_kwargs)
                summary_tensors.append(summary.squeeze()[-max_new_tokens:])

            batch["response"] = [self.tokenizer.decode(r.squeeze()) for r in summary_tensors]

            query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]
            query_response_pairs = self.validate_and_truncate(query_response_pairs)
            reward_kwargs={"top_k": None, "function_to_apply": "none", "batch_size": 16}
            rewards = self.sentiment_pipe(query_response_pairs, **reward_kwargs)

            reward_tensors = [torch.tensor(reward[0]["score"]) for reward in rewards]

            stats = self.ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
            self.ppo_trainer.log_stats(stats, batch, reward_tensors)

            print(f'objective/kl: {stats["objective/kl"]}')
            print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
            print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
            print('-' * 100)

    def save_model(self, path):
        """
        Saves the trained model and tokenizer to a specified path.

        Args:
            path (str): Path to save the model and tokenizer files.
        """

        torch.save(self.ppo_trainer.model.state_dict(), path + "_model.pt")
        torch.save(self.tokenizer, path + "_tokenizer.pt")

    def load_model(self, path):
            """
            Loads a pre-trained model and tokenizer from a specified path.

            Args:
                path (str): Path to the saved model and tokenizer files.
            """

            self.ppo_trainer.model.load_state_dict(torch.load(path + "_model.pt"))
            self.tokenizer = torch.load(path + "_tokenizer.pt")

    def generate(self, prompt, max_length=200):
        """
        Generates a response to a given prompt using the trained model.

        Args:
            prompt (str): The prompt to generate a response for.
            max_length (int, optional): Maximum length for the generated response. Defaults to 200.

        Returns:
            str: The generated response.
        """

        prompt_tensor = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generation_kwargs = {"max_length": max_length, "min_length": 5, "top_k": 0.0, "top_p": 1.0, "do_sample": True}
        response = self.ppo_trainer.generate(prompt_tensor, **generation_kwargs)
        return self.tokenizer.decode(response.squeeze(), skip_special_tokens=True)

    def evaluate(self, test_dataset, metrics=["bleu"]):
        """
        Evaluates the trained model on a held-out test set.

        Args:
            test_dataset (dict): Dictionary containing test data.
            metrics (list, optional): List of metrics to use for evaluation (e.g., "bleu", "rouge"). Defaults to ["bleu"].
        """

        # Implement evaluation logic here using the test_dataset and metrics
        # This might involve generating responses for prompts in the test set
        # and comparing them to human-generated references using the specified metrics.

        # Example using the transformers 'evaluate' function (assuming compatibility)
        from transformers import evaluate

        generated_responses = []
        for prompt in test_dataset["prompts"]:
            response = self.generate(prompt)  # Call your generation function here
            generated_responses.append(response)

        references = test_dataset["references"]
        evaluation_results = evaluate(references=references, predictions=generated_responses, metrics=metrics)

        # Print or return the evaluation results
        print(f"Evaluation Results:")
        for metric, score in evaluation_results.items():
            print(f"\t- {metric}: {score}")