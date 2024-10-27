# Generating-Less-Toxic-Content-from-LLM

## Project Description
Large language models (LLMs) have grown significantly in capability, enabling them to 
produce complex and nuanced text. However, this power also raises concerns, as these 
models can inadvertently generate harmful content, including hate speech and 
misinformation. Addressing this issue is crucial to ensure that the use of LLMs aligns with 
ethical standards and promotes positive societal impacts.

Our approach is to mitigate this risk is using reinforcement learning (RL), a method within 
machine learning. In RL, a model is trained to minimize toxic output by implementing a 
system of rewards and penalties: it receives incentives for producing safe, non-toxic content 
and consequences for generating unacceptable content. 

## Summary
This project leverages reinforcement learning (RL) techniques to mitigate the risk of harmful 
content generation by large language models (LLMs), specifically focusing on reducing hate 
speech and misinformation. Using the T5 model as a base, the approach involves fine-tuning with 
a Parameter Efficient Fine-Tuning (PEFT) method enhanced by Quantized Low-Rank Adaptation 
(QLoRA) for efficiency. The model undergoes further refinement through Proximal Policy 
Optimization (PPO) to prioritize non-toxic outputs, guided by a reward model trained to detect 
hate speech. This setup uses a combination of the Hugging Face Transformers library and 
specialized datasets like DialogSum for training and evaluation. The process is characterized by a 
meticulous tuning and evaluation phase where model responses are adjusted based on toxicity 
scores and optimized continuously for reduced harmful content, demonstrating a significant 
decrease in toxicity levels post-detoxification, which supports the goal of aligning LLM outputs 
with ethical standards and improving societal impact.