# Goal

The major goal of this set of experiments is to learn more about the following (in order of importance):

1. Large(ish) scale training include multi-step training (pre/mid/fine/RL) and observing how models abilities change as a function of their training
2. Engineering the LLM model to be somewhat interactive

# TODO

- Add CORE eval
- Research KV caching
- Research memory capacity requirements

# Estimation of number of iterations

K suggests three methods of setting the number of iterations: (1) direct, (2) target FLOPS, (3) parameter to data ratio. Interestingly (according to Claude) the concept of epochs is not popular with these larger models so we'll ignore and only focus on iterations and assume the datasets are so large we don't get a change to revisit.

1. Just set it based on vibes? IDK 10K?
2. If we assume a target number of FLOPS of 4e19, and a token requires 6 FLOPS x num_parameters (roughly 2 for forward pass, 4 for backwards pass, don't trust my numbers), and batch size = 524288 we get roughly 28K iterations.
3. Assume a optimal 20:1 ratio, if our LLM has 461M parameters, that requires 9.2B tokens (data = tokens always). If batch size = 524288 we get 17.5K iterations.

Further breaking down the training, we must understand the concept of multi-GPU. If we have 8XH100 GPUs with batch_size capability for the H100, we have 8 * batch_size which is good for (1) speed and (2) model quality. Next, remember that for a sequence, when it is passed into the tokenizer we always pad/truncate to a particular max_length = 2048. It is good to think of us thus having max_length x batch_size x num_gpu tokens per iteration. Furthermore, if we have gradient accumulation we squeeze out an additional x accumulation_steps of compute. 

So determining how many iterations we may run for sould probably start with (1) estimate of the optimal number of FLOPS using the 20:1 ratio followed by (2) refining that estimate based on your budget (where we can get $/FLOP based on the GPU and provider).

# Datasets and training

K uses 3 datasets for his example and 4 steps of training (let's ignore RL for now).

1. Pretrain on FineWeb-Edu: large, 14.5B tokens used in training, derived from common crawl, generally high quality websites and educational resources. Use causal language modeling loss function where logit from model predicts likleihood of next token given vocabulary. This is good for general understanding of grammar, facts, reasoning. 
2. Midtrain on FineWeb-Edu + SmolTalk Mix: medium, 5B tokens used in training at 70:30 split of FineWeb-Edu and SmolTalk (which uses Q:A conversations in coding, math, creative writing, advice, chitchat). Uses same loss function as pretraining. Mixing is useful to stop model from overfitting on new data (might lose world knowledge in favor of conversational knowledge).
3. SFT on SmolTalk: medium, 2-3B tokens of 100% SmolTalk, uses masked lanaguage modeling loss function to only generate the assistant response (the answer). 

Remember that attention masking is already used in training GPT-style models, so part (3) masking is more of a fine tune on particular tasks to only give reward to model for particular behavior.

# Getting started

Let's start understanding the datasets. In K's code the download process is kicked off using python -m nanochat.dataset -n num_shards so let's just copy that (this loads the pretraining data from Huggingface). Speaking of data, let's plan the overall size of the model. When starting, user should download the shards they need and store them as parquet files. I then went ahead and incorporated the major aspects of K's code into pytorch lightning because I find dealing with the multi-GPU process tedious. I have yet to scale up to the proper model size so it is unclear if the dataloaders have been implemented correctly but we will see. I also tried to use his distributed Muon optimizer with lightning but that required a full custom training step which I don't want to deal with quite yet. In it's place I had Claude extract out the underlying logic of the optimizer (basically splitting the LLM into slow and fast optimizations) and created a good baseline using AdamW. I am still trying to figure out how much we should abstract away from K's very low level implementation and still aline with Goal 2 above.