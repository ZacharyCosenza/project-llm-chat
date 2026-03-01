# Goal

The major goal of this set of experiments is to learn more about the following (in order of importance):

1. Large(ish) scale training include multi-step training (pre/mid/fine/RL) and observing how models abilities change as a function of their training
2. Engineering the LLM model to be somewhat interactive

# Estimation of number of iterations

K suggests three methods of setting the number of iterations: (1) direct, (2) target FLOPS, (3) parameter to data ratio. Interestingly (according to Claude) the concept of epochs is not popular with these larger models so we'll ignore and only focus on iterations and assume the datasets are so large we don't get a chance to revisit.

1. Just set it based on vibes? IDK 10K?
2. If we assume a target number of FLOPS of 4e19, and a token requires 6 FLOPS x num_parameters (roughly 2 for forward pass, 4 for backwards pass, don't trust my numbers), and batch size = 524288 we get roughly 28K iterations.
3. Assume a optimal 20:1 (token:parameter) ratio, if our LLM has 524M parameters, that requires ~10B tokens. If batch size = 524288 we get 17.5K iterations. Given 1.7K shards, at roughly 53K rows of data that gives us ~100B GPT2 tokens available in training. For a model of 461M parameters we only need ~10% of this data. 

Further breaking down the training, we must understand the concept of multi-GPU. If we have 8XH100 GPUs with batch_size capability for the H100, we have 8 * batch_size which is good for (1) speed and (2) model quality. Next, remember that for a sequence, when it is passed into the tokenizer we always pad/truncate to a particular max_length = 2048. It is good to think of us thus having max_length x batch_size x num_gpu tokens per iteration. Furthermore, if we have gradient accumulation we squeeze out an additional x accumulation_steps of compute. 

So determining how many iterations we may run for sould probably start with (1) estimate of the optimal number of FLOPS using the 20:1 ratio followed by (2) refining that estimate based on your budget (where we can get $/FLOP based on the GPU and provider).

# Datasets and training

K uses 3 datasets for his example and 4 steps of training (let's ignore RL for now).

1. Pretrain on FineWeb-Edu: large, 14.5B tokens used in training, derived from common crawl, generally high quality websites and educational resources. Use causal language modeling loss function where logit from model predicts likleihood of next token given vocabulary. This is good for general understanding of grammar, facts, reasoning. 
2. Midtrain on FineWeb-Edu + SmolTalk Mix: medium, 5B tokens used in training at 70:30 split of FineWeb-Edu and SmolTalk (which uses Q:A conversations in coding, math, creative writing, advice, chitchat). Uses same loss function as pretraining. Mixing is useful to stop model from overfitting on new data (might lose world knowledge in favor of conversational knowledge).
3. SFT on SmolTalk: medium, 2-3B tokens of 100% SmolTalk, uses masked language modeling loss function to only generate the assistant response (the answer). 

Remember that attention masking is already used in training GPT-style models, so part (3) masking is more of a fine tune on particular tasks to only give reward to model for particular behavior.

# Getting started

Let's start understanding the datasets. In K's code the download process is kicked off using python -m nanochat.dataset -n num_shards so let's just copy that (this loads the pretraining data from Huggingface). Speaking of data, let's plan the overall size of the model. When starting, user should download the shards they need and store them as parquet files. I then went ahead and incorporated the major aspects of K's code into pytorch lightning because I find dealing with the multi-GPU process tedious. I have yet to scale up to the proper model size so it is unclear if the dataloaders have been implemented correctly but we will see. I also tried to use his distributed Muon optimizer with lightning but that required a full custom training step which I don't want to deal with quite yet. In it's place I had Claude extract out the underlying logic of the optimizer (basically splitting the LLM into slow and fast optimizations) and created a good baseline using AdamW. I am still trying to figure out how much we should abstract away from K's very low level implementation and still aline with Goal 2 above. I also managed to get a basic version of the model working at around 325M parameters with max_seq_len = 128. Next step is scaling up to the proper parameter size.

# Scaling up

One of the critical aspect of this study is to examine how to create large-scale ML experiments. With K's batch size of 32, token size of 2048, and ~400M parameter model, this is a small-scale LLM project but fairly large-scale project relative to the things I have done. Stealing K's dataloader allowed me to do this on 8 A40 GPUs (distribute samples according to rank, stream etc), and easily run experiments with fewer GPUs. Trying to simplify the dataloader to use a more classic pytorch dataloader resulted in OOM errors even at smaller batch sizes and smaller context length, so clearly the dataset is too large to hold in memory. I went on to keep closer track of VRAM during training using torch.cuda. After logging memory usage I realized allocated and reserved memory increase monotonically, indicating leakage somewhere (see image below).

[Figure 1](/images/fig1_badmemory.webp)

Helpful commands for RAM: du -h /workspace | sort -h | tail -20
Helpful scripts for VRAM: torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)

I also had Claude rewrite the entire script, using the loading of parquet files as a starting point, and that seems to help (see below). 

[Figure 2](/images/fig2_goodmemory.png)

However, I am still running into an issue where, under certain conditions like various number of GPUs, the script itself starts but hangs. The issue seemed to resolve itself by running without P2P communication via the following:

NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=X run_X.py

In further experiments, the VRAM memory issue, while resolved from a leakage perspective, still came up as the memory usage sits near the edge of OOM. So when certain validation steps pushed memory outside this bound the program fails. I find this odd for two reasons: (1) we are not keeping any artifacts of validation in memory and (2) the differences between validation steps in terms of memory is small. I removed the external validation steps as they required downloading artifacts, which may be held in memory in uncontrolled ways. Next I ran into a crash on one of the dataloader workers so I am setting num_workers = 0. This should increase stability. With these fixes I can get stable training with batch_size = 16 and max_seq_length = 2024/4, with lack of compute being the only thing holding us back. We are finally getting learning as the model goes from uniform random samples, to modal words (mostly "the" and "\n"), to modal sentence pieces ("United States", "from the") at ~500 iterations. I then switched from A40s to H100s and the speedup (maybe because the lack of multi-GPU communications when comparing 1XH100 vs 8XA40) was dramatic (~20x faster, ~33% lower cost). One issue is that I am using runpod.io service for GPU instances but I often cannot get H100s. I am now testing out moving to lambda.ai as they seem to have many more high quality GPUs.

# Re-estimate of the size/scope of pre-training

After futher experiments, I find that I can use a context window of 2048 for my model and hold it in memory on an H200 GPU with batch_size 22 at ~2.35s/iteration. This is less than the 32 that K uses and I am still investigating this. Let's say I want to keep my model at ~500M parameters and reach the 1e19 FLOP threshhold, this will require ~10B tokens (or about 170 shards). Each iteration FLOP count can be calculated as (batch_size = 22) * (context = 2048) * (num_parameters = 500M) * (forward + backwards = 6) * (gradient accumulation = 4) = 4e14. This assumed a single H200 with grad_accum=4, giving 22 × 2048 × 4 = 180,224 tokens/optimizer step. Thus, to get 1e19 FLOP total we need 1e5 (25,000) iterations. Using Chinchilla optimiality I get 20:1 -> 500M --> ~10B tokens --> 55k iterations. Let's split the difference and go with 40k iterations.

I tried to scale to 8XH100 but perhaps due to overhead issues I needed to reduce the batch_size to 18 to still get ~2.4s / iteration. This would then require ~15,000 iterations. The logical next step is to compare the loss curve in single vs multi-GPU setup to validate this approach works, but my loss curves are identical even when increasing the learning rates. I ran some experiments and found that it was likley that was an artifact of training with a low learning rate early in pre-training. I just looked up what a common GPT2-style optimization technique looks like and used that, which seemed to work.

# Pre-training

Session 1: I will start pre-training. I first starting running on 2XH100 using batch_size = 22 for ~4k iterations while watching The Brutalist. That gives me 720,896,000 tokens (~720.9M), so ~9.28B to go (id = qfkhmres)!

Session 2: Updated with improved saving. Let's reload the checkpoint and start at iteration 4000. If we use 4XH100. To finish training we'd need only 18k iterations here, but I needed to reduce batch_size = 18, so we're back up to needing 22k iterations. However I only got through ~500 iterations because I'm in a coffee shop with bad internet connection. Trained 147,456,000 tokens (~147.5M), cumulative ~868.4M, ~9.13B to go (id = 46acw8eb). [actually 73M tokens]

Session 3: 4500 starting, 10000 ending, batch_size 18 and 4 GPUs. Trained 1,622,016,000 tokens (~1.62B), cumulative ~2.49B, ~7.51B to go (id = carxia2m). [actually 811M tokens]

Session 4: starting 10000, batch_size 18 and 4 GPUs, ended 16500. Trained 1,916,928,000 tokens (~1.92B), cumulative ~4.41B, ~5.59B to go (id = 56c2n5c6). [actually 958M tokens]

Session 5: started 16500 ended checkpoint_35800.pt (iteration 35800), batch_size 18 and 4 GPUs (id = iec3w33j). Trained 5,689,344,000 tokens (~5.69B), cumulative ~10.10B, COMPLETED! I was at Trees Bar near Park Slope enjoying Wine Wednesdays after PPTC run club and got to watch my child (who I have named ZAC-GPT) learn how to say full sentences! [actually 2.8B tokens]

Session 6: Okay so I deleted iec3w33j by accident so let's restart at 56c2n5c6, which starts at iteration 16500 with batch_size 18 on 4 GPUs and ended at iteration 38200 (id = j651hrgg). Trained 6,394,368,000 tokens (~6.39B), cumulative ~10.80B, COMPLETED for ZAC-GPT-2 (with extra tokens for good measure)! [actually 3.2B tokens]

Absolutley facinating observations by the end. The LLM goes from random words (due to random weights), to most common words (often "the") after a few hundred iterations, to slowly being able to form larger sentences (~10000 iterations) and paragraphs (~30000 iterations). I knew that pre-training would be helpful for spelling, grammar, and basic sentence construction, but running a set of world knowledge tasks on them I get these results: 

The tallest mountain in the world is:

Iteration 4000: ...the world’s tallest mountain, the world’s tallest mountain.

Iteration 10000: ...the Sargasso Glacier, which is the highest mountain in the world.

Iteration 38200: ...the Mount Everest. The mountain is a part of Nepal, India, Nepal, Bhutan, Nepal

Indicating that because the pre-training dataset contains a QA and world knowledge in it's distribution, even a simple pre-trained GPT2-style LLM can generate plausable responses without mid-training or fine-tuning. Cool!

# Creating a set for validation

The paradigm here is only having training and validation. We hold out a parquet file for validation. I had been using perplexity loss for this, but I want to add some additional tests. I have started with a set of sentence completions that will be printed as well as some basic world knowledge compeltion tasks. This required adding a predict function to the model to make autoregressive temperature-normalized predictions until a max token limit. Eventually Claude Code got good enough models to easily integrate K's CORE evaluation metrics into my setup. Here is how the validation set up works:

(1) Open ended sentence completion (for fun)
(2) World knowledge (harder but still fun)
(3) Perplexity (raw score, good for tracking pre-training but not details)
(4) CORE Metric (K-shots available):
    - Multiple Choice: context + different answers is tokenized for k answers, the choice with the lowest mean loss (for answer only) is picked, accuracy computed as whether you matched the gold standard. Inductive bias here is ICL of answer space.
    - Schema: different context + same answer is tokenized, same metric as above, against loss computed as mean for context segment. Inductive bias here is ICL of context space, requireing more world knowledge and reasoning.
    - Language Modeling: regular continuation with accuracy of argmax token. Inductive bias is strict recall.

# Mid-training

Session 1: With pre-training done let's test out mid-training (I know K ended up removing mid-training but whatever this is fun). The major changes here are (i) introducing special tokens (BOS, roles) and (ii) training on conversational data. I'm going to use 30% FineWebEdu  (same data as pre-training) and 70% mix of SmolTalk (synthetic multi-turn QA in technical domains) and UltaChatGen (synthetic dialogue and currated discussions). Goal is to hit 5B tokens with this distribution so with batch_size 18 and 4XH100 GPUs that's ~20k iterations (id = pu94vo4r).

Session 2: Something I realized during training was the CORE metric was both very noisy and not increasing much. I originally took this as due to high learning rate, but realized (i) few sequences were making up the CORE metric and (ii) I was padding, and not packing, the mid-training data. Playing around with more complex QA and knowledge checks it gets the sentence structure and flavor of the answer, but hardly the exact answer. Let's try this again. If we still want to get to 5B mid-training tokens, previous session was really ~50%, so we have to do another 2.5B, or 10k iterations of the new packed dataset (id = 771w9gdd).

To fetch do the following:

scp -r -P 15085 -i ~/.ssh/id_ed25519 root@216.243.220.220:/workspace/project-llm-chat/logs/7mkc9emw/ /home/zaccosenza/code/project-llm-chat/logs/7mkc9emw/

# Some mess ups and prep for SFT

I noticed some things that give me pause for the next phase of training: 

(1) re-reading some documentation on K's repo I noticed he says 350 shards are needed for pre-training on 10B tokens. I did some sampling of the FineWebEdu dataset and get ~56M tokens/shard, so to get 10B tokens I need 178 shards. I re-checked my notes and got the same answer, but checking the server I only had 100 shards downloaded. This might explain why my CORE score was 0.19 despite passing through mid-training. Also it turns out I was not accounting for grad_accmulation = max(1, 4 // world_size). With world_size = 4 I underestimated the number of tokens per iteration, so pre-training I really only trained on ~6.2B tokens and mid-training I really only trained on ~3B tokens. Lession learned, be very detailed in the training dataset sizes to match expected specs! 

(2) Chatting with 771w9gdd I noticed it (i) lies, (ii) runs on too long and (iii) mimics the user tokens. I could be wrong, but it seems to me that at minimum (ii) and (iii) could be solved with SFT and RLHF. (i) might be solved by going back and re-pre/mid-training. Seeing as this is expensive, we've learned a lot about large-scale training, and I want to start exploring other topics, I think we can start SFT and RLHF. Part of the value of this project was to struggle with these unfamilar topics and actually learn them, rather than rely on perfect vibecoding and other peoples code. 

(3) I have the ability to resume training, but I never exclude already seen shards from training. So while the data is stratified across GPUs and then randomly selected, some data may have been seen beforehand. 

(4) After some re-engineering in preparation for SFT I found that I was appending EOS token to end of entire document, and the default EOS was being appended to the end of rows of text in FineWebEdu. The inductive bias here for EOS will therefore be a weak end-of-the-line signal, rather than a genuine EOS token. However, happily, mid-train was done with assistant-to-user passing, so I can re-purpose user as an EOS token during QA.

I think all together (1) (2) (3) all point to a poor model with low performance due to overfitting on a smaller less diverse distribution of pre-trained data. I'm going to run an experiment to see how much we can recover by solving the above issues. If I don't see a significant drop in validation we might move forward with the mulligan and do some more experiments on our poor dumb little model.

# SFT

From experiments on conversational data (ultachat and smoltalk) we have ~ 30 + 8 + 7 M tokens / shard * 30 shards = ~1.35B tokens. Assuming 2048 tokens / seq, 18 batch size, 4 GPUs and our logic of accumulate_grad_batches = max(1, 4 // world_size) = 1, I get 8.8k iterations to run through all data in a single epoch. Reading the literature I tend to see SFT being done on basis of number of examples rather than number of tokens, of which we have ~1.4M examples in our datasets. So the question is do we start SFT with many synthetic examples or few high quality examples? The usualy answer is few high quality, but as we've already established, my model is ~40% undertrained, so something inbetween mid-training and SFT might be best here. 

Session 1: trained above model for 10k iterations. Works much better as a conversational agent compared to during mid-training. Looks like the assistant masking worked! I'd like to run a quick fine tune on the LIMA data next (id = ieqhwwrl)

Session 2: I also got the LIMA dataset so let's train for 3 epochs. At 683k tokens with ~20% padding that's 683000 / 2048 / (1 GPU) / 18 (batch size) / (1 - 0.2) = 69 iterations.

# Infrastructure and multi-GPU setup

Through trial and error on RunPod/Lambda cloud GPU instances, I arrived at a working multi-GPU configuration captured in startup.sh and run_pytorch.sh. These are worth documenting because they represent hours of debugging hangs and crashes.

The setup targets cloud GPU pods (Ubuntu-based) with CUDA 12.4. Key choices:
- Python 3.11 in a venv (not system Python) for isolation
- PyTorch installed from the cu124 wheel index to match the driver
- hf_transfer installed for faster HuggingFace dataset downloads (the default requests-based download is painfully slow for 170+ parquet shards)
- tmux for keeping training alive after SSH disconnects (critical for multi-hour sessions on cloud instances)

The cloud GPU instances I used (RunPod, Lambda) don't have InfiniBand or NVLink between GPUs, so NCCL's default communication strategy fails or hangs. The working set of flags:

- `NCCL_P2P_DISABLE=1` — Disable peer-to-peer GPU memory access. Without this, training hangs indefinitely at the first all-reduce. This was the original fix discovered during scaling up (mentioned above).
- `NCCL_IB_DISABLE=1` — Disable InfiniBand transport. The cloud instances don't have IB hardware, but NCCL tries to use it by default and fails silently.
- `NCCL_SOCKET_IFNAME=lo` — Force NCCL to use the loopback interface. On multi-GPU single-node setups, there's no need for network traffic between GPUs — everything goes through shared memory or sockets on localhost.
- `NCCL_NET=Socket` and `NCCL_NET_PLUGIN=none` — Force plain socket transport and disable any network plugins. This is the most conservative/compatible transport and avoids issues with missing or misconfigured network plugins on cloud instances.
- `NCCL_DEBUG=WARN` — Only show warnings, not the firehose of INFO messages. Useful during debugging (can set to INFO or TRACE) but too noisy for normal training.

Together these flags tell NCCL: "don't try anything fancy, just use sockets over loopback." The tradeoff is some communication overhead vs. NVLink/P2P, but for 4-GPU training the bottleneck is compute not communication, so the impact is negligible.

Using `torchrun --standalone` for single-node multi-GPU. The `--standalone` flag handles the rendezvous internally (no need for a separate etcd or c10d store). Each GPU gets its own process with RANK, WORLD_SIZE, and LOCAL_RANK set automatically.