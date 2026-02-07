# Building Predictive Concept Decoders (PCD) on a Student Budget

Date: Feb. 7th, 2026

Github: 

Paper: 

- Attempted 1 day hackathon to build PCDs on a Student Budget

???+ note "Motivation"
    - I had the opportunity to see Sarah Schwettmann present PCDs at the NeurIPS 2025 MechInterp Workshop! I didn't have the Interp understanding at the time to grasp the concepts (other than, *wow this is really cool!*)
    - It's been 62 days since NeurIPS 2025, looking to try my hand at reproducing SOTA research
        - Also been wanting to work on this since my week at LISA but have been battling catching up with the school I missed while in the UK! Looking to timebox this into a single Saturday at the library!
    
## High Level Plan

Goal: Reproduce the most important aspects of PCDs (1) Encoder/Decoder Architecture (2) Pretraining/FT (3) Understand $L_{aux}$
    - Cutting out baseline benchmarking

0. Understand math/intuition behind PCDs (maybe take a quick look at Activation Oracles (AOs))

1. Implement Encoder/Decoder Architecture (w/ soft tokens from $l_{read}$ to $l_{write}$ and Top-K) with unitialized weights

2. Setup training infrastructure (LoRA), $L_{aux}$

3. Attempt to uncover a hidden goal using PCD corroborated by Encoder Concepts

4. Open source streamlined version/tutorial/ReadMe (maybe LessWrong post?)

- Use AI models (e.g. Claude) to help understand and scaffold code but not Coding Agents (e.g. Claude Code) to get the intuitions and practice of building by hand

## What about "Student Budget"?

- I hope that this reproduction will be accessible to other people (especially students like me!)

- Training on a GPU seems fun to setup, there should be some pretty economical options (ARBOx used Runpod/Vast)!

- To Pareto Principle (80-20) this, we will skip the evals/baselines section. Assuming we've followed the paper correctly, we do not need to verify we're doing better than baselines (also this is more for educational purposes)

## Math Deep Dive

- Feel free to skp this section! This is for me to understand the math/intuition behind how PCDs work and hopefully generate a very cool and intuitive walkthrough explanation!

### Architecture

$\mathcal{S}$ = Subject Model we're trying to interpret

$\mathcal{E}$ = Encoder, maps activation $a$ from subject model $S$ to a sparse set of concepts

$\mathcal{D}$ = Decoder from sparse set of concepts to natural language; same architecture/initialized to $\mathcal{S}$ with a LoRA adapter

Figure 2 from the paper:
![PCD Architecture](../../assets/projects/pcd/pcd_arch.png)

#### Encoder

let $a^{(i)} \in \mathbb{R}^{d}$ be an activation from layer $l_{read}$ in $\mathcal{S}$ at token position $i$ where $d$ is the dimensionality of the activation (usually residual stream)

let $m$ be the number of directions or `concepts` in the encoder's concept dictionary

let $W_{enc} \in \mathbb{R}^{m \times d}$, each row represents a concept ("template") in the Encoder space

$$\mathcal{E}(a) = \text{how strongly each of the m concepts are expressed} \in \mathbb{R}^m$$

$$\mathcal{E}(a) = (W_{enc}(a^{(i)})+ b_{enc}) \in \mathbb{R}^m$$

<small>Quick aside: The encoder-decoder architecture is similar to the 1-layer MLP from SAEs (similar sparsity penalty). However, the big difference is the training objective, SAEs $W_{enc}$ learn concepts useful to reconstruction the activations (hence `auto` in `auto-encoder`), meanwhile PCDs $W_{enc}$ learn concepts useful for the decoder to decode model behavior into natural language</small>

#### Sparsity $TopK$

- Sparsity constraint; $k=16$ in the paper

$$ TopK(\cdot) = \text{zero out all but the $k$ largest entries}$$

#### Decoder 

let $W_{emb} \in \mathbb{R}^{d \times m}$ map the sparse concept list into activation space for the decoder $\mathcal{D}$ to use
    - Note that the `emb` name comes from how the output of the encoder-decoder $a'^(i)$ is patched into the embedding of decoder $\mathcal{D}$

let $a'^{(i)} \in \mathbb{R}^d$ be the patched activation into the decoder $\mathcal{D}$, written to decoder layer $l_{write}$

$$ \mathcal{D}(a) = W_{emb} (TopK(\mathcal{E}(a)))$$

Formula (1) from the paper:
$$ a'^{(i)} = W_{emb}(TopK(W_{enc}(a^{(i)}) + b_{enc})) $$

### Training

Adapted from bullet list at end of Section 2 in paper:

| Training Phase | Task | Intuition | Example | 
| -------------- | ---- | --- | --- |
| Pretraining    | Jointly train $\mathcal{E}$ and $\mathcal{D}$ on next-token prediction of FineWeb | Have $\mathcal{E}$ learn a sparse concept dictionary by predicting completions (what comes next) | "Curious George and the Man in the Yellow Hat went to the zoo. They saw `<COMPLETION>`", $\mathcal{E}$ should learn general concepts (e.g. agents, objects, animals) to predict the `<COMPLETION>` |
| Finetuning | Freeze $\mathcal{E}$ and finetune $\mathcal{D}$ on QA about model beliefs | In pretraining we already learned concept dictionaries, now interpret this concept dictionary to talk about model beliefs; imagine a person given a really good dictionary of concepts for a story and asked to write a story about what the model is thinking | "Curious George and the Man in the Yellow Hat went to the zoo. They saw `<COMPLETION>`"; the completion is now "the model believes there are two characters and is thinking about animals, specifically zebras and lions" |

#### Pretraining

let $n_{prefix}$, $n_{middle}$, and $n_{suffix}$ be the lengths of the tokens of the prefix, middle, and suffix of the prompt $P$

$$ P = p^({1:n_{prefix}}) + p^({n_{prefix}:n_{middle}}) + p^({n_{middle}:n_{suffix}}) $$

To make our lives easier:

- let $s^({1:n_{suffix}})$ be the suffix (in finetuning, the explanation of model behavior) that the decoder $\mathcal{D}$ is trying to predict, represented above as the corresponding tokens ($p^({n_{middle}:n_{suffix}})$)

- let $ a^{(1:n_{middle})} \in \mathbb{R}^d$ be the activations from $\mathcal{S}$ at layer $l_{read}$ for the middle tokens (don't include the activations from the prefix $p^({1:n_{prefix}})$!). 

From $a^{(1:n_{middle})}$, predict suffix tokens $s^({1:n_{suffix}})$.

Formally from equation 2 in the paper the loss $\mathcal{L}_{next-tokens}$:

$$ \mathcal{L}_{next-tokens} = - \sum_{t=1}^{n_{suffix}} \log p_{\mathcal{D}} (s^{(t)} | s^{(1:t-1)}, \mathcal{E}(a^{(1:n_{middle})})) $$

- In plain English: we minimize loss when predicting the suffix tokens from the previous suffix tokens and the encoded middle activations

![PCD Pretraining](../../assets/projects/pcd/PCD_Pretraining.png)

??? note "Aside on`communication bottleneck`"
    We never explicitly see the activations of the prefix tokens. Why?!

    1. In the pretraining task, the decoder is doing good ol' next-token prediction.

    2. If the prefix tokens were also passed into the decoder, since the decoder's pretraining task is predicting the suffix (e.g. `a` `bomb`), the encoder could learn to pass through a version of the token embedding rather than interpretable/useful concepts

    3. If we take away the TopK, given that $m$ (the size of the up projection of the encoder) is much larger than $d$, then we could literally pass the token embedding. Although we do have a TopK which enforces sparsity, the same idea applies that the encoder could not actually learn meaningful information.

    ![Thought Experiment: PCD Pretraining with Prefix Token Activations](../../assets/projects/pcd/PCD_Pretraining_with_Prefix.png)

#### Auxiliary Loss `Maintaining Concept Activity`

let $I$ be the set of inactive concepts with the highest dot product with $a$ (almost activated/selected but not quite), where inactive means the concept has not been selected in the TopK for the last 1M tokens

let $k_{aux} = |I|$

let $\epsilon_{aux}$ be a scalar of the gradient to backpropogate into the inactive concept selected, in the paper $\epsilon_{aux} = 10^{-4}$

We have divide by $k_{aux}$ in $ \frac{\epsilon_{aux}}{k_{aux}} $ because imagine if we have more inactive concepts, we don't want to overload the loss (normalize the aux loss by the number of inactive concepts we want to help revive)

Intuition: For concepts that are very close to being activated but not selected, let those concepts get a little bit of gradient so they can become more aligned with being useful for the decoder

$$ \mathcal{L}_{aux} = - \frac{\epsilon_{aux}}{k_{aux}} \sum_{i \in I} W_{enc, i}(a) $$

#### Finetuning