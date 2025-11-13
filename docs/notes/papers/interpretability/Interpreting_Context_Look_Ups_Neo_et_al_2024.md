# Interpreting Context Look Ups
Authors: Clement Neo, Shay B. Cohen, Fazl Barez

Publication Date: 2024-10-23

Full Paper: [Interpreting Context Look-ups in Transformers: Investigating Attention-MLP Interactions](https://arxiv.org/abs/2402.15055)

## Summary
This paper aims to bridge the gap between MLP and Attention Interpretability by focusing on (1) identifying next-token neurons and (2) find which attention heads activate that neuron.

Limitation: The paper only focuses on single words and single neurons; future work could consider more complex neural firing patterns

Working on trying to reproduce this paper, we will focus on using GPT-2 small (though the paper also uses GPT-2 Large and Pythia as well)

See [handwritten notes/derivations (section Reproducing Neo et al. 2024)](https://share.goodnotes.com/s/XVJdMIolpMuDVClIVAADEl) for drawings of matrices / geometric intuition

## High Level Pipeline 
1. Identify next-token neurons; find prompts that highly activate them
2. Determine Attention heads most responsible for activating each next-token neuron using head attribution score
3. Use GPT-4 to generate explanations for activity patterns of these attention heads
4. Evaluate response quality by using GPT-4 zero shot classifier for head activity on a new prompt 

## Identifying Neurons and MLP Interpretability

### MLP Architecture
???+ example "MLP Algorithm"
    let Up Projection Weight Matrix $ = W_{up} \in \mathbb{R}^{d_{up}, d_{model}}$

    let Down Projection Weight Matrix $ = W_{down} \in \mathbb{R}^{d_{model}, d_{up}}$

    let Activation Function $ \sigma = GELU $ <small>[ReLU but more differentiable](https://arxiv.org/abs/1606.08415)</small>

    let biases = $b_{up} \in \mathbb{R}^{d_{up}}$ and $b_{down} \in \mathbb{R}^{d_{down}}$

    $$ MLP(h) = W_{down} \sigma(W_{up} h + b_{up}) + b_{down} \in \mathbb{R}^{d_{model}}$$

    Note that `MLP(h)` must be in $\mathbb{R}^{d_{model}}$ since it is written back to the "working memory" residual stream

I like to call this part the "Secret Life of a Hidden Neuron":

???+ example "Step 1: Feature Weighting"
    To find the `next-token neuron` $a_i$ or weight of information vector $i$, we calculate $a$:
    
    $$a = W_{up} h + b_{up} \in \mathbb{R}^{d_{up}}$$

    1. There are $d_{up}$ features to weigh; each weight $a_i$ gates a specific information basis vector $W_{down}[:,i] \in \mathbb{R}^{d_{model}}$ to be passed onto the residual stream
    2. $W_{up}$ is 

    
| Math   | Interpretation  |
|------|------|
| $W_{up}$ | `Read`/`feature extract` from residual stream and decide whether or not to activate information in $W_{down}$ |
| $a_i$ | `Gate`/`weight`/`how much` of information from $W_{down}[:,i]$ |



     that $a_i$ is the information gate, $W_j^{out}$ is the information (basis vector in $d_{model}$ space) and $