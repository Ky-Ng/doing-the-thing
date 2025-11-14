# Interpreting Context Look Ups
Authors: Clement Neo, Shay B. Cohen, Fazl Barez

Publication Date: 2024-10-23

Full Paper: [Interpreting Context Look-ups in Transformers: Investigating Attention-MLP Interactions](https://arxiv.org/abs/2402.15055)

In Progress Reproduction: [Github Repo](https://github.com/Ky-Ng/reproducing-neo-et-al-2024)

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

## Identifying Next-token Neurons and MLP Interpretability

### MLP Architecture
??? example "MLP Algorithm"
    let Up Projection Weight Matrix $ = W_{up} \in \mathbb{R}^{d_{up}, d_{model}}$

    let Down Projection Weight Matrix $ = W_{down} \in \mathbb{R}^{d_{model}, d_{up}}$

    let Activation Function $ \sigma = GELU $ <small>[ReLU but more differentiable](https://arxiv.org/abs/1606.08415)</small>

    let biases = $b_{up} \in \mathbb{R}^{d_{up}}$ and $b_{down} \in \mathbb{R}^{d_{down}}$

    $$ MLP(h) = W_{down} \sigma(W_{up} h + b_{up}) + b_{down} \in \mathbb{R}^{d_{model}}$$

    Note that `MLP(h)` must be in $\mathbb{R}^{d_{model}}$ since it is written back to the "working memory" Residual Stream

### Next-Token Neuron Algorithm
I like to call this part the "Secret Life of a Hidden Neuron" (also known as finding the `next-token neuron`). This consists of three steps:

1. Feature Weighting (Reading from Residual Stream)
2. Down Projection Matrix Columns as Information (Writing to Residual Stream)
3. Next-token neuron as max congruence between information and unembedding

??? info "Concept 1: Feature Weighting"
    To find the `next-token neuron` $a_i$ or weight of information vector $W_{down}[:,i]$, we calculate $a$:
    
    $$a = \sigma(W_{up} h + b_{up}) \in \mathbb{R}^{d_{up}}$$

    1. Crucially, the consequence of the formula is that $a_i = \langle W_{up}[i], h \rangle$

        a. The row $W_{up}[i]$ helps us to compute the "gating neuron" $a_i$

        b. $W_{up}[i]$ "reads" from the Residual Stream

    2. Each weight $a_i$` decides to include`/`gates`/`weights` a specific information basis vector $W_{down}[:,i] \in \mathbb{R}^{d_{model}}$ to be passed onto the Residual Stream $h$
    3. There are $d_{up}$ features to weigh; high dimensionality $d_{up}$ allows the model to store $d_{up}$ "vectors of information"

??? info "Concept 2: $W_{down}[:,i]$ as information basis vectors"
    In this section, we assume the following interpretation of matrices (based on 3Blue1Brown [Change of basis | Chapter 13, Essence of linear algebra](https://www.youtube.com/watch?v=P2LTAUO1TdA)):
    
    1. Matrices encode linear transformations 
    2. The columns of a matrix are being basis vectors of `Source Space (SS)` expressed in `Target Space (TS)`

        a. A basis vector $b$ defines how a `SS` vector component along axis $b$ should stretch/transform in `TS` (see handwritten notes for visual intuition)
    
    Interpretation of basis vectors as encoding information:

    1. Matrix $W_{down} \in \mathbb{R}^{d_{model}, d_{up}}$ has $d_{up}$ basis vectors in `TS` (we can call this `MLP space` if we like) from `SS` $d_{model}$ (we can call this `Residual space` if we like)
    2. Each column $W_{down}[:,i]$ is a basis vector in `MLP space` representing potential information to add to $h_k$ in the Residual Stream of layer $k$
    3. $a_i$ `chooses` which information basis vectors $W_{down}[:,i]$ should be added to add to $h_k$ in the Residual Stream

    The update from the previous Residual Stream $h_{k}$ to the new Residual Stream $h_{k+1}$ is mathematically expressed as the weighted sum of all the $d_{up}$ potential information to include:

    $$h_{k+1} = \sum_{i=1}^{d_{up}} a_i \cdot W_{down}[:,i] $$

??? example "Concept 3: Next-token Neuron = Max Congruence Score"
    Armed with an understanding of $a_i$  $W_{down}[:,i]$ and $ W_{up}[i]$, we define the `potential` for a neuron $a_i$ to activate token $t$ as $s_i$:

    $$ s_i = \underset{t \in \mathbb{V}}{max} \langle W_{down}[:, i], e^{(t)} \rangle $$

    where $e^{(t)}$ = the unembedding vector for token $t$ <small>[LLM Basics: Embedding Spaces - Transformer Token Vectors Are Not Points in Space by AI Alignment Forum](https://www.alignmentforum.org/posts/pHPmMGEMYefk9jLeh/llm-basics-embedding-spaces-transformer-token-vectors-are) has a very nice introduction</small> with vocabulary $\mathbb{V}$

    Since $s_i$ is not dependent on the activation/"neuron firing" of $a_i$, $s_i$ represents the congruence of the information basis vector $W_{down}[:, i]$ with the unembedding $e^{(t)}$ if the neuron were to "fire".

    Thus, the next-token neuron for a token $t$ is the largest $i$ with the largest $s_i$ mathematically expressed as:

    $$ \text{next-token neuron for token t} = \underset{i \in 1:d_{up}}{argmax} (s_i)$$

### Summary Table: Math + Interpretation
| Math   | Interpretation  |
|------|------|
| $W_{up}[i]$ | `Read`/`feature extract` from Residual Stream $h_k$ and decide whether or not to activate information in $W_{down}$ |
| $W_{down}[:, i]$ | Information to add to the residual stream $h_{k+1}$ |
| $\langle a_i, W_{down}[:, i] \rangle$  | `Gate`/`weight`/`how much` of information from $W_{down}[:,i]$ |
| $h_{k+1}$ | Weighted sum of information, $W_{down}[:, i]$ to add to residual stream |
| $s_i$ | Congruence between information in $W_{down}[:, i]$ and unembedding vector for token $t$, $e^{(t)}$ used to calcualte next-token neuron

## Head Attribution
??? info "Chain of Activations"
    Given a next-token neuron $i$ at layer $l$:

    1. The definition of a next-token neuron is that the $i$ chosen has max $s_i$ for all tokens
    2. Max $s_i$ means max 