# Multilingual Semantics Probe

Started: 2025-12-18

<!-- Writeup: [Interpreting Context Look-ups in Transformers: Investigating Attention-MLP Interactions](https://arxiv.org/abs/2402.15055) -->

In Progress: [Github Repo](https://github.com/Ky-Ng/reproducing-neo-et-al-2024)

???+ tip "Timelog"
    To timebox this project, we'll follow a [Nanda MATS stream](https://tinyurl.com/neel-mats-app) style 16-20 hour project.

    Hopefully, this rigorous/pragmatic approach can lead to some ambitious outcomes in understanding model representations of the scope ambiguity phenomenon.

    | Hour | Progress |
    | ---- | --------- |
    | 0-1 | Scaffold Project using GPT Project |
    | 1-2 | Setup Github, generate stimuli, start understanding log probs | 
    | 2-5.5 | Understand log probs, vectorized logic for Continuations Log Probs | 
    | 5.5-7.5 | Setup aggregation/comparing Log Probs for surface vs. inverse Prompts | 

## High Level Summary
I am interested in investigating how (multilingual) LLMs represent [`Quantifier Scope Ambiguity`](https://www.sfu.ca/~jeffpell/Ling324/fjpSlides7.pdf) cross-linguistically. 

Examples:

| Language | Sentence                                                      | Expected Interpretation                                                                                     |
| -------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| English  | A shark ate every pirate                                      | Ambiguous<br>(1) There is exactly 1 shark (Surface Scope)<br>(2) There are 1 or more sharks (Inverse Scope) |
| Mandarin | yǒu yì tiáo shāyú chī le měi yí gè hǎidào<br><br>有一條鯊魚吃了每一個海盜 | Unambiguous<br>(2) There is exactly 1 shark (Surface Scope)                                                 |

???+ note "Research Questions"
    1. Do LLMs allow for Ambiguous (Surface/Inverse) quantifier scope in English but only Surface Scope in Mandarin?
    2. Recent research on model size has found "shared circuitry increases with model scale" ([Brinkmann et al. 2025](https://arxiv.org/abs/2501.06346)); does model size impact whether an LLMs applies the correct language-specific interpretation?
        1. E.g. larger models with a multi-lingual semantic space would incorrectly apply the Inverse scope to Mandarin while a small model wouldn't
    3. Is there an interpretable hidden representation for quantifier scope? Can this be used to steer the model to get a specific interpretation in a particular language?

???+ example "Hypothesis"
    Given a multiple-choice situation (`please choose exactly 1 answer, (1) there is exactly 1 shark | (2) there are 1 or more sharks`):

    - Larger models will over-generalize English Inverse Scope semantics to Mandarin while smaller models will correctly apply language-specific semantic interpretations

??? note "Original Study: Abstract of Scrontas et al."
    
    Accurately recognizing and resolving ambiguity is a hallmark of linguistic ability. English is a language with scope ambiguities in doubly-quantified sentences like A shark ate every pirate; this sentence can either describe a scenario with a single shark eating all of the pirates, or a scenario with many sharks—a potentially-different one eating each pirate. In Mandarin Chinese, the corresponding sentence is unambiguous, as it can only describe the single-shark scenario. We present experimental evidence to this effect, comparing native speakers of English with native speakers of Mandarin in their interpretations of doubly-quantified sentences. Having demonstrated the difference between these two languages in their ability for inverse scope interpretations, we then probe the robustness of the grammar of scope by extending our experiments to English-dominant adult heritage speakers of Mandarin. Like native speakers of Mandarin, heritage Mandarin speakers lack inverse scope in Mandarin. Crucially, these speakers also lack inverse scope in English, their dominant language in adulthood. We interpret these results as evidence for the pressure to simplify the grammar of scope, decreasing ambiguity when possible. In other words, when two systems meet—as in the case of heritage speakers—the simpler system prevails.

### Project Scaffold

1. Create a stimulus dataset of inverse/surface quantifier scope with various lexical items (words, aka not just sharks and pirates)
2. Create pipeline for evaluating model log-probs for surface/inverse scope
3. Inspect models that show inverse scope (Interp techniques TBD keeping pragmaticism in mind; likely linear probe + steering vector)

## Progress Details

??? note "Hour 0-1: Scaffolding with AI"
    
    1 hour brainstorming session with GPT on roadmap for executing the experiment
    
    - Started looking into the interpretability technique (linear probe) contenders and models (Gemma, Llama, Qwen, DeepSeek)
    
    - A new TODO is to understand how Reasoning models work mathematically (high level intuition)


??? note "Hour 1-2: Setup Github, generate stimuli, start understanding log probs"
    
    1. Evaluation methods: Use same language continuations instead of MCQs to test latent linguistic knowledge rather than meta-linguistic judgement. Specifically, MCQs may be testing a models reasoning capabilities which are often skewed towards English reasoning (doesn't exactly tell us something interesting about the models underlying representations if we look at the likely English reasoning space)

    2. Setup ipynb to generate `stimuli,jsonl` and `stimuli_with_continuations.jsonl`

    3. Next Step: Understand how to compare continuations log probs from first principles

??? note "Hour 2-5.5: Understand log probs, vectorized logic for Continuations Log Probs"
    
    1. Work out Log Probs/comparisons from first principles on pencil and paper

    2. Extract Log Probs of inverse/surface continuations through a batched operation

    - Probably should take more breaks/sleep when working on this instead of grinding through the night

??? note "Hour 5.5-7.5: Setup aggregation/comparing Log Probs for surface vs. inverse Prompts"
    
    1. Setup pipeline to compare surface vs. inverse log sum/mean difference and odds (exponentiate the log differences)

    2. Save outputs for model specifics

    The key finding from working on GPT-2 is that the model always prefers surface scope. Next, we will try to see if this extrapolates to other models that have more than just pretraining.

    Additional follow up:
    
    - After running some experiments as background jobs, the models `Qwen3-0.6B`, `Llama-3.2-1B`, and `Llama-3.2-1B-Instruct`
    - Interestingly enough, `gemma-3-1b` prefers `inverse` scope for Mandarin for most examples whereas for English both surface and inverse are preferred. This is counter to intuitions from Natural Language semantics where `inverse` scope is not available
        - This could be due to pragmatics ("a child made every parent smile" is more likely to have inverse scope than "a president made every citizen happy")
        
    TODO:

    - Perhaps investigate this pragmatic infludence with MCQ style questions?

    - Investiage whether the prompts in different languages show the same inverse/surface scope; if both Mandarin and English prompt translations give the same scope, this is likely affects of pragmatics (and the most "likely"/"first" interpretation)