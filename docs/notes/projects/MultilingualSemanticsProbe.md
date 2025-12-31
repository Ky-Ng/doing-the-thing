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
    | 7.5-10.5 | Debug scripts to add (1) bfloat16 support large models (>27b) and (2) comparison of EN vs. ZH | 
    | 10.5-12 | Read [Fang et al.](https://arxiv.org/abs/2509.10860v1) and [Schut 2025 et al.](https://arxiv.org/abs/2502.15603), test GPT-2-Chinese (`uer_gpt2-xlarge-chinese-cluecorpussmall`), `aya-23-35B`, `Gemma2-27B`, and `aya-expanse-32b` | 
    | 12-12.5 | Verified that Existential-Universal and Universal-Existential prefer the same continuation preferences |
    | 12.5-13 | Find that English preference for inverse and Chinese preference for surface is statistically significant (p=1.948e-18)|
    | 13 - TBD | Plan Mechanistic Interpretability techniques to use |

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

??? note "Hour 7.5-10.5: Debug scripts to add (1) bfloat16 support large models (>27b) and (2) comparison of en vs. zh"
    
    1. Models greater than 12b (e.g. `Gemma-3-12b`) are too large to fit on a High-RAM A100 on Collab in fp32
        - Some back of the napkin calculation
        $$ 12 \text{ b params} \cdot \frac{32 \text{ bits}}{1 \text{ param}} \cdot \frac{1 \text{ byte}}{8 
        \text{ bits}} \cdot \frac{1 \text{ G}}{1 \text{ B}} 96 \text{ Gb}$$
        - Since a A100 High-RAM GPU has 167 GB of CPU RAM but 80 GB of HBM (GPU RAM), then FP32 will not fit on device
        - Pivoting to use FP16 halfs the footprint to 48GB but the decreased range causes logits to go to NaNs
        - Thus, using BF16 solves the memory footprint and range issues
            - We also upcast logits to FP32 during post-processing
    
    2. The results in the tabs below show that for models >4B, surface is always preferred; smaller models (270M and 1B) prefer surface for en and incorrectly prefer inverse for zh

    | Model Size   | `en` Preference to Surface | `en` Preference to Inverse | `zh` Preference to Surface | `zh` Preference to Inverse | Takeaway                                                                                                                      |
    | ------------ | -------------------------- | -------------------------- | -------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
    | Gemma-3-27B  | 59                         | 5                          | 51                         | 13                         | Strong surface preference in both English and Mandarin; large model behaves conservatively and consistently across languages. |
    | Gemma-3-12B  | 51                         | 13                         | 37                         | 27                         | Surface preference remains, but Mandarin shows degradation and increased inverse scope relative to English.                   |
    | Gemma-3-4B   | 47                         | 17                         | 41                         | 23                         | Both languages show weakened surface bias; Mandarin drifts further toward inverse interpretations.                            |
    | Gemma-3-1B   | 45                         | 19                         | 14                         | 50                         | English still surface-biased, but Mandarin strongly prefers inverse—opposite of theoretical expectation for small models.     |
    | Gemma-3-270M | 64                         | 0                          | 29                         | 35                         | English collapses entirely to surface scope; Mandarin slightly prefers inverse, showing extreme cross-lingual divergence.     |


    ??? example "Gemma-3-27b results"
        Takeaway: Model prefers surface form for both zh and en
        
        | model                 | language   |   inverse |   surface |   total | p_inverse   |
        |-----------------------|------------|-----------|-----------|---------|-------------|
        | google_gemma-3-27b-it | en         |         5 |        59 |      64 | 7.8%        |
        | google_gemma-3-27b-it | zh         |        13 |        51 |      64 | 20.3%       |

        | model                 | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |-----------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | google_gemma-3-27b-it | en         |                  64 |             -0.858 |               -0.863 |             0.546 |                  64 |              0.491 |                0.422 |             0.281 |
        | google_gemma-3-27b-it | zh         |                  64 |             -0.961 |               -1.098 |             1.238 |                  64 |              0.842 |                0.333 |             1.313 |

        | model                 | agreement_rate   |
        |-----------------------|------------------|
        | google_gemma-3-27b-it | 71.9%            |

        | model                 | pattern                |   count |
        |-----------------------|------------------------|---------|
        | google_gemma-3-27b-it | surface_EN__surface_ZH |      46 |
        | google_gemma-3-27b-it | surface_EN__inverse_ZH |      13 |
        | google_gemma-3-27b-it | inverse_EN__surface_ZH |       5 |

    ??? example "Gemma-3-12b results"
        Takeaway: Model prefers surface form for both zh and en; zh degraded performance

        | model                 | language   |   inverse |   surface |   total | p_inverse   |
        |-----------------------|------------|-----------|-----------|---------|-------------|
        | google_gemma-3-12b-it | en         |        13 |        51 |      64 | 20.3%       |
        | google_gemma-3-12b-it | zh         |        27 |        37 |      64 | 42.2%       |

        | model                 | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |-----------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | google_gemma-3-12b-it | en         |                  64 |             -0.496 |               -0.549 |             0.521 |                  64 |              0.699 |                0.578 |             0.392 |
        | google_gemma-3-12b-it | zh         |                  64 |              0.13  |               -0.517 |             1.634 |                  64 |              4.511 |                0.598 |             7.996 |

        | model                 | agreement_rate   |
        |-----------------------|------------------|
        | google_gemma-3-12b-it | 37.5%            |

        | model                 | pattern                |   count |
        |-----------------------|------------------------|---------|
        | google_gemma-3-12b-it | surface_EN__inverse_ZH |      27 |
        | google_gemma-3-12b-it | surface_EN__surface_ZH |      24 |
        | google_gemma-3-12b-it | inverse_EN__surface_ZH |      13 |
    
    ??? example "Gemma-3-4b results"
        Takeaway: Both zh/en prefer inverse slightly more. Degradation in Mandarin performance

        | model                | language   |   inverse |   surface |   total | p_inverse   |
        |----------------------|------------|-----------|-----------|---------|-------------|
        | google_gemma-3-4b-it | en         |        17 |        47 |      64 | 26.6%       |
        | google_gemma-3-4b-it | zh         |        23 |        41 |      64 | 35.9%       |

        | model                | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |----------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | google_gemma-3-4b-it | en         |                  64 |             -0.386 |               -0.602 |             1.155 |                  64 |              1.857 |                0.548 |             4.793 |
        | google_gemma-3-4b-it | zh         |                  64 |             -0.849 |               -0.861 |             1.955 |                  64 |              1.602 |                0.423 |             2.271 |

        | model                | agreement_rate   |
        |----------------------|------------------|
        | google_gemma-3-4b-it | 56.2%            |

        | model                | pattern                |   count |
        |----------------------|------------------------|---------|
        | google_gemma-3-4b-it | surface_EN__surface_ZH |      30 |
        | google_gemma-3-4b-it | surface_EN__inverse_ZH |      17 |
        | google_gemma-3-4b-it | inverse_EN__surface_ZH |      11 |
        | google_gemma-3-4b-it | inverse_EN__inverse_ZH |       6 |
    
    ??? example "Gemma-3-1b results"
        Takeaway: zh heavily prefers inverse while en prefers surface. Opposite of expected behavior; theoretically if aligned with [Brinkmann et al. 2025](https://arxiv.org/abs/2501.06346), then smaller models would prefer only surface form.

        | model                | language   |   inverse |   surface |   total | p_inverse   |
        |----------------------|------------|-----------|-----------|---------|-------------|
        | google_gemma-3-1b-it | en         |        19 |        45 |      64 | 29.7%       |
        | google_gemma-3-1b-it | zh         |        50 |        14 |      64 | 78.1%       |

        | model                | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |----------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | google_gemma-3-1b-it | en         |                  64 |             -0.432 |               -0.416 |             0.622 |                  64 |              0.772 |                0.66  |             0.455 |
        | google_gemma-3-1b-it | zh         |                  64 |              0.704 |                0.492 |             0.934 |                  64 |              3.56  |                1.636 |             5.84  |

        | model                | agreement_rate   |
        |----------------------|------------------|
        | google_gemma-3-1b-it | 45.3%            |

        | model                | pattern                |   count |
        |----------------------|------------------------|---------|
        | google_gemma-3-1b-it | surface_EN__inverse_ZH |      33 |
        | google_gemma-3-1b-it | inverse_EN__inverse_ZH |      17 |
        | google_gemma-3-1b-it | surface_EN__surface_ZH |      12 |
        | google_gemma-3-1b-it | inverse_EN__surface_ZH |       2 |
    
    ??? example "Gemma-3-270m results"
        Takeaway: en only predicts surface while slight preference to inverse for zh

        | model                  | language   |   inverse |   surface |   total | p_inverse   |
        |------------------------|------------|-----------|-----------|---------|-------------|
        | google_gemma-3-270m-it | en         |         0 |        64 |      64 | 0.0%        |
        | google_gemma-3-270m-it | zh         |        35 |        29 |      64 | 54.7%       |

        | model                  | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |------------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | google_gemma-3-270m-it | en         |                  64 |             -4.226 |               -3.902 |             1.644 |                  64 |              0.04  |                 0.02 |             0.057 |
        | google_gemma-3-270m-it | zh         |                  64 |              0.386 |                0.376 |             2.904 |                  64 |             24.851 |                 1.46 |            62.733 |

        | model                  | agreement_rate   |
        |------------------------|------------------|
        | google_gemma-3-270m-it | 45.3%            |

        | model                  | pattern                |   count |
        |------------------------|------------------------|---------|
        | google_gemma-3-270m-it | surface_EN__inverse_ZH |      35 |
        | google_gemma-3-270m-it | surface_EN__surface_ZH |      29 |

??? note "Hour 10.5-12: Read [Fang et al.](https://arxiv.org/abs/2509.10860v1) [Schut 2025 et al.](https://arxiv.org/abs/2502.15603), test GPT-2-Chinese (`uer_gpt2-xlarge-chinese-cluecorpussmall`), `aya-23-35B`, `Gemma2-27B`, and `aya-expanse-32b`"  
    
    Based on 
    1. [Schut 2025 et al.](https://arxiv.org/abs/2502.15603): Do Multilingual LLMs Think In English?
        a. LLMs use English representation to reason/ 
    2. [Fang et al.](https://arxiv.org/abs/2509.10860v1): Quantifier Scope Interpretation in Language Learners and LLMs

    Find that `gpt2-chinese` prefers surface for Chinese and inverse for English!

    ??? example "aya-23-35B results"
        | model                 | language   |   inverse |   surface |   total | p_inverse   |
        |-----------------------|------------|-----------|-----------|---------|-------------|
        | CohereLabs_aya-23-35B | en         |         5 |        59 |      64 | 7.8%        |
        | CohereLabs_aya-23-35B | zh         |        31 |        33 |      64 | 48.4%       |

        | model                 | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |-----------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | CohereLabs_aya-23-35B | en         |                  64 |             -0.89  |               -0.964 |             0.485 |                  64 |              0.468 |                0.382 |             0.283 |
        | CohereLabs_aya-23-35B | zh         |                  64 |              0.035 |               -0.018 |             0.389 |                  64 |              1.111 |                0.982 |             0.405 |

        | model                 | agreement_rate   |
        |-----------------------|------------------|
        | CohereLabs_aya-23-35B | 56.2%            |

        | model                 | pattern                |   count |
        |-----------------------|------------------------|---------|
        | CohereLabs_aya-23-35B | surface_EN__surface_ZH |      32 |
        | CohereLabs_aya-23-35B | surface_EN__inverse_ZH |      27 |
        | CohereLabs_aya-23-35B | inverse_EN__inverse_ZH |       4 |
        | CohereLabs_aya-23-35B | inverse_EN__surface_ZH |       1 |

    ??? example "gpt2-xlarge-chinese-cluecorpussmall results"
        | model                                   | language   |   inverse |   surface |   total | p_inverse   |
        |-----------------------------------------|------------|-----------|-----------|---------|-------------|
        | uer_gpt2-xlarge-chinese-cluecorpussmall | en         |        64 |         0 |      64 | 100.0%      |
        | uer_gpt2-xlarge-chinese-cluecorpussmall | zh         |         0 |        64 |      64 | 0.0%        |

        | model                                   | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |-----------------------------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | uer_gpt2-xlarge-chinese-cluecorpussmall | en         |                  64 |              0.471 |                0.467 |             0.08  |                  64 |              1.607 |                1.596 |             0.129 |
        | uer_gpt2-xlarge-chinese-cluecorpussmall | zh         |                  64 |             -0.703 |               -0.715 |             0.243 |                  64 |              0.51  |                0.489 |             0.125 |

        | model                                   | agreement_rate   |
        |-----------------------------------------|------------------|
        | uer_gpt2-xlarge-chinese-cluecorpussmall | 0.0%             |

        | model                                   | pattern                |   count |
        |-----------------------------------------|------------------------|---------|
        | uer_gpt2-xlarge-chinese-cluecorpussmall | inverse_EN__surface_ZH |      64 |

    ??? example "aya-expanse-32b results"
        | model                      | language   |   inverse |   surface |   total | p_inverse   |
        |----------------------------|------------|-----------|-----------|---------|-------------|
        | CohereLabs_aya-expanse-32b | en         |         0 |        64 |      64 | 0.0%        |
        | CohereLabs_aya-expanse-32b | zh         |        61 |         3 |      64 | 95.3%       |

        | model                      | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |----------------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | CohereLabs_aya-expanse-32b | en         |                  64 |             -1.681 |               -1.636 |             0.366 |                  64 |              0.198 |                0.195 |             0.071 |
        | CohereLabs_aya-expanse-32b | zh         |                  64 |              0.82  |                0.922 |             0.487 |                  64 |              2.532 |                2.514 |             1.198 |

        | model                      | agreement_rate   |
        |----------------------------|------------------|
        | CohereLabs_aya-expanse-32b | 4.7%             |

        | model                      | pattern                |   count |
        |----------------------------|------------------------|---------|
        | CohereLabs_aya-expanse-32b | surface_EN__inverse_ZH |      61 |
        | CohereLabs_aya-expanse-32b | surface_EN__surface_ZH |       3 |

    ??? example "gpt2-chinese-cluecorpussmall (aka small) results"
        | model                            | language   |   inverse |   surface |   total | p_inverse   |
        |----------------------------------|------------|-----------|-----------|---------|-------------|
        | uer_gpt2-chinese-cluecorpussmall | en         |        64 |         0 |      64 | 100.0%      |
        | uer_gpt2-chinese-cluecorpussmall | zh         |         0 |        64 |      64 | 0.0%        |

        | model                            | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |----------------------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | uer_gpt2-chinese-cluecorpussmall | en         |                  64 |              1.014 |                1.036 |             0.206 |                  64 |              2.813 |                2.819 |             0.55  |
        | uer_gpt2-chinese-cluecorpussmall | zh         |                  64 |             -0.923 |               -0.871 |             0.185 |                  64 |              0.404 |                0.418 |             0.071 |

        | model                            | agreement_rate   |
        |----------------------------------|------------------|
        | uer_gpt2-chinese-cluecorpussmall | 0.0%             |

        | model                            | pattern                |   count |
        |----------------------------------|------------------------|---------|
        | uer_gpt2-chinese-cluecorpussmall | inverse_EN__surface_ZH |      64 |

??? note "Hour 12-13: Verify (1) Existential-Universal and Universal-Existential preference (2) statistical significance"

    1. Universal-Existential and Existential-Universal

        a. Verify that the surface/inverse scope preference hold for either a Existential-Universal and Universal-Existential test

    2. Using a Wilcox Signed-Rank Test 

        | Scope Type | Language | p-value |
        | ---------- | -------- | ------- |
        | Existential-Universal | en | 1.948e-18 |
        | Existential-Universal | zh | 1.948e-18 |
        | Universal-Existential | en | 1.674e-15 |
        | Universal-Existential | zh | 1.948e-18 |

??? note "Hour 13-14: Learn math of Steering Vectors" 
    Potential Mechanistic Interpretability techniques:
    
    <small>The results above are super exciting! In fact, this will be my first time working on any SOTA probe type of technology from the MechInterp literature. A key focus right now is to make sure that I stay pragmatic considering I'm already at hour 13 of this project</small>
    
    Suggested approaches from AI:
    
    1. Linear probes (per layer, per token)
    
    2. Difference-of-means direction
    
    3.Steering intervention
    
    4. One clean ablation experiment

    Note: I haven't ever used any of these techniques so to me it is more important to also learn what these techniques imply/shortcomings.

    Side note on hypotheses: I assumed that we are looking for a `scope vector` represented in the model. Ideally this `scope vector` would be active in both the zh and en sentences (available cross linguistically) and be steerable so that we can cause inverse scope preference for continuation. (since these GPT-2 models are only pretrained and not instruction tuned, it's not possible to affect the models QA/instruction following capabilities)

    Derived the math for Steering Vectors from first principles with help from AI.

??? note "Hour 14-15: Setback--GPT2-Chinese is unreliable"
    Turns out the clean results from GPT2-Chinese judgements are a result of the model being undertrained.'

    For example, this is the next most likely set of tokens in for `uer/gpt2-xlarge-chinese-cluecorpussmall`. I realized this when looking at the vocab size for the GPT-2-Chinese models being ~20k whereas the original GPT2-vocab is ~50k. Thus, this got me suspicious and sent me down a probing route. I am surprised how [Fang et al.](https://arxiv.org/abs/2509.10860v1) was able to use these models for their double quantifier paper.

    Thus, it is likely that the model does not actually have a good understanding of the language.
    ```
    'a shark ate everyday.the##n.a.n.a.a.a.a.a.a.a.a.a.a.a.a.a.'
    ```

    ```
    每一条都是经过反复推敲的，不是随便说说的。[UNK][UNK][UNK][UNK][UNK][UNK][UNK][UNK][UNK][UNK][UNK][UNK][UNK][UNK][UNK][UNK][UNK]
    ```

    
    `qwen2.5-3B` outputs are quite reasonable:
    ```
    每一條路線的車站數量、車站名稱、車站位置、車站間的距離、車站間的時間、車站間的運行時間、車站間的運行速度、車站間的運

    The number of stations on each route; the station names; station locations; the distances between stations; the time between stations; the operating time between stations; the operating speed between stations;
    ```

    ```
    A Shark ate every pirate on a ship. The pirates were divided into 3 groups. The first group had 10 pirates, the second group had 15 pirates, and the third group had 20 pirates. If each pirate had 2 eyes, how
    ```

??? note "Hour 15-16: Qwen2.5 Surface/Inverse Scope Judgements"
    Goal: Try SOTA models which likely have high Mandarin proficiency (e.g. produced by Chinese AI Labs)

    Note: here we are making a critical assumption that Qwen and Deepseek models have acquired Mandarin and English grammars; we safely make this assumption since these models are SOTA. However, we are also testing the models through a sanity check script to ensure the model produces reasonable completions.

    After an hour of experimenting, Qwen2.5-{0.5, 1.5, 3, 7, 14, 32}B follow these trends:

    1. Existential-Universal (EU) for English prefers Surface but can accept some inverse scope readings starting model 1.5B and larger. Mandarin stimuli accept only surface forms
        a. Interestingly, the English inverse preference occurs when the subject is `kangaroo`. Perhaps some more stimui are needed to narrow down this behavior
    2. Universal-Existential (UE) show Surface preference for English and Mandarin. Surprisingly, Mandarin also accepts inverse scope in UE.
    3. All preferences for surface are significant

    Next steps: 
    
    - Since our goal is to use these sentences as a way to probe for steering vectors and then causality, the stats/stimlui need not be perfect. 
    - Instead, we can look to find those steering vectors to test the influence of hypothesized steering vectors.

    ??? example "Qwen/qwen2.5-0.5b eu scope"
        | model             | language   |   inverse |   surface |   total | p_inverse   |
        |-------------------|------------|-----------|-----------|---------|-------------|
        | Qwen_Qwen2.5-0.5B | en         |         7 |        93 |     100 | 7.0%        |
        | Qwen_Qwen2.5-0.5B | zh         |         8 |        92 |     100 | 8.0%        |

        | model             | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |-------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | Qwen_Qwen2.5-0.5B | en         |                 100 |             -0.779 |               -0.797 |             0.536 |                 100 |              0.528 |                0.45  |             0.292 |
        | Qwen_Qwen2.5-0.5B | zh         |                 100 |             -0.58  |               -0.564 |             0.621 |                 100 |              0.669 |                0.569 |             0.494 |

        | model             | agreement_rate   |
        |-------------------|------------------|
        | Qwen_Qwen2.5-0.5B | 85.0%            |

        | model             | pattern                |   count |
        |-------------------|------------------------|---------|
        | Qwen_Qwen2.5-0.5B | surface_EN__surface_ZH |      85 |
        | Qwen_Qwen2.5-0.5B | surface_EN__inverse_ZH |       8 |
        | Qwen_Qwen2.5-0.5B | inverse_EN__surface_ZH |       7 |

        | Language   |   n |   Statistic |   p-value |   Mean Δ |   Median Δ | Preference   |
        |------------|-----|-------------|-----------|----------|------------|--------------|
        | en         | 100 |         100 |  3.78e-17 |  -0.7787 |    -0.7974 | surface      |
        | zh         | 100 |         380 |  8.2e-14  |  -0.5802 |    -0.5642 | surface      |

    ??? example "Qwen/qwen2.5-0.5b ue scope"
        | model             | language   |   inverse |   surface |   total | p_inverse   |
        |-------------------|------------|-----------|-----------|---------|-------------|
        | Qwen_Qwen2.5-0.5B | en         |         6 |        94 |     100 | 6.0%        |
        | Qwen_Qwen2.5-0.5B | zh         |        29 |        71 |     100 | 29.0%       |

        | model             | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |-------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | Qwen_Qwen2.5-0.5B | en         |                 100 |             -0.428 |               -0.441 |             0.242 |                 100 |              0.671 |                0.644 |             0.169 |
        | Qwen_Qwen2.5-0.5B | zh         |                 100 |             -0.258 |               -0.301 |             0.394 |                 100 |              0.833 |                0.74  |             0.323 |

        | model             | agreement_rate   |
        |-------------------|------------------|
        | Qwen_Qwen2.5-0.5B | 69.0%            |

        | model             | pattern                |   count |
        |-------------------|------------------------|---------|
        | Qwen_Qwen2.5-0.5B | surface_EN__surface_ZH |      67 |
        | Qwen_Qwen2.5-0.5B | surface_EN__inverse_ZH |      27 |
        | Qwen_Qwen2.5-0.5B | inverse_EN__surface_ZH |       4 |
        | Qwen_Qwen2.5-0.5B | inverse_EN__inverse_ZH |       2 |

        | Language   |   n |   Statistic |   p-value |   Mean Δ |   Median Δ | Preference   |
        |------------|-----|-------------|-----------|----------|------------|--------------|
        | en         | 100 |          49 |  8.45e-18 |  -0.4277 |    -0.4407 | surface      |
        | zh         | 100 |         947 |  2.89e-08 |  -0.2577 |    -0.3014 | surface      |

    ??? example "Qwen/qwen2.5-1.5b eu scope"
        | model             | language   |   inverse |   surface |   total | p_inverse   |
        |-------------------|------------|-----------|-----------|---------|-------------|
        | Qwen_Qwen2.5-1.5B | en         |        14 |        86 |     100 | 14.0%       |
        | Qwen_Qwen2.5-1.5B | zh         |         0 |       100 |     100 | 0.0%        |

        | model             | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |-------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | Qwen_Qwen2.5-1.5B | en         |                 100 |             -0.413 |               -0.436 |             0.309 |                 100 |              0.695 |                0.647 |             0.231 |
        | Qwen_Qwen2.5-1.5B | zh         |                 100 |             -0.8   |               -0.779 |             0.217 |                 100 |              0.459 |                0.459 |             0.093 |

        | model             | agreement_rate   |
        |-------------------|------------------|
        | Qwen_Qwen2.5-1.5B | 86.0%            |

        | model             | pattern                |   count |
        |-------------------|------------------------|---------|
        | Qwen_Qwen2.5-1.5B | surface_EN__surface_ZH |      86 |
        | Qwen_Qwen2.5-1.5B | inverse_EN__surface_ZH |      14 |

        | Language   |   n |   Statistic |   p-value |   Mean Δ |   Median Δ | Preference   |
        |------------|-----|-------------|-----------|----------|------------|--------------|
        | en         | 100 |         184 |  4.17e-16 |  -0.4127 |    -0.4358 | surface      |
        | zh         | 100 |           0 |  1.95e-18 |  -0.8003 |    -0.7793 | surface      |

    ??? example "Qwen/qwen2.5-1.5b ue scope"
        | model             | language   |   inverse |   surface |   total | p_inverse   |
        |-------------------|------------|-----------|-----------|---------|-------------|
        | Qwen_Qwen2.5-1.5B | en         |         9 |        91 |     100 | 9.0%        |
        | Qwen_Qwen2.5-1.5B | zh         |        22 |        78 |     100 | 22.0%       |

        | model             | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |-------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | Qwen_Qwen2.5-1.5B | en         |                 100 |             -0.285 |               -0.282 |             0.218 |                 100 |              0.769 |                0.754 |             0.166 |
        | Qwen_Qwen2.5-1.5B | zh         |                 100 |             -0.432 |               -0.407 |             0.452 |                 100 |              0.714 |                0.665 |             0.298 |

        | model             | agreement_rate   |
        |-------------------|------------------|
        | Qwen_Qwen2.5-1.5B | 73.0%            |

        | model             | pattern                |   count |
        |-------------------|------------------------|---------|
        | Qwen_Qwen2.5-1.5B | surface_EN__surface_ZH |      71 |
        | Qwen_Qwen2.5-1.5B | surface_EN__inverse_ZH |      20 |
        | Qwen_Qwen2.5-1.5B | inverse_EN__surface_ZH |       7 |
        | Qwen_Qwen2.5-1.5B | inverse_EN__inverse_ZH |       2 |

        | Language   |   n |   Statistic |   p-value |   Mean Δ |   Median Δ | Preference   |
        |------------|-----|-------------|-----------|----------|------------|--------------|
        | en         | 100 |         140 |  1.2e-16  |  -0.2854 |    -0.2825 | surface      |
        | zh         | 100 |         486 |  1.19e-12 |  -0.4319 |    -0.4074 | surface      |

    ??? example "Qwen/qwen2.5-3b eu scope"
        | mode`l           | language   |   inverse |   surface |   total | p_inverse   |
        |-----------------|------------|-----------|-----------|---------|-------------|
        | Qwen_Qwen2.5-3B | en         |        16 |        84 |     100 | 16.0%       |
        | Qwen_Qwen2.5-3B | zh         |         0 |       100 |     100 | 0.0%        |

        | model           | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |-----------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | Qwen_Qwen2.5-3B | en         |                 100 |             -0.394 |               -0.404 |             0.365 |                 100 |              0.72  |                0.668 |             0.27  |
        | Qwen_Qwen2.5-3B | zh         |                 100 |             -0.831 |               -0.861 |             0.247 |                 100 |              0.449 |                0.423 |             0.119 |

        | model           | agreement_rate   |
        |-----------------|------------------|
        | Qwen_Qwen2.5-3B | 84.0%            |

        | model           | pattern                |   count |
        |-----------------|------------------------|---------|
        | Qwen_Qwen2.5-3B | surface_EN__surface_ZH |      84 |
        | Qwen_Qwen2.5-3B | inverse_EN__surface_ZH |      16 |

        | Language   |   n |   Statistic |   p-value |   Mean Δ |   Median Δ | Preference   |
        |------------|-----|-------------|-----------|----------|------------|--------------|
        | en         | 100 |         318 |  1.62e-14 |  -0.3936 |    -0.4037 | surface      |
        | zh     `    | 100 |           0 |  1.95e-18 |  -0.8312 |    -0.8612 | surface      |

    ??? example "Qwen/qwen2.5-3b ue scope"
        | model           | language   |   inverse |   surface |   total | p_inverse   |
        |-----------------|------------|-----------|-----------|---------|-------------|
        | Qwen_Qwen2.5-3B | en         |         6 |        94 |     100 | 6.0%        |
        | Qwen_Qwen2.5-3B | zh         |        27 |        73 |     100 | 27.0%       |

        | model           | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |-----------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | Qwen_Qwen2.5-3B | en         |                 100 |             -0.49  |               -0.475 |             0.309 |                 100 |              0.641 |                0.622 |             0.194 |
        | Qwen_Qwen2.5-3B | zh         |                 100 |             -0.167 |               -0.18  |             0.28  |                 100 |              0.878 |                0.836 |             0.241 |

        | model           | agreement_rate   |
        |-----------------|------------------|
        | Qwen_Qwen2.5-3B | 77.0%            |

        | model           | pattern                |   count |
        |-----------------|------------------------|---------|
        | Qwen_Qwen2.5-3B | surface_EN__surface_ZH |      72 |
        | Qwen_Qwen2.5-3B | surface_EN__inverse_ZH |      22 |
        | Qwen_Qwen2.5-3B | inverse_EN__inverse_ZH |       5 |
        | Qwen_Qwen2.5-3B | inverse_EN__surface_ZH |       1 |

        | Language   |   n |   Statistic |   p-value |   Mean Δ |   Median Δ | Preference   |
        |------------|-----|-------------|-----------|----------|------------|--------------|
        | en         | 100 |          46 |  7.73e-18 |  -0.4904 |    -0.4748 | surface      |
        | zh         | 100 |        1008 |  9.14e-08 |  -0.1675 |    -0.1796 | surface      |

    ??? example "Qwen/qwen2.5-7b eu scope"
        | model           | language   |   inverse |   surface |   total | p_inverse   |
        |-----------------|------------|-----------|-----------|---------|-------------|
        | Qwen_Qwen2.5-7B | en         |        10 |        90 |     100 | 10.0%       |
        | Qwen_Qwen2.5-7B | zh         |         4 |        96 |     100 | 4.0%        |

        | model           | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |-----------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | Qwen_Qwen2.5-7B | en         |                 100 |             -0.64  |               -0.688 |             0.363 |                 100 |              0.565 |                0.502 |             0.224 |
        | Qwen_Qwen2.5-7B | zh         |                 100 |             -0.525 |               -0.542 |             0.28  |                 100 |              0.614 |                0.581 |             0.172 |

        | model           | agreement_rate   |
        |-----------------|------------------|
        | Qwen_Qwen2.5-7B | 86.0%            |

        | model           | pattern                |   count |
        |-----------------|------------------------|---------|
        | Qwen_Qwen2.5-7B | surface_EN__surface_ZH |      86 |
        | Qwen_Qwen2.5-7B | inverse_EN__surface_ZH |      10 |
        | Qwen_Qwen2.5-7B | surface_EN__inverse_ZH |       4 |

        | Language   |   n |   Statistic |   p-value |   Mean Δ |   Median Δ | Preference   |
        |------------|-----|-------------|-----------|----------|------------|--------------|
        | en         | 100 |          68 |  1.48e-17 |  -0.64   |    -0.6885 | surface      |
        | zh         | 100 |          14 |  2.97e-18 |  -0.5253 |    -0.5424 | surface      |

    ??? example "Qwen/qwen2.5-7b ue scope"
        | model           | language   |   inverse |   surface |   total | p_inverse   |
        |-----------------|------------|-----------|-----------|---------|-------------|
        | Qwen_Qwen2.5-7B | en         |         3 |        97 |     100 | 3.0%        |
        | Qwen_Qwen2.5-7B | zh         |        24 |        76 |     100 | 24.0%       |

        | model           | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |-----------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | Qwen_Qwen2.5-7B | en         |                 100 |             -0.432 |               -0.424 |             0.265 |                 100 |              0.672 |                0.655 |             0.179 |
        | Qwen_Qwen2.5-7B | zh         |                 100 |             -0.259 |               -0.271 |             0.396 |                 100 |              0.836 |                0.763 |             0.355 |

        | model           | agreement_rate   |
        |-----------------|------------------|
        | Qwen_Qwen2.5-7B | 77.0%            |

        | model           | pattern                |   count |
        |-----------------|------------------------|---------|
        | Qwen_Qwen2.5-7B | surface_EN__surface_ZH |      75 |
        | Qwen_Qwen2.5-7B | surface_EN__inverse_ZH |      22 |
        | Qwen_Qwen2.5-7B | inverse_EN__inverse_ZH |       2 |
        | Qwen_Qwen2.5-7B | inverse_EN__surface_ZH |       1 |

        | Language   |   n |   Statistic |   p-value |   Mean Δ |   Median Δ | Preference   |
        |------------|-----|-------------|-----------|----------|------------|--------------|
        | en         | 100 |          54 |  9.8e-18  |  -0.4321 |    -0.4238 | surface      |
        | zh         | 100 |         911 |  1.43e-08 |  -0.2593 |    -0.2711 | surface      |

    ??? example "Qwen/qwen2.5-14b eu scope"
        | model            | language   |   inverse |   surface |   total | p_inverse   |
        |------------------|------------|-----------|-----------|---------|-------------|
        | Qwen_Qwen2.5-14B | en         |         7 |        93 |     100 | 7.0%        |
        | Qwen_Qwen2.5-14B | zh         |         0 |       100 |     100 | 0.0%        |

        | model            | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | Qwen_Qwen2.5-14B | en         |                 100 |             -0.485 |               -0.519 |             0.291 |                 100 |              0.643 |                0.595 |             0.196 |
        | Qwen_Qwen2.5-14B | zh         |                 100 |             -0.962 |               -0.892 |             0.31  |                 100 |              0.4   |                0.41  |             0.116 |

        | model            | agreement_rate   |
        |------------------|------------------|
        | Qwen_Qwen2.5-14B | 93.0%            |

        | model            | pattern                |   count |
        |------------------|------------------------|---------|
        | Qwen_Qwen2.5-14B | surface_EN__surface_ZH |      93 |
        | Qwen_Qwen2.5-14B | inverse_EN__surface_ZH |       7 |

        | Language   |   n |   Statistic |   p-value |   Mean Δ |   Median Δ | Preference   |
        |------------|-----|-------------|-----------|----------|------------|--------------|
        | en         | 100 |          61 |  1.21e-17 |  -0.485  |    -0.5192 | surface      |
        | zh         | 100 |           0 |  1.95e-18 |  -0.9621 |    -0.8918 | surface      |

    ??? example "Qwen/qwen2.5-14b ue scope"
        | model            | language   |   inverse |   surface |   total | p_inverse   |
        |------------------|------------|-----------|-----------|---------|-------------|
        | Qwen_Qwen2.5-14B | en         |         7 |        93 |     100 | 7.0%        |
        | Qwen_Qwen2.5-14B | zh         |         7 |        93 |     100 | 7.0%        |

        | model            | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | Qwen_Qwen2.5-14B | en         |                 100 |             -0.274 |               -0.258 |             0.205 |                 100 |              0.776 |                0.772 |             0.162 |
        | Qwen_Qwen2.5-14B | zh         |                 100 |             -0.694 |               -0.835 |             0.456 |                 100 |              0.554 |                0.434 |             0.259 |

        | model            | agreement_rate   |
        |------------------|------------------|
        | Qwen_Qwen2.5-14B | 92.0%            |

        | model            | pattern                |   count |
        |------------------|------------------------|---------|
        | Qwen_Qwen2.5-14B | surface_EN__surface_ZH |      89 |
        | Qwen_Qwen2.5-14B | inverse_EN__surface_ZH |       4 |
        | Qwen_Qwen2.5-14B | surface_EN__inverse_ZH |       4 |
        | Qwen_Qwen2.5-14B | inverse_EN__inverse_ZH |       3 |

        | Language   |   n |   Statistic |   p-value |   Mean Δ |   Median Δ | Preference   |
        |------------|-----|-------------|-----------|----------|------------|--------------|
        | en         | 100 |         131 |  9.26e-17 |  -0.2742 |    -0.2584 | surface      |
        | zh         | 100 |          61 |  1.21e-17 |  -0.6941 |    -0.8354 | surface      |

    ??? example "Qwen/qwen2.5-32b eu scope"
        | model            | language   |   inverse |   surface |   total | p_inverse   |
        |------------------|------------|-----------|-----------|---------|-------------|
        | Qwen_Qwen2.5-32B | en         |        10 |        90 |     100 | 10.0%       |
        | Qwen_Qwen2.5-32B | zh         |         0 |       100 |     100 | 0.0%        |

        | model            | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | Qwen_Qwen2.5-32B | en         |                 100 |             -0.415 |               -0.401 |             0.329 |                 100 |              0.696 |                0.67  |             0.231 |
        | Qwen_Qwen2.5-32B | zh         |                 100 |             -1.192 |               -1.077 |             0.463 |                 100 |              0.334 |                0.341 |             0.135 |

        | model            | agreement_rate   |
        |------------------|------------------|
        | Qwen_Qwen2.5-32B | 90.0%            |

        | model            | pattern                |   count |
        |------------------|------------------------|---------|
        | Qwen_Qwen2.5-32B | surface_EN__surface_ZH |      90 |
        | Qwen_Qwen2.5-32B | inverse_EN__surface_ZH |      10 |

        | Language   |   n |   Statistic |   p-value |   Mean Δ |   Median Δ | Preference   |
        |------------|-----|-------------|-----------|----------|------------|--------------|
        | en         | 100 |         172 |  2.97e-16 |  -0.4151 |    -0.4008 | surface      |
        | zh         | 100 |           0 |  1.95e-18 |  -1.1921 |    -1.0773 | surface      |

    ??? example "Qwen/qwen2.5-32b ue scope"
        | model            | language   |   inverse |   surface |   total | p_inverse   |
        |------------------|------------|-----------|-----------|---------|-------------|
        | Qwen_Qwen2.5-32B | en         |         2 |        98 |     100 | 2.0%        |
        | Qwen_Qwen2.5-32B | zh         |        26 |        74 |     100 | 26.0%       |

        | model            | language   |   delta_mean__count |   delta_mean__mean |   delta_mean__median |   delta_mean__std |   ratio_mean__count |   ratio_mean__mean |   ratio_mean__median |   ratio_mean__std |
        |------------------|------------|---------------------|--------------------|----------------------|-------------------|---------------------|--------------------|----------------------|-------------------|
        | Qwen_Qwen2.5-32B | en         |                 100 |             -0.349 |               -0.295 |             0.199 |                 100 |              0.719 |                0.744 |             0.132 |
        | Qwen_Qwen2.5-32B | zh         |                 100 |             -0.65  |               -0.815 |             0.672 |                 100 |              0.652 |                0.443 |             0.438 |

        | model            | agreement_rate   |
        |------------------|------------------|
        | Qwen_Qwen2.5-32B | 74.0%            |

        | model            | pattern                |   count |
        |------------------|------------------------|---------|
        | Qwen_Qwen2.5-32B | surface_EN__surface_ZH |      73 |
        | Qwen_Qwen2.5-32B | surface_EN__inverse_ZH |      25 |
        | Qwen_Qwen2.5-32B | inverse_EN__inverse_ZH |       1 |
        | Qwen_Qwen2.5-32B | inverse_EN__surface_ZH |       1 |

        | Language   |   n |   Statistic |   p-value |   Mean Δ |   Median Δ | Preference   |
        |------------|-----|-------------|-----------|----------|------------|--------------|
        | en         | 100 |           5 |  2.27e-18 |  -0.349  |    -0.2952 | surface      |
        | zh         | 100 |         539 |  4.29e-12 |  -0.6502 |    -0.8149 | surface      |