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