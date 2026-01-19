# Doing The Thing: From Mechanical Turk to Mech Interp
<sub> For motivation/momentum purposes: Started AI Safety journey **{{ dlog_num_days() }} days ago** with current daily streak of "doing the thing" for **{{ dlog_consecutive_days() }} consecutive days**! </sub>

???+ tip "Updates"
    1. In the UK working from [LISA](https://www.safeai.org.uk/) from **Monday, January 19th to Friday, January 23rd**; worked from Trajan House, Oxford Monday, January 5th to Friday, January 16th
        
        a. Looking to meet more AI Safety folks! Please let me know if you would like to meet up; happy to travel to meet!
    
    2. Reflections from [ARBOx](./notes/reflections/ARBOx.md) including work in progress Theory of Change

    3. Currently working on an Automated Interpretability Project extending [Neo et al. 2024 Interpreting Context Look Ups](./notes/papers/interpretability/Neo_et_al_2024.md). I'm also interested in:
        
        a. Automated Circuit Tracing (e.g. developing Agents to automatically find circuits in Neuronpedia like attribution graphs)

        b. Cross-lingual alignment (e.g. Does aligning a model in English mean it's aligned in Chinese?)

??? note "TLDR of Things I've Done Since Starting My AI Safety Journey {{ dlog_num_days() }} Days Ago"
    **Projects**

    1. Cross-Linguistic Alignment: Does LoRA Fine Tuning a model on a task (e.g. respond in all CAPS) translate cross-linguistically? ([Summary](https://docs.google.com/presentation/d/16jQDJhF4orOrSzpVKMJwtU52u-vlzT77pCK8gZGwO9g/edit?usp=sharing) && [Github](https://github.com/leungchristopher/arbox_project))

    2. Reproducing [Neo et al. 2024 Interpreting Context Look Ups](./notes/papers/interpretability/Neo_et_al_2024.md)

    3. [Multilingual Semantics Probe](./notes/projects/MultilingualSemanticsProbe.md): Looking for Steering Vectors for semantically ambiguous sentences in English but not Mandarin 

    3. Syntactic Dependencies in Transformers: Attention Patterns for Balanced Parentheses (Dyck) Language ([Github](https://github.com/Ky-Ng/Dyck-Interp-Probe))

    **Programs**

    1. [ARBOx](./notes/reflections/ARBOx.md): 2 weeks of compressed ARENA curriculum; project on Cross-linguistic generalization of fine-tuning
    
    2. Attended NeurIPS [Mech Interp 2025 Workshop](https://mechinterpworkshop.com) and found some cool [takeaways](./notes/reflections/meetings/NeurIPS_2025_Mech_Interp_Workshop.md)!

    3. Started being mentored by [Sudhanshu Kasewa](https://www.linkedin.com/in/skasewa/?originalSubdomain=uk) from [80,000 Hours](https://80000hours.org/)

## What is this?
Here it goes! I'm Kyle, a 4th year Computational Linguistics student at USC. This is the start of my AI Safety journey! I am very greatful to be mentored by [Prof. Khalil Iskarous](https://dornsife.usc.edu/profile/khalil-iskarous/) at USC.

I learned about AI Safety from the Seattle Llama4 Hackathon on June 21st, 2025 where I learned of [AI 2027](https://ai-2027.com). After finishing an awesome summer of [engineering](https://en.wikipedia.org/wiki/Annapurna_Labs), I realized the problems which excite me the most lie at the crossroads of engineering and science (computation and linguistics). 

The urgent need for understanding increasingly capable AI models coupled with a burning passion for working at the interdisciplinary intersection of NLP, linguistics, and engineering at scale has **sharpened my goal: to become an AI Safety researcher in Mechanistic Interpretability**.

## Working Backwards :octicons-clock-16:
Sometimes (often) I get analysis paralysis or want to wait for the perfect {time, situation, background, preparation} to start which makes it difficult to get into pursuing my goals (and dreams). So this time around, I know my goal **to become an AI Safety/Mech Interp** researcher! After finding David Quarel's [do the thing](https://davidquarel.github.io/2024/02/04/Do-the-thing.html#fn:audience) I decided that this site is a place where I will keep myself acountable for *doing the thing*.

- Doing --> Working through math problems, reading papers, writing down lists of possible intersections of linguistics and NLP
- The Thing --> Any of the above for at least 1 hour every day, with consistent (though not perfect) progress.

![Mechanical Turk to Mech Interp](./assets/homepage/Mechanical_Turk_to_Mech_Interp.png)

## What is **dlog**? 
I am starting this daily log or *dlog* where every day I will document my progress. I hope that the daily act of documenting will make me more resilient and help prove to myself how badly I want to be an interpetability researcher. With the help of AI, a macro pulls the daily logs into the summary you see below:

??? example "Extra Stuff (click me)" 

    ### Wait What's Computational Linguistics?

    As a Computational Linguistics student, I see Computational Linguistics as three parts:

    1. Linguistics = study of *human* language processing / cognition
    2. Mechanistic Interpretability = study of *LLM* language processing / cognition
    3. Computational Linguistics = Interdisciplinary approach to studying LLM language processing 

    The Computational Linguistics topics that pull me at 9.8 m / s^2 are concepts like Information Theory and Probabilistic Phonology in addition to Theoretical Machine Learning and NLP.

    ### What does AI Safety and Mechanistic Interpretability mean to me?
    - I hope that having deep knowledge in both the fields of linguistics and ML/NLP can help me build a more holistic understanding of LLM cognition and language processing. 
    - I see Mechanistic Interpretability as a sort of [psycholinguistics](https://www.britannica.com/science/psycholinguistics) (the study of real-time processing of language) for LLMs. 
    - Furthermore, I see Mechanistic Interpretability as a foundational basis for understanding AI systems. Perhaps understanding models (such as like biological organisms) can support the other branches of AI Safety (alignment, control, governance, and more).

## **dlog**: {{ dlog_num_days() }} Days and Counting
**Total time focused so far:** {{ dlog_total_time()["hrs"] }} hrs {{ dlog_total_time()["mins"] }} mins throughout {{ dlog_num_days() }} days of learning

Below are the latest updates (auto-generated). 

### Latest entries
{{ dlog_cards(limit=10) }}

## Todo List
The todo list started getting too beefy and has been moved to its own [todo page](notes/Todo.md)!
