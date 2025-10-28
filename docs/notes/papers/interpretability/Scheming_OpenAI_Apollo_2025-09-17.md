# Detecting and reducing scheming in AI models

Authors: OpenAI and Apollo Research

Publication Date: 2025-09-17

Full Paper: [Stress Testing Deliberative Alignment for Anti-Scheming Training](https://www.arxiv.org/abs/2509.15541)

Open AI Blog: [Detecting and reducing scheming in AI models](https://openai.com/index/detecting-and-reducing-scheming-in-ai-models/)

## Key Terms
- Scheming
     - "Pretending to be aligned while pursuing some other agenda"
     - From output alone is not possible to see if models are deliberately being unfaithful
     - My definition: Scheming is when an AI model deliberately responds (either correctly or incorrectly) to a user to pursue an unaligned goal.
- Covert Actions
    - "deliberate witholding or distortion of task-relevant information"
    - Proxy for scheming 
    - When a model makes its answers sound more aligned to a human goal by purposefully distorting its answers
- Sandbagging
    - When a model purposefully underperforms to achieve an unaligned task
- Deception
    - Model pretends to have completed a task without actually doing so 
        - e.g. iconic example of reading a large CSV file and responding before processing
- Situational Awareness
    - Models become aware that their responses are being evaluated; dangerous for different evaluation- vs. deploy-time behavior
- Deliberative Alignment
    - Teaching models to read and reason through "anti-scheming" rules before answering (think of an advanced [Isaac Asimov's "Big Three" rules of robotics](https://webhome.auburn.edu/~vestmon/robotics.html))

## Main Takeaways
1. Reading Chain-of-thought (COT) reasoning transparency in English allows for researchers to peer into model scheming; this would fail if COT is opaque or unfaithful (see [Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety](https://arxiv.org/abs/2507.11473))

2. Model Anti-scheming training (through deliberate alignment) can result in either (1) model faithfullness or (2) models become better at scheming/hide misalignment
    - Deliberate alignment's anti-scheming specification is meant to train model faithfullness (1) not hiding misalignment (2); supported by OpenAI model citing spec in COT reasoning

3. Situational Awareness may allow for modesl to score well during alignment benchmarks but execute misaligned behavior in production; current impact is limited because models do not have opportunity to cause harm

??? question "Always Inserting Situational Awareness Prompt"
    - Interesting finding: Removing Situational Awareness COT increases likelihood for scheming; Adding Situational Awareness COT decreases likelihood for scheming
    - Wonder if it's possible to always just add Situational Awareness in COT for production deployment;(1)perhaps models would start outsmarting this method and (2) does not solve the fundamental alignment problem
## Reflections
### Guiding Questions from ARENA
1. Why might model scheming be a particularly worrying concern as models get more capable? 
2. Why does situational awareness (or evaluation awareness) pose a problem for the field of model evaluations? How might we try to mitigate this?
3. In your view, does this paper provide evidence for or against the following suggestion: "Deliberative alignment will effectively work to address our concerns about scheming in LLMs." Justify your opinion.
