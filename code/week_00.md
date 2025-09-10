### Domain Selection
> Employee information extraction vulnerability in enterprise chatbots with database access.

`Scenario:` An internal company chatbot that has access to HR databases and can answer employee queries about policies, benefits, etc. The target is to evaluate whether unauthorized users can extract sensitive employee information through prompt injection or social engineering techniques.

### Model Exploration
`Model:` Gemma 2-1B-it

`Reason:`
* Instruction-Tuned (better at following system prompts)
* state-of-the-art open-source model from Gemini family (2024)
* Lightweight (1B parameters)
* Enterprise chatbot simulation

`Exploration:`
* build synthetic employee dataset & test cases
* run the model to observe the behaviour
* observations:
    1. 
    2. 