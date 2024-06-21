def depth_expansion_init_prompt(global_taxo=None, k=5):
    init_prompt = f"""You are an assistant that is helping to build a topical taxonomy by identifying {k} children topics given a "parent" topic (tag: "parent"). Your outputted set of child topics should all be (1) specifically relevant to the "parent" topic, and (2) distinct (minimal potential overlap between the sibling topics) and at the same level of depth/complexity."""

    if global_taxo is not None:
        init_prompt += f"""\n\nYou should also ensure that your outputted children are distinct from the other nodes already present within the existing taxonomy (tag: "existing_taxonomy:"):\n\nexisting_taxonomy:\n{global_taxo}"""

    init_prompt += f"""
We provide an example of a given example_parent and the expected taxonomy for example_parent in YAML format below:

Example Input:
---
example_parent: Methodology for Reinforcement Learning
---

Example Output:
---
example_child_topic_1: 
  label: Model-Free Methods
  description: Approaches in reinforcement learning that do not require a model of the environment, focusing on learning policies directly from interactions with the environment.
  terms: [Q-learning, SARSA, Temporal Difference, Policy Gradient, Monte Carlo Methods, Actor-Critic, Deep Q-Network, Experience Replay, Target Networks, Off-Policy Learning]

example_child_topic_2:
  label: Model-Based Methods
  description: Techniques that use a model of the environment to simulate outcomes and make decisions, often leading to more sample-efficient learning.
  terms: [Planning, Model Predictive Control, Dyna-Q, Bayesian Networks, Transition Models, World Models, Imagination-Augmented Agents, Value Iteration, Policy Iteration, Simulation]

example_child_topic_3:
  label: Hierarchical Reinforcement Learning
  description: Methods that involve decomposing tasks into hierarchies of sub-tasks, enabling more scalable and efficient learning for complex problems.
  terms: [Options Framework, MAXQ, Feudal Reinforcement Learning, Subgoal Discovery, Temporal Abstraction, Skills, Task Decomposition, Hierarchical Policies, Intrinsic Motivation, Multi-Level Learning]

example_child_topic_4:
  label: Multi-Agent Reinforcement Learning
  description: Strategies for reinforcement learning where multiple agents interact within the same environment, learning to collaborate or compete.
  terms: [Cooperative Learning, Competitive Learning, Nash Equilibrium, Joint Policy Learning, Decentralized Control, Communication Protocols, Agent Modeling, Self-Play, Multi-Agent Coordination, Teamwork]

example_child_topic_5:
  label: Meta-Reinforcement Learning
  description: Techniques aimed at creating agents that can learn how to learn, adapting more quickly to new tasks by leveraging prior experience.
  terms: [Learning to Learn, Fast Adaptation, Meta-Policy, Few-Shot Learning, Transfer Learning, Contextual Policies, Task Agnostic Learning, Recurrent Neural Networks, Gradient-Based Meta-Learning, Hyperparameter Optimization]
---

"""
    return init_prompt


depth_expansion_prompt = lambda parent: f"""For the parent topic below (tag: "parent"), output your answer following the same logic used for the example provided above:
parent: {parent}

Output your answer in the following YAML format:

---
child_topic_1: <(label (one line), description (one line), list of 10 terms associated with topic (one line)) of child_topic_1; child_topic_1 is under "parent">
child_topic_2: <(label (one line), description (one line), list of 10 terms associated with topic (one line)) of child_topic_2; child_topic_2 is under "parent">
...
child_topic_5: <(label (one line), description (one line), list of 10 terms associated with topic (one line)) of child_topic_5; child_topic_5 is under "parent">
---
"""