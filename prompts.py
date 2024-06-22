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
  label: model-free_methods
  description: Approaches in reinforcement learning that do not require a model of the environment, focusing on learning policies directly from interactions with the environment.
  terms: [q-learning, sarsa, temporal_difference, policy_gradient, monte_carlo_methods, actor-critic, deep_q-network, experience_replay, target_networks, off-policy_learning]

example_child_topic_2:
  label: model-based_methods
  description: Techniques that use a model of the environment to simulate outcomes and make decisions, often leading to more sample-efficient learning.
  terms: [planning, model_predictive_control, dyna-q, bayesian_networks, transition_models, world_models, imagination-augmented_agents, value_iteration, policy_iteration, simulation]

example_child_topic_3:
  label: hierarchical_reinforcement_learning
  description: Methods that involve decomposing tasks into hierarchies of sub-tasks, enabling more scalable and efficient learning for complex problems.
  terms: [options_framework, maxq, feudal_reinforcement_learning, subgoal_discovery, temporal_abstraction, skills, task_decomposition, hierarchical_policies, intrinsic_motivation, multi-level_learning]

example_child_topic_4:
  label: multi-agent_reinforcement_learning
  description: Strategies for reinforcement learning where multiple agents interact within the same environment, learning to collaborate or compete.
  terms: [cooperative_learning, competitive_learning, nash_equilibrium, joint_policy_learning, decentralized_control, communication_protocols, agent_modeling, self-play, multi-agent_coordination, teamwork]

example_child_topic_5:
  label: meta-reinforcement_learning
  description: Techniques aimed at creating agents that can learn how to learn, adapting more quickly to new tasks by leveraging prior experience.
  terms: [learning_to_learn, fast_adaptation, meta-policy, few-shot_learning, transfer_learning, contextual_policies, task_agnostic_learning, recurrent_neural_networks, gradient-based_meta-learning, hyperparameter_optimization]
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