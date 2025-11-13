## Distilling a Degradation Model for MoE 

**Rationale:**

Mixture-of-Experts (MoE) is a sparsification strategy applied to the MLP layers within the LLaMA architecture. By introducing a token dispatcher, different tokens can be routed to different MLP parameter sets (experts), thereby enabling more diverse and specialized representations.

However, this diversity of experts comes at the cost of increased memory consumption. To scale up the number of experts, it is common to distribute them across multiple GPUs, i.e., to use expert parallelism. This, in turn, introduces substantial communication overhead.

During online inference, this architecture exhibits several issues. Even though training typically incorporates an expert load-balancing loss, the token-to-expert assignments at inference time can still be highly imbalanced. This leads to uneven utilization of memory and compute across GPUs, creating both resource bottlenecks and performance bottlenecks.

To address this, prior work has proposed Speculative Dispatch, which constructs a probabilistic model by analyzing both within-layer token–expert affinity and inter-layer expert–expert affinity. Specialized collective communication operators are then designed for these speculative routing paths. When performing collective communication, tokens can be pre-dispatched speculatively, allowing overlap with the original gating computation and, with some probability, reducing the actual communication volume required after the final gating decisions.

This study aims to explore a more aggressive approach: when expert load becomes high, we propose to introduce a compact speculative model that directly approximates the end-to-end effect of dispatch plus expert computation. This smaller model can be obtained via **knowledge distillation** from the MoE structure or through joint training. The goal is to achieve comparable generalization performance and end-to-end inference quality, while alleviating the load and communication bottlenecks in large MoE systems.

