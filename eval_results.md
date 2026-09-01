# Evaluation Results

## Q1: How does self-attention work?

**Answer:**

Self-attention works by projecting corpus-level co-occurrence statistics into sequence-specific contexts via a selector matrix Q [Source 1, Page: 16]. This projection captures how words interact in context, bridging the gap between classical distributional approaches (such as GloVe) and modern neural architectures [Source 3, Page: 1]. The query-key-value mechanism in self-attention arises as the natural asymmetric extension for modeling directional relationships, while positional encodings and multi-head attention follow as structured refinements of this projection principle [Source 1, Page: 16].


**References:**

```
[Source 1] Self-Attention as Distributional Projection: A Unified Interpretation of Transformer Architecture — Nihal Mehta (p.16, arXiv:2511.13780v1)
[Source 2] Self-Attention as Distributional Projection: A Unified Interpretation of Transformer Architecture — Nihal Mehta (p.15, arXiv:2511.13780v1)
[Source 3] Self-Attention as Distributional Projection: A Unified Interpretation of Transformer Architecture — Nihal Mehta (p.1, arXiv:2511.13780v1)
[Source 4] Attention Guided CAM: Visual Explanations of Vision Transformer Guided by Self-Attention — Saebom Leem, Hyunseok Seo (p.2, arXiv:2402.04563v1)
[Source 5] On the Surprising Effectiveness of Attention Transfer for Vision Transformers — Alexander C. Li, Yuandong Tian, Beidi Chen, Deepak Pathak, Xinlei Chen (p.5, arXiv:2411.09702v1)
```


---

## Q2: What is multi-head attention?

**Answer:**

Multi-head attention is a method used in Transformers to capture different types of contextual relations in language. It corresponds to factorizing the self-attention matrix S into multiple low-rank relational components S(1), ..., S(h), each specializing in a different pattern, such as syntactic dependency, coreference, topical cohesion, discourse linkage, etc. [Source 2, Page: 14]

In multi-head attention, each head i computes the same exact attention operation, but with different parameters for the query, key, and value matrices. The outputs of each head are then concatenated and linearly recombined to produce the final output [Source 2, Page: 14].


**References:**

```
[Source 1] On the Surprising Effectiveness of Attention Transfer for Vision Transformers — Alexander C. Li, Yuandong Tian, Beidi Chen, Deepak Pathak, Xinlei Chen (p.17, arXiv:2411.09702v1)
[Source 2] Self-Attention as Distributional Projection: A Unified Interpretation of Transformer Architecture — Nihal Mehta (p.14, arXiv:2511.13780v1)
[Source 3] On the Surprising Effectiveness of Attention Transfer for Vision Transformers — Alexander C. Li, Yuandong Tian, Beidi Chen, Deepak Pathak, Xinlei Chen (p.16, arXiv:2411.09702v1)
[Source 4] Attention Guided CAM: Visual Explanations of Vision Transformer Guided by Self-Attention — Saebom Leem, Hyunseok Seo (p.4, arXiv:2402.04563v1)
[Source 5] Efficient Mixed Transformer for Single Image Super-Resolution — Ling Zheng, Jinchen Zhu, Jinpeng Shi, Shizhuang Weng (p.4, arXiv:2305.11403v5)
```


---

## Q3: How do vision transformers use attention mechanisms?

**Answer:**

Vision transformers use attention mechanisms to project global distributional structure into a sequence-specific, directional, and order-aware subspace [Source 4, Page 16]. This is done through the self-attention mechanism, which refines the same core operation by projecting global distributional structure into a sequence-specific subspace at each step [Source 4, Page 16].

The attention mechanism in vision transformers can be understood as a natural consequence of projecting corpus-level associations into sequence-specific contexts with directional asymmetry [Source 4, Page 16]. This is achieved through the query-key-value structure, which is a key component of the self-attention mechanism [Source 4, Page 16].

In vision transformers, the attention mechanism is used to model long-range inter-dependencies between tokens and to capture rich context [Source 5, Page 2]. However, traditional vision transformers impose a constraint on the attention distance due to the fixed window size [Source 5, Page 2]. To recover the transformer's ability to model long-term dependency, techniques such as deformable sampling have been explored [Source 5, Page 2].


**References:**

```
[Source 1] On the Surprising Effectiveness of Attention Transfer for Vision Transformers — Alexander C. Li, Yuandong Tian, Beidi Chen, Deepak Pathak, Xinlei Chen (p.1, arXiv:2411.09702v1)
[Source 2] On the Surprising Effectiveness of Attention Transfer for Vision Transformers — Alexander C. Li, Yuandong Tian, Beidi Chen, Deepak Pathak, Xinlei Chen (p.9, arXiv:2411.09702v1)
[Source 3] Dilated Neighborhood Attention Transformer — Ali Hassani, Humphrey Shi (p.2, arXiv:2209.15001v3)
[Source 4] Self-Attention as Distributional Projection: A Unified Interpretation of Transformer Architecture — Nihal Mehta (p.16, arXiv:2511.13780v1)
[Source 5] Vision Transformer with Quadrangle Attention — Qiming Zhang, Jing Zhang, Yufei Xu, Dacheng Tao (p.2, arXiv:2303.15105v1)
```


---

## Q4: What is the computational complexity of self-attention?

**Answer:**

The computational complexity of self-attention is O(n^2d) and O(n^2) for the attention weights, where n is the number of tokens and d is the embedding dimension [Source 1].


**References:**

```
[Source 1] Dilated Neighborhood Attention Transformer — Ali Hassani, Humphrey Shi (p.3, arXiv:2209.15001v3)
[Source 2] Efficient Content-Based Sparse Attention with Routing Transformers — Aurko Roy, Mohammad Saffar, Ashish Vaswani, David Grangier (p.1, arXiv:2003.05997v5)
[Source 3] Vision Transformer with Quadrangle Attention — Qiming Zhang, Jing Zhang, Yufei Xu, Dacheng Tao (p.6, arXiv:2303.15105v1)
[Source 4] Self-Attention as Distributional Projection: A Unified Interpretation of Transformer Architecture — Nihal Mehta (p.15, arXiv:2511.13780v1)
[Source 5] Vision Transformer with Quadrangle Attention — Qiming Zhang, Jing Zhang, Yufei Xu, Dacheng Tao (p.4, arXiv:2303.15105v1)
```


---

## Q5: How does dilated attention differ from standard local attention?

**Answer:**

Dilated attention, as introduced in Dilated Neighborhood Attention Transformer (DiNA), differs from standard local attention in that it allows the receptive field to grow exponentially, rather than linearly [Source 4, p. 3]. This is achieved by dilating neighborhoods in the standard Neighborhood Attention (NA) mechanism, enabling DiNA to capture more global context without additional computational cost [Source 4, p. 3].


**References:**

```
[Source 1] Dilated Neighborhood Attention Transformer — Ali Hassani, Humphrey Shi (p.1, arXiv:2209.15001v3)
[Source 2] Energy-Gated Attention and Wavelet Positional Encoding: Complementary Inductive Biases for Transformer Attention — Athanasios Zeris (p.1, arXiv:2605.26355v1)
[Source 3] Vision Transformer with Quadrangle Attention — Qiming Zhang, Jing Zhang, Yufei Xu, Dacheng Tao (p.13, arXiv:2303.15105v1)
[Source 4] Dilated Neighborhood Attention Transformer — Ali Hassani, Humphrey Shi (p.3, arXiv:2209.15001v3)
[Source 5] Dilated Neighborhood Attention Transformer — Ali Hassani, Humphrey Shi (p.2, arXiv:2209.15001v3)
```


---

## Q6: What is the capital of France?

**Answer:**

Unfortunately, the provided sources do not mention the capital of France.


**References:**

```
[Source 1] Forgetting Transformer: Softmax Attention with a Forget Gate — Zhixuan Lin, Evgenii Nikishin, Xu Owen He, Aaron Courville (p.19, arXiv:2503.02130v2)
[Source 2] Efficient Content-Based Sparse Attention with Routing Transformers — Aurko Roy, Mohammad Saffar, Ashish Vaswani, David Grangier (p.19, arXiv:2003.05997v5)
[Source 3] Efficient Content-Based Sparse Attention with Routing Transformers — Aurko Roy, Mohammad Saffar, Ashish Vaswani, David Grangier (p.15, arXiv:2003.05997v5)
[Source 4] On the Surprising Effectiveness of Attention Transfer for Vision Transformers — Alexander C. Li, Yuandong Tian, Beidi Chen, Deepak Pathak, Xinlei Chen (p.17, arXiv:2411.09702v1)
[Source 5] Dilated Neighborhood Attention Transformer — Ali Hassani, Humphrey Shi (p.17, arXiv:2209.15001v3)
```


---
