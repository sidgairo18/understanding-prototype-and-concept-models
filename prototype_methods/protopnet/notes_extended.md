# ProtoPNet — extended paper notes (Section 2.2 + Supplement S1, S2)

> Companion to [notes.md](notes.md) (the implementation log). This file is the **theory
> deep-dive**: a careful, self-contained walkthrough of the *training algorithm*
> (main paper §2.2) and the two supplementary results that justify it — **S1** (the proof
> that prototype projection is "safe") and **S2** (the probabilistic reading of the whole
> model). The aim is mechanistic understanding, faithful to the paper's notation.
>
> Source PDFs (git-ignored, local):
> `papers/prototype_methods/This Looks Like That- Deep Learning for Interpretable Image Recognition.pdf`
> and `papers/prototype_methods/supp_this_looks_like_that.pdf`.

**Contents**
1. [Notation recap (from §2.1)](#0-notation-recap-from-21)
2. [§2.2 — The training algorithm](#1-22--the-training-algorithm)
3. [S1 — Proof of Theorem 2.1 (projection is safe)](#2-s1--proof-of-theorem-21)
4. [S2 — A probabilistic interpretation of ProtoPNet](#3-s2--a-probabilistic-interpretation-of-protopnet)
5. [How the three pieces fit together](#4-how-the-three-pieces-fit-together)

---

## 0. Notation recap (from §2.1)

You need a handful of objects from §2.1 before §2.2 makes sense. (See also
[README.md](README.md) and the shapes docstring in [model.py](model.py).)

| Symbol | Meaning |
|---|---|
| $f$ | the conv backbone **+ add-on 1×1 convs**; $\mathbf z = f(\mathbf x)$ is a $D\times h\times w$ feature map |
| **patch** $\tilde{\mathbf z}$ | one spatial cell of $\mathbf z$, a $D\text{-vector}$ with a receptive field back in the image |
| $\mathbf p_j$ | the $j\text{-th}$ **prototype**, a $D\text{-vector}$ (same shape as a patch) |
| $\mathbf P_k\subseteq\mathbf P$ | the prototypes **allocated to class $k$**; there are $m_k$ of them (paper: 10/class) |
| $m$ | total prototypes $=\sum_k m_k$ |
| $g_{\mathbf p_j}$ | the **prototype unit**: similarity of the best patch to $\mathbf p_j$ |
| $h$ | the final **fully-connected** layer; $w_h$ its weight matrix |
| $w_h^{(k,j)}$ | weight from prototype unit $j$ to the logit of class $k$ |

The **similarity / activation** a prototype unit produces is (note: *monotonically
decreasing* in the distance):

$$
g_{\mathbf p_j}(\mathbf z) = \max_{\tilde{\mathbf z}\in\text{patches}(\mathbf z)} \log\!\left(\frac{\lVert\tilde{\mathbf z}-\mathbf p_j\rVert_2^2+1}{\lVert\tilde{\mathbf z}-\mathbf p_j\rVert_2^2+\epsilon}\right).
$$

Small distance ⇒ large similarity. The `max` over patches answers "*is this prototypical
part present **anywhere** in the image?*" and also records **where** (which patch attained
the max). Logits are $w_h$ times the vector of these $m$ similarities, then softmax. This
is implemented as `_distance_to_similarity` in [model.py](model.py#L75-L78).

---

## 1. §2.2 — The training algorithm

Training cycles through **three stages**. They can be repeated more than once
(the full schedule is the algorithm chart in supplement §S9.3):

```
   (1) SGD of layers before the last layer   ──┐
   (2) projection ("push") of prototypes        │  repeat 1→2→3
   (3) convex optimization of the last layer  ──┘
```

Each stage trains a *different* slice of the network and optimizes a *different*
objective. Read them as a division of labor:

| Stage | What is trained | What is frozen | Objective |
|---|---|---|---|
| 1 | $w_{\text{conv}}$ (backbone+add-on) **and** prototypes $\mathbf P$ | last layer $h$ | CE + cluster + separation |
| 2 | prototypes $\mathbf P$ (snapped, not gradient) | everything else | nearest-patch projection |
| 3 | last layer $h$ only | $w_{\text{conv}}$, $\mathbf P$ | CE + L1 sparsity |

### Stage 1 — SGD of the layers before the last layer

**Goal.** Carve out a *meaningful latent space*: the most informative patches of an image
should **cluster** (in $L^2$ distance) around prototypes of that image's *true* class, and
those clusters should be **well-separated** from prototypes of *other* classes. We jointly
optimize the conv parameters $w_{\text{conv}}$ and the prototypes $\mathbf P=\{\mathbf p_j\}_{j=1}^m$
by SGD, **holding the last layer $w_h$ fixed**.

With training set $D=\{(\mathbf x_i,y_i)\}_{i=1}^n$, the objective is:

$$
\min_{\mathbf P,\,w_{\text{conv}}} \frac1n\sum_{i=1}^n \mathrm{CrsEnt}\big(h\circ g_{\mathbf p}\circ f(\mathbf x_i),\,y_i\big) + \lambda_1\,\mathrm{Clst} + \lambda_2\,\mathrm{Sep},
$$

where the two structure terms are

$$
\mathrm{Clst}=\frac1n\sum_{i=1}^n \underbrace{\min_{j:\,\mathbf p_j\in\mathbf P_{y_i}}}_{\text{nearest own-class prototype}} \underbrace{\min_{\mathbf z\in\text{patches}(f(\mathbf x_i))}}_{\text{nearest patch to it}} \lVert \mathbf z-\mathbf p_j\rVert_2^2,
$$

$$
\mathrm{Sep}=-\,\frac1n\sum_{i=1}^n \underbrace{\min_{j:\,\mathbf p_j\notin\mathbf P_{y_i}}}_{\text{nearest other-class prototype}} \min_{\mathbf z\in\text{patches}(f(\mathbf x_i))} \lVert \mathbf z-\mathbf p_j\rVert_2^2 .
$$

Reading the two nested `min`s is the whole trick:

- **inner `min`** (over patches) = the distance from a prototype to the *single closest patch*
  in the image — this is exactly `min_distances` returned by `forward` in
  [model.py](model.py#L80-L82);
- **outer `min`** (over prototypes) = pick the *single closest prototype* (own-class for
  Clst, other-class for Sep).

**Cluster cost (Clst).** For each training image, find the nearest (patch, **same-class**
prototype) pair and *pull them together*. Effect: every training image ends up with **at
least one patch that strongly matches some prototype of its own class**. Note it only acts
on the *one* closest own-class prototype per image — which is precisely **why each class
needs several prototypes**: different images (and different parts: head, wing, …) get
captured by different prototypes of the class.

**Separation cost (Sep).** The leading minus sign means minimizing $\mathrm{Sep}$ =
*maximizing* the distance to the nearest **other-class** prototype. Effect: patches of a
$\text{class-}k$ image are pushed **away** from every $\text{non-class-}k$ prototype. (The paper notes
the separation cost is *new to this work* — earlier prototype methods used only a
cluster-style term.)

> Both terms are computed in `cluster_and_separation_costs`
> ([model.py](model.py#L85-L89)) from `min_distances` and the `prototype_class` buffer
> ([model.py](model.py#L54-L59)), which records which class each prototype belongs to.

**Why freeze $h$, and how it is initialized.** In stage 1 the last layer is held at a
hand-set value. For a $\text{class-}k$ logit:

$$
w_h^{(k,j)} = \begin{cases} +1 & \text{if } \mathbf p_j\in\mathbf P_k \quad(\text{own-class connection}) \\ -0.5 & \text{if } \mathbf p_j\notin\mathbf P_k \quad(\text{cross-class connection}) \end{cases}
$$

Intuition: a **positive** own-class connection means "this image looking like a $\text{class-}k$
prototype *raises* its $\text{class-}k$ probability"; a **negative** cross-class connection means
"looking like a $\textit{non-}k$ prototype *lowers* $\text{class-}k$ probability." Fixing $h$ this way
*forces the conv layers to do the work* — if a $\text{class-}k$ image's patch drifts too close to
a $\text{non-}k$ prototype, the frozen negative weight inflates the cross-entropy loss, so SGD has
to fix the latent space instead of papering over it with the classifier.

Note the **separation cost and the negative connection push in the same direction**:
together they make a $\text{class-}k$ prototype represent a concept that is *characteristic of
class $k$ but absent from other classes*. (If a $\text{class-}k$ prototype captured a concept also
present in $\text{non-}k$ images, those images would activate it strongly → larger separation
cost *and* larger cross-entropy.)

### Stage 2 — Projection ("push") of prototypes

**Goal: make prototypes *visualizable*.** A prototype learned by SGD is just a point in
$\mathbb R^D$; nothing forces it to coincide with any real image patch. To be able to *show*
a prototype as an image region, we **snap** each $\mathbf p_j$ (of class $k$) onto the
**nearest latent patch from a training image of the same class $k$**:

$$
\mathbf p_j \leftarrow \arg\min_{\mathbf z\in\mathcal Z_j}\lVert \mathbf z-\mathbf p_j\rVert_2, \qquad \mathcal Z_j=\{\tilde{\mathbf z}:\tilde{\mathbf z}\in\text{patches}(f(\mathbf x_i))\ \forall i\text{ s.t. } y_i=k\}.
$$

After this update **every prototype literally *is* a patch of a real training image**, so it
can be rendered (how to crop the source region is §2.3 / supplement §S7). This is the move
that turns ProtoPNet from "prototype-*regularized*" into genuinely **interpretable**.

> Implemented as `push_prototypes` in [push.py](push.py); this is the POC's signature
> interpretability step.

Two practical notes from the paper:
- Projection has **the same time complexity as a forward pass** of a conv layer followed by
  global average pooling — it introduces no extra training-time cost.
- Snapping changes the prototype vectors, so it could in principle *change predictions*.
  **It (almost) doesn't** — and that guarantee is exactly **Theorem 2.1**, proved in S1
  below. The cluster cost from stage 1 is what keeps prototypes already close to real
  patches, so the "move" during projection is small — which is the precondition the theorem
  needs.

### Stage 3 — Convex optimization of the last layer

**Goal: sparsify the cross-class connections.** With the conv layers and prototypes now
**frozen**, we optimize *only* $w_h$ so that the cross-class weights, initialized at $-0.5$,
are driven toward $0$:

$$
\min_{w_h} \frac1n\sum_{i=1}^n \mathrm{CrsEnt}\big(h\circ g_{\mathbf p}\circ f(\mathbf x_i),\,y_i\big) + \lambda\sum_{k=1}^{K}\sum_{j:\,\mathbf p_j\notin\mathbf P_k}\big|w_h^{(k,j)}\big|.
$$

The L1 penalty applies **only to cross-class weights** (own-class connections are left
near $+1$). Why we want $w_h^{(k,j)}\approx 0$ for $\mathbf p_j\notin\mathbf P_k$: it makes
the model rely **less on negative reasoning** of the form *"this is a $\text{class-}k^{\prime}$ bird
because it contains a patch that is **not** prototypical of class $k$."* We prefer
explanations built from *positive* evidence ("this looks like that $\text{class-}k$ part").

**Why "convex".** Everything feeding $h$ is frozen, so the logits are a *linear* function
of the now-fixed similarities; cross-entropy of a linear model plus an L1 term is a convex
program in $w_h$. This stage improves accuracy **without disturbing the latent space or the
(now visualizable) prototypes**.

> **POC mapping.** Stage 1 ⇒ the loss in [model.py](model.py#L85-L89) + the SGD loop in
> [train.py](train.py). Stage 2 ⇒ [push.py](push.py). Stage 3 ⇒ a last-layer-only optimizer
> pass with L1 on cross-class weights. See the alternating-schedule TODO in
> [notes.md](notes.md).

---

## 2. S1 — Proof of Theorem 2.1

> **In one sentence:** *if* prototypes barely move during projection (assured by the cluster
> cost) *and* the image was already classified correctly with a comfortable margin, *then*
> the projection in Stage 2 does **not** change the prediction. This is the formal license
> for the interpretability move.

### 2.1 New notation for the proof

The proof re-indexes prototypes **by class**: $\mathbf p_l^{k}$ is the $l\text{-th}$ prototype of
class $k$, so $\mathbf P^k=\{\mathbf p_l^{k}\}_{l=1}^{m_k}$. For a fixed input image
$\mathbf x$ with correct label $c$:

| Symbol | Meaning |
|---|---|
| $\mathbf b_l^{k}$ | value of prototype $\mathbf p_l^{k}$ **before** projection |
| $\mathbf a_l^{k}$ | value **after** projection |
| $\mathbf z_l^{k}$ | nearest patch of $f(\mathbf x)$ to $\mathbf b_l^{k}$ (before projection), $=\arg\min_{\tilde{\mathbf z}}\lVert\tilde{\mathbf z}-\mathbf b_l^{k}\rVert_2$ |
| $c$ | correct class of $\mathbf x$ |
| $m^{\prime}$ | number of prototypes per class (assumed equal across classes) |

### 2.2 Assumptions (and what each one *means*)

- **(A1)** $\mathbf z_l^{k}$ is *also* the nearest patch after projection (i.e.
  $\mathbf z_l^{k}=\arg\min_{\tilde{\mathbf z}}\lVert\tilde{\mathbf z}-\mathbf a_l^{k}\rVert_2$).
  *"The patch a prototype points at doesn't change just because we nudged the prototype."*
- **(A2)** there is a $\delta\in(0,1)$ such that:
  - **(A2a, wrong-class prototypes $k\neq c$):**
    $\lVert\mathbf a_l^{k}-\mathbf b_l^{k}\rVert_2\le\theta\,\lVert\mathbf z_l^{k}-\mathbf b_l^{k}\rVert_2-\sqrt\epsilon$,
    with $\theta=\min\!\big(\sqrt{1+\delta}-1,\ 1-\tfrac{1}{\sqrt{2-\delta}}\big)$.
    *"A prototype moves only a little relative to how far it already was from the image's
    nearest patch."* (The `min` makes the single $\theta$ tight enough for **both** factors
    of the bound below.)
  - **(A2b, correct-class prototypes):**
    $\lVert\mathbf a_l^{c}-\mathbf b_l^{c}\rVert_2\le(\sqrt{1+\delta}-1)\lVert\mathbf z_l^{c}-\mathbf b_l^{c}\rVert_2$
    **and** $\lVert\mathbf z_l^{c}-\mathbf b_l^{c}\rVert_2\le\sqrt{1-\delta}$.
    *"Correct-class prototypes are already close to a patch of $\mathbf x$ (so $\mathbf z\approx\mathbf b$) and move little."*
- **(A3)** equal #prototypes per class, $m^{\prime}$.
- **(A4)** the last layer is in its **sparse final state**: $w_h^{(k,j)}=1$ for
  $\mathbf p_j\in\mathbf P_k$ and $w_h^{(k,j)}=0$ for $\mathbf p_j\notin\mathbf P_k$.
  (This is what Stage 3 drives the weights toward, so the theorem describes the trained model.)

### 2.3 The logit under (A4)

Because cross-class weights are $0$, the $\text{class-}k$ logit is just the **sum of its own
prototypes' activations**:

$$
L_k\big(\mathbf x,\{\mathbf p_l^{k}\}\big) = \sum_{l=1}^{m^{\prime}}\log\!\left(\frac{\lVert\mathbf z_l^{k}-\mathbf p_l^{k}\rVert_2^2+1}{\lVert\mathbf z_l^{k}-\mathbf p_l^{k}\rVert_2^2+\epsilon}\right).
$$

Define the **per-prototype multiplicative change** of the logit caused by projection
($\mathbf b\to\mathbf a$):

$$
\Delta_k=L_k(\mathbf x,\{\mathbf a_l^{k}\})-L_k(\mathbf x,\{\mathbf b_l^{k}\})=\sum_{l=1}^{m^{\prime}}\log\Psi_l^{k}, \qquad \Psi_l^{k}=\frac{\lVert\mathbf z_l^{k}-\mathbf a_l^{k}\rVert_2^2+1}{\lVert\mathbf z_l^{k}-\mathbf b_l^{k}\rVert_2^2+1}\cdot\frac{\lVert\mathbf z_l^{k}-\mathbf b_l^{k}\rVert_2^2+\epsilon}{\lVert\mathbf z_l^{k}-\mathbf a_l^{k}\rVert_2^2+\epsilon}.
$$

So the whole proof reduces to **bounding the single ratio $\Psi_l^{k}$** — from *below* for
the correct class (logit can't drop too much) and from *above* for wrong classes (logits
can't rise too much).

### 2.4 Correct class: $\Psi_l^{c}\ge\dfrac{1}{(1+\delta)(2-\delta)}$

Write $z,a,b$ for $\mathbf z_l^{c},\mathbf a_l^{c},\mathbf b_l^{c}$. Bound the two factors.

**Factor 1** — use the second part of (A2b), $\lVert z-b\rVert_2^2\le 1-\delta$:

$$
\frac{\lVert z-a\rVert^2+1}{\lVert z-b\rVert^2+1} \ge \frac{1}{\lVert z-b\rVert^2+1} \ge \frac{1}{2-\delta}. \quad (1)
$$

**Factor 2** — triangle inequality $\lVert z-a\rVert\le\lVert z-b\rVert+\lVert a-b\rVert$, then
the first part of (A2b) $\lVert a-b\rVert\le(\sqrt{1+\delta}-1)\lVert z-b\rVert$ gives
$\lVert z-a\rVert\le\sqrt{1+\delta}\,\lVert z-b\rVert$, i.e. $\lVert z-a\rVert^2\le(1+\delta)\lVert z-b\rVert^2$.
Since also $\epsilon\le(1+\delta)\epsilon$,

$$
\lVert z-a\rVert^2+\epsilon\le(1+\delta)(\lVert z-b\rVert^2+\epsilon) \implies \frac{\lVert z-b\rVert^2+\epsilon}{\lVert z-a\rVert^2+\epsilon}\ge\frac{1}{1+\delta}. \quad (2)
$$

Multiplying (1)·(2): $\Psi_l^{c}\ge\frac{1}{(1+\delta)(2-\delta)}$. Summing the logs over the
$m^{\prime}$ correct-class prototypes:

$$
\Delta_c=\sum_{l}\log\Psi_l^{c}\ge m^{\prime}\log\frac{1}{(1+\delta)(2-\delta)} \quad\Longleftrightarrow\quad -\Delta_c\le \underbrace{m^{\prime}\log\big((1+\delta)(2-\delta)\big)}_{=\,\Delta_{\max}}.
$$

**The correct-class logit can drop by at most $\Delta_{\max}$.**

### 2.5 Wrong class: $\Psi_l^{k}\le(1+\delta)(2-\delta)$

The mirror image, now with $z,a,b=\mathbf z_l^{k},\mathbf a_l^{k},\mathbf b_l^{k}$ for $k\neq c$,
using **(A2a)**.

**Factor 1 $\le 1+\delta$.** Triangle inequality + (A2a) (drop the $-\sqrt\epsilon$ slack)
give $\lVert z-a\rVert^2\le(1+\delta)\lVert z-b\rVert^2$, hence

$$
\frac{\lVert z-a\rVert^2+1}{\lVert z-b\rVert^2+1} \le \frac{(1+\delta)\lVert z-b\rVert^2+1}{\lVert z-b\rVert^2+1} \le \frac{(1+\delta)\lVert z-b\rVert^2+(1+\delta)}{\lVert z-b\rVert^2+1}=1+\delta. \quad (3)
$$

**Factor 2 $\le 2-\delta$.** Reverse triangle inequality $\lVert z-a\rVert\ge\lVert z-b\rVert-\lVert a-b\rVert>0$
(positivity from (A2a)), squared, together with the $\big(1-\tfrac1{\sqrt{2-\delta}}\big)$
branch of (A2a) and the $\sqrt\epsilon$ slack, yields

$$
\frac{\lVert z-b\rVert^2+\epsilon}{\lVert z-a\rVert^2+\epsilon}\le 2-\delta. \quad (7)
$$

(Inequalities (3) and (7) keep the paper's numbering; the in-between algebra is routine
triangle-inequality manipulation — supplement eqs. (4)–(9).) Multiplying (3)·(7):
$\Psi_l^{k}\le(1+\delta)(2-\delta)$, so

$$
\Delta_k=\sum_{l}\log\Psi_l^{k}\le m^{\prime}\log\big((1+\delta)(2-\delta)\big)=\Delta_{\max}.
$$

**Every wrong-class logit can rise by at most $\Delta_{\max}$.**

### 2.6 Putting the two bounds together (the margin argument)

Suppose **before** projection the correct class led by at least $2\Delta_{\max}$:

$$
L_c(\mathbf x,\{\mathbf b_l^{c}\})\ge L_k(\mathbf x,\{\mathbf b_l^{k}\})+2\Delta_{\max}\quad\forall k\neq c.
$$

Then, chaining "correct drops $\le\Delta_{\max}$" with "wrong rises $\le\Delta_{\max}$":

$$
\begin{aligned} L_c(\mathbf x,\{\mathbf a_l^{c}\}) &\ge L_c(\mathbf x,\{\mathbf b_l^{c}\})-\Delta_{\max} \\ &\ge L_k(\mathbf x,\{\mathbf b_l^{k}\})+\Delta_{\max} \\ &\ge L_k(\mathbf x,\{\mathbf a_l^{k}\}). \end{aligned}
$$

So after projection class $c$ still wins. **Projection does not change the prediction.** ∎

### 2.7 Interpretation of Theorem 2.1 (the concrete $\delta=\frac{9}{16}$ check)

To get a feel for the assumptions, plug in $\delta=\tfrac9{16}$:

- (A2a) becomes $\lVert\mathbf a_l^{k}-\mathbf b_l^{k}\rVert_2\le\big(1-\tfrac{4}{\sqrt{23}}\big)\lVert\mathbf z_l^{k}-\mathbf b_l^{k}\rVert_2-\sqrt\epsilon$,
  and $1-\tfrac{4}{\sqrt{23}}\approx 0.17$.
- (A2b) becomes $\lVert\mathbf a_l^{c}-\mathbf b_l^{c}\rVert_2\le\tfrac14\lVert\mathbf z_l^{c}-\mathbf b_l^{c}\rVert_2$
  and $\lVert\mathbf z_l^{c}-\mathbf b_l^{c}\rVert_2\le\tfrac{\sqrt7}{4}$.

The paper observes the requirement $\lVert\mathbf z^c-\mathbf b^c\rVert_2\le\tfrac{\sqrt7}{4}$
is **empirically always satisfied**, and that the conditions are *tighter on the wrong
classes than the correct one* — which is exactly what you'd expect: a $\text{class-}c$ image has
**no** patch very close to a $\text{non-}c$ prototype, while projection pushes each $\text{non-}c$
prototype onto the closest patch *in its own class*, so $\lVert\mathbf a^k-\mathbf b^k\rVert$
(movement) is generally much smaller than $\lVert\mathbf z^k-\mathbf b^k\rVert$ (the
already-large distance). Hence the assumptions are reasonable, and **when they hold the
classifier's decision provably does not get worse over a large region of the image domain.**

> **Takeaway for the POC.** This theorem is *why* the alternating schedule works: the
> cluster cost (Stage 1) keeps prototypes hugging real patches, which makes the projection
> move (Stage 2) tiny, which (by 2.1) keeps accuracy. If after a push your accuracy
> *collapses*, the usual culprit is **forgetting Stage 3** (last-layer re-opt) — see the
> gotcha in [notes.md](notes.md).

---

## 3. S2 — A probabilistic interpretation of ProtoPNet

> **In one sentence:** the exact logit/softmax computation ProtoPNet uses at inference is a
> **special case of Bayesian class-conditional density estimation**, with the prototypes
> acting as the *modes* of per-prototype densities in latent space. This reframes "ProtoPNet
> is a useful heuristic" as "ProtoPNet is principled inference under stated assumptions."

### 3.1 Start from Bayes, then move to latent space

Classification = estimate $P(Y=k\mid \mathbf X=\mathbf x)$. By Bayes' theorem this is a
**class-conditional density** problem:

$$
P(Y=k\mid\mathbf X=\mathbf x)=\frac{P(\mathbf X=\mathbf x\mid Y=k)\,P(Y=k)}{\sum_{c=1}^K P(\mathbf X=\mathbf x\mid Y=c)\,P(Y=c)}.
$$

But estimating a density over the **image space** $\mathcal X$ is *harder* than the original
classification. The whole point of S2 is to **shift the density estimation into the latent
patch space** $\Omega$ (the domain of patches/prototypes), which is far more tractable.

### 3.2 The bridge: the "closest-patch" functions $f_l^{k}$

Define, for each prototype $\mathbf p_l^{k}$, the function returning the image's **closest
latent patch** to it:

$$
f_l^{k}(\mathbf x):=\arg\min_{\mathbf z\in\text{patches}(f(\mathbf x))}\lVert\mathbf z-\mathbf p_l^{k}\rVert_2 \ \in\ \Omega.
$$

Assuming a unique closest patch, $f_l^{k}(\mathbf X)$ (a function of the random image
$\mathbf X$) is itself a **random variable** in latent space. The conv weights and the
prototypes are treated as *deterministic distribution parameters* (frequentist view), so each
$f_l^{k}$ is well-defined.

### 3.3 Factor the class density and kill the image-space term

Knowing $\mathbf X=\mathbf x$ determines every $f_l^{k}(\mathbf x)$ with probability 1, so

$$
P(\mathbf X=\mathbf x\mid Y=k)= \underbrace{P\big(\mathbf X=\mathbf x \mid f_1^{k}(\mathbf X)=f_1^{k}(\mathbf x),\dots,Y=k\big)}_{\text{image-space term}} \cdot \underbrace{P\big(f_1^{k}(\mathbf X)=f_1^{k}(\mathbf x),\dots\mid Y=k\big)}_{\text{latent-space term}}.
$$

**Assumption (i).** For *every* image $\mathbf x$ and *any* two classes $a,b$, the
image-space term is the same:

$$
P\big(\mathbf X=\mathbf x\mid f_1^{a}(\mathbf X)=f_1^{a}(\mathbf x),\dots,Y=a\big) = P\big(\mathbf X=\mathbf x\mid f_1^{b}(\mathbf X)=f_1^{b}(\mathbf x),\dots,Y=b\big).
$$

In words: *knowing an image's closest patches to a class's prototypes (plus that the image
is of that class) leaves the **same** residual uncertainty about the image, no matter which
class we condition on.* Because it is class-independent, this term **cancels** between
numerator and denominator of Bayes' rule. (A simpler, more restrictive version: the closest
patches to $\text{class-}c$ prototypes already pin down the pixel-level uncertainty, so the class
label adds nothing.)

After (i), the posterior depends **only on the latent-space term**:

$$
P(Y=k\mid\mathbf X=\mathbf x) \propto P\big(f_1^{k}(\mathbf X)=f_1^{k}(\mathbf x),\dots,f_{m_k}^{k}(\mathbf X)=f_{m_k}^{k}(\mathbf x)\mid Y=k\big)\,P(Y=k).
$$

The density estimation now lives in $\Omega^{m_k}$ instead of $\mathcal X$ — exactly the
intended simplification.

### 3.4 Factor across prototypes (Assumption ii)

**Assumption (ii)** — conditional independence of the per-prototype variables *at the
observed closest-patch values*:

$$
P\big(f_1^{k}(\mathbf X)=f_1^{k}(\mathbf x),\dots\mid Y=k\big)=\prod_{l=1}^{m_k}P\big(f_l^{k}(\mathbf X)=f_l^{k}(\mathbf x)\mid Y=k\big).
$$

(This is *not* claiming global independence of the $f_l^{k}(\mathbf X)$ — only on the
subspace of actually-attained closest-patch tuples.) Now the posterior is a clean product:

$$
P(Y=k\mid\mathbf X=\mathbf x) \propto \Big[\prod_{l=1}^{m_k}P\big(f_l^{k}(\mathbf X)=f_l^{k}(\mathbf x)\mid Y=k\big)\Big]\,P(Y=k).
$$

### 3.5 Choose the per-prototype density to encode case-based reasoning

We *want* high probability for class $k$ exactly when the image has parts close to $\text{class-}k$
prototypes. So make each per-prototype density **decrease with distance to the prototype**:

$$
P\big(f_l^{k}(\mathbf X)=\mathbf z\mid Y=k\big)=d_l^{k}\big(\lVert\mathbf z-\mathbf p_l^{k}\rVert_2\big),
$$

where $d_l^{k}:[0,\infty)\to[0,\infty)$ is **monotonically decreasing** and normalized
($\int_\Omega d_l^{k}\,dz=1$). Such a density is **spherically symmetric with its mode
exactly at the prototype $\mathbf p_l^{k}$** — the prototype *is* the most likely latent
patch for that class. Substituting (eq. 10 in the supplement):

$$
P(Y=k\mid\mathbf X=\mathbf x)=\frac{\big[\prod_{l=1}^{m_k}d_l^{k}(\lVert f_l^{k}(\mathbf x)-\mathbf p_l^{k}\rVert_2)\big]\,P(Y=k)}{\sum_{c=1}^K\big[\prod_{l=1}^{m_c}d_l^{c}(\lVert f_l^{c}(\mathbf x)-\mathbf p_l^{c}\rVert_2)\big]\,P(Y=c)}. \quad (10)
$$

### 3.6 ProtoPNet's actual softmax *is* equation (10)

Take ProtoPNet with last-layer weights $1$ (own-class) / $0$ (cross-class) — which is what
Stage 3 produces. Its $\text{class-}k$ logit is $\sum_l g_{\mathbf p_l^{k}}(f(\mathbf x))$, and
using $\frac{d^2+1}{d^2+\epsilon}=1+\frac{1-\epsilon}{d^2+\epsilon}$, the softmax probability is

$$
P(Y=k\mid\mathbf X=\mathbf x)=\frac{\exp\!\Big(\sum_{l}\log\big(1+\tfrac{1-\epsilon}{\lVert f_l^{k}(\mathbf x)-\mathbf p_l^{k}\rVert_2^2+\epsilon}\big)\Big)}{\sum_{c}\exp\!\Big(\sum_{l}\log\big(1+\tfrac{1-\epsilon}{\lVert f_l^{c}(\mathbf x)-\mathbf p_l^{c}\rVert_2^2+\epsilon}\big)\Big)},
$$

which simplifies (turn $\exp\sum\log$ into a product) to (eq. 11):

$$
P(Y=k\mid\mathbf X=\mathbf x)=\frac{\prod_{l=1}^{m_k}\dfrac{\lVert f_l^{k}(\mathbf x)-\mathbf p_l^{k}\rVert_2^2+1}{\lVert f_l^{k}(\mathbf x)-\mathbf p_l^{k}\rVert_2^2+\epsilon}}{\sum_{c=1}^K\prod_{l=1}^{m_c}\dfrac{\lVert f_l^{c}(\mathbf x)-\mathbf p_l^{c}\rVert_2^2+1}{\lVert f_l^{c}(\mathbf x)-\mathbf p_l^{c}\rVert_2^2+\epsilon}}. \quad (11)
$$

**Comparing (10) and (11)**, ProtoPNet is the special case where:
1. $\Omega=[0,1]^{H_1\times W_1\times D}$ (the bounded latent-patch domain);
2. **uniform class prior** $P(Y=c)=\tfrac1K$;
3. $d_l^{k}(r)=C_l^{k}\cdot\dfrac{r^2+1}{r^2+\epsilon}$, with normalizer $C_l^{k}$ making
   $\int_\Omega d_l^{k}=1$.

When every class has the **same number of prototypes**, the $\prod_l C_l^{k}$ factors
approximately cancel between numerator and denominator, leaving exactly the expression
ProtoPNet computes. So the model's inference **is** Bayesian density estimation with these
choices.

### 3.7 Remarks / why this matters

- **The similarity function is a modeling choice, not a given.** Different densities
  $d_l^{k}$ ⇒ different activation functions. E.g. a **Gaussian** class-conditional marginal
  corresponds to the activation $g_{\mathbf p_j}(\mathbf z)=\max_{\tilde{\mathbf z}}-\lVert\tilde{\mathbf z}-\mathbf p_j\rVert_2^2$.
  The probabilistic frame opens up a whole design space (the paper leaves empirical
  comparison as future work).
- **It re-explains training and projection.** Learning prototypes by SGD before projection
  = learning the **parametric modes** of the $f_l^{k}(\mathbf X)$ densities; projection
  (Stage 2) = **moving each mode to the nearest training latent patch**. If the mode barely
  moves, the effect is minimal — *the same precondition as Theorem 2.1*. The two
  supplementary results describe the **same fact** from two angles (geometric margin in S1,
  density mode in S2).
- **It explains ensembling.** Adding the logits of several ProtoPNet models = taking the
  **product of their class-conditional marginals** $f_l^{k}(\mathbf X)\mid Y=k$. Including
  more random variables in the joint density generally **improves** prediction.

---

## 4. How the three pieces fit together

```
 §2.2 Stage 1  (cluster + separation)
        │  shapes a latent space where each image hugs an own-class prototype
        │  and avoids other-class prototypes  →  prototypes sit ON real patches
        ▼
 §2.2 Stage 2  (projection/push)             ── interpretability move
        │  snap prototypes onto real training patches (now visualizable)
        │
        ├── S1 / Theorem 2.1 : because Stage 1 kept the move small AND the
        │                       margin was ≥ 2·Δ_max, the prediction is UNCHANGED
        │
        └── S2 / probabilistic: the snap = moving the MODE of a latent density
                                 to the nearest training patch (minimal effect)
        ▼
 §2.2 Stage 3  (convex last-layer opt, L1)
           drive cross-class weights → 0  ⇒  positive-evidence explanations,
           and exactly the w=1/0 last layer that S1 (A4) and S2 (§3.6) assume.
```

The single thread running through all of it: **the cluster cost keeps prototypes close to
real patches**, which (a) makes projection visualizable, (b) makes S1's "small move"
assumption true so accuracy survives the push, and (c) is what S2 calls "the mode doesn't
move." Stage 3's sparse last layer is, in turn, the exact configuration both supplementary
proofs assume in order to write the logit as a clean sum/product over a class's own
prototypes.
