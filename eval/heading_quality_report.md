# Documents with unusable headings

Every ACTIVE Chapter and Section title in the corpus, checked against five failure modes seen in the data. A title is counted unusable if it matches any of them.

- documents scanned: **998**
- headings scanned: **30,077**
- unusable: **1,869 (6%)**
- documents at or above 30% unusable: **68**

## Why they fail

| Failure mode | Headings |
|---|---|
| no words at all | 749 |
| line-break fragment | 394 |
| mostly digits | 348 |
| body sentence | 294 |
| equation fragment | 263 |
| mid-sentence fragment | 202 |
| table-grid row | 31 |

_A heading can fail more than one test, so these sum to more than the total._

## Examples, by failure mode

What the parser produced, with the document it came from. These are the concrete cases to work against.

### no words at all

| Document | Stored heading |
|---|---|
| doc_arxiv_adam | `7 1 0 2 n a J 0 3 ] G L .s c[ 9 v0 89 6. 2 1 41 : v i X ra` |
| doc_arxiv_adam | `T ≥ 1.` |
| doc_arxiv_adam | `0.40 ts` |
| doc_arxiv_adam | `2 2 α t β 1 ,t β 1 ,t ) 2 m t,i 2` |
| doc_arxiv_adam | `λt−1t ≤ X (1 β ) t=1 − 1` |
| doc_arxiv_bert | `y ca 82 ru cc A v 80 e D I L N 78 M` |
| doc_arxiv_gan | `G D` |
| doc_arxiv_gan | `G 2 G 2` |

### line-break fragment

| Document | Stored heading |
|---|---|
| doc_arxiv_bert | `TPU-now-offers-preemptible-pricing-and-global-` |
| doc_arxiv_bert | `https://cloudplatform.googleblog.com/2018/06/Cloud-` |
| doc_arxiv_resnet | `reducing of the training error3. The reason for such opti-` |
| doc_arxiv_resnet | `CNN step are both trained for 240k iterations with a learn-` |
| doc_arxiv_2608_19762 | `Since the adaptive map is coordinate-separable, its deriva-` |
| doc_arxiv_2608_19762 | `Thus, positive local curvature gives a negative momentum-` |
| doc_arxiv_2608_19762 | `The reference-surrogate ISO instead constructs a determin-` |
| doc_arxiv_2608_20055 | `The previous section presents a manually designed multi-` |

### mostly digits

| Document | Stored heading |
|---|---|
| doc_irs_p596 | `Form 8862` |
| doc_arxiv_adam | `0.40 ts` |
| doc_irs_p514 | `Form 1116` |
| doc_arxiv_bert | `200 400 600 800 1,000 Pre-training Steps (Thousands)` |
| doc_arxiv_resnet | `VGG-19` |
| doc_arxiv_resnet | `1 ResNet-101` |
| doc_arxiv_resnet | `Networks, 5(2):157–166, 1994.` |
| doc_arxiv_seq2seq | `EL 30` |

### body sentence

| Document | Stored heading |
|---|---|
| doc_arxiv_resnet | `Figure 4. Training on ImageNet. Thin curves denote training error, and bold curves denote ` |
| doc_arxiv_resnet | `Table 2. Top-1 error (%, 10-crop testing) on ImageNet validation. Here the ResNets have no` |
| doc_arxiv_resnet | `layer residual nets (ResNets). The baseline architectures are the same as the above plain ` |
| doc_arxiv_resnet | `34-layer plain net has higher training error throughout the whole training procedure, even` |
| doc_nist_sp800_218 | `NIST SP 800-218, Secure Software Development Framework (SSDF) Version 1.1: Recommendations` |
| doc_nist_sp800_161 | `Relationship to NIST SP 800-161, Rev. 1, Cybersecurity Supply Chain Risk Management Practi` |
| doc_irs_p541 | `Calculation and Reporting for the API 1-Year Distributive Share Amount and 3-Year Distribu` |
| doc_irs_p541 | `The Owner Taxpayer Reporting of the Recharacterization Amount on Schedule D (Form 1040) or` |

### equation fragment

| Document | Stored heading |
|---|---|
| doc_arxiv_adam | `T ≥ 1.` |
| doc_arxiv_adam | `λt−1t ≤ X (1 β ) t=1 − 1` |
| doc_arxiv_gan | `4.1 Global Optimality of pg = pdata` |
| doc_arxiv_2608_19762 | `Πθ := I 0 0` |
| doc_arxiv_2608_19762 | `≤ h ≤ H 1 ≤ h ≤ H` |
| doc_arxiv_2608_19762 | `≤ h ≤ H` |
| doc_arxiv_2608_19762 | `j=t+1` |
| doc_arxiv_2608_19762 | `≤ h ≤ H K` |

### mid-sentence fragment

| Document | Stored heading |
|---|---|
| doc_arxiv_bert | `TriviaQA-Wiki formed of the ﬁrst 400 tokens in documents,` |
| doc_arxiv_2608_19762 | `When the loss is twice differentiable in this region,` |
| doc_arxiv_2608_19762 | `Then, for every fixed h ≥ 1,` |
| doc_arxiv_2608_19762 | `AAAI Conference on Artificial Intelligence, volume 36,` |
| doc_arxiv_2608_19762 | `For every future step s ≥ t + 1,` |
| doc_arxiv_2608_19762 | `At the first post-shock state,` |
| doc_arxiv_2608_19762 | `By induction,` |
| doc_arxiv_2608_19762 | `We formulate localized minibatch influence as a signed,` |

### table-grid row

| Document | Stored heading |
|---|---|
| doc_arxiv_2608_19762 | `\|r (h)\| ≤ X e⊤v \|w∗e \| \|λ \|h−1. θ i i k k i` |
| doc_arxiv_2608_19043 | `\|ψ′1⟩ = √ XX\|x⟩\|0⟩\|t⟩, and (9) m2nt t x` |
| doc_arxiv_2608_19043 | `\|ψ2⟩ = √ X (−1) f(x) \|x⟩ (24) 2n x Z2 ∈ n` |
| doc_arxiv_2608_19043 | `\|ψ⟩ = √ X f(x) \|x⟩. (46) 2n x Zn ∈ 2` |
| doc_arxiv_2608_19043 | `\|y⟩ = √ X X ay \|x, z⟩, x,z (47)` |
| doc_arxiv_2608_19298 | `\|𝑇̂ \| > \|𝑇 \|` |
| doc_arxiv_2608_19879 | `\|𝜆𝑟 − 𝜆\| ≤ 𝑘𝑟𝑚𝑎𝑥{𝜆𝑜 − 𝑎, 𝑏 − 𝜆𝑜}, ∀𝑟 ≥ 1. (9)` |
| doc_arxiv_2608_19890 | `\|ST \|` |

## What the failures look like

Two patterns account for most of the debris, and both are recoverable in
the parser rather than downstream.

**Rotated text read as a horizontal run.** Chart axis labels are drawn
vertically, and come out as character runs:

    'y ca 82 ru cc A v 80 e D I L N 78 M'   <- "Accuracy" / "MNLI Dev"
    ')s µ( yr 105 e u q e p it n'           <- "query time (µs)"
    '0.59 E - n e k oT0.4'

Each is one text run per glyph column. A rotation check on the span, or
dropping spans whose glyphs share an x-coordinate, removes this whole
class -- it is the largest single source of "no words at all" and
"mostly digits".

**Bibliography entries promoted to headings.** Reference lists are
short lines in a distinct style, which reads like a heading run:

    'Adepoju, S., David, S.: An Intelligent API Framework for Real-time'
    'Education, 2023, 54-60, https://doi.org/10.1145/3587102.3588794.'

Anything after a "References"/"Bibliography" heading is a citation, not a
section, and can be excluded by position alone.

The remaining modes -- line-break fragments, mid-sentence fragments and
body sentences -- are all one thing: a body line promoted because it was
short or styled like a heading. A heading does not end in a hyphen or a
comma, which is what those two tests key on.


## Worst documents

| Document | Source file | Headings | Unusable | % | Example |
|---|---|---|---|---|---|
| doc_arxiv_2608_20281 | 233d8e1d91374234a93931aa9dbc5adf_a | 8 | 6 | 75% | `Qian Kou 1 ∗† , Xiaofeng Shi 1 ∗† ,` |
| doc_arxiv_2608_20019 | d91a83677ecf49d28453f6b1056c38e9_a | 22 | 15 | 68% | `Contrastive Mixed Prompt Learning for Incomplete Multim` |
| doc_arxiv_2608_12845 | 61a8577cacdf4955b984a6f85c8132c2_a | 6 | 4 | 67% | `L L` |
| doc_arxiv_2608_15407 | 721e33e194b14f0693eac137859cb544_a | 3 | 2 | 67% | `Chameleon: An Adaptive AI-Driven Honeypot Architecture ` |
| doc_arxiv_2608_11524 | af927ff12a6542b983a1a14bc45d3425_a | 3 | 2 | 67% | `2.5 sl` |
| doc_arxiv_2608_19953 | dfc1b6686a4142c48bc4d88e8fc987d8_a | 52 | 34 | 65% | `0 Mixed-Integer Linear Programming (MILP) is a fundamen` |
| doc_arxiv_2608_17941 | dccf6026716f4b759007b6e7143921cd_a | 22 | 14 | 64% | `K − 1` |
| doc_arxiv_2608_10517 | a1c00ab23fbd4210af12178cab44ce03_a | 38 | 24 | 63% | `6 2 0 2 g u A 1 1 ]I N .s c [ 1 v 7 1 50 1 . 80 6 2 :v ` |
| doc_arxiv_2608_02095 | 1d3da0bce25a42cebcb4b63d7bc84083_a | 8 | 5 | 62% | `S S (` |
| doc_arxiv_2608_06447 | 6a349f2adfdc42258bd690a6964e1fd2_a | 8 | 5 | 62% | `ACCURACY = (3) 𝐹𝑃+𝑇𝑃+𝐹𝑁+𝑇𝑁` |
| doc_arxiv_2608_15410 | 403058fc39ec4d9397425dcbd2b14828_a | 18 | 11 | 61% | `to generate the pixel-level mask associated with a natu` |
| doc_arxiv_2608_19637 | 56d2304008ff40a380bf5dcbc8ed404e_a | 15 | 9 | 60% | `Appendix A.1 More Dataset and Benchmark Details Product` |
| doc_arxiv_2608_20208 | 99e9b3a9eac7420da4f104c2a29d8ddd_a | 14 | 8 | 57% | `L H` |
| doc_arxiv_2608_18673 | 5938aab2a051418e8827abe56f15d8d9_a | 16 | 9 | 56% | `12 result is the unweighted macro-average over MASA [17` |
| doc_arxiv_2608_19621 | ac8deb89553341c4b5d5aa8bccb41496_a | 33 | 17 | 52% | `Figure 2: Overview of LifeMem. (a) Individual life traj` |
| doc_arxiv_2608_18919 | 1488143bf7724c94b4f44b3ebfe2fdf4_a | 38 | 19 | 50% | `To define performance dimensions relative to the data-` |
| doc_arxiv_2608_19808 | 506603985a55499b9a42eda4435b2895_a | 24 | 12 | 50% | `We evaluate FAR-DPO with PepGLAD and PepFlow to de-` |
| doc_arxiv_2608_20256 | 3708b95b0ec1467ab327934ee215d4fb_a | 20 | 10 | 50% | `Pi=1 Ni i=1 t=1` |
| doc_arxiv_2608_20005 | dc1141789f4f4845b7e5e25c9ae47cd0_a | 10 | 5 | 50% | `Table III. On ETT, removing SA consistently degrades pe` |
| doc_arxiv_2608_11502 | 32cdd98842ce472dbc1a82de4f9d0c51_a | 4 | 2 | 50% | `CD D 5` |
| doc_arxiv_2607_23636 | 6c929f0b2a844df7b6ba19f24ab63f95_a | 43 | 21 | 49% | `\|VY ,VX1 \|` |
| doc_arxiv_2608_19298 | f6a2412052ac458ebfc9954dd22cb096_a | 29 | 14 | 48% | `SceneGTMM: A Conformal Mapping-based Scene-Aware Transf` |
| doc_arxiv_2608_16587 | 7e87498299cd47ecb012e224a1f0c99b_a | 11 | 5 | 45% | `L + 1` |
| doc_arxiv_2608_19043 | 0798cf5598784278842c21df5de6c806_a | 80 | 36 | 45% | `Our generalised construction, as illustrated in Figure ` |
| doc_arxiv_2608_10391 | 96e67fbf8ba4400ebddd5fc2c4b47f24_a | 20 | 9 | 45% | `ET ER` |
| doc_arxiv_2608_08425 | 0e38b81654d04cc993ad55fa494887f7_a | 18 | 8 | 44% | `) B M (` |
| doc_arxiv_2608_02031 | 47bb486c8aeb49a589d5cff271322e6a_a | 41 | 18 | 44% | `𝑇$,! (1)` |
| doc_arxiv_2608_09640 | 46f20b2920b345a383822aaa9e9c1813_a | 7 | 3 | 43% | `Islices = ([W1, W2, ..., WN], FSlicer)` |
| doc_arxiv_2608_19674 | 6829042dfc524577bec89b3b60ef0a8f_a | 7 | 3 | 43% | `L P c v 6 2 r` |
| doc_arxiv_2608_19966 | a704c241940d4f4f95c7f355a8a0c013_a | 80 | 34 | 42% | `(1) (𝑅)` |
| doc_arxiv_2608_20255 | 8f59ca32ba794550a557bb6857b36e75_a | 125 | 53 | 42% | `1 n 1 / 2` |
| doc_arxiv_2608_19536 | 9de11a13a64942bbbb6df705d76696dc_a | 19 | 8 | 42% | `6 2 c .8 06 2 X r` |
| doc_arxiv_2608_19879 | 77bab9bb276446b0b8059ebdf0ca1038_a | 51 | 21 | 41% | `A Repeated Measurements Approach to 𝑆𝑜𝐻 Battery Modelli` |
| doc_arxiv_resnet | arxiv_resnet.pdf | 22 | 9 | 41% | `VGG-19` |
| doc_arxiv_2608_19890 | fadbf7eb19bb453c8690d5aef8dbddf3_a | 72 | 29 | 40% | `I O` |
| doc_arxiv_2608_11037 | 9385f579bbc74d2ca01ecf43323998b1_a | 50 | 20 | 40% | `100%` |
| doc_arxiv_2608_17613 | e70d08aef8ee4e888f552ed793eab1a5_a | 40 | 16 | 40% | `Yang Hu , Jingui Ma , Ning Li , Jiangling Qin , Yanming` |
| doc_arxiv_2608_20331 | 93977afcedc449a88bd2d51b9dfca207_a | 20 | 8 | 40% | `However, precision alone may favor overly short respons` |
| doc_arxiv_2608_10555 | 3db09345435a4dd4ab48ba0166a3f3e4_a | 15 | 6 | 40% | `1 ]I N` |
| doc_arxiv_2608_13292 | 828240282f804874bd98a3f94ba1b1ef_a | 10 | 4 | 40% | `[Experimental Results for RQ-1.2]: Surface-level con- p` |
| doc_arxiv_2607_24021 | cdc171e64b58412ba40eab5455922635_a | 43 | 17 | 40% | `if Σ = ∅ then` |
| doc_arxiv_2608_17632 | 26e70eaadbcf41c0b342ed4a0c5357db_a | 23 | 9 | 39% | `ℒ +·ℒ` |
| doc_arxiv_2608_18840 | 21ef771328fd4a9a8e54cf68d76d16d7_a | 39 | 15 | 38% | `6 2 c 88 1 v` |
| doc_arxiv_2608_12184 | b05e93557b244dc2afece5d491a9065d_a | 21 | 8 | 38% | `Q K` |
| doc_arxiv_2608_09221 | 36f9ea964bcb470c91e4b7c8971d8960_a | 35 | 13 | 37% | `6 2 02 g u ] G L c 1 v 1 0 6 2 i` |
| doc_arxiv_2608_09748 | 4aab81601d724d92b49eb1b3eb7210bb_a | 27 | 10 | 37% | `6 7` |
| doc_arxiv_2608_05639 | e1b99b2c5fb04b55964e719c4fc12929_a | 22 | 8 | 36% | `cost of using all configured opportunities. Proactive H` |
| doc_arxiv_2608_20122 | 710845ac39f340cf98d3c19792e2fc3d_a | 31 | 11 | 35% | `Annotation and QA Construction We construct AdvSpot thr` |
| doc_arxiv_2608_14011 | 102ed171dd2d4c63a2f3fc5dbae33eba_a | 17 | 6 | 35% | `Generative recommendation (GR) has recently emerged as ` |
| doc_arxiv_2608_19889 | 561e346e1da544029b8a9b222f4ea799_a | 71 | 25 | 35% | `6 2 0 2 I 9 1 .` |
| doc_arxiv_2608_20117 | 31b288b2b28843c187f3562b1482cd57_a | 37 | 13 | 35% | `❌` |
| doc_arxiv_2608_19762 | 511ac4287dc74bdb89a48d6daf25dc58_a | 80 | 28 | 35% | `When the loss is twice differentiable in this region,` |
| doc_arxiv_2608_18952 | c0771e91efe1432991e8487bab006c71_a | 59 | 20 | 34% | `We formalize next-item ranking as conditional gener-` |
| doc_arxiv_2608_20305 | f6373fdb077a44059ad909017124d44c_a | 27 | 9 | 33% | `CalcSeg: Confidence-aware 3D Latent Context Curriculum ` |
| doc_arxiv_2608_04050 | 5c7612f6e2a54e24bc23d12649f39ec6_a | 24 | 8 | 33% | `SM 0.1` |
| doc_arxiv_2607_23632 | e100edc4df994439a5a40b6dc6537fca_a | 15 | 5 | 33% | `Y A T F` |
| doc_arxiv_2608_09308 | 7c6b0cf63d5e4f68be78e07967eba469_a | 15 | 5 | 33% | `1.5k` |
| doc_arxiv_2608_20322 | f5c7ce45ebca476086b65516d1d60c9d_a | 12 | 4 | 33% | `A comparison between ceiling-mounted FMCW, IR-UWB and W` |
| doc_arxiv_2608_18746 | 1b88488dbb4a420e9dd3a81d787ef181_a | 12 | 4 | 33% | `0: H` |
| doc_arxiv_2608_14386 | 88fe28e6627b47349aa5b70c1079bb7a_a | 9 | 3 | 33% | `9 103` |
| doc_arxiv_2608_19000 | 22cb5760ae01423c926952f5548cfe93_a | 46 | 15 | 33% | `We formulate element-conditioned design generation as a` |
| doc_arxiv_2608_13077 | c7056e1932124bccaaf9339be18f0ee5_a | 37 | 12 | 32% | `Recent benchmarks have attempted to strengthen specifi-` |
| doc_arxiv_2608_16536 | 8f0243b4d19d4a55afb48028f5312513_a | 29 | 9 | 31% | `Subsequently, in the interaction stage, the augmented s` |
| doc_arxiv_2608_19730 | 2cfdc32617a046188446adef99463ce5_a | 42 | 13 | 31% | `For the small-scale channel gain, h, we adopt the Nakag` |
| doc_arxiv_2608_14995 | 9be9ab8654f64f15aaf127ca70deeb41_a | 13 | 4 | 31% | `8.7` |
| doc_arxiv_2608_08277 | a39daa48efce4371b12875810e02e6b9_a | 23 | 7 | 30% | `1 P` |
| doc_arxiv_2608_18984 | 0e776224d1ef499c8446458454b0f89a_a | 53 | 16 | 30% | `6 2 02 g u A 9 1 ] V C .s c [ 1 v4 8 98 1. 8 0 6 2: v i` |
| doc_arxiv_2608_16508 | a46cdd1f709144e593b0dda836d5918e_a | 40 | 12 | 30% | `The proposed framework consists of three main com-` |

## Clean

**717** documents have no unusable heading (of 986 with three or more).

