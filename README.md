<h1 align="center">🎥 Echo-Forcing</h1>
<p align="center"><sub><b></b></sub></p>

<p align="center">
A Scene Memory Framework for Interactive Long Video Generation
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.05293">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square" alt="Paper"/>
  </a>
  <a href="https://github.com/wlfeng0509/Fast-SAM3D">
    <img src="https://img.shields.io/badge/GitHub-Code-blue?style=flat-square&logo=github" alt="Code"/>
  </a>
</p>

<div align="center">
  <strong>
    Mingqiang Wu<sup>1,2,*</sup> · 
    Weilun Feng<sup>1,2,*</sup> · 
    Zhefeng Zhang<sup>3</sup> · 
    Chuanguang Yang<sup>1,✉</sup>
  </strong>
  <br>
  <strong>
    Haotong Qin<sup>4</sup> · 
    Yuqi Li<sup>5</sup> · 
    Guoxin Fan<sup>1,2</sup> · 
    Xiaokun Liu<sup>1,2</sup>
  </strong>
  <br>
  <strong>
    Zhulin An<sup>1,✉</sup> · 
    Libo Huang<sup>1</sup> · 
    Yongjun Xu<sup>1</sup>
  </strong>
</div>
<p align="center">
  <sup>*</sup> Equal Contribution &nbsp;&nbsp; 
  <sup>✉</sup> Corresponding Authors
</p>
<p align="center">
  State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences<br>
  University of Chinese Academy of Sciences<br>
  China University of Mining and Technology, Beijing<br>
  ETH Zürich<br>
  City College of New York, City University of New York
</p>
---

<p align="center">
  <img src="assets/teaser.png" width="95%" alt="Echo-Forcing Teaser"/>
</p>
<p align="center">
Echo-Forcing enables training-free interactive long-video generation with preserve, recall, and forget scene memories.
</p>

## 🎬 Visualization

<p align="center">
  <video src="https://github.com/user-attachments/assets/91158ce5-7a18-4f0b-a420-97be84288f22" width="95%" controls></video>
</p>

<p align="center">
  <strong>"Interstellar"</strong>: a demo video with a scene transition every 10 seconds, for a total of 6 transitions.
</p>

## 📰 News

- **[2026/05/13]** 🎉 Paper released. Code coming soon.

---

## 📖 Abstract

Autoregressive video diffusion models enable open-ended generation through local attention and KV caching. However, existing training-free long-video optimization methods mainly focus on stable extension under a single prompt, making them difficult to handle interactive scenarios involving prompt switching, old-scene forgetting, and historical scene recall.

We identify the core bottleneck as the functional entanglement of historical KV states: stable anchors and recent dynamics are handled by the same cache policy, leading to outdated background contamination, delayed response to new prompts, and loss of long-range memory.

To address this issue, we propose **Echo-Forcing**, a training-free scene-memory framework specifically designed for interactive long-video generation. Echo-Forcing introduces three core mechanisms:

- **Hierarchical Temporal Memory**, which decouples stable anchors, compressed history, and recent windows under relative RoPE.
- **Scene Recall Frames**, which compress historical scenes into spatially structured KV representations for long-term recall.
- **Difference-aware Memory Decay**, which adaptively forgets conflicting tokens according to the discrepancy between old and new scenes.

With these designs, Echo-Forcing uniformly supports long-horizon generation, smooth transitions, hard cuts, and long-range scene recall under a bounded cache budget.



## 🔍 Method Overview

<p align="center">
  <img src="assets/overview.png" width="80%" alt="Echo-Forcing Teaser"/>
</p>

<p align="center">
  <strong>Overview of the proposed Echo-Forcing framework. </strong>Our method integrates three scene-
memory modules to preserve temporal continuity, recall historical scenes, and suppress conflicting
memories during interactive long-video generation.
</p>

## 📊 Results

<p align="center">
  <img src="assets/long.png" width="80%" alt="Echo-Forcing Teaser"/>
</p>

<p align="center">
Long-video generation on VBench-Long. We compare Echo-Forcing with training-free
long-video baselines at 60s and 120s. Echo-Forcing improves visual fidelity and temporal stability
while maintaining competitive inference throughput.
</p>

<p align="center">
  <img src="assets/inter.png" width="80%" alt="Echo-Forcing Teaser"/>
</p>

<p align="center">
Interactive video generation. We evaluate smooth transition, hard cut, and scene recall
under both non-fine-tuned and fine-tuned settings. Echo-Forcing consistently improves prompt
responsiveness and scene consistency across interaction modes.
</p>


## 📧 Contact

For questions or suggestions, please open an issue or contact:

- Mingqiang Wu wumingqiang25e@ict.ac.cn
- Weilun Feng: [fengweilun24s@ict.ac.cn](fengweilun24s@ict.ac.cn)
- Chuanguang Yang: [yangchuanguang@ict.ac.cn](mailto:yangchuanguang@ict.ac.cn)
- Zhulin An: [anzhulin@ict.ac.cn](mailto:anzhulin@ict.ac.cn)
