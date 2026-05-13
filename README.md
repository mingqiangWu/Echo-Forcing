<h1 align="center">🚀 Echo-Forcing</h1>
<p align="center"><sub><b>A Scene Memory Framework for Interactive Long Video Generation</b></sub></p>

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
  <strong>Echo-Forcing enables training-free interactive long-video generation with preserve, recall, and forget scene memories.</strong>
</p>



## 📰 News

- **[Coming Soon]** Code release.

---

## 📖 Abstract

Autoregressive video diffusion models enable open-ended generation through local attention and KV caching. However, existing training-free long-video optimization methods mainly focus on stable extension under a single prompt, making them difficult to handle interactive scenarios involving prompt switching, old-scene forgetting, and historical scene recall.

We identify the core bottleneck as the functional entanglement of historical KV states: stable anchors and recent dynamics are handled by the same cache policy, leading to outdated background contamination, delayed response to new prompts, and loss of long-range memory.

To address this issue, we propose **Echo-Forcing**, a training-free scene-memory framework specifically designed for interactive long-video generation. Echo-Forcing introduces three core mechanisms:

- **Hierarchical Temporal Memory**, which decouples stable anchors, compressed history, and recent windows under relative RoPE.
- **Scene Recall Frames**, which compress historical scenes into spatially structured KV representations for long-term recall.
- **Difference-aware Memory Decay**, which adaptively forgets conflicting tokens according to the discrepancy between old and new scenes.

With these designs, Echo-Forcing uniformly supports long-horizon generation, smooth transitions, hard cuts, and long-range scene recall under a bounded cache budget.

## 🎬 Visualization

<p align="center">
  <video src="assets/demo.mp4" width="95%" controls></video>
</p>


<p align="center">
  <strong>Echo-Forcing supports multiple interactive long-video generation modes, including long-horizon rollout, smooth transition, hard cut, and long-range scene recall.</strong>
</p>



## 📊 Results

<p align="center">
  <img src="assets/results.png" width="95%" alt="Echo-Forcing Results"/>
</p>


<p align="center">
  Echo-Forcing improves long-horizon stability and interactive scene control across smooth transition, hard cut, scene recall, and long-video generation.
</p>

## 📧 Contact

For questions or suggestions, please open an issue or contact:

- Mingqiang Wu wumingqiang25e@ict.ac.cn
- Weilun Feng: [fengweilun24s@ict.ac.cn](fengweilun24s@ict.ac.cn)
- Chuanguang Yang: [yangchuanguang@ict.ac.cn](mailto:yangchuanguang@ict.ac.cn)
- Zhulin An: [anzhulin@ict.ac.cn](mailto:anzhulin@ict.ac.cn)
