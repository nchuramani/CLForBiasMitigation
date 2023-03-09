# Domain-Incremental Continual Learning for Mitigating Bias in Facial Expression and Action Unit Recognition

This is a PyTorch implementation for the [Domain-Incremental Continual Learning for Mitigating Bias in Facial Expression and Action Unit Recognition](https://ieeexplore.ieee.org/document/9792455) paper published in the IEEE Transactions on Affective Computing. 
The paper presents benchmarking results comparing Continual Learning for bias mitigation in Facial Affect Analyses with state-of-the-art bias mitigatio methods. 
Code for individual benchmark experiments with the BP4D and RAF-DB datasets can be found in the respective folders. 

## Abstract
As Facial Expression Recognition (FER) systems become integrated into our daily lives, these systems need to prioritise making fair decisions instead of only aiming at higher individual accuracy scores. From surveillance systems, to monitoring the mental and emotional health of individuals, these systems need to balance the accuracy versus fairness trade-off to make decisions that do not unjustly discriminate against specific under-represented demographic groups. Identifying bias as a critical problem in facial analysis systems, different methods have been proposed that aim to mitigate bias both at data and algorithmic levels. In this work, we propose the novel use of Continual Learning (CL), in particular, using Domain-Incremental Learning (Domain-IL) settings, as a potent bias mitigation method to enhance the fairness of Facial Expression Recognition (FER) systems. We compare different non-Continual Learning (CL)-based and CL-based methods for their performance and fairness scores on expression recognition and Action Unit (AU) detection tasks using two popular benchmarks, the RAF-DB and BP4D datasets, respectively. Our experimental results show that CL-based methods, on average, outperform other popular bias mitigation techniques on both accuracy and fairness metrics.

## Citation

```
@article{Churamani2022CL4BiasMitigation,
  author={Churamani, Nikhil and Kara, Ozgur and Gunes, Hatice},
  journal={IEEE Transactions on Affective Computing}, 
  title={{Domain-Incremental Continual Learning for Mitigating Bias in Facial Expression and Action Unit Recognition}}, 
  year={2022},
  pages={1-15},
  doi={10.1109/TAFFC.2022.3181033}
}

@inproceedings{Kara2021Towards,
  title={{Towards Fair Affective Robotics: Continual Learning for Mitigating Bias in Facial Expression and Action Unit Recognition}},
  author={Kara, Ozgur and Churamani, Nikhil and Gunes, Hatice},
  booktitle={{Workshop on Lifelong Learning and Personalization in Long-Term Human-Robot Interaction (LEAP-HRI), 16th ACM/IEEE International Conference on Human-Robot Interaction (HRI)}},
  year={2021}
}
```

## Acknowledgement
For the purpose of open access, the author(s) has applied a Creative Commons Attribution (CC BY) license to any Accepted Manuscript version arising.

**Funding:** N. Churamani is funded by the EPSRC under grant EP/R513180/1 (ref. 2107412). H. Gunes’ work is sup- ported by the EPSRC under grant ref. EP/R030782/1 and partially by the European Union’s Horizon 2020 Research and Innovation program project WorkingAge under grant agreement No. 826232. O. Kara contributed to this work during a summer research study at the Department of Computer Science and Technology, University of Cambridge. 

**Data Access Statement:** This study involved secondary analyses of pre-existing datasets. All datasets are described in the text and cited accordingly. Licensing restrictions prevent sharing of the datasets. The authors thank Prof Lijun Yin from the Binghamton University (USA) for providing access to the BP4D Dataset (https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) and relevant race attributes; and Shan Li, Prof Weihong Deng and JunPing Du from the Beijing University of Posts and Telecommunications (China) for providing access to RAF-DB (http://www.whdeng.cn/raf/model1.html).

**Code:** This repository adapts the code published under the [Continual-Learning-Benchmark](https://github.com/GT-RIPL/Continual-Learning-Benchmark) and [
continual-learning
](https://github.com/GMvandeVen/continual-learning) repositories. Please also consider citing the following if you re-use the code:

```
@article{vandeven2019three,
  title={Three scenarios for continual learning},
  author={van de Ven, Gido M and Tolias, Andreas S},
  journal={arXiv preprint arXiv:1904.07734},
  year={2019}
}

@article{vandeven2018generative,
  title={Generative replay with feedback connections as a general strategy for continual learning},
  author={van de Ven, Gido M and Tolias, Andreas S},
  journal={arXiv preprint arXiv:1809.10635},
  year={2018}
}

@inproceedings{Hsu18_EvalCL,
  title={Re-evaluating Continual Learning Scenarios: A Categorization and Case for Strong Baselines},
  author={Yen-Chang Hsu and Yen-Cheng Liu and Anita Ramasamy and Zsolt Kira},
  booktitle={NeurIPS Continual learning Workshop },
  year={2018},
  url={https://arxiv.org/abs/1810.12488}
}
```
