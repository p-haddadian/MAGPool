# MAGPool: Multi-hop Attention-based Graph Pooling

## Overview
This directory contains the code required to execute the MAGPool algorithm. MAGPool is a hierarchical graph pooling algorithm that utilizes Personalized PageRank and an Attention mechanism. This combination is designed to integrate multi-hop connections existing within the graph into the processes of message passing and information aggregation.

For further information you can refer to [Multi-hop Attention-based Graph Pooling: A Personalized PageRank Perspective](https://ieeexplore.ieee.org/abstract/document/10454077).

## Requirements
MAGPool is implemented with Pytorch and Pytorch Geometric

    Pytorch >= 2.0.1+cu118
    Pytorch Geometric >= 2.3.1
Install the reqirements:

    pip install -r requirements.txt
    
## Usage

    python3 train.py

NOTE: there are various arguments which you can modify, please see `train.py` for further details.

## Cite

If you make advantage of the MAGPool, please cite the following in your manuscript:

     @INPROCEEDINGS{10454077,
      author={Haddadian, Parsa and Booryaee, Roya and Abedian, Rooholah and Moeini, Ali},
      booktitle={2024 Third International Conference on Distributed Computing and High Performance Computing (DCHPC)}, 
      title={Multi-hop Attention-based Graph Pooling: A Personalized PageRank Perspective}, 
      year={2024},
      volume={},
      number={},
      pages={1-7},
      keywords={Representation learning;High performance computing;Robustness;Graph neural networks;Task analysis;Distributed computing;Resilience;Graph Representation Learning;Graph Pooling;Attention Mechanism;Personalized PageRank},
      doi={10.1109/DCHPC60845.2024.10454077}}

## Contact Us

Please do not hesitate to contact [Parsa](mailto:p.haddadian@ut.ac.ir) or [Roya](mailto:roya.boryaee@ut.ac.ir), if you had any further questions


