# snntorch-LSM
This is an **snntorch** implementation of Liquid State Machine (LSM) networks. The parameters of the example network implemented in **main.py** are derived from the paper **"MAdapter: A Multimodal Adapter for Liquid State Machines configures the Input Layer for the same Reservoir to enable Vision and Speech Classification"** [link](10.1109/IJCNN54540.2023.10191376)

## Requirements
Pytorch, Tonic and snntorch

## Description
1. **lsm_weight_definitions.py** - contains definitions of connectivity (Input->Reservoir and Recurrent Reservoir weights)
2. **lsm_models.py** - contains the LSM model definition.
3. **main.py** - contains an example implementation with the N-MNIST dataset. Network execution must be run with **torch.no_grad()** for LSM operation
