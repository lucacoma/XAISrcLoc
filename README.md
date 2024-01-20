# Interpreting End-to-End Deep Learning Models for Acoustic Source Localization using Layer-wise Relevance Propagation

Code repository for the paper _Interpreting End-to-End Deep Neural Networks Models for Acoustic Source Localization using Layer-wise Relevance Propagation_
[[1]](#references). 

For any further info feel free to contact me at [luca.comanducci@polimi.it](luca.comanducci@polimi.it)

- [Dependencies](#dependencies)
- [Data Generation](#data-generation)
- [Network Training](#network-training)
- [Results Computation](#results-computation)

### Dependencies
- Python, it has been tested with version 3.9.18
- Numpy, tqdm, matplotlib
- Pytorch 2.1.2+cu118
- [zennit](https://github.com/chr5tphr/zennit)
- [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR)

### Data generation
The _generate_data.py_ script generates the data using the room parameters contained in _params.py_.

The command-line arguments are the following
- T60: float, Reverberation Time (T60)
- SNR: Int, Signal to Noise Ratio (SNR)
- gpu: Int, number of the chosen GPU (if multiple are available)


### Network training
The _train.py_ script trains the network. 

The command-line arguments are the following
- T60: float, Reverberation Time (T60)
- SNR: Int, Signal to Noise Ratio (SNR)
- gpu: Int, number of the chosen GPU (if multiple are available)
- data_path: String, path to where dataset is saved
- log_dir: String, Path to where to store tensorboard logs

### Results computation
To perform the XAI experiments:

The _perturbation_experiment.py_ performs manipulation of input features.

The command-line arguments are the following
- T60: float, Reverberation Time (T60)
- SNR: Int, Signal to Noise Ratio (SNR)
- gpu: Int, number of the chosen GPU (if multiple are available)

The _tdoa_experiment.py_ performs the time-delay estimation experiment.

The command-line arguments are the following
- T60: float, Reverberation Time (T60)
- SNR: Int, Signal to Noise Ratio (SNR)
- gpu: Int, number of the chosen GPU (if multiple are available)

The jupyter notebooks _Input_visualization_paper.ipynb_ and _Plot_Perturbation.ipynb_ can be used to obtain the same figures presented in the paper.0

N.B. pre-trained models used to compute the results shown in [[1]](#references) can be found in folder _models_

# References
>[1] L.Comanducci, F.Antonacci, A.Sarti, Interpreting End-to-End Deep Learning Models for Acoustic Source Localization using Layer-wise Relevance Propagation
