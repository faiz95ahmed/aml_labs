#!/bin/bash

conda remove --name aml_prac1 --all
conda create --name aml_prac1
conda activate aml_prac1
conda install -c anaconda networkx
conda install pytorch torchvision torchaudio cpuonly -c pytorch
python3 -m pip install --upgrade pip
python3 -m pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
python3 -m pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
python3 -m pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
python3 -m pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
python3 -m pip install torch-geometric
