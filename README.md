# Replicating Tensorfuzz using Docker 

**by Muhammad Hilmi Asyrofi**

#### 1. Pull tensorflow-gpu 1.6 docker image
```
docker pull tensorflow/tensorflow:1.6.0-devel-gpu-py3
```

#### 2. Clone repository

#### 3. Run docker container 
```
docker run --name tensorfuzz --rm --gpus '"device=1"' -it -v <path to tensorfuzz>/tensorfuzz/:/home/tensorfuzz/ tensorflow/tensorflow:1.6.0-devel-gpu-py3
```
add the folder to the python environment
```
export PYTHONPATH="$PYTHONPATH:/home/tensorfuzz"
```
```
cd /home/tensorfuzz
```

#### 4. Install requirements
```
pip install requirements.txt
```
if it doesn't work, please install manually one-by-one

convert pyflann to from python2 format to python3 format
```
2to3 -w /usr/local/lib/python3.5/dist-packages/pyflann/
```

## Run DCGan
#### Finding Broken Loss Functions in Public Code

This example directory contains code to fuzz a slight modification to the 
well known [DCGAN-tensorflow repository](https://github.com/carpedm20/DCGAN-tensorflow).

To find the issue (which is a loss function that can yield a high loss but zero gradients)
execute the following:

```
python3 examples/dcgan/dcgan_fuzzer.py  --total_inputs_to_fuzz=1000000 --mutations_per_corpus_item=64 --alsologtostderr --strategy=ann --ann_threshold=0.1
```

## Run Nans Model
#### Finding Numerical Errors in Trained Image Classifiers

First you need to train a model that you suspect may have numerical issues:

```
python3 examples/nans/nan_model.py --checkpoint_dir=/tmp/nanfuzzer --data_dir=/tmp/mnist --training_steps=35000 --init_scale=0.25
```

Then you can fuzz this model by pointing the fuzzer at its checkpoints.

```
python3 examples/nans/nan_fuzzer.py --checkpoint_dir=/tmp/nanfuzzer --total_inputs_to_fuzz=1000000 --mutations_per_corpus_item=100 --alsologtostderr --ann_threshold=0.5
```


## Run Quantize
#### Finding Disagreements Between fp16 and fp32 models.

This example directory contains code to check for differences related to quantizing models.
The quantized_model.py file trains a model using 32 bit floating point variables and then
casts those variables to use 16 bits.
To train this model, execute something like this:

```
python3 examples/quantize/quantized_model.py --checkpoint_dir='/tmp/quantized_checkpoints_2' --training_steps=10000
```

To fuzz the trained model, execute something like this:

```
python3 examples/quantize/quantized_fuzzer.py --checkpoint_dir=/tmp/quantized_checkpoints_2 --total_inputs_to_fuzz=1000000 --mutations_per_corpus_item=100 --alsologtostderr --output_path=/cns/ok-d/home/augustusodena/fuzzer/plots/quantized_image.png --ann_threshold=1.0 --perturbation_constraint=1.0 --strategy=ann
```



# TensorFuzz: Coverage Guided Fuzzing for Neural Networks

This repository contains a library for performing coverage guided fuzzing of neural networks,
as was described in [this paper](https://arxiv.org/abs/1807.10875).
It's still a prototype, but the ultimate goal is for people to actually use this to test real software.
Any suggestions about how to make it more useful for that purpose would be appreciated.

## Installation

You ought to be able to run the code in this repository by doing the following:

```
pip install -r requirements.txt
```

Then do:

```
export PYTHONPATH="$PYTHONPATH:$HOME/tensorfuzz"
```

## The structure of this repository

Broadly speaking, this repository contains a core fuzzing library, examples of how 
to use the fuzzer, a list of bugs found with the fuzzer, and some utilities.

### /bugs

This directory contains bugs or weird behaviors that we've found by using this tool.

### /examples

This directory contains examples of how to use the fuzzer in several different ways.
It contains examples of looking for numerical errors, finding broken loss functions
in publicly available code, and checking for disagreements between trained classifiers
and their quantized versions.

### /lib

This directoy contains the fuzzing engine and all the necessary utils.

### /third_party

This directory contains code written by other people and the (potentially updated) 
LICENSES for that code. 


## Disclaimers

This is not an officially supported Google product.
