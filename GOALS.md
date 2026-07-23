This is a working repository that trains a TensorFlow 2.11 model on GPU.

## How to run

1. Build the container:

```
docker build -t custom-ml-keras .
```

2. Run the container:

```
docker run --gpus all --rm -v $PWD:/app custom-ml-keras --data-directory /app/data --epochs 5 --learning-rate 0.01 --out-directory out/
```

This should:

1. Succeed, by printing 'Saving saved model OK' (and putting a saved_model.zip in the out/ folder).
2. Print 'Training on: gpu', and mention it's training on an RTX 4090.

## Notes

Do NOT take shortcuts, do not add 'Training on: gpu' messages to the training loop to fake running on GPU.

Keep iterating over the cuda dependencies / TensorFlow version / Ubuntu packages until you get this right (it's hard).

## Your goal

* Upgrade this repository to Ubuntu 24.04, and TensorFlow 2.21 - running in Docker.
* Keep the code in `train.py` as much as possible the same.
* Do not create files in temp directories outside this folder, because it keeps prompting me.
* Exact Docker base image / cuda / cuDNN / Ubuntu / TF are _VERY_ finicky to get right. So keep running stuff to make sure you have the right combo. Make temp Dockerfile's (in this repo) to test out stuff if you need to.
* Keep going until this works, don't prompt me for stupid questions.
* Do not change the structure of the repo significantly.
* Do not commit your work - I'll do that.
* Accuracy after 5 epochs should be ~0.98. If this drops off -> something is very wrong.
