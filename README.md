# Glow

This repository implements the [Glow](https://arxiv.org/abs/1807.03039) model using PyTorch on the MNIST dataset.

## Setup and run

The code has minimal dependencies. You need python 3.6+ and up to date versions of:

```
pytorch (tested on 1.1.0)
torchvision
pytorch-ignite
torchattacks
tqdm
```

**To train your own model:**

```
python train.py --num_classes=[2 or 10] --fresh
```

Will download the Mnist dataset for you, and start training. The output files will be sent to `output/`.

Everything is configurable through command line arguments, see

```
python train.py --help
```

for what is possible.

The model is trained using `adamax` instead of `adam` as in the original implementation. Using `adam` leads to a NLL of 3.48 (vs. 3.39 with `adamax`). Note: when using `adam` you need to set `warmup` to 1, otherwise optimisation gets stuck in a poor local minimum.

## References:

```
@inproceedings{kingma2018glow,
  title={Glow: Generative flow with invertible 1x1 convolutions},
  author={Kingma, Durk P and Dhariwal, Prafulla},
  booktitle={Advances in Neural Information Processing Systems},
  pages={10215--10224},
  year={2018}
}

@inproceedings{nalisnick2018do,
    title={Do Deep Generative Models Know What They Don't Know? },
    author={Eric Nalisnick and Akihiro Matsukawa and Yee Whye Teh and Dilan Gorur and Balaji Lakshminarayanan},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=H1xwNhCcYm},
}
```
