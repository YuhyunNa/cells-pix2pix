

# Cell pix2pix

Cell pix2pix 는 [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 를 기반으로 한다.

microblock들을 같은 공간에 두었을 때(before), 시간이 지나서 잘 모이면, Tissue Module로 변화 하는데 성공 한 경우이고, 잘 모이지 않는다면 실패한 경우이다.(after)


o_images_final example  
  
x_images_final example

before 사진들과 after 사진들을 1456*1456크기의 정사각형으로 자르고,
before 사진과 after사진을 가로로 이어 붙여서 2912*1456 사진을 만들었다.


성공과 실패, 두 가지의 경우에 대하여 각각 학습 시켰다.
그렇기에 2가지의 체크포인트가 제공 된다.

1. ./checkpoint/o_300_epochs
2. ./checkpoint/x_300_epochs



Task의 목표는 Image-to-Image Translation를 통해,  before 사진을 input으로 하여, after 사진을 예측하여 그리는 것이다.

Before 사진과 성공/실패 여부를 알려주면, 그 여부에 맞게 예측하여 아웃풋을 낸다.


test results

o example 사진
x example 사진








## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN


## train/test

|        |  o_images_final  |  x_images_final |  
|-------:|:----------------:|----------------:|  
|  train |       548        |             487 |  
|    val |        69        |              61 |  
|   test |        69        |              61 |

-   To view training results and loss plots, run `python -m visdom.server` and click the URL [http://localhost:8097](http://localhost:8097/).
- Train a model:
```bash
python train.py --dataroot ./cell_images/o_images_final --name o_300_epochs --n_epochs 300 --n_epochs_decay 0 --model pix2pix --direction AtoB --gpu_ids 2
```

```bash
python train.py --dataroot ./cell_images/x_images_final --name x_300_epochs --n_epochs 300 --n_epochs_decay 0 --model pix2pix --direction AtoB --gpu_ids 2
```
To see more intermediate results, check out  `./checkpoints/facades_pix2pix/web/index.html`.
- jupyter notebook 사용자들은 checkpoint_display_img.ipynb를 통해 results를 볼 수 있다.  
- train이 다 되고, 마지막 pth파일만 남기고 싶으면, delete_pth_file.ipynb를 이용 할 수 있다.

- Test the model:
```bash
python test.py --dataroot ./cell_images/o_images_final --name o_300_epochs --model pix2pix --direction AtoB --gpu_ids 2
```

```bash
python test.py --dataroot ./cell_images/x_images_final --name x_300_epochs --model pix2pix --direction AtoB --gpu_ids 2
```
- The test results will be saved to a html file here: `./results/facades_pix2pix/o_images_final/index.html`. You can find more scripts at `scripts` directory.
- jupyter notebook 사용자들은 test_display_img.ipynb를 통해 results를 볼 수 있다.




## Citation
**CycleGAN: [Project](https://junyanz.github.io/CycleGAN/) | [Paper](https://arxiv.org/pdf/1703.10593.pdf) 

**Pix2pix: [Project](https://phillipi.github.io/pix2pix/) | [Paper](https://arxiv.org/pdf/1611.07004.pdf) 

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.  
[Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)*, [Taesung Park](https://taesung.me/)*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In ICCV 2017. (* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)

Image-to-Image Translation with Conditional Adversarial Networks.  
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In CVPR 2017. [[Bibtex]](https://www.cs.cmu.edu/~junyanz/projects/pix2pix/pix2pix.bib)
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```