# DAC-Net (BMVC 2021 Oral Presentation)

Pytorch implementation of DAC-Net (["Zhongying Deng, Kaiyang Zhou, Yongxin Yang, Tao Xiang. Domain Attention Consistency for Multi-Source Domain Adaptation. BMVC 2021"](https://www.bmvc2021-virtualconference.com/assets/papers/0353.pdf))

## Installation

Please install the [Dassl codebase](https://github.com/KaiyangZhou/Dassl.pytorch#get-started), and then copy the files in this repository to Dassl. 
When asked to overwrite, say yes (some `__init__.py` files may be overwritten, these files is modified to include the backbone or trainer used for DAC-Net).
Then you will install `pytorch 1.7.1 + cuda 10.1, python 3.7`.

## Training

Create a folder like `output/dacnet_pacs` (under the Dassl root path) where checkpoint and log can be saved.

Then
```bash
bash train_dacnet.sh /path/to/your/dataset
```

Run the above bash script, then the experiments on PACS will be running. In the script, `$DATA` denotes the location where datasets are installed. For experiments on Digit-Five and DomainNet, modify the `--source-domains`, `--target-domains` and related config files, such as `--dataset-config-file configs/datasets/da/digit5_ca.yaml` and `--config-file configs/trainers/da/dacnet/digit5.yaml`.

The detailed training settings are in the folder named `configs`, such as datasets and backbone name used for DAC-Net (see `configs/datasets/da`), and lr, optimizer etc. (see `configs/trainers/da/dacnet`)

Some important files are under the folder of `dassl`: 
* Implementation of DAC-Net can be found in `dassl/engine/da/dacnet.py`;
* The backbone CNN model of our DAC-Net can be found in `dassl/modeling/backbone/resnet_ca.py` (for PACS and DomainNet where ResNet is adopted as backbone) and `dassl/modeling/backbone/cnn_digit5_m3sda_ca.py` (for Digit-Five);
* Config definition for DAC-Net can be found in `dassl/config/defaults.py` (see last 5 lines);

Trained model on Sketch domain of PACS can be found [here](https://drive.google.com/file/d/1VLWc0K9WVC4Nx6ZZ4uKM_YmCn_s4PjgO/view?usp=sharing). This model gives 84.88% on the Sketch domain.

## Test

Similar to `train_dacnet.sh`, testing can be done like this:

```
DATA=/root_path/to/your/dataset
CUDA_VISIBLE_DEVICES=0 python tools/train.py --root $DATA --trainer DACNet \
 --source-domains cartoon art_painting photo --target-domains sketch \
 --dataset-config-file configs/datasets/da/pacs_ca.yaml --config-file configs/trainers/da/dacnet/pacs.yaml \
 --output-dir output/dacnet_pacs/sketch \
 --eval-only \
 --model-dir output/dacnet_pacs/sketch \
 --load-epoch 30 \
 MODEL.INIT_HEAD_WEIGHTS output/dacnet_pacs/sketch/classifier/model.pth.tar-30
```

## Citation

If you find this code useful, please consider citing the following paper:
```
@article{deng2021domain,
  title={Domain Attention Consistency for Multi-Source Domain Adaptation},
  author={Deng, Zhongying and Zhou, Kaiyang and Yang, Yongxin and Xiang, Tao},
  journal={arXiv preprint arXiv:2111.03911},
  year={2021}
}
```
