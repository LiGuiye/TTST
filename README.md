# TTST (IEEE TIP 2024)
### 📖[**Paper**](https://ieeexplore.ieee.org/abstract/document/9530280) | 🖼️[**PDF**](/fig/TTST.pdf)

PyTorch codes for "[TTST: A Top-k Token Selective Transformer for Remote Sensing Image Super-Resolution](https://ieeexplore.ieee.org/abstract/document/9530280)", **IEEE Transactions on Image Processing (TIP)**, 2024.

- Authors: [Yi Xiao](https://xy-boy.github.io/), [Qiangqiang Yuan*](http://qqyuan.users.sgg.whu.edu.cn/), [Kui Jiang](https://homepage.hit.edu.cn/jiangkui?lang=zh), [Jiang He](https://jianghe96.github.io/), [Chia-Wen Lin](https://www.ee.nthu.edu.tw/cwlin/), and [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)<br>
- Wuhan University, Harbin Institute of Technology, and National Tsinghua University

## Abstract
> Transformer-based method has demonstrated promising performance in image super-resolution tasks, due to its long-range and global aggregation capability. However, the existing Transformer brings two critical challenges for applying it in large-area earth observation scenes: (1) redundant token representation due to most irrelevant tokens; (2) single-scale representation which ignores scale correlation modeling of similar ground observation targets. To this end, this paper proposes to adaptively eliminate the interference of irreverent tokens for a more compact self-attention calculation. Specifically, we devise a Residual Token Selective Group (RTSG) to grasp the most crucial token by dynamically selecting the top-k keys in terms of score ranking for each query. For better feature aggregation, a Multi-scale Feed-forward Layer (MFL) is developed to generate an enriched representation of multi-scale feature mixtures during feed-forward process. Moreover, we also proposed a Global Context Attention (GCA) to fully explore the most informative components, thus introducing more inductive bias to the RTSG for an accurate reconstruction. In particular, multiple cascaded RTSGs form our final Top-k Token Selective Transformer (TTST) to achieve progressive representation. Extensive experiments on simulated and real-world remote sensing datasets demonstrate our TTST could perform favorably against state-of-the-art CNN-based and Transformer-based methods, both qualitatively and quantitatively. In brief, TTST outperforms the state-of-the-art approach (HAT-L) in terms of PSNR by 0.14 dB on average, but only accounts for 47.26\% and 46.97\% of its computational cost and parameters.
## Network  
 ![image](/fig/network.png)
 
## 🧩 Install
```
git clone https://github.com/XY-boy/TTST.git
```

## Environment
 * CUDA 11.1
 * Python 3.9.13
 * PyTorch 1.9.1
 * Torchvision 0.10.1

## 🎁 Dataset
Please download the following remote sensing benchmarks:
| Data Type | [AID](https://captain-whu.github.io/AID/) | [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) | [DIOR](https://www.sciencedirect.com/science/article/pii/S0924271619302825) | [NWPU-RESISC45](https://ieeexplore.ieee.org/abstract/document/7891544)
| :----: | :-----: | :----: | :----: | :----: |
|Training | [Download](https://captain-whu.github.io/AID/) | None | None | None |
|Testing | [Download](https://captain-whu.github.io/AID/) | [Download](https://captain-whu.github.io/DOTA/dataset.html) | [Download](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC) | [Download](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp)
## 🧩 Usage
### Train
```
python train_4x.py
```
### Test
```
python eval_4x.py
```
## 🖼️ Results
### Quantitative
 ![image](/fig/red.png)

### Visual
 ![image](/fig/dota.png)

## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: xiao_yi@whu.edu.cn; Tel: (+86) 15927574475 (WeChat)

## Citation
If you find our work helpful in your research, please consider citing it. We appreciate your support！😊

```
@article{xiao2024ttst,
  title={TTST: A Top-k Token Selective Transformer for Remote Sensing Image Super-Resolution},
  author={Xiao, Yi and Yuan, Qiangqiang and Jiang, Kui and He, Jiang and Chia-Wen Lin and Zhang, Liangpei},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  volume={33},
  pages={1-15},
  doi={10.1109/TGRS.2023.3341437}
}
```
