
# YOLOv9を使用して領収証認識システムを構築する 

## Summary

<div style="max-width: 600px; word-wrap: break-word;">


進化し続ける人工知能（AI）と機械学習の分野において、領収書から重要な情報を効率的かつ正確に抽出する能力は、ビジネスにとってますます重要になっています。本記事では、YOLO v9モデルを使用して、スーパーでの買い物領収書を認識するシステムの構築方法について説明します。この記事では、約130枚の領収書を収集し、3つのラベル（クラス）を付けてデータをアノテーションし、モデルをトレーニングしました。その結果、YOLO v9モデルは指定した3つのクラスを正確に認識することができました

このプロジェクトは、YOLOv9を参考にして作成されています。詳細については以下のリポジトリをご参照ください：
https://github.com/WongKinYiu/yolov9

</div>



## はじめに

### YOLO v9について
<div style="max-width: 600px; word-wrap: break-word;">

YOLOは「You Only Look Once」の略で、速度と精度の高さで知られる最先端のオブジェクト検出モデルです。このモデルの第9バージョン（v9）は、「Programmable Gradient Information（PGI）」や「Generalized Efficient Layer Aggregation Network（GELAN）」といった新しいアーキテクチャを導入し、機能と性能がさらに強化されています。YOLO v9は、モデルの学習能力を向上させるだけでなく、検出プロセス全体で重要な情報を保持することを可能にし、卓越した精度と性能を実現しています。この記事はYOLO v9モデルを使用して、スーパーでの買い物領収書を認識するシステムの構築

</div>


## 準備

### 環境を選ぶ
<div style="max-width: 600px; word-wrap: break-word;">

#### トーレニングはGoogle Colabを使用します。

Google Colabは、GPUへの無料アクセスを提供するクラウドベースのプラットフォームであり、ディープラーニングモデルのトレーニングに最適な選択肢です。


#### データ準備

データの準備は最初のステップです。アノテーションツールにはいくつかの選択肢がありますが、その中でもLabelImgやLabelmeを使用した経験があります。ただし、Roboflowは非常に使いやすいツールです。今回は135枚の領収書を収集し、それらをRoboflowを使ってアノテーションしました。

Roboflowについて
Roboflowはデータセットのアノテーションを簡単かつ迅速に行い、さまざまなモデルのトレーニングに適した形式に変換できるソリューションを提供しています。無料で利用可能なRoboflowのパブリックバージョンを使用することができます。https://app.roboflow.com/

<div align="medium">
    <img src="images/dataset1.png" alt="YOLO" width="100%">
</div>


ここでは、135枚の領収書サンプルをRoboflowアカウントにアップロードしてアノテーションを行います。このプロジェクトでは、領収書から以下の3つのラベルを抽出します：『会社ロゴ』、『領収書合計』、『領収書の日付』、『店の電話』。そのため、それぞれのラベル名を入力し、各領収書サンプルのアノテーションを作ります。


<div align="medium">
    <img src="images/dataset2.png" alt="YOLO" width="100%">


アノテーションが完了したら、いくつかの前処理を行います。例えば、データセットを訓練用、検証用、テスト用に分割することや、画像の拡張（既存の画像から新しい訓練用サンプルを作成するプロセス）を行います。拡張手法としては、彩度の調整、反転、グレースケール変換などがあります。また、画像を640x640サイズにリサイズします（現在のYOLO v9は640x640サイズの画像しか処理できないため、このステップは必須です）。

これらの前処理を行った結果、データセットの87枚が訓練用、26枚を検証用、12枚はテスト用として準備されました。


<div align="medium">
    <img src="images/dataset3.png" alt="YOLO" width="100%">

</div>


データセットの作成が完了したら、Roboflowの「Export Dataset」オプションを使用して、YOLO v9モデル用のデータセットをエクスポートできます。エクスポートしたデータセットコードを取得し、このコードをColab上で使用してモデルの訓練を行います。


## Google Colabでトーレニングの流れ

### Mount your Google Drive for storage

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Clone and Install

Clone YOLO v9 repository into your Google Drive.


```python
!git clone https://github.com/SkalskiP/yolov9.git
%cd yolov9
!pip install -r requirements.txt -q
```

**NOTE:** Install roboflow package for importing dataset


```python
!pip install -q roboflow
```

## Imports

```python
import roboflow

from IPython.display import Image
```

## Download model weights

Download available YOLO model weights from the github repo


```python
!mkdir -p {HOME}/weights
```


```python
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt
```


```python
!ls -la {HOME}/weights
```


## Download the Dataset

**NOTE:** The dataset must be saved inside the `{HOME}/yolov9` directory, otherwise, the training will not succeed. Paste your YOLO v9 export dataset code from the roboflow here


```python
%cd {HOME}/yolov9


!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YMynYD3vFskFi5oZwlSk")
project = rf.workspace().project("invoice_extraction-9emnx")
version = project.version(3)
dataset = version.download("yolov9")
```

    


## Train YOLO Model on Invoices

We are training our dataset with the gelan-c model. Before training, please access the gelan-c.yaml file and change the no. of anchors into the labels you need to train. Here we are training for only 4 labels so change anchors into 4


```python
%cd {HOME}/yolov9

!python train.py \
--batch 8 --epochs 100 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data invoice_extraction-3/data.yaml \
--weights {HOME}/weights/gelan-c.pt \
--cfg models/detect/gelan-c.yaml \
--hyp hyp.scratch-high.yaml
```



```python
%cd {HOME}/yolov9

!python train.py \
--batch 8 --epochs 50 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data invoice_extraction-3/data.yaml \
--weights {HOME}/weights/gelan-c.pt \
--cfg models/detect/gelan-c.yaml \
--hyp hyp.scratch-high.yaml
```



## Examine Training Results


```python
!ls {HOME}/yolov9/runs/train/exp/
```

    confusion_matrix.png				    labels_correlogram.jpg  PR_curve.png  weights
    events.out.tfevents.1731123877.a7d028d62c1f.3069.0  labels.jpg		    R_curve.png
    F1_curve.png					    opt.yaml		    results.csv
    hyp.yaml					    P_curve.png		    results.png



```python
Image(filename=f"{HOME}/yolov9/runs/train/exp2/results.png", width=1000)
```




    
![png](output_28_0.png)
    




```python
Image(filename=f"{HOME}/yolov9/runs/train/exp2/confusion_matrix.png", width=1000)
```




    
![png](output_29_0.png)
    




```python
Image(filename=f"{HOME}/yolov9/runs/train/exp2/labels.jpg", width=1000)
```




    
![jpeg](output_30_0.jpg)
    



## Validate Trained Model


```python
%cd {HOME}/yolov9

!python val.py \
--img 640 --batch 8 --conf 0.50 --iou 0.7 --device 0 \
--data invoice_extraction-3/data.yaml \
--weights {HOME}/yolov9/runs/train/exp2/weights/best.pt
```

    /content/yolov9
    [34m[1mval: [0mdata=invoice_extraction-3/data.yaml, weights=['/content/yolov9/runs/train/exp2/weights/best.pt'], batch_size=8, imgsz=640, conf_thres=0.5, iou_thres=0.7, max_det=300, task=val, device=0, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=False, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False, min_items=0
    WARNING ⚠️ confidence threshold 0.5 > 0.001 produces invalid results
    YOLOv5 🚀 1e33dbb Python-3.10.12 torch-2.5.0+cu121 CUDA:0 (Tesla T4, 15102MiB)
    
    /content/yolov9/models/experimental.py:75: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
    Fusing layers... 
    gelan-c summary: 467 layers, 25412502 parameters, 0 gradients, 102.5 GFLOPs
    [34m[1mval: [0mScanning /content/yolov9/valid/labels.cache... 10 images, 0 backgrounds, 0 corrupt: 100% 10/10 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  50% 1/2 [00:01<00:01,  1.68s/it]Exception in thread Thread-3 (plot_images):
    Traceback (most recent call last):
      File "/usr/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
      File "/usr/lib/python3.10/threading.py", line 953, in run
        self._target(*self._args, **self._kwargs)
      File "/content/yolov9/utils/plots.py", line 300, in plot_images
        annotator.box_label(box, label, color=color)
      File "/content/yolov9/utils/plots.py", line 86, in box_label
        w, h = self.font.getsize(label)  # text width, height
    AttributeError: 'FreeTypeFont' object has no attribute 'getsize'
    Exception in thread Thread-4 (plot_images):
    Traceback (most recent call last):
      File "/usr/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
      File "/usr/lib/python3.10/threading.py", line 953, in run
        self._target(*self._args, **self._kwargs)
      File "/content/yolov9/utils/plots.py", line 300, in plot_images
        annotator.box_label(box, label, color=color)
      File "/content/yolov9/utils/plots.py", line 86, in box_label
        w, h = self.font.getsize(label)  # text width, height
    AttributeError: 'FreeTypeFont' object has no attribute 'getsize'
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:02<00:00,  1.03s/it]
                       all         10         20          1       0.75      0.875      0.688
                      Date         10         10          1        0.9       0.95      0.726
                    Telnum         10         10          1        0.6        0.8      0.651
    Speed: 0.3ms pre-process, 69.2ms inference, 92.7ms NMS per image at shape (8, 3, 640, 640)
    Exception in thread Thread-5 (plot_images):
    Traceback (most recent call last):
      File "/usr/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
      File "/usr/lib/python3.10/threading.py", line 953, in run
        self._target(*self._args, **self._kwargs)
      File "/content/yolov9/utils/plots.py", line 300, in plot_images
        annotator.box_label(box, label, color=color)
      File "/content/yolov9/utils/plots.py", line 86, in box_label
        w, h = self.font.getsize(label)  # text width, height
    AttributeError: 'FreeTypeFont' object has no attribute 'getsize'
    Exception in thread Thread-6 (plot_images):
    Traceback (most recent call last):
      File "/usr/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
      File "/usr/lib/python3.10/threading.py", line 953, in run
        self._target(*self._args, **self._kwargs)
      File "/content/yolov9/utils/plots.py", line 300, in plot_images
        annotator.box_label(box, label, color=color)
      File "/content/yolov9/utils/plots.py", line 86, in box_label
        w, h = self.font.getsize(label)  # text width, height
    AttributeError: 'FreeTypeFont' object has no attribute 'getsize'
    Results saved to [1mruns/val/exp[0m


## Evaluate Model Performance on test image

---




```python
!python detect.py \
--img 640 --conf 0.5 --device 0 \
--weights {HOME}/yolov9/runs/train/exp2/weights/best.pt \
--source {HOME}/yolov9/valid/images
```

    [34m[1mdetect: [0mweights=['/content/yolov9/runs/train/exp2/weights/best.pt'], source=/content/yolov9/valid/images, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.5, iou_thres=0.45, max_det=1000, device=0, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1
    YOLOv5 🚀 1e33dbb Python-3.10.12 torch-2.5.0+cu121 CUDA:0 (Tesla T4, 15102MiB)
    
    /content/yolov9/models/experimental.py:75: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
    Fusing layers... 
    gelan-c summary: 467 layers, 25412502 parameters, 0 gradients, 102.5 GFLOPs
    image 1/10 /content/yolov9/valid/images/20240821_093350_jpg.rf.e3a2367318e9fb1a89ef898811d70b67.jpg: 640x640 1 Date, 1 Telnum, 50.9ms
    image 2/10 /content/yolov9/valid/images/20240821_093451_jpg.rf.e35a6b18d3877f64908e994349f8c874.jpg: 640x640 1 Date, 1 Telnum, 41.6ms
    image 3/10 /content/yolov9/valid/images/20240821_093930_jpg.rf.d0053590bf32d74b43269a085b34596d.jpg: 640x640 (no detections), 41.6ms
    image 4/10 /content/yolov9/valid/images/20240821_094025_jpg.rf.cdfa5a90107b32da1ee9199b58496d3a.jpg: 640x640 1 Date, 1 Telnum, 41.5ms
    image 5/10 /content/yolov9/valid/images/20240930_161302_jpg.rf.8027f70ce3d1f342f527c2888f096ed2.jpg: 640x640 1 Date, 1 Telnum, 32.9ms
    image 6/10 /content/yolov9/valid/images/20240930_161345_jpg.rf.88266e8d4f9c87312f72e5ee3023f23c.jpg: 640x640 1 Date, 1 Telnum, 27.7ms
    image 7/10 /content/yolov9/valid/images/20240930_161501_jpg.rf.e656eece8574ddfd27931a064f017d22.jpg: 640x640 1 Date, 1 Telnum, 27.8ms
    image 8/10 /content/yolov9/valid/images/20240930_161547_jpg.rf.a8d8bbf2ee8b55cfc87c96eb7d66aad8.jpg: 640x640 1 Date, 27.7ms
    image 9/10 /content/yolov9/valid/images/20240930_161606_jpg.rf.62e27ac42fb4954b91eefdce17c4b6a3.jpg: 640x640 1 Date, 27.6ms
    image 10/10 /content/yolov9/valid/images/20240930_161636_jpg.rf.e26ae1d08416b4bbee49682c84d20b9d.jpg: 640x640 1 Date, 27.5ms
    Speed: 0.5ms pre-process, 34.7ms inference, 55.1ms NMS per image at shape (1, 3, 640, 640)
    Results saved to [1mruns/detect/exp2[0m



```python
import glob
from IPython.display import Image, display

for image_path in glob.glob(f'{HOME}/yolov9/runs/detect/exp2/*.jpg')[:3]:
      display(Image(filename=image_path, width=600))
      print("\n")
```


    
![jpeg](output_35_0.jpg)
    


    
    



    
![jpeg](output_35_2.jpg)
    


    
    



    
![jpeg](output_35_4.jpg)
    


    
    



```python

```


```python

```
