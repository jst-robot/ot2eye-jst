# ot2eye

## About ot2eye

- Opentrons OT-2 の底面の画像からラボウェアの位置を検出する
- 検出した位置をbounding boxとして元の画像に重ねた画像を出力する
- チップラックがあった場合、チップの位置を検出する



## Tutorial

### Installation

```
$ git clone --recursive git@github.com:bioinfo-tsukuba/ot2eye.git
$ cd ot2eye/yolov5/
$ python3 -m pip install -r requirements.txt 
```

### Confirmation of Operation

~~~~
$ cd ot2eye/
$ python3 ot2eye.py dataset/20220718_large/
~~~~

It will take some time to complete. The installation is successful if the "out" directory is created without any errors. The "out" directory contains label files and images including bounding boxes for the detected labware.



## Usage of Labware Detection

### Execution Command

~~~~
$ python3 ot2eye.py <image_dir>
~~~~

Arguments

* image_dir: Directory path of the images for which you want to detect labware. Multiple images can be detected at once.

Options

| Option               | Explanation                                                  | Default                                       |
| -------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| --out-dir            | Directory path of the output files.                          | out                                           |
| --model-labware      | YOLO model for detecting labware other than tips.            | model/detect_labware_20220624/weights/best.pt |
| --model-tip          | YOLO model for detecting tips.                               | model/detect_tip_20220624/weights/best.pt     |
| --threshold          | Threshold for class determination. Same as option "conf" in YOLO v5's detect.py. | 0.7                                           |
| --labware-train-yaml | Path of the yaml file used for training of labware other than tip detection in YOLO. | model/dataset_20220624_small_notip.yaml       |



### Required files

* 



### output files

output

ot2eye/out/

out1, out2, ...



### Example

~~~~
$ python3 ot2eye.py dataset/20220718_large
~~~~



### Requirement



## How to detect labware



## Creating a detection model



## File structure

ot2eye/  
    |---- model/  
        |---- detect_labware_20220624/  
        |---- detect_tip_20220624/  
        |---- dataset_20220624_small_notip.yaml  
    |---- scripts/  
        |---- trim_tip_rack.py  
        |---- obj_rec_eval.py  
    |---- yolov5/  
    |---- LICENSE  
    |---- README.md  
    |---- ot2eye.py  



## Evaluation mode

The accuracy of labware detection can be evaluated numerically.

### Required files

* To evaluate the accuracy of the detection, Answer label files are required.
  Answer label files. This must be in the same directory and in the same format as the YOLO training data. Refer to "1.2 Create Labels" on this page  (https://docs.ultralytics.com/yolov5/train_custom_data/#11-collect-images.) for details on the format of the YOLO training data.
* Put "classes.txt" in the same directory as the answer label files. This is usually generated automatically when the label files are created.

### Commands

If used evaluation mode, it is specified by a command option.

```
$ python3 ot2eye.py <img_dir> --evaluate <answer_labels> 
```

* img_dir: **説明追加**
* answer labels: directory path of the answer labels and classes.txt

### Output files

* evaluation.csv
  * example
  
    | #Image_file | #Labware | #Recall | #Precision | #Precision |
    | ----------- | -------- | ------- | ---------- | ---------- |
    |             |          |         |            |            |
    |             |          |         |            |            |
    |             |          |         |            |            |
    |             |          |         |            |            |
  
    
  * #Image_file #Labware #Recall #Precision #Precision
  * **例（表）**
* images_evaluation/ (image_files)
  * An answer bbox surrounded by a dashed rectangle is added to the detection result image. The label names of the answer bboxes other than the tip are displayed in the lower right corner. 
  * The extension is the same as that of the input image.
  * **例の画像**

### Specifications

* 評価値の計算方法．bboxの被った判定のしかた．nanがでるとき．csvの形式の説明
* 



## Required library

- OpenCV

  ```
  $ python3 -m pip install opencv-python
  $ python3 -m pip install opencv-contrib-python
  ```

- argparse

- subprocess

- csv

- yaml

- shutil

- glob





### Usage Enviroment

* 



## ドキュメントに書くこと

- 画像に対する想定
  - 画像に対する想定現在のモデルやソフトウェアでは、入力として、Opentrons OT-2 の真上から撮影した画像を想定しています。
  - この画像は斜めになっていないことが想定されています（the images is assumed not to be titled.）



ユーザー名データ等が残らないようにする（yamlには残ってるから，履歴が公開されないように公開）


