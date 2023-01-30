# ot2eye

## About ot2eye

- Opentrons OT-2 の底面の画像からラボウェアの位置を検出する
- 検出した位置をbounding boxとして元の画像に重ねた画像を出力する
- チップラックがあった場合、チップの位置を検出する



## Example

input

output

画像



## installation

```
$ git clone --recursive git@github.com:bioinfo-tsukuba/ot2eye.git
$ cd ot2eye/yolov5/
$ python3 -m pip install -r requirements.txt 
```



## Usage

### Detect mode

execution command

~~~~
$ python3 ot2eye.py <image_dir>
~~~~

output

ot2eye/out/

out1, out2, ...

option

- --out-dir
  - directory path of output files.
  - defailt: "out"
- --model-labware
- --model-tip
- --threshold
- --labware-train-yaml

~~~~
$ python3 ot2eye.py <>
~~~~



### example

~~~~
$ python3 ot2eye.py dataset/20220718_large
~~~~





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



## Required library

- OpenCV

  ```
  $ python3 -m pip install opencv-python
  $ python3 -m pip install opencv-contrib-python
  ```

- csv

- yaml

- argparse





### Usage Enviroment

* 



## ドキュメントに書くこと

- 画像に対する想定
  - 画像に対する想定現在のモデルやソフトウェアでは、入力として、Opentrons OT-2 の真上から撮影した画像を想定しています。
  - この画像は斜めになっていないことが想定されています（the images is assumed not to be titled.）





## 開発メモ

### 出力形式

- out/
  - images_labware/
    - hoge.jpeg
      ※非リサイズ画像にラボウェアとチップのbbox
      ※bboxはラベルと確率あり（ただしチップはラベル無し）
    - fuga.jpeg
  - labels_labware/
    - hoge.txt
      ※ラボウェアとチップを統合したラベル
    - fuga.txt
  - images_tip/
    - hoge_0.jpeg
    - hoge_1.jpeg
    - fuga_0.jpeg
  - labels_tip/
    - hoge_0.txt
    - hoge_1.txt
    - fuga_0.txt
  - 評価.csv





## 変更点

yolov5/utils/plots.py

参考：https://qiita.com/fujioka244kogacity/items/e5acf8d9bb728e7d1bcc

* 99行目をコメントアウトし，100行目追加

  ```
  cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
  ```

  ↓

  ```
  cv2.rectangle(self.im, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
  ```

* 106行目コメントアウト

  ```
  cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
  ```

  ↓

  ```
  # cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
  ```

* 107〜113行目をコメントアウトし，114〜119行目追加

  ```
  cv2.putText(self.im,
              label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
              0,
              self.lw / 3,
              txt_color,
              thickness=tf,
              lineType=cv2.LINE_AA)
  ```

  ↓

  ```
  cv2.putText(self.im,
              label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
              cv2.FONT_HERSHEY_SIMPLEX,
              self.lw / 6,
              txt_color,
              thickness=1)
  ```

  