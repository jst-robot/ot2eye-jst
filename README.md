# ot2eye



## インストール

```
git clone --recursive git@github.com:bioinfo-tsukuba/ot2eye.git
cd ot2eye/yolov5/
python3 -m pip install -r requirements.txt 
```



## 必要ライブラリ

- PyTorch

  ```
  python3 -m pip install torch
  ```

- OpenCV

  ```
  python3 -m pip install opencv-python
  python3 -m pip install opencv-contrib-python
  ```

- csv

- yaml

- argparse



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

* 107~113行目をコメントアウトし，114~119行目追加

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

  

```
cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
```

↓