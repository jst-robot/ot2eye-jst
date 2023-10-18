import os
from os import sep
import sys
import time
from datetime import datetime as dt
import subprocess
import argparse
import cv2
from scripts.obj_rec_eval import Obj_Rec_Eval
from ot2eye import OT2Eye


class OT2Eye_WebCam():
	def __init__(self, args):
		cam_number = args.camera_number
		img_dir_ori = "record" # 画像保存ディレクトリ
		interval = 10 # 撮影間隔

		# カメラ設定
		self.cam_setting(cam_number, 1920, 1080, 30, 0)


		# 撮影開始
		try:
			start_time=time.time()
			while True:
				ret, frame = self.cap.read()  # 画像を取得

				if not ret:  # 画像取得が失敗した場合
					print('fail to get image')
					break

				# 保存
				if time.time() - start_time >= interval:
					start_time = time.time()

					# 保存ディレクトリ生成
					img_dir = self.make_output_dir(img_dir_ori)

					# 画像保存
					out_img_file_name = dt.now().strftime("%Y%m%d%H%M%S")+".jpeg"
					cv2.imwrite(img_dir+sep+out_img_file_name, frame)

					# ot2eye実行
					ot2eye = OT2Eye(img_dir, args.out_dir,\
							args.model_labware, args.model_tip, args.threshold,\
							args.labware_train_yaml, args.evaluate)

					# 評価モード
					if args.evaluate != None:
						obj_rec_eval = Obj_Rec_Eval(\
								ot2eye.out_dir,\
								ot2eye.out_dir+sep+"labels_merged",\
								ot2eye.out_dir+sep+"images_merged",\
								args.evaluate)

				cv2.imshow("web_cam", cv2.resize(frame, (640,480)))
				if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' キーが押されたらループを終了
					break

		except KeyboardInterrupt:
			print("Finish monitoring")


	def make_output_dir(self, out_dir):
		MAX_SEQ_NUM = 1024 # 最大連番数

		if not os.path.isdir(out_dir): #存在しなければ生成
			os.mkdir(out_dir)
			print("The directory \"{}\" was made.".format(out_dir))
		else: #存在すれば連番で代わりを生成
			while out_dir[-1]==sep:
				out_dir = out_dir[:-1] # ディレクトリパスの最後に"/"があれば消す
			# 連番ディレクトリ名存在確認
			for i in range(1,MAX_SEQ_NUM):
				out_dir_cdd = out_dir + str(i)
				if not os.path.isdir(out_dir_cdd):
					os.mkdir(out_dir_cdd)
					print("The directory \"{}\" already exists. \"{}\" was made instead.".format(out_dir, out_dir_cdd))
					out_dir = out_dir_cdd
					break

		# ディレクトリ名戻し
		return out_dir

	def cam_setting(self, num, width, height, fps, auto_focus):
		# self.auto_focus(auto_focus, f'/dev/video{num}')

		self.cap = cv2.VideoCapture(num)  # 0はデフォルトカメラのインデックス
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)   # 幅設定
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) # 高さ設定
		self.cap.set(cv2.CAP_PROP_FPS, fps)  # FPS設定

	def auto_focus(self, on_off, device):
		subprocess.run(['v4l2-ctl', '-d', device, '-c', f'focus_auto={on_off}'])



if __name__ == '__main__':
	#
	# argument
	#
	prs = argparse.ArgumentParser()
	# image dir to detect
	# prs.add_argument("image_dir", type=str,
	# 		help="directory path of image files.")
	# camera number
	prs.add_argument("--camera-number", type=int, required=False, default=0,
			help="number of web cam")
	# output dir path
	prs.add_argument("--out-dir", type=str, required=False, default="out",
			help="directory path of output files.")
	# labware detection model
	prs.add_argument("--model-labware", type=str, required=False,
			default="."+sep+"model"+sep+"detect_labware_20220624"+sep+"weights"+sep+"best.pt",
			help="file path of labware detection model.")
	# tip detection model
	prs.add_argument("--model-tip", type=str, required=False,
			default="."+sep+"model"+sep+"detect_tip_20220624"+sep+"weights"+sep+"best.pt",
			help="file path of labware detection model.")
	# threshold of detection
	prs.add_argument("--threshold", type=float, required=False,
			default=0.7,
			help="confidence threshold of detection. (default: 0.7)")
	# yaml file of training labware
	prs.add_argument("--labware-train-yaml", type=str, required=False,
			default="."+sep+"model"+sep+"dataset_20220624_small_notip.yaml",
			help="yaml file path of training labware.")
	# evaluation mode
	prs.add_argument("--evaluate", type=str, required=False, default=None,
			help="directory path of answer labels file.")
	args = prs.parse_args()

	#
	# detection
	#
	ot2eye_webcam = OT2Eye_WebCam(args)


	#
	# evaluation mode
	#


