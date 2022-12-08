#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import argparse
import cv2
import shutil
from glob import glob
from scripts.trim_tip_rack import Trim_Tip_Rack


class OT2Eye():
	def __init__(self, img_dir, out_dir, model_labware, model_tip, threshold, train_yaml):
		#
		# パラメータ
		#
		# img_dir: 推論画像ディレクトリ
		# out_dir: 結果保存ディレクトリ
		# model_labware: ラボウェアの検出モデル
		# model_tip: チップの検出モデル
		# threshold: 検出閾値
		# train_yaml: 訓練データに用いたyamlファイル
		WIDTH_SMALL = 640 # 縮小後画像幅
		HIGHT_SMALL = 480 # 縮小後画像高
		TIP_RACK_LABEL_NAME = "tip_rack"
		OUT_DIR_TMP = "tmp"
		OUT_DIR_DETECT = "detect"
		self.OUT_DIR_IMG_RESIZE = OUT_DIR_TMP+"/"+"images_resize" #縮小画像保存ディレクトリ
		self.OUT_DIR_DETECT_LABWARE = OUT_DIR_TMP+"/"+"detect_labware" #ラボウェア検出結果ディレクトリ
		self.OUT_DIR_IMG_TRIM = OUT_DIR_TMP+"/"+"image_trim_tip_rack"# チップラックトリミング画像ディレクトリ
		self.OUT_DIR_DETECT_TIP = OUT_DIR_TMP+"/"+"detect_tip" #チップ検出結果ディレクトリ


		#
		# ディレクトリ処理
		#
		# 入力画像ディレクトリ存在確認
		if not os.path.isdir(img_dir):
			print("No such directory \"{}\"".format(img_dir))
			return

		# 保存先ディレクトリ処理
		if not os.path.isdir(out_dir): #存在しなければ生成
			os.mkdir(out_dir)
		else: #存在すれば連番で代わりを生成
			# ディレクトリパスの最後に"/"があれば消す
			while out_dir[-1]=="/":
				out_dir = out_dir[:-1]
			# 連番ディレクトリ名存在確認
			for i in range(1,256):
				out_dir_cdd = out_dir+str(i)
				if not os.path.isdir(out_dir_cdd):
					os.mkdir(out_dir_cdd)
					print("The directory \"{}\" already exists. \"{}\" was made instead.".format(out_dir, out_dir_cdd))
					out_dir = out_dir_cdd
					break
		# 各種サブ出力ディレクトリ生成
		os.mkdir(out_dir+"/"+OUT_DIR_TMP)
		os.mkdir(out_dir+"/"+OUT_DIR_DETECT)


		#
		# 画像の縮小＆保存
		#
		self.resize_save_img(img_dir, out_dir, WIDTH_SMALL)


		#
		# リサイズ画像からラボウェア検出
		#
		subprocess.run(["python3", "yolov5/detect.py",\
				# 検出対象画像ディレクトリ
				"--source", out_dir+"/"+self.OUT_DIR_IMG_RESIZE,\
				# 検出結果出力先ディレクトリ
				"--project", out_dir,\
				# 検出結果ディレクトリ名
				"--name", self.OUT_DIR_DETECT_LABWARE,\
				# 検出モデル
				"--weights", model_labware,\
				# 検出閾値
				"--conf", str(threshold),\
				# 推論結果ラベル出力
				"--save-txt",\
				# 推論結果確率出力
				"--save-conf",\
				# 検出結果上書き
				"--exist-ok"])


		#
		# 検出結果からトリミング画像生成
		#
		if not os.path.isdir(out_dir+"/"+self.OUT_DIR_IMG_TRIM): #存在しなければ生成
			os.mkdir(out_dir+"/"+self.OUT_DIR_IMG_TRIM)
		trim = Trim_Tip_Rack(\
				# トリミング前画像ディレクトリ
				img_dir,\
				# ラボウェア検出結果ラベルディレクトリ
				out_dir+"/"+self.OUT_DIR_DETECT_LABWARE+"/labels/",\
				# 学習時yamlファイル
				train_yaml,\
				# トリミング結果画像ディレクトリ
				out_dir+"/"+self.OUT_DIR_IMG_TRIM+"/",\
				# チップラックのラベル名
				TIP_RACK_LABEL_NAME,\
				# トリミング結果画像幅
				WIDTH_SMALL,\
				# トリミング結果画像高
				HIGHT_SMALL)


		#
		# トリミング画像からチップ検出
		#
		subprocess.run(["python3", "yolov5/detect.py",\
				# 検出対象画像ディレクトリ
				"--source", out_dir+"/"+self.OUT_DIR_IMG_TRIM,\
				# 検出結果出力先ディレクトリ
				"--project", out_dir,\
				# 検出結果ディレクトリ名
				"--name", self.OUT_DIR_DETECT_TIP,\
				# 検出モデル
				"--weights", model_tip,\
				# 検出閾値
				"--conf", str(threshold),\
				# 推論結果ラベル出力
				"--save-txt",\
				# 推論結果確率出力
				"--save-conf",\
				# 検出結果上書き
				"--exist-ok"])


		#
		# 総合検出結果出力
		#
		# 画像と同じ名前のディレクトリ名を生成
		#	ラボウェアとチップのラベルファイル
		#	ラボウェアbboxとチップの*を合わせた画像生成
		#	チップラックを抜き出したラベルファイル
		#	チップラックをbbox画像

		imgs = os.listdir(path=img_dir) # リサイズ前画像
		for num in range(len(imgs)):
			base_name = imgs[num].rsplit(".",1)[0]

			# ベース名のサブディレクトリ生成
			os.mkdir(out_dir+"/"+OUT_DIR_DETECT+"/"+base_name)

			#
			# ラボウェア検出ラベルファイル生成
			#
			# ラボウェア検出ラベルがなければスキップ
			if not os.path.isfile(out_dir+"/"+self.OUT_DIR_DETECT_LABWARE+"/"+"labels"+"/"+base_name+".txt"):
				# リサイズ前画像ファイルコピー
				shutil.copy(img_dir+"/"+imgs[num], out_dir+"/"+OUT_DIR_DETECT+"/"+base_name)
				continue
			# ラボウェア検出ラベルディレクトリ生成
			os.mkdir(out_dir+"/"+OUT_DIR_DETECT+"/"+base_name+"/labels_labware")
			# ラボウェア検出ラベルファイルコピー
			shutil.copy(out_dir+"/"+self.OUT_DIR_DETECT_LABWARE+"/"+"labels"+"/"+base_name+".txt",\
					out_dir+"/"+OUT_DIR_DETECT+"/"+base_name+"/labels_labware")

			#
			# チップ検出ラベルファイル生成
			#
			# チップ検出ラベルディレクトリ生成
			if 0 != len(glob(out_dir+"/"+self.OUT_DIR_DETECT_TIP+"/"+"labels"+"/"+base_name+"*")):
				os.mkdir(out_dir+"/"+OUT_DIR_DETECT+"/"+base_name+"/labels_tip")
			# チップ検出ラベルファイルコピー
			for tip_label in glob(out_dir+"/"+self.OUT_DIR_DETECT_TIP+"/"+"labels"+"/"+base_name+"*"):
				shutil.copy(tip_label, out_dir+"/"+OUT_DIR_DETECT+"/"+base_name+"/labels_tip")

			# リサイズ前画像ファイルコピー
			shutil.copy(img_dir+"/"+imgs[num], out_dir+"/"+OUT_DIR_DETECT+"/"+base_name)
			#
			# リサイズ前画像とラベルからbbox画像生成
			#



		# 検出ラベルファイルコピー
		# labels = os.listdir(path=out_dir+"/"+self.OUT_DIR_DETECT_LABWARE+"/"+"labels")
		# for num in range(len(labels)):
		# 	base_name = labels[num].rsplit(".",1)[0]
		# 	# ラボウェア検出ラベルファイルコピー
		# 	shutil.copy(out_dir+"/"+self.OUT_DIR_DETECT_LABWARE+"/"+"labels"+"/"+labels[num],\
		# 			out_dir+"/"+OUT_DIR_DETECT+"/"+base_name+"/labels_labware")
		# 	# チップ検出ラベルファイルコピー
		# 	for tipl in glob(out_dir+"/"+self.OUT_DIR_DETECT_TIP+"/"+"labels"+"/"+base_name+"*"):
		# 		shutil.copy(tipl, out_dir+"/"+OUT_DIR_DETECT+"/"+base_name+"/labels_tip")

		# チップ検出ラベルファイルコピー
		# labels = os.listdir(path=out_dir+"/"+self.OUT_DIR_DETECT_TIP+"/"+"labels")
		# for num in range(len(labels)):
		# 	base_name = labels[num].rsplit(".",1)[0]
		# 	shutil.copy(out_dir+"/"+self.OUT_DIR_DETECT_TIP+"/"+"labels"+"/"+base_name+"*",\
		# 			out_dir+"/"+OUT_DIR_DETECT+"/"+base_name)
		# labels = os.listdir(path=out_dir+"/"+self.OUT_DIR_DETECT_LABWARE+"/"+"labels")
		# for num in range(len(labels)):
		# 	base_name = labels[num].rsplit(".",1)[0]
		# 	shutil.copy(out_dir+"/"+self.OUT_DIR_DETECT_LABWARE+"/"+"labels"+"/"+labels,\
		# 			out_dir+"/"+OUT_DIR_DETECT+"/"+base_name)
			# チップ検出ラベルファイルコピー
			# 存在チェック
			# shutil.copy(out_dir+"/"+self.OUT_DIR_DETECT_TIP+"/"+"labels"+"/"+base_name+"_0.txt",\
			# 		out_dir+"/"+OUT_DIR_DETECT+"/"+base_name)




	# 画像の縮小＆保存
	def resize_save_img(self, img_dir, out_dir, width):
		imgs = os.listdir(path=img_dir)

		for num in range(len(imgs)):
			# read image
			img = cv2.imread(img_dir+"/"+imgs[num])
			# resize
			h, w = img.shape[:2]
			height = round(h * (width / w))
			img = cv2.resize(img, dsize=(width, height))
			# save
			dir_name = out_dir+"/"+self.OUT_DIR_IMG_RESIZE
			if not os.path.isdir(dir_name):
				os.mkdir(dir_name)
			cv2.imwrite(dir_name+"/"+imgs[num], img)



if __name__ == '__main__':
	# argument
	prs = argparse.ArgumentParser()
	# prs.add_argument("--model-dir", type=str, required=True,
	# 		help="directory path of detection model file.")
	prs.add_argument("--image-dir", type=str, required=True,
			help="directory path of image files.")
	prs.add_argument("--out-dir", type=str, required=True,
			help="directory path of output files.")
	args = prs.parse_args()

	# ot2eye = OT2Eye(args.image_dir, args.out_dir, "./model/detect_labware_20220624/weights/best.pt", "./model/detect_tip_20220624/weights/best.pt", 0.7, "./dataset_20220624_small_notip.yaml")
	ot2eye = OT2Eye(args.image_dir, args.out_dir, "../obj_rec/model/exp_20220624_small_notip/weights/best.pt", "./model/detect_tip_20220624/weights/best.pt", 0.7, "./model/dataset_20220624_small_notip.yaml")

	# example
	# python3 ot2eye.py --image-dir dataset/20220718_large/ --out-dir out
