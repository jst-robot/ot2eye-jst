#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import argparse
import csv
import yaml
import cv2
import shutil
from glob import glob
from scripts.trim_tip_rack import Trim_Tip_Rack
from scripts.obj_rec_eval import Obj_Rec_Eval


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
		DIR_TMP = "tmp" #出力一時ディレクトリ
		DIR_TMP_IMG_RESIZE =     DIR_TMP+"/"+"images_resize" #縮小画像保存ディレクトリ
		DIR_TMP_IMG_TRIM =       DIR_TMP+"/"+"images_trim"   #チップラックトリミング画像ディレクトリ
		DIR_TMP_DETECT_LABWARE = DIR_TMP+"/"+"detect_labware"#ラボウェア検出結果ディレクトリ
		DIR_TMP_DETECT_TIP =     DIR_TMP+"/"+"detect_tip"    #チップ検出結果ディレクトリ
		DIR_OUT_LABWARE_IMG = "images_labware"
		DIR_OUT_LABWARE_LBL = "labels_labware"
		DIR_OUT_TIP_IMG =     "images_tip"
		DIR_OUT_TIP_LBL =     "labels_tip"


		#
		# ディレクトリ処理
		#
		# 入力画像ディレクトリ存在確認
		if not os.path.isdir(img_dir):
			print("No such directory \"{}\"".format(img_dir))
			return
		# 保存先ディレクトリ生成
		out_dir = self.make_output_dir(out_dir)
		# 各種サブ出力ディレクトリ生成
		os.mkdir(out_dir+"/"+DIR_TMP)


		#
		# 画像の縮小＆保存
		#
		print("##########################")
		print("# generate resize images #")
		print("##########################")
		self.resize_save_img(img_dir, out_dir+"/"+DIR_TMP_IMG_RESIZE, WIDTH_SMALL)
		# print("# Success!")


		#
		# リサイズ画像からラボウェア検出
		#
		print("##################")
		print("# detect labware #")
		print("##################")
		subprocess.run(["python3", "yolov5/detect.py",\
				# 検出対象画像ディレクトリ
				"--source", out_dir+"/"+DIR_TMP_IMG_RESIZE,\
				# 検出結果出力先ディレクトリ
				"--project", out_dir,\
				# 検出結果ディレクトリ名
				"--name", DIR_TMP_DETECT_LABWARE,\
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
		# print("# Success!")


		#
		# 検出結果からトリミング画像生成
		#
		print("###########################")
		print("# generate triming images #")
		print("###########################")
		if not os.path.isdir(out_dir+"/"+DIR_TMP_IMG_TRIM): #存在しなければ生成
			os.mkdir(out_dir+"/"+DIR_TMP_IMG_TRIM)
		trim = Trim_Tip_Rack(\
				# トリミング前画像ディレクトリ
				img_dir,\
				# ラボウェア検出結果ラベルディレクトリ
				out_dir+"/"+DIR_TMP_DETECT_LABWARE+"/labels",\
				# 学習時yamlファイル
				train_yaml,\
				# トリミング結果画像ディレクトリ
				out_dir+"/"+DIR_TMP_IMG_TRIM,\
				# チップラックのラベル名
				TIP_RACK_LABEL_NAME,\
				# トリミング結果画像幅
				WIDTH_SMALL,\
				# トリミング結果画像高
				HIGHT_SMALL)
		# print("# Success!")


		#
		# トリミング画像からチップ検出
		#
		print("##############")
		print("# detect tip #")
		print("##############")
		subprocess.run(["python3", "yolov5/detect.py",\
				# 検出対象画像ディレクトリ
				"--source", out_dir+"/"+DIR_TMP_IMG_TRIM,\
				# 検出結果出力先ディレクトリ
				"--project", out_dir,\
				# 検出結果ディレクトリ名
				"--name", DIR_TMP_DETECT_TIP,\
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
		# print("# Success!")


		#
		# 総合検出結果出力
		#
		print("####################")
		print("# result integrate #")
		print("####################")
		# サブ出力ディレクトリ生成
		os.mkdir(out_dir+"/"+DIR_OUT_LABWARE_IMG)
		os.mkdir(out_dir+"/"+DIR_OUT_TIP_IMG)

		#ラボウェア検出結果ファイルコピー
		self.make_bbox_image(img_dir, out_dir+"/"+DIR_TMP_DETECT_LABWARE+"/labels/",
				out_dir+"/"+DIR_OUT_LABWARE_IMG, train_yaml)
		#チップ検出結果ファイルコピー
		self.make_bbox_image(out_dir+"/"+DIR_TMP_IMG_TRIM, out_dir+"/"+DIR_TMP_DETECT_TIP+"/labels/",
				out_dir+"/"+DIR_OUT_TIP_IMG, "tip")

		# ラベル結果コピー
		shutil.copytree(out_dir+"/"+DIR_TMP_DETECT_LABWARE+"/labels", out_dir+"/"+DIR_OUT_LABWARE_LBL)
		shutil.copytree(out_dir+"/"+DIR_TMP_DETECT_TIP+"/labels", out_dir+"/"+DIR_OUT_TIP_LBL)






	# 検出データからbbox付き画像生成
	def make_bbox_image(self, ori_img_dir, label_dir, save_dir, train_yaml_path):
		object_name = None # 検出オブジェクト名配列
		mode = None

		# 訓練yaml取得(ラベルデータ取得)
		if train_yaml_path == "tip":
			object_name = ["tip"]
			mode = "tip"
		else:
			with open(train_yaml_path, 'r') as yamlfile:
				obj = yaml.safe_load(yamlfile)
				object_name = obj["names"]
			mode = "standard"

		# 元画像ループ
		for img_name in os.listdir(ori_img_dir):
			base_name = img_name.rsplit(".",1)[0]
		
			# ラベルの有無確認
			if not os.path.isfile(label_dir+"/"+base_name+".txt"): # ラベルがなければスキップ
				continue
			else: # ラベルがあれば以下の処理
				# 画像取得
				img = cv2.imread(ori_img_dir+"/"+img_name)
				# ラベルループ
				for label_row in self.label_file_to_arr(label_dir+"/"+base_name+".txt"):
					if mode == "standard":
						self.label_row_to_bbox(img, label_row, object_name)
					elif mode == "tip":
						self.label_row_to_bbox(img, label_row, object_name, "tip")
				# 保存
				cv2.imwrite(save_dir+"/"+img_name, img)
	

	# ラベル情報からbbox描画
	def label_row_to_bbox(self, img, label_row, object_name, mode="standard"):
		# 画像サイズ取得
		height, width = img.shape[:2]
		# ラベル情報取得
		obj_name = object_name[int(label_row[0])]
		obj_pos_w  = width  * float(label_row[1])
		obj_pos_h  = height * float(label_row[2])
		obj_size_w = width  * float(label_row[3])
		obj_size_h = height * float(label_row[4])
		obj_prob = label_row[5]
		obj_l = round(obj_pos_w - obj_size_w/2.0)
		obj_r = round(obj_pos_w + obj_size_w/2.0)
		obj_t = round(obj_pos_h - obj_size_h/2.0)
		obj_b = round(obj_pos_h + obj_size_h/2.0)

		# 描画モード
		if mode == "standard":
			# 色設定
			bbox_col,txt_col = self.gen_2_color(int(label_row[0]))
			# bbox設定
			lw = max(round(sum(img.shape) / 2 * 0.003), 2) # line width (yolov5と同じ)
			# テキスト設定
			label_text  = obj_name+" "+obj_prob[0:4] # 確率は小数点以下2桁まで
			myFontFace  = cv2.FONT_HERSHEY_SIMPLEX
			myFontSlace = lw / 5.0
			myFontThickness = max(lw-4, 1)
			(txt_w, txt_h),tmp = cv2.getTextSize(label_text, myFontFace, myFontSlace, myFontThickness)

			# bbox描画
			cv2.rectangle(img, (obj_l, obj_t), (obj_r, obj_b), bbox_col, lw)
			# ラベル背景描画
			cv2.rectangle(img, (obj_l, obj_t-txt_h-lw*2), (obj_l+txt_w+lw*2, obj_t), bbox_col, -1)
			# ラベルテキスト描画
			cv2.putText(img, label_text, (obj_l+lw, obj_t-lw),
					myFontFace, myFontSlace, txt_col, myFontThickness)

		elif mode == "tip":
			#描画設定
			bbox_col = (0,255,0) # green
			lw = max(round(sum(img.shape) / 2 * 0.003), 2) # line width (yolov5と同じ)
			# bbox描画
			cv2.rectangle(img, (obj_l, obj_t), (obj_r, obj_b), bbox_col, 1)
			# 中心点描画
			cv2.drawMarker(img, (round(obj_pos_w), round(obj_pos_h)), bbox_col,
					cv2.MARKER_CROSS, round(min(obj_size_w, obj_size_h)*0.5))



	# 色2セット生成
	def gen_2_color(self, seed):
		if int(seed) % 6 == 0:
			col1 = (0,255,0) # green
			col2  = (255,255,255) # white
		elif int(seed) % 6 == 1:
			col1 = (255,0,0) # blue
			col2  = (255,255,255) # white
		elif int(seed) % 6 == 2:
			col1 = (0,0,255) # red
			col2  = (255,255,255) # white
		elif int(seed) % 6 == 3:
			col1 = (255,255,0) # cyan
			col2  = (255,255,255) # white
		elif int(seed) % 6 == 4:
			col1 = (255,0,255) # magenta
			col2  = (255,255,255) # white
		elif int(seed) % 6 == 5:
			col1 = (0,255,255) # yellow
			col2  = (0,0,0) # bloack

		return col1, col2


	

	# ラベルファイルから配列へ変換
	def label_file_to_arr(self, file_path):
		SEPARATOR = " "
		labels = None
		with open(file_path, 'r') as txtfile:
			reader = csv.reader(txtfile, delimiter=SEPARATOR)
			labels = [row for row in reader]

		return labels



	def make_integrate_label(self, out_dir, dir_dtc_labware, dir_dtc_tip, dir_out_labware_lbl):
		# copy labware label
		shutil.copytree(out_dir+"/"+dir_dtc_labware+"/labels", out_dir+"/"+dir_out_labware_lbl)

		#labware label file loop
		for label_labware in os.listdir(out_dir+"/"+dir_out_labware_lbl):
			base_name_labware = label_labware.rsplit(".",1)[0]
			print(label_labware)

			# tip label file loop
			for label_tip in os.listdir(out_dir+"/"+dir_dtc_tip+"/labels"):
				# labwareラベルのベースネームとtipラベルの先頭が一致すれば
				if label_tip.startswith(base_name_labware):
					print(label_tip)




	# 保存先ディレクトリ生成
	def make_output_dir(self, out_dir):
		MAX_SEQ_NUM = 1024 # 最大連番数

		if not os.path.isdir(out_dir): #存在しなければ生成
			os.mkdir(out_dir)
			print("The directory \"{}\" was made.".format(out_dir))
		else: #存在すれば連番で代わりを生成
			# ディレクトリパスの最後に"/"があれば消す
			while out_dir[-1]=="/":
				out_dir = out_dir[:-1]
			# 連番ディレクトリ名存在確認
			for i in range(1,MAX_SEQ_NUM):
				out_dir_cdd = out_dir+str(i)
				if not os.path.isdir(out_dir_cdd):
					os.mkdir(out_dir_cdd)
					print("The directory \"{}\" already exists. \"{}\" was made instead.".format(out_dir, out_dir_cdd))
					out_dir = out_dir_cdd
					break

		# ディレクトリ名戻し
		return out_dir




	# 画像の縮小＆保存
	def resize_save_img(self, img_dir, out_dir, width):
		imgs = os.listdir(path=img_dir)

		# make save dir
		if not os.path.isdir(out_dir):
			os.mkdir(out_dir)

		for num in range(len(imgs)):
			# read image
			img = cv2.imread(img_dir+"/"+imgs[num])
			# resize
			h, w = img.shape[:2]
			height = round(h * (width / w))
			img = cv2.resize(img, dsize=(width, height))
			cv2.imwrite(out_dir+"/"+imgs[num], img)



if __name__ == '__main__':
	#
	# argument
	#
	prs = argparse.ArgumentParser()
	# image dir to detect
	prs.add_argument("image_dir", type=str,
			help="directory path of image files.")
	# output dir path
	prs.add_argument("--out-dir", type=str, required=False, default="out",
			help="directory path of output files.")
	# labware detection model
	prs.add_argument("--model-labware", type=str, required=False,
			default="./model/detect_labware_20220624/weights/best.pt",
			help="file path of labware detection model.")
	# tip detection model
	prs.add_argument("--model-tip", type=str, required=False,
			default="./model/detect_tip_20220624/weights/best.pt",
			help="file path of labware detection model.")
	# threshold of detection
	prs.add_argument("--threshold", type=float, required=False,
			default=0.7,
			help="confidence threshold of detection.")
	# yaml file of training labware
	prs.add_argument("--labware_train_yaml", type=str, required=False,
			default="./model/dataset_20220624_small_notip.yaml",
			help="yaml file path of training labware.")
	# evaluation mode
	prs.add_argument("--evaluate", action="store_true", required=False,
			help="flag of evaluation mode.")
	args = prs.parse_args()


	#
	# detection
	#
	ot2eye = OT2Eye(args.image_dir, args.out_dir, args.model_labware, args.model_tip, args.threshold, args.labware_train_yaml)

	#
	# evaluation mode
	#
	if args.evaluate:
		print("##############")
		print("# evaluation #")
		print("##############")
		obj_rec_eval = Obj_Rec_Eval(\
				"out1/tmp/detect_tip/labels",\
				"out1/tmp/images_trim",\
				"dataset/20220718_trim_answer/labels",\
				".jpeg")

	# example
	# python3 ot2eye.py dataset/20220718_large/
