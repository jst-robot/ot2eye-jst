#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import csv
import yaml
import argparse

class Trim_Tip_Rack():
	def __init__(self, origin_dir_path, label_dir_path, dataset_yaml_path, output_path,
			tip_label="tip_label", trimmed_img_size_w=640.0, trimmed_img_size_h=480.0):

		TRIMMED_IMG_SIZE_W = trimmed_img_size_w
		TRIMMED_IMG_SIZE_H = trimmed_img_size_h

		img_name = os.listdir(path=origin_dir_path)
		FILE_NUM = len(img_name)
		label_name = [""] * FILE_NUM

		# get tip rack label name form dataset yaml
		tr_label_num = 0
		with open(dataset_yaml_path, 'r') as yamlfile:
			obj = yaml.safe_load(yamlfile)
			tr_label_num = obj['names'].index(tip_label)


		# loop for all file in original image directory
		for file_num in range(FILE_NUM):
			# label_name[file_num] = img_name[file_num][:img_name[file_num].rfind(".")]+".txt"
			label_name[file_num] = img_name[file_num].rsplit(".",1)[0]+".txt"

			# image size
			ori_img = cv2.imread(origin_dir_path + img_name[file_num])
			# get image size
			ori_height, ori_width = ori_img.shape[:2]

			# label text 
			labels = None

			# ラベルがなければ飛ばす
			if not os.path.isfile(label_dir_path+label_name[file_num]):
				continue

			with open(label_dir_path+label_name[file_num], 'r') as txtfile:
				reader = csv.reader(txtfile, delimiter=" ")
				labels = [row for row in reader]

			# get tip rack coordinate
			index = 0
			for row in labels: # loop for all row in detected label text file
				if int(row[0]) == tr_label_num: # if this row is about tip rack 
					# get detected object info
					obj_cnt_w  = ori_width* float(row[1])
					obj_cnt_h  = ori_height*float(row[2])
					obj_size_w = ori_width* float(row[3])
					obj_size_h = ori_height*float(row[4])
					obj_aspect = obj_size_h / obj_size_w

					# trimming from original image
					rate = 0
					if obj_aspect <= TRIMMED_IMG_SIZE_H / TRIMMED_IMG_SIZE_W: #横長
						rate =  TRIMMED_IMG_SIZE_W / obj_size_w
					else:
						rate =  TRIMMED_IMG_SIZE_H / obj_size_h
					trim_img = cv2.resize(ori_img, dsize=None, fx=rate, fy=rate)
					trim_h = [round(obj_cnt_h*rate - TRIMMED_IMG_SIZE_H/2.0),\
							  round(obj_cnt_h*rate + TRIMMED_IMG_SIZE_H/2.0)]
					trim_w = [round(obj_cnt_w*rate - TRIMMED_IMG_SIZE_W/2.0),\
							  round(obj_cnt_w*rate + TRIMMED_IMG_SIZE_W/2.0)]
					trim_img = trim_img[trim_h[0]:trim_h[1], trim_w[0]:trim_w[1]]

					# output trimmed image
					out_file_name = img_name[file_num][:img_name[file_num].rfind(".")]\
							+ "_{}".format(index) +\
							img_name[file_num][img_name[file_num].rfind("."):]
					cv2.imwrite(output_path+out_file_name, trim_img)

					index += 1





if __name__ == '__main__':
	# argument
	prs = argparse.ArgumentParser()
	prs.add_argument("--origin-dir", type=str, required=True,
			help="directory path of \"not\" compressed image file.")
	prs.add_argument("--label-dir", type=str, required=True,
			help="directory path of label text file.")
	prs.add_argument("--dataset-yaml", type=str, required=True,
			help="yaml file path of dataset.")
	prs.add_argument("--out", type=str, required=True,
			help="directory path of trimed image file.")
	prs.add_argument("--tip-label", type=str, default="tip_rack",
			help="label name of tip rack.")
	prs.add_argument("--trim-w", type=float, default="640",
			help="width of trimmed image.")
	prs.add_argument("--trim-h", type=float, default="480",
			help="height of trimmed image.")
	args = prs.parse_args()

	trim = Trim_Tip_Rack(args.origin_dir, args.label_dir, args.dataset_yaml, args.out, args.tip_label, args.trim_w, args.trim_h)
