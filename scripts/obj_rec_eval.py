#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from os import sep
import cv2
import csv
from glob import glob

class Obj_Rec_Eval():
	def __init__(self, out_file_path, DTC_LABEL_DIR_PATH, IMG_DIR_PATH, ANS_LABEL_DIR_PATH):
		self.SEPARATOR=" " # labelファイルのセパレータ

		# output csv array
		result_arr = []
		# result_arr.append(["#Image_file", "#Labware", "#Recall", "#Precision", "#F-value"])
		result_arr.append(["Image_file", "Labware", "N_pos", "TP", "FP", "Recall", "Precision", "F-value"])
		# label names
		label_name_arr = []

		# get label number and name
		with open(ANS_LABEL_DIR_PATH+sep+"classes.txt", "r") as file:
			label_name_arr = [s.strip() for s in file.readlines()]

		# loop for all file in detect label directory
		for dtc_label_file_name in sorted(os.listdir(DTC_LABEL_DIR_PATH)):
			# file name
			base_name     = dtc_label_file_name.rsplit(".",1)[0]
			IMG_EXT       = glob(IMG_DIR_PATH+os.sep+base_name+"*")[0].rsplit(".",1)[1]
			img_file_name = base_name + "." + IMG_EXT

			# get image
			img = cv2.imread(IMG_DIR_PATH + sep +  img_file_name)

			# get detected and answer label data
			dtc_label_arr = self.label_file_to_arr(DTC_LABEL_DIR_PATH+sep+dtc_label_file_name)
			ans_label_arr = self.label_file_to_arr(ANS_LABEL_DIR_PATH+sep+dtc_label_file_name)

			# 各ラベルごとにループ
			for obj_num in range(len(label_name_arr)):
				# evaluation calculation
				eval_N_pos, eval_TP,eval_FP = self.calc_TP_FP(ans_label_arr, dtc_label_arr, img, obj_num)

				# calc precision
				try:
					eval_Precision = eval_TP / float(eval_TP + eval_FP)
				except ZeroDivisionError:
					eval_Precision = float("nan")
				# calc recall
				try:
					eval_Recall = eval_TP / float(eval_N_pos)
				except ZeroDivisionError:
					eval_Recall = float("nan")
				# calc F value
				try:
					eval_F = 2.0 / (1.0/eval_Precision + 1.0/eval_Recall)
				except ZeroDivisionError:
					eval_F = float("nan")


				result_arr.append([\
						img_file_name,\
						label_name_arr[obj_num],\
						"{:d}".format(eval_N_pos),\
						"{:d}".format(eval_TP),\
						"{:d}".format(eval_FP),\
						"{:.04f}".format(eval_Recall),\
						"{:.04f}".format(eval_Precision),\
						"{:.04f}".format(eval_F)])


		# output evalutate file
		with open(out_file_path+sep+"evaluation.csv", "w") as out_file:
			writer = csv.writer(out_file, delimiter="\t")
			writer.writerows(result_arr)

		return





	# lebel file to array
	def label_file_to_arr(self, label_file_path):
		try:
			labels = None

			with open(label_file_path, 'r') as txtfile:
				reader = csv.reader(txtfile, delimiter=self.SEPARATOR)
				labels = [row for row in reader]

			return labels
		except FileNotFoundError:
			return []
	

	# calc TP and FP
	def calc_TP_FP(self, ans_labels, dtc_labels, img, obj_num):
		img_height, img_width = img.shape[:2]
		num_all_dtc = 0
		eval_N_pos  = 0
		eval_TP = 0
		eval_FP = 0

		# calc N pos
		for ans_row in ans_labels:
			if int(ans_row[0]) == obj_num:
				eval_N_pos += 1

		# calc TP and FP
		for dtc_row in dtc_labels: # loop for all row in detect label file
			if int(dtc_row[0]) != obj_num:
				continue
			else:
				num_all_dtc += 1

			for ans_row in ans_labels: # loop for all row in answer label file
				if int(ans_row[0]) != obj_num:
					continue
				# 推論bboxの中点が正解bboxに含まれているかどうか
				if(self.point_is_in_bbox(ans_row, dtc_row, img_width, img_height)):
					eval_TP += 1
		
		eval_FP = num_all_dtc - eval_TP

		return (eval_N_pos, eval_TP, eval_FP)


	# check point is in bbox
	def point_is_in_bbox(self, p_row, b_row, img_width, img_height):
		# get point object info
		p_label, p_pos_w, p_pos_h, p_size_w, p_size_h\
				= self.get_obj_info(p_row, img_width, img_height)
		# get bbox object info
		b_label, b_pos_w, b_pos_h, b_size_w, b_size_h\
				= self.get_obj_info(b_row, img_width, img_height)

		if p_label != b_label:
			return False
		else:
			b_l = b_pos_w - b_size_w*0.5
			b_r = b_pos_w + b_size_w*0.5
			b_t = b_pos_h - b_size_h*0.5
			b_b = b_pos_h + b_size_h*0.5

			return\
					(p_label == b_label and\
					b_l < p_pos_w and p_pos_w < b_r and\
					b_t < p_pos_h and p_pos_h < b_b)


	# get object info from label array
	def get_obj_info(self, row, img_width, img_height):
		obj_label  = int(row[0])
		obj_cnt_w  = img_width  * float(row[1])
		obj_cnt_h  = img_height * float(row[2])
		obj_size_w = img_width  * float(row[3])
		obj_size_h = img_height * float(row[4])

		return obj_label, obj_cnt_w, obj_cnt_h, obj_size_w, obj_size_h








def main():
	obj_rec_eval = Obj_Rec_Eval(\
			"./out",\
			"./out/labels_merged",\
			"./dataset/20220718_trim/images",\
			"./dataset/20220718_large_answer")

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass
