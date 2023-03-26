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
		result_arr.append(["#Image_file", "#Labware", "#Recall", "#Precision", "#F-value"])
		# label names
		label_name_arr = []

		# get label number and name
		with open(ANS_LABEL_DIR_PATH+sep+"classes.txt", "r") as file:
			label_name_arr = [s.strip() for s in file.readlines()]

		# loop for all file in detect label directory
		for dtc_label_file_name in os.listdir(DTC_LABEL_DIR_PATH):
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


				# print(dtc_label_file_name)
				# print("evaluation of {}".format(dtc_label_file_name))
				# print("\tlabel: {}".format(label_name_arr[obj_num]))
				# print("\tN_pos: {}".format(eval_N_pos))
				# print("\tTP:    {}".format(eval_TP))
				# print("\tFP:    {}".format(eval_FP))
				# print("\tPrecision: {}".format(eval_Precision))
				# print("\tRecall:    {}".format(eval_Recall))
				# print("\tF value:   {}".format(eval_F))
				# print("")

				result_arr.append([\
						img_file_name,\
						label_name_arr[obj_num],\
						"{:.06f}".format(eval_Recall),\
						"{:.06f}".format(eval_Precision),\
						"{:.06f}".format(eval_F)])


		# output evalutate file
		with open(out_file_path+sep+"evaluation.csv", "w") as out_file:
			writer = csv.writer(out_file, delimiter=" ")
			writer.writerows(result_arr)

		return


		#	#
		#	# show detect img
		#	#
		#	self.show_dtc_ans_img(img, ans_labels, dtc_labels)
		#	cv2.imshow("{}".format(img_name[file_num]),img)
		#	cv2.waitKey(0)

		#cv2.destroyAllWindows()



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


	# # calc TP
	# def calc_TP(self, ans_labels, dtc_labels, img, obj_name):
	# 	eval_TP = 0
	# 	img_height, img_width = img.shape[:2]

	# 	for ans_row in ans_labels: # loop for all row in answer label file
	# 		# 所望のラベルでなければスキップ
	# 		# if ans_row[0] != obj_name:
	# 		# 	continue
	# 		for dtc_row in dtc_labels: # loop for all row in detect label file
	# 			# 正解bboxの中点が推論bboxに含まれているかどうか
	# 			if(self.point_is_in_bbox(ans_row, dtc_row, img_width, img_height)):
	# 				eval_TP += 1
		
	# 	return eval_TP


	# # calc FP
	# def calc_FP(self, ans_labels, dtc_labels, img):
	# 	img_height, img_width = img.shape[:2]
	# 	num_all_dtc = len(dtc_labels)
	# 	num_dtc = 0

	# 	for dtc_row in dtc_labels: # loop for all row in detect label file
	# 		for ans_row in ans_labels: # loop for all row in answer label file
	# 			# 推論bboxの中点が正解bboxに含まれているかどうか
	# 			if(self.point_is_in_bbox(ans_row, dtc_row, img_width, img_height)):
	# 				num_dtc += 1
	
	# 	return num_all_dtc-num_dtc



	
	# show detect img
	def show_dtc_ans_img(self, img, ans_labels, dtc_labels):
		img_height, img_width = img.shape[:2]

		# show ans
		for ans_row in ans_labels: # loop for all row in answer label text file
			# get answer object info
			ans_obj_label, ans_obj_cnt_w, ans_obj_cnt_h, ans_obj_size_w, ans_obj_size_h\
					= self.get_obj_info(ans_row, img_width, img_height)
			# add ans marker to img
			cv2.drawMarker(img, (round(ans_obj_cnt_w),round(ans_obj_cnt_h)), (0,0,255), markerSize=10)
			# add ans rectangle to img
			# ans_obj_l = ans_obj_cnt_w - ans_obj_size_w/2.0
			# ans_obj_r = ans_obj_cnt_w + ans_obj_size_w/2.0
			# ans_obj_t = ans_obj_cnt_h - ans_obj_size_h/2.0
			# ans_obj_b = ans_obj_cnt_h + ans_obj_size_h/2.0
			# cv2.rectangle(img, (round(ans_obj_l), round(ans_obj_t)), (round(ans_obj_r), round(ans_obj_b)), (0,0,255))

		# show dtc
		for dtc_row in dtc_labels: # loop for all row in detect label text file
			# get answer object info
			dtc_obj_label, dtc_obj_cnt_w, dtc_obj_cnt_h, dtc_obj_size_w, dtc_obj_size_h\
					= self.get_obj_info(dtc_row, img_width, img_height)
			# add dtc marker to img
			# cv2.drawMarker(img, (round(dtc_obj_cnt_w),round(dtc_obj_cnt_h)), (0,255,0), markerSize=10)
			# add detected rectangle to img
			dtc_obj_l = dtc_obj_cnt_w - dtc_obj_size_w/2.0
			dtc_obj_r = dtc_obj_cnt_w + dtc_obj_size_w/2.0
			dtc_obj_t = dtc_obj_cnt_h - dtc_obj_size_h/2.0
			dtc_obj_b = dtc_obj_cnt_h + dtc_obj_size_h/2.0
			cv2.rectangle(img, (round(dtc_obj_l), round(dtc_obj_t)), (round(dtc_obj_r), round(dtc_obj_b)), (0,255,0))







def main():
	obj_rec_eval = Obj_Rec_Eval(\
			"./out",\
			"./out/labels_merged",\
			"./dataset/20220718_trim/images",\
			"./dataset/20220718_large_answer",\
			".jpeg")

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass
