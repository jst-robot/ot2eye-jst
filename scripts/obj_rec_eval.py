#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import csv

class Obj_Rec_Eval():
	def __init__(self, DTC_LABEL_DIR_PATH, IMG_DIR_PATH, ANS_LABEL_DIR_PATH, IMG_EXT):

		# get number of detect labels files
		dtc_label_name = os.listdir(path=DTC_LABEL_DIR_PATH)
		FILE_NUM = len(dtc_label_name)
		img_name = [""] * FILE_NUM
		ans_label_name = [""] * FILE_NUM

		# loop for all file in detect label directory
		for file_num in range(FILE_NUM):
			# classes.txt file is skip 
			if(dtc_label_name[file_num] == "classes.txt"):
				continue

			# variable of evaluation
			eval_N_pos = 0
			eval_TP = 0
			eval_FP = 0
			eval_Precision = 0
			eval_Recall = 0
			eval_F = 0

			# set file name
			img_name[file_num] = dtc_label_name[file_num][:dtc_label_name[file_num].rfind(".")]+IMG_EXT
			ans_label_name[file_num] = dtc_label_name[file_num]

			# get image size
			img = cv2.imread(IMG_DIR_PATH + "/" +  img_name[file_num])
			img_height, img_width = img.shape[:2]
			# get detect label file
			dtc_labels = self.file_to_arr(DTC_LABEL_DIR_PATH, dtc_label_name, file_num)
			# get answer label file
			ans_labels = self.file_to_arr(ANS_LABEL_DIR_PATH, ans_label_name, file_num)

			#
			# evaluation
			#
			# calc N_pos & TP & FP
			eval_N_pos = len(ans_labels)
			eval_TP = self.calc_TP(ans_labels, dtc_labels, img)
			eval_FP = self.calc_FP(ans_labels, dtc_labels, img)
			# calc Precision & Recall
			eval_Precision = eval_TP/float(eval_TP+eval_FP)
			eval_Recall    = eval_TP/float(eval_N_pos)
			eval_F         = 2.0 / (1.0/eval_Precision + 1.0/eval_Recall)

			# show calc result
			print("evaluation of {}".format(dtc_label_name[file_num]))
			print("\tN_pos: {}".format(eval_N_pos))
			print("\tTP:    {}".format(eval_TP))
			print("\tFP:    {}".format(eval_FP))
			print("\tPrecision: {}".format(eval_Precision))
			print("\tRecall:    {}".format(eval_Recall))
			print("\tF value:   {}".format(eval_F))
			print("")

			#
			# show detect img
			#
			self.show_dtc_ans_img(img, ans_labels, dtc_labels)
			cv2.imshow("{}".format(img_name[file_num]),img)
			cv2.waitKey(0)

		cv2.destroyAllWindows()


	# lebel file to array
	def file_to_arr(self, dir_path, label_name, file_num):
		dtc_labels = None
		with open(dir_path + "/" + label_name[file_num], 'r') as txtfile:
			reader = csv.reader(txtfile, delimiter=" ")
			labels = [row for row in reader]

		return labels
	

	# get object info from label array
	def get_obj_info(self, row, img_width, img_height):
		obj_label  = int(row[0])
		obj_cnt_w  = img_width  * float(row[1])
		obj_cnt_h  = img_height * float(row[2])
		obj_size_w = img_width  * float(row[3])
		obj_size_h = img_height * float(row[4])

		return obj_label, obj_cnt_w, obj_cnt_h, obj_size_w, obj_size_h

	
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

	# check point is in bbox
	def point_is_in_bbox(self, p_row, b_row, img_width, img_height):
		# get point object info
		p_label, p_cnt_w, p_cnt_h, p_size_w, p_size_h\
				= self.get_obj_info(p_row, img_width, img_height)
		# get bbox object info
		b_label, b_cnt_w, b_cnt_h, b_size_w, b_size_h\
				= self.get_obj_info(b_row, img_width, img_height)

		b_l = b_cnt_w - b_size_w/2.0
		b_r = b_cnt_w + b_size_w/2.0
		b_t = b_cnt_h - b_size_h/2.0
		b_b = b_cnt_h + b_size_h/2.0

		return\
				(p_label == b_label and\
				b_l < p_cnt_w and p_cnt_w < b_r and\
				b_t < p_cnt_h and p_cnt_h < b_b)


	# calc TP
	def calc_TP(self, ans_labels, dtc_labels, img):
		eval_TP = 0
		img_height, img_width = img.shape[:2]

		for ans_row in ans_labels: # loop for all row in answer label text file
			for dtc_row in dtc_labels: # loop for all row in detect label text file
				# 正解bboxの中点が推論bboxに含まれているかどうか
				if(self.point_is_in_bbox(ans_row, dtc_row, img_width, img_height)):
					eval_TP += 1
		
		return eval_TP


	# calc FP
	def calc_FP(self, ans_labels, dtc_labels, img):
		# eval_FP = 0
		img_height, img_width = img.shape[:2]
		num_all_dtc = len(dtc_labels)
		num_dtc = 0

		for dtc_row in dtc_labels: # loop for all row in detect label text file
			for ans_row in ans_labels: # loop for all row in answer label text file
				# 推論bboxの中点が正解bboxに含まれているかどうか
				# if(self.point_is_in_bbox(dtc_row, ans_row, img_width, img_height)):
				if(self.point_is_in_bbox(ans_row, dtc_row, img_width, img_height)):
					# eval_FP += 1
					num_dtc += 1
		
		return num_all_dtc-num_dtc




def main():
	obj_rec_eval = Obj_Rec_Eval(\
			"./detect/hoge/labels",\
			"./dataset/20220718_trim/images",\
			"./dataset/20220718_trim/labels",\
			# "./detect/exp_20220624_small_notip_train/labels",\
			# "./dataset/20220624_small_notip/train/images",\
			# "./dataset/20220624_small_notip/train/labels",\
			".jpeg")

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass
