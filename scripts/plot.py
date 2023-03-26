#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2

class Plot():

	def __init__(self):
		pass

	# ラベル情報からbbox描画
	def label_row_to_bbox(self, img, label_row, object_name):
		# 画像情報取得
		height, width = img.shape[:2]
		obj_pos_w  = width  * float(label_row[1])
		obj_pos_h  = height * float(label_row[2])
		obj_size_w = width  * float(label_row[3])
		obj_size_h = height * float(label_row[4])
		obj_prob = label_row[5]
		# bboxの角の座標
		obj_l = round(obj_pos_w - obj_size_w*0.5)
		obj_r = round(obj_pos_w + obj_size_w*0.5)
		obj_t = round(obj_pos_h - obj_size_h*0.5)
		obj_b = round(obj_pos_h + obj_size_h*0.5)

		# 色設定
		bbox_col,txt_col = self.gen_2_color(int(label_row[0]))
		# bbox設定
		lw = max(round(sum(img.shape) / 2 * 0.003), 2) # line width (yolov5と同じ)

		# 描画
		if int(label_row[0]) >= len(object_name) or object_name == ["tip"]: #tip
			# bbox描画
			cv2.rectangle(img, (obj_l, obj_t), (obj_r, obj_b), bbox_col, 1)
			# 中心点描画
			cv2.drawMarker(img, (round(obj_pos_w), round(obj_pos_h)), bbox_col,
					cv2.MARKER_CROSS, round(min(obj_size_w, obj_size_h)*0.5))
		else: #tip以外
			obj_name = object_name[int(label_row[0])]
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



	# 色2セット生成
	def gen_2_color(self, seed):
		# 1色目
		if int(seed) % 6 == 0:
			col1 = (0,255,0) # green
		elif int(seed) % 6 == 1:
			col1 = (255,0,0) # blue
		elif int(seed) % 6 == 2:
			col1 = (0,0,255) # red
		elif int(seed) % 6 == 3:
			col1 = (255,255,0) # cyan
		elif int(seed) % 6 == 4:
			col1 = (255,0,255) # magenta
		elif int(seed) % 6 == 5:
			col1 = (0,255,255) # yellow

		# 2色目
		if int(seed) % 6 == 5:
			col2  = (0,0,0) # black
		else:
			col2  = (255,255,255) # white

		return col1, col2

