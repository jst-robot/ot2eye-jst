#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import torch


class OT2Eye():
	def __init__(self):
		model = torch.hub.load("ultralytics/yolov5", "yolov5s")
		# img = "model/detect_tip_20220624/weights/best.pt"
		# img = "./yolov5/data/images/bus.jpg"
		img = "./dataset/20220718_small/IMG_0278.jpeg"

		results = model(img)

		results.show()

if __name__ == '__main__':
	# argument
	# prs = argparse.ArgumentParser()
	# prs.add_argument("--model-dir", type=str, required=True,
	# 		help="directory path of detection model file.")
	# prs.add_argument("--image-dir", type=str, required=True,
	# 		help="directory path of image files.")
	# prs.add_argument("--out-dir", type=str, required=True,
	# 		help="directory path of output files.")
	# args = prs.parse_args()

	ot2eye = OT2Eye()
