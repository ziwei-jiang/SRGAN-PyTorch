import cv2
import argparse
import numpy as np
import json

if __name__ == '__main__':
	refPt = []
	cropping = False
	parser = argparse.ArgumentParser()

	parser.add_argument('--image', type=str, help='input image to be croped')
	parser.add_argument('--coords', type=str, default= None,
	 help='Loading the bounding box coordinates from saved file. Manual selecting boxes when no saved coordinates is used.')
	opt = parser.parse_args()
	img_name = opt.image
	image = cv2.imread(img_name)
	cv2.namedWindow('image')
	cv2.setMouseCallback('image', lambda: None)
	clone = image.copy()
	if img_name.find('/') == -1:
		out_dir = ''
	else:
		out_dir = img_name[:img_name.rfind('/')+1]

	if opt.coords == None:
		r = cv2.selectROIs('image', image)
		print(r)
		cv2.destroyAllWindows()
		bb_colors = np.random.random((len(r), 3))*255

		saved_crops = {'roi_coords': r,
						'bb_colors': bb_colors
						}
		np.save(out_dir + img_name[img_name.rfind('/')+1:img_name.find('.')]+'.npy', saved_crops, allow_pickle=True)



	else:	
		saved_crops = np.load(opt.coords, allow_pickle=True).item()

		r = saved_crops["roi_coords"]
		bb_colors = saved_crops['bb_colors']		
	uls = [(r[i][0], r[i][1]) for i in range(len(r))]
	lrs = [(r[i][0]+r[i][2], r[i][1]+r[i][3]) for i in range(len(r))]
	rois = [image[r[i][1]:r[i][1]+r[i][3], r[i][0]:r[i][0]+r[i][2]] for i in range(len(r))]

	for i in range(len(r)):
		row = int(0.05* rois[i].shape[0])
		col = int(0.05* rois[i].shape[1])

		img_b = cv2.copyMakeBorder(rois[i], row, row, col, col, borderType=cv2.BORDER_CONSTANT, value=bb_colors[i])
		cv2.rectangle(clone, uls[i], lrs[i], bb_colors[i], 5)
		cv2.imwrite(out_dir + str(i)+'_crop_'+img_name[img_name.rfind('/')+1:], img_b)

	cv2.imwrite(out_dir +'annotated_' + img_name[img_name.rfind('/')+1:], clone)
