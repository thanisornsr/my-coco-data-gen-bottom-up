import os
import json
import math
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf 
from skimage.transform import resize
import skimage.io as io
import random
from pycocotools.coco import COCO 

class Coco_datagen_bu:
	def __init__(self,data_dir,anno_type,model_input_shape,model_output_shape,batch_size_select=64):
		self.img_dir = data_dir

		self.kps_and_valid = []
		self.kps = []

		self.valids = []
		self.img_ids = []
		self.imgs = []
		self.start_idx = []
		self.end_idx = []
		self.input_shape = model_input_shape
		self.output_shape = model_output_shape
		self.n_imgs = None
		self.batch_size = batch_size_select
		self.n_batchs = None
		self.img_wh = []

		dataDir = '.'
		dataType = anno_type
		annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
		self.coco_kps = COCO(annFile)
		print('Initiate COCO: Done ...')

		self.annIds = self.coco_kps.getAnnIds()
		self.anns = self.coco_kps.loadAnns(self.annIds)

		self.img_ids = [ann['image_id'] for ann in self.anns]
		self.unique_img_ids = list(set(self.img_ids))
		print('Get image id: Done ...')
		self.n_imgs = len(self.unique_img_ids)
		self.kps, self.valids = self.get_kps_valids_by_id()
		print('Split kps and valids: Done ...')
		self.imgs = self.coco_kps.loadImgs(self.unique_img_ids)
		print('Load images: Done ...')
		self.img_wh = self.get_wh()
		print('Get image width and height: Done ...')

		self.start_idx, self.end_idx, self.n_batchs = self.get_start_end_idx()
		print('Get start and end index: Done ...')

		self.limb_list = [(0,1),(2,0),(0,3),(0,4),(3,5),(4,6),(5,7),(6,8),(4,3),(3,9),(4,10),(10,9),(9,11),(10,12),(11,13),(12,14)]
		self.n_keypoints = 15
		self.n_limbs = 16
		print('Create datagen: Done ...')



	def get_target_valid_joint(self,input_kav):
		splited_kps = []
		splited_valids = []
		for temp_anno_kp in input_kav:
			temp_x = np.array(temp_anno_kp[0::3])
			temp_y = np.array(temp_anno_kp[1::3])
			temp_valid = np.array(temp_anno_kp[2::3])
			temp_valid = temp_valid > 0
			temp_valid = temp_valid.astype('float32')
			temp_target_coord = np.stack([temp_x,temp_y],axis=1)
			temp_target_coord = temp_target_coord.astype('float32')

			splited_kps.append(temp_target_coord)
			splited_valids.append(temp_valid)

		return splited_kps,splited_valids

	def get_kps_valids_by_id(self):
		temp_kps = []
		temp_vs = []
		for i in range(self.n_imgs):
			temp_id = self.unique_img_ids[i]
			temp_k_and_v = [ann['keypoints'] for ann in self.anns if ann['image_id'] == temp_id]
			# print(temp_k_and_v)
			t_k,t_v = self.get_target_valid_joint(temp_k_and_v)
			for i in range(len(t_k)):
				temp_k = t_k[i]
				temp_v = t_v[i]
				# print(temp_k.shape)
				t_k[i] = np.delete(temp_k,[1,2],0)
				t_v[i] = np.delete(temp_v,[1,2])
				# print(t_k[0].shape)
				# print(t_v[0].shape)
			temp_kps.append(t_k)
			temp_vs.append(t_v)

		return temp_kps, temp_vs

	def get_wh(self):
		temp_wh = []
		for i in range(self.n_imgs):
			temp_img_data = self.imgs[i]
			temp_wh.append((temp_img_data['width'],temp_img_data['height']))
		return temp_wh

	def get_start_end_idx(self):
		max_idx = self.n_imgs
		temp_batch_size = self.batch_size
		l = list(range(max_idx))
		temp_start_idx = l[0::temp_batch_size]
		def add_batch_size(num,max_id=max_idx,bz=temp_batch_size):
			return min(num+bz,max_id)
		temp_end_idx = list(map(add_batch_size,temp_start_idx))
		temp_n_batchs = len(temp_start_idx)

		return temp_start_idx, temp_end_idx, temp_n_batchs
	def render_heatmap(self,i_grid_x,i_grid_y,i_kp,i_v,sigma,accumulate_confid_map,i_h,i_w):
		if i_v != 0:
			y_range = [i for i in range(int(i_grid_y))]
			x_range = [i for i in range(int(i_grid_x))]
			xx, yy = np.meshgrid(x_range,y_range)
			t_x = i_kp[0] * i_grid_x / i_w
			t_y = i_kp[1] * i_grid_y / i_h
			# print(t_kp)
			d2 = (xx - t_x)**2 + (yy - t_y)**2
			exponent = d2 / 2.0 / sigma / sigma
			mask = exponent <= 4.6052
			cofid_map = np.exp(-exponent)
			cofid_map = np.multiply(mask, cofid_map)
			accumulate_confid_map += cofid_map
			accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
			return accumulate_confid_map
		else:
			return accumulate_confid_map

	def render_paf(self,i_grid_x,i_grid_y,i_kp_A,i_kp_B,i_v_A,i_v_B,accumulate_vec_map,i_h,i_w):
		if (i_v_A != 0) and (i_v_B !=0):
			i_kp_A = i_kp_A.astype('float')
			i_kp_B = i_kp_B.astype('float')
			thre = 1.5  # limb width

			centerA = (i_kp_A[0] * i_grid_x / i_w,i_kp_A[1] * i_grid_y / i_h)
			centerB = (i_kp_B[0] * i_grid_x / i_w,i_kp_B[1] * i_grid_y / i_h)

			limb_vec = i_kp_B - i_kp_A
			norm = np.linalg.norm(limb_vec)
			if (norm == 0.0):
				# print('limb is too short, ignore it...')
				return accumulate_vec_map

			limb_vec_unit = limb_vec / norm

			min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
			max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), i_grid_x)
			min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
			max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), i_grid_y)
			range_x = list(range(int(min_x), int(max_x), 1))
			range_y = list(range(int(min_y), int(max_y), 1))
			xx, yy = np.meshgrid(range_x, range_y)
			ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
			ba_y = yy - centerA[1]
			limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
			mask = limb_width < thre  # mask is 2D

			vec_map = np.copy(accumulate_vec_map) * 0.0
			vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
			vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

			mask = np.logical_or.reduce((np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))
			accumulate_vec_map += vec_map

			return accumulate_vec_map
		else:
			# print('not valid')
			return accumulate_vec_map

	def gen_batch(self,batch_order):
		batch_imgs = []
		batch_heatmaps = []
		batch_pafs = []
		batch_valids = []
		b_start = self.start_idx[batch_order]
		b_end = self.end_idx[batch_order]
		temp_output_shape = self.output_shape
		temp_input_shape = self.input_shape
		temp_valids = self.valids

		temp_imgs = self.imgs
		temp_kps = self.kps
		temp_img_dir = self.img_dir

		for idx in range(b_start,b_end):
			channels_heat = self.n_keypoints
			channels_paf = 2 * self.n_limbs
			#valid
			i_valid = temp_valids[idx]
			# i_ones = np.ones((*temp_output_shape,17),dtype = np.float32)
			# o_valid = i_ones*

			temp_wh = self.img_wh[idx]
			temp_w = temp_wh[0]
			temp_h = temp_wh[1]

			grid_y = self.output_shape[1]
			grid_x = self.output_shape[0]

			#heatmap
			i_kp = temp_kps[idx]
			heatmaps = np.zeros((int(grid_y), int(grid_x), channels_heat))
			for i in range(len(i_kp)):
				for j in range(channels_heat):
					heatmaps[:,:,j] = self.render_heatmap(grid_x,grid_y,i_kp[i][j],i_valid[i][j],2,heatmaps[:,:,j],temp_h,temp_w)

			#paf
			pafs = np.zeros((int(grid_y), int(grid_x), channels_paf))
			temp_limb_list = self.limb_list
			for i in range(len(i_kp)):
				for j in range(len(temp_limb_list)):
					limb = temp_limb_list[j]
					idx_A = limb[0]
					idx_B = limb[1]
					pafs[:,:,[j,j+self.n_limbs]] = self.render_paf(grid_x,grid_y,i_kp[i][idx_A],i_kp[i][idx_B],i_valid[i][idx_A],i_valid[i][idx_B],pafs[:,:,[j,j+self.n_limbs]],temp_h,temp_w)

			#imgs
			i_img = temp_imgs[idx]
			o_img = io.imread('./'+ temp_img_dir + '/' + i_img['file_name'])
			r_img = resize(o_img,temp_input_shape)
			r_img = r_img.astype('float32')
			# print(r_img.shape)
			if len(r_img.shape) > 2:
				batch_imgs.append(r_img)
				batch_heatmaps.append(heatmaps)
				batch_valids.append(i_valid)
				batch_pafs.append(pafs)
		batch_imgs = np.array(batch_imgs)
		# print(batch_imgs.shape)
		batch_heatmaps = np.array(batch_heatmaps)
		batch_pafs = np.array(batch_pafs)

		return batch_imgs, batch_heatmaps, batch_pafs, batch_valids

	def shuffle_order(self):
		temp_kps = self.kps
		temp_valids = self.valids
		temp_imgs = self.imgs
		temp_unique_img_ids = self.unique_img_ids
		temp_img_wh = self.img_wh
		to_shuffle = list(zip(temp_kps, temp_valids, temp_imgs, temp_unique_img_ids, temp_img_wh))
		random.shuffle(to_shuffle)
		temp_kps, temp_valids, temp_imgs, temp_unique_img_ids, temp_img_wh = zip(*to_shuffle)

		self.kps = temp_kps
		self.valids = temp_valids
		self.imgs = temp_imgs
		self.unique_img_ids = temp_unique_img_ids
		self.img_wh = temp_img_wh






