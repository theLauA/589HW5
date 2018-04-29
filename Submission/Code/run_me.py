# Import modules

import numpy as np
from scipy import misc

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,AgglomerativeClustering
from util import recreate_image,rec_err
from sklearn.neighbors import kneighbors_graph,BallTree
import k_means

def read_scene():
	data_x = misc.imread('../../Data/umass_campus.jpg')

	return (data_x)

if __name__ == '__main__':
	
	################################################
	# K-Means

	data_x = read_scene()/255
	
	print('X = ', data_x.shape)

	flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
	print('Flattened image = ', flattened_image.shape)

	print('Implement AHC here ...')
	#k_list = np.array([2, 5, 10, 25, 50, 75, 100, 200])
	k_list = np.array([2, 5, 10])
	fig = plt.figure(figsize=(8,8))

	#Original Picture
	fig.add_subplot(3,3,1)
	plt.axis('off')
	plt.imshow(data_x)

	count = 2


	rec_errs = np.zeros( k_list.shape[0] )
	for idx, k in enumerate(k_list):
		HAC = AgglomerativeClustering(n_clusters=k,linkage='complete',affinity='cosine')
		labels = HAC.fit_predict(flattened_image)
		rec_errs[idx] = rec_err(kmeans.cluster_centers_,labels,flattened_image)
		fig.add_subplot(3,3,count)
		plt.axis('off')
		plt.title("{0} cluster".format(k))
		plt.imshow(recreate_image(kmeans.cluster_centers_, labels, 400, 400))
		
		print('{0} cluster finished'.format(k))

		count = count+1

	plt.show()	
	print('Implement k-means here ...')
	#k_means.run(data_x,flattened_image)

	reconstructed_image = flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
	print('Reconstructed image = ', reconstructed_image.shape)

