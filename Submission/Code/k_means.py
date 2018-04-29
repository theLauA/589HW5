import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from util import recreate_image,rec_err

def run(data_x, flattened_image):
    
    k_list = np.array([2, 5, 10, 25, 50, 75, 100, 200])
    fig = plt.figure(figsize=(8,8))

    #Original Picture
    fig.add_subplot(3,3,1)
    plt.axis('off')
    plt.imshow(data_x)

    count = 2


    rec_errs = np.zeros( k_list.shape[0] )
    for idx, k in enumerate(k_list):
        kmeans = KMeans(n_clusters=k,random_state=0).fit(flattened_image)
        labels = kmeans.predict(flattened_image)
        rec_errs[idx] = rec_err(kmeans.cluster_centers_,labels,flattened_image)
        fig.add_subplot(3,3,count)
        plt.axis('off')
        plt.imshow(recreate_image(kmeans.cluster_centers_, labels, 400, 400))
        print('{0} cluster finished'.format(k))

        count = count+1

    plt.show()		
    return rec_errs