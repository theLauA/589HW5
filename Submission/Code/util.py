import numpy as np

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def rec_err(codebook, labels, true_labels):
    temp = np.zeros( (1,3) )
    
    for idx, label in enumerate(labels):
        temp = temp + (codebook[labels[idx]] - true_labels[idx])
    
    return np.sqrt(np.sum(temp, axis=1).mean())