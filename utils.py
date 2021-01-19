import imageio
import numpy as np
import scipy
from scipy.fftpack import dct

def img_to_blocks(img, block_size=8):
    block_rows = int(img.shape[0]/block_size)
    block_cols = int(img.shape[1]/block_size)
    blocks = np.zeros((block_rows, block_cols, block_size, block_size))
    
    for i in range(block_rows):
        for j in range(block_cols):
            blocks[i,j] = img[i*block_size:(i+1)*block_size,
                              j*block_size:(j+1)*block_size]
    
    return blocks
    
def blocks_to_img(blocks):
    block_rows, block_cols, block_size = blocks.shape[0:3]
    img = np.zeros((block_rows * block_size, block_cols * block_size), np.float)
    for i in range(block_rows):
        for j in range(block_cols):
            img[i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size] = blocks[i,j]
        
    return img

def dct2(blocks):
    """Computes the 2D-DCT on each block of a segmented image"""
    assert len(blocks.shape) == 4 and blocks.shape[2] == blocks.shape[3]
    block_rows, block_cols, block_size = blocks.shape[0:3]

    dct_blocks = np.zeros(blocks.shape)
    for i in range(block_rows):
        for j in range(block_cols):
            dct_blocks[i,j] = dct(dct(blocks[i,j]-128, axis=0, norm = 'ortho'),
                                  axis=1, norm = 'ortho') 

    return dct_blocks

def idct2(blocks):
    """Computes the inverse 2D-DCT on each block of a segmented image"""
    assert len(blocks.shape) == 4 and blocks.shape[2] == blocks.shape[3]
    block_rows, block_cols, block_size = blocks.shape[0:3]
    
    idct_blocks = np.zeros(blocks.shape)
    for i in range(block_rows):
        for j in range(block_cols):
            idct_blocks[i,j] = dct(dct(blocks[i,j], axis=0, type=3, norm = 'ortho'), 
                                   axis=1, type=3, norm = 'ortho')+128
            
    return idct_blocks


if __name__ == "__main__":
    # Implementation Checks
    im1 = imageio.imread('../img/test1.png')

    # Assert block and unblock operations reverse one another
    np.testing.assert_array_equal(blocks_to_img(img_to_blocks(block_size, im1)), im1)

    # Assert that idct reverses dct
    blocks = img_to_blocks(block_size, im1)
    np.testing.assert_allclose(blocks, idct2(dct2(blocks)), atol=1e-10)