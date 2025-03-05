import numpy as np
import torch
"""
def random_cutout(imgs, min_cut=10,max_cut=30):
    
    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, size=(n,c))
    h1 = np.random.randint(min_cut, max_cut, size=(n,c))
    x1 = np.random.randint(0, w-31, size=(n,c))
    y1 = np.random.randint(0, h-31, size=(n,c))
    
    cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    rand_box = np.random.randint(0, 255, size=(n, c)) / 255.
    for i in range(n):
        for j in range(c):
            cut_img = imgs[i][j].copy()
            if np.random.uniform(low=0, high=1) < 0.5:
                cut_img[x1[i,j]:x1[i,j]+w1[i,j], y1[i,j]:y1[i,j]+h1[i,j]] = np.tile(rand_box[i,j].reshape(-1,1,1), 
                            (w1[i,j], h1[i,j]))
            cutouts[i][j] = cut_img
    return cutouts

def random_jiterring(imgs, jitter_p = 0.02):
    n, c, h, w = imgs.shape
    jitter_noise = np.random.randint(-255*int((1-jitter_p) / jitter_p), 255, size=(n, c, h, w)) / 255.
    jitter_noise = np.clip((jitter_noise*2-1), 0, 1)
    jitters = imgs.copy()
    jitters = np.clip((jitters + jitter_noise), 0, 1)
    return jitters

def random_jiterring(imgs, jitter_p = 0.02):
    n, c, h, w = imgs.shape
    jitters = np.empty((n, c, h, w), dtype=imgs.dtype)

    for i in range(n):
        for j in range(c):
            cut_img = imgs[i][j].copy()
            if np.random.uniform(low=0, high=1) < 0.8:
                jitter_noise = np.random.randint(-255*int((1-jitter_p) / jitter_p), 255, size=(h, w)) / 255.
                jitter_noise = np.clip((jitter_noise*2-1), 0, 1)
                cut_img = np.clip((cut_img + jitter_noise), 0, 1)
            jitters[i][j] = cut_img
    return jitters

"""
# weaker data aug
"""
def random_cutout(imgs, min_cut=10,max_cut=30):
    
    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, size=(n,c))
    h1 = np.random.randint(min_cut, max_cut, size=(n,c))
    x1 = np.random.randint(0, w-31, size=(n,c))
    y1 = np.random.randint(0, h-31, size=(n,c))
    
    #cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    rand_box = np.random.randint(0, 255, size=(n, c)) / 255.
    cutouts = imgs.copy()
    for i in range(n):
        for j in range(c):
            if np.random.uniform(low=0, high=1) < 0.5:
                cutouts[i][j][x1[i,j]:x1[i,j]+w1[i,j], y1[i,j]:y1[i,j]+h1[i,j]] = np.tile(rand_box[i,j].reshape(-1,1,1), 
                            (w1[i,j], h1[i,j]))
    return cutouts

def random_jiterring(imgs):
    n, c, h, w = imgs.shape
    jitters = imgs.copy()
    for i in range(n):
        for j in range(c):
            if np.random.uniform(low=0, high=1) < 0.5:
                for _ in range(20):
                    jitter_x, jitter_y = np.random.randint(0, 256, size=(2))
                    jitters[i][j][jitter_x][jitter_y] = 1
    return jitters

"""
# Stronger data aug

def random_cutout(imgs, min_cut=10,max_cut=30):
    
    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, size=(n,c))
    h1 = np.random.randint(min_cut, max_cut, size=(n,c))
    x1 = np.random.randint(0, w-31, size=(n,c))
    y1 = np.random.randint(0, h-31, size=(n,c))

    w2 = np.random.randint(min_cut, max_cut, size=(n,c))
    h2 = np.random.randint(min_cut, max_cut, size=(n,c))
    x2 = np.random.randint(0, w-31, size=(n,c))
    y2 = np.random.randint(0, h-31, size=(n,c))
    
    #cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    rand_box = np.random.randint(0, 255, size=(n, c)) / 255.
    rand_box2 = np.random.randint(0, 255, size=(n, c)) / 255.
    cutouts = imgs.copy()
    for i in range(n):
        for j in range(c):
            if np.random.uniform(low=0, high=1) < 0.5:
                cutouts[i][j][x1[i,j]:x1[i,j]+w1[i,j], y1[i,j]:y1[i,j]+h1[i,j]] = np.tile(rand_box[i,j].reshape(-1,1,1), 
                            (w1[i,j], h1[i,j]))
                cutouts[i][j][x2[i,j]:x2[i,j]+w2[i,j], y2[i,j]:y2[i,j]+h2[i,j]] = np.tile(rand_box2[i,j].reshape(-1,1,1), 
                            (w2[i,j], h2[i,j]))
    return cutouts

def random_jiterring(imgs):
    n, c, h, w = imgs.shape
    jitters = imgs.copy()
    for i in range(n):
        for j in range(c):
            if np.random.uniform(low=0, high=1) < 0.5:
                for _ in range(20):
                    jitter_x, jitter_y = np.random.randint(0, 255, size=(2))
                    jitters[i][j][jitter_x][jitter_y] = 1
                    jitters[i][j][jitter_x+1][jitter_y] = 1
                    jitters[i][j][jitter_x][jitter_y+1] = 1
                    jitters[i][j][jitter_x+1][jitter_y+1] = 1
    return jitters