#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

TESTEI AQUI O L2 DAS CAMADAS INTERNAS (input_l2)

- regularization (L2, dropout) e outros parametros menos importantes

Rodar SSNP com GT e KMeans, MNIST,n_classes*2 variando:
-- regularização L2 = [0.0, 0.1, 0.5, 1.0, 2.0]
-- ativação interna (parametro act da classe SSNP): relu, sigmoid, tanh, leakyrelu
-- init: glorot_uniform, he_uniform, e mais um
-- early stopping: patience = [3, 5, 10]
"""

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    #tf.config.experimental.set_memory_growth(logical_gpus[0], True)
    for i in range(len(gpus)):
        tf.config.experimental.set_memory_growth(gpus[i], True)
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
    print('error!!!')


import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from glob import glob
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
from PIL import Image, ImageDraw, ImageFont
from skimage import io
from sklearn.cluster import (DBSCAN, AffinityPropagation,
                             AgglomerativeClustering, Birch, KMeans)
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import Isomap, MDS
from umap import UMAP
from tqdm import tqdm

import ae
import metrics
import ssnp
import nnproj

plt.switch_backend('agg')

def compute_all_metrics(X, X_2d, D_high, D_low, y, X_inv=None):
    T = metrics.metric_trustworthiness(X, X_2d, D_high, D_low)
    C = metrics.metric_continuity(X, X_2d, D_high, D_low)
    R = metrics.metric_shepard_diagram_correlation(D_high, D_low)
    S = metrics.metric_normalized_stress(D_high, D_low)
    N = metrics.metric_neighborhood_hit(X_2d, y, k=3)

    if X_inv is not None:
        MSE = metrics.metric_mse(X, X_inv)
    else:
        MSE = -99.0
    
    return T, C, R, S, N, MSE

def plot(X, y, figname=None):
    if len(np.unique(y)) <= 10:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = plt.get_cmap('tab20')

    fig, ax = plt.subplots(figsize=(20, 20))
    
    for cl in np.unique(y):
        ax.scatter(X[y==cl,0], X[y==cl,1], c=[cmap(cl)], label=cl, s=20)
        ax.axis('off')

    if figname is not None:
        fig.savefig(figname)

    plt.close('all')
    del fig
    del ax


if __name__ == '__main__':
    patience = 5
    epochs = 200
    
    min_delta = 0.05

    verbose = 2
    results = []

    output_dir = 'results_direct6_L2'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir ='../data'
    data_dirs = ['mnist']

    epochs_dataset = {}
    epochs_dataset['mnist'] = 10

    classes_mult = {}
    classes_mult['mnist'] = 2

    L2_weights = [0.0, 0.1, 0.5, 1.0, 2.0]

    for l2w in L2_weights:
        for d in tqdm(data_dirs):
            dataset_name = d

            X = np.load(os.path.join(data_dir, d, 'X.npy'))
            y = np.load(os.path.join(data_dir, d, 'y.npy'))

            print('------------------------------------------------------')
            print('Dataset: {0}'.format(dataset_name))
            print(X.shape)
            print(y.shape)
            print(np.unique(y))

            n_classes = len(np.unique(y)) * classes_mult[dataset_name]
            n_samples = X.shape[0]

            train_size = min(int(n_samples*0.9), 5000)

            X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size, random_state=420, stratify=y)
            D_high = metrics.compute_distance_list(X_train)

            epochs = epochs_dataset[dataset_name]

            ssnpgt = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear', input_l2=l2w)
            ssnpkm = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear', input_l2=l2w)

            projectors = [
                ssnpgt,
                ssnpkm,
            ]

            c_algs = [
                None, #SSNP_GT
                "KMeans(n_clusters=n_classes)",
            ]

            labels = [
                f'SSNP-GT-{l2w:.2f}',
                f'SSNP-KMeans-{l2w:.2f}',
                ]

            Xs = [] # Projected points
            Ds = [] # distances
            Ts = [] # time elapsed

            # print("Fitting ssnp GT")
            # ssnpgt = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear')
            # ssnpgt.fit(X_train, y_train)
            # print("Transforming GT")
            # X_ssnpgt = ssnpgt.transform(X_train)
            # Xs.append(X_ssnpgt)

            for P_, c, label in zip(projectors, c_algs, labels):
                hist = None
                if c is not None:
                    if c.startswith('DBSCAN'):
                        c = f"DBSCAN(eps=dbscan_eps['{d}'])"
                        print(f"Fit predict {c}")
                        y_ += 1 # adjust for the outliers in DBSCAN with label -1
                    print(f"Fit predict {c}")
                    C = eval(c)
                    y_ = C.fit_predict(X_train)
                else:
                    y_ = y_train
                print(f"Trying to fit transform {label}")
                try:
                    X_ = P_.fit_transform(X_train)
                except:
                    print(f"Fit_transform failed.")
                    print(f"Fitting {label}")
                    t0 = perf_counter()
                    try:
                        if isinstance(P_, ssnp.SSNP):
                            hist = P_.fit(X_train, y_)
                        else:
                            P_.fit(X_train, y_)
                    except:
                        P_.fit(X_train)
                    elapsed_time = perf_counter() - t0
                    print(f"Transforming {label}")
                    X_ = P_.transform(X_train)
                Xs.append(X_)
                Ds.append(metrics.compute_distance_list(X_))
                Ts.append(elapsed_time)
                if hist is not None:
                    # convert the history.history dict to a pandas DataFrame:
                    # ref: https://stackoverflow.com/a/55901240/3562468
                    hist_df = pd.DataFrame(hist.history)
                    hist_csv_file = f'history-{label}.csv'
                    with open(hist_csv_file, mode='w') as f:
                        hist_df.to_csv(os.path.join(output_dir,f))

            
            # print("Fitting TSNE")
            # tsne = TSNE(n_jobs=4, random_state=420)
            # print("Transforming TSNE")
            # X_tsne = tsne.fit_transform(X_train)

            # print("Fitting UMAP")
            # ump = UMAP(random_state=420)
            # print("Transforming UMAP")
            # X_umap = ump.fit_transform(X_train)

            # print("Fitting NNProj")
            # nnp = nnproj.NNProj(init=TSNE(n_jobs=4, random_state=420))
            # nnp.fit(X_train)
            # print("Transforming NNProj")
            # X_nnp = nnp.transform(X_train)

            for X_, label, D_, T_ in zip(Xs, labels, Ds, Ts):
                results.append((dataset_name, label, T_,) + compute_all_metrics(X_train, X_, D_high, D_, y_train))
                fname = os.path.join(output_dir, '{0}_{1}.png'.format(dataset_name, label))
                print(fname)
                plot(X_, y_train, fname)


            

        df = pd.DataFrame(results, columns=[    'dataset_name',
                                                'test_name',
                                                'time_elapsed',
                                                'T_train',
                                                'C_train',
                                                'R_train',
                                                'S_train',
                                                'N_train',
                                                'MSE_train'])

        df.to_csv(os.path.join(output_dir, 'metrics.csv'), header=True, index=None)

        #don't plot NNP
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 50)
        pri_images = [l for l in labels if l != "NNP"]

        images = glob(output_dir + '/*.png')    
        base = 2000

        for d in data_dirs:
            dataset_name = d
            to_paste = []    

            for i, label in enumerate(pri_images):
                to_paste += [f for f in images if os.path.basename(f) == '{0}_{1}.png'.format(dataset_name, label)]

            img = np.zeros((base, base*6, 3)).astype('uint8')
            
            for i, im in enumerate(to_paste):
                tmp = io.imread(im)
                img[:,i*base:(i+1)*base,:] = tmp[:,:,:3]

            pimg = Image.fromarray(img)
            pimg.save(output_dir + '/composite_full_{0}.png'.format(dataset_name))

            for i, label in enumerate(pri_images):
                print('/composite_full_{0}.png'.format(dataset_name), "{0} {1}".format(dataset_name, label))


        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 50)
        pri_images = ['SSNP-KMeans']

        images = glob(output_dir + '/*.png')    
        base = 2000

        for d in data_dirs:
            dataset_name = d
            to_paste = []    

            for i, label in enumerate(pri_images):
                to_paste += [f for f in images if os.path.basename(f) == '{0}_{1}.png'.format(dataset_name, label)]

            img = np.zeros((base, base*3, 3)).astype('uint8')
            
            for i, im in enumerate(to_paste):
                tmp = io.imread(im)
                img[:,i*base:(i+1)*base,:] = tmp[:,:,:3]

            pimg = Image.fromarray(img)
            pimg.save(output_dir + '/composite_{0}.png'.format(dataset_name))

            for i, label in enumerate(pri_images):
                print('/composite_{0}.png'.format(dataset_name), "{0} {1}".format(dataset_name, label))
