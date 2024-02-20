import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix


def sharing(dev, weight, shape, module, fc_bits, conv_bits, kmp: bool = False):
    if len(shape) == 2:  # FC layer

        sum1 = 0
        for _ in np.unique(weight):
            sum1 += 1
        print("Unique element number:", sum1)

        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num=2 ** fc_bits)

        if kmp:
            kmeans = KMeans(n_clusters=len(space), init='k-means++', n_init=1)
        else:
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1)

        kmeans.fit(mat.data.reshape(-1, 1))

        sum2 = 0
        for _ in np.unique(kmeans.cluster_centers_[kmeans.labels_]):
            sum2 += 1
        print("Unique element number after weight sharing:", sum2)

        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)

        # np.copyto(mat, new_weight)
        mat.data = new_weight
        module.weight.data = torch.from_numpy(mat.toarray()).to(dev)

    elif len(shape) == 4:  # Conv layer

        sum1 = 0
        for _ in np.unique(weight):
            sum1 += 1
        print("Unique element number before weight sharing:", sum1)

        mat = weight.flatten()
        co_mat_ones = np.ones(mat.size)
        co_mat_zeros = np.zeros(mat.size)
        co_mat = np.where(mat == 0, co_mat_zeros, co_mat_ones)
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num=2 ** conv_bits)

        if kmp:
            kmeans = KMeans(n_clusters=len(space), init='k-means++', n_init=1)
        else:
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1)

        kmeans.fit(mat.reshape(-1, 1))

        sum2 = 0
        for _ in np.unique(kmeans.cluster_centers_[kmeans.labels_]):
            sum2 += 1
        print("Unique element number after weight sharing:", sum2)

        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)

        np.copyto(mat, new_weight)
        mat = mat * co_mat
        mat = mat.reshape(shape)
        module.weight.data = torch.from_numpy(mat).float().to(dev)

    else:
        print(f"Module: {module} cannot be quantized.")


def apply_weight_sharing(model, fc_bits: int = 5, conv_bits: int = 5, kmp: bool = False):
    r"""
    Applies weight sharing to the given model
    """

    for module in model.children():
        try:
            dev = module.weight.device
            weight = module.weight.data.cpu().numpy()
            shape = weight.shape

            sharing(dev, weight, shape, module, fc_bits, conv_bits, kmp)

        except:
            print(f"Module: {module} cannot be quantized.")
            continue


def apply_weight_sharing_block(model, fc_bits: int = 5, conv_bits: int = 5, kmp: bool = False):
    r"""
    Applies weight sharing to the given model
    """

    flag1 = flag2 = flag3 = flag4 = flag5 = True

    for level1 in model.children():
        try:
            dev = level1.weight.device
            weight = level1.weight.data.cpu().numpy()
            shape = weight.shape
            sharing(dev, weight, shape, level1, fc_bits, conv_bits, kmp)
        except:
            for level2 in level1.children():
                try:
                    dev = level2.weight.device
                    weight = level2.weight.data.cpu().numpy()
                    shape = weight.shape
                    sharing(dev, weight, shape, level2, fc_bits, conv_bits, kmp)
                    flag1 = False
                except:
                    for level3 in level2.children():
                        try:
                            dev = level3.weight.device
                            weight = level3.weight.data.cpu().numpy()
                            shape = weight.shape
                            sharing(dev, weight, shape, level3, fc_bits, conv_bits, kmp)
                            flag1 = flag2 = False
                        except:
                            for level4 in level3.children():
                                try:
                                    dev = level4.weight.device
                                    weight = level4.weight.data.cpu().numpy()
                                    shape = weight.shape
                                    sharing(dev, weight, shape, level4, fc_bits, conv_bits, kmp)
                                    flag1 = flag2 = flag3 = False
                                except:
                                    for level5 in level4.children():
                                        try:
                                            dev = level5.weight.device
                                            weight = level5.weight.data.cpu().numpy()
                                            shape = weight.shape
                                            sharing(dev, weight, shape, level5, fc_bits, conv_bits, kmp)
                                            flag1 = flag2 = flag3 = flag4 = flag5 = False
                                        except:
                                            if flag5:
                                                print(f"Module: {level5} cannot be quantized.")
                                    pass
                                    if flag4:
                                        print(f"Module: {level4} cannot be quantized.")
                            pass
                            if flag3:
                                print(f"Module: {level3} cannot be quantized.")
                    pass
                    if flag2:
                        print(f"Module: {level2} cannot be quantized.")
            pass
            if flag1:
                print(f"Module: {level1} cannot be quantized.")
