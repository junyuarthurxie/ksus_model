import numpy as np


def random2d(h, w, sparsity):
    mask = np.random.rand(h ,w)           
    return np.where(mask > np.percentile(mask, (1 - sparsity) * 100) , 1, 0)

def random1d(h, w, sparsity):
    mask = np.random.rand(h ,w)     
    for i in range(h):
        mask[i, :] = np.mean(mask[i, :])
    return np.where(mask > np.percentile(mask, (1 - sparsity) * 100) , 1, 0)

def random2dc(h, w, sparsity, r = 8):
    mask = np.random.rand(h,w)   
    lower_bound = np.int(np.ceil((r - 1) * h / (2 * r)))
    upper_bound = np.int((r + 1) * h / (2 * r))
    mask[lower_bound : upper_bound, lower_bound : upper_bound] = 1 
    #sparsity_eff = (sparsity - 1 / (r ** 2)) / (1 - 1 / (r ** 2))     
    return np.where(mask > np.percentile(mask, (1 - sparsity) * 100) , 1, 0)

def random1dc(h, w, sparsity, r = 8):
    mask = np.random.rand( h ,w)   
    for i in range(h):
        mask[i, :] = np.mean(mask[i, :])
    lower_bound = np.int(np.ceil((r - 1) * h / (2 * r)))
    upper_bound = np.int((r + 1) * h / (2 * r))
    mask[lower_bound : upper_bound, lower_bound : upper_bound] = 1 
    #sparsity_eff = (sparsity - 1 / (r ** 2)) / (1 - 1 / (r ** 2))      
    return np.where(mask > np.percentile(mask, (1 - sparsity) * 100) , 1, 0)

def uniform1dc(h, w, sparsity, r = 8):
    sparsity_eff = (sparsity - 1 / (r ** 2)) / (1 - 1 / (r ** 2))
    mask = np.zeros(( h ,w))   
    count = 0
    for i in range(h):
        if (i - (h % np.int(1 / sparsity_eff)) / 2 + 1) * sparsity_eff > count:
            mask[i, :] = 1
            count += 1
    lower_bound = np.int(np.ceil((r - 1) * h / (2 * r)))
    upper_bound = np.int((r + 1) * h / (2 * r))
    mask[lower_bound : upper_bound, lower_bound : upper_bound] = 1       
    return mask

def uniform1d(h, w, sparsity): 
    mask = np.zeros(( h ,w))   
    count = 0
    for i in range(h):
        if (i - (h % np.int(1 / sparsity)) / 2 - 1) * sparsity > count:
            mask[i, :] = 1
            count += 1    
    return mask

def gauss2d(h, w, sparsity, sig_ratio = 4.5):
    sig = h / sig_ratio
    def gauss(x, y, sig):
        return np.exp( -1 * (x ** 2 + y ** 2) / (2 * sig ** 2))
    hseq = np.linspace(- h / 2, h / 2, h)
    wseq = np.linspace(- w / 2, w / 2, w)
    x, y = np.meshgrid(hseq, wseq)
    z = gauss(x, y, sig) 
    mask = z / np.sum(z) * sparsity * h * w - np.random.rand(h, w)

    return np.where(mask > np.percentile(mask, (1 - sparsity) * 100) , 1, 0)

def gauss2dc(h, w, sparsity, r = 8, sig_ratio = 4.5):
    sig = h / sig_ratio
    def gauss(x, y, sig):
        return np.exp( -1 * (x ** 2 + y ** 2) / (2 * sig ** 2))
    hseq = np.linspace(- h / 2, h / 2, h)
    wseq = np.linspace(- w / 2, w / 2, w)
    x, y = np.meshgrid(hseq, wseq)
    z = gauss(x, y, sig) 
    mask = z / np.sum(z) * sparsity * h * w - np.random.rand(h, w)
    lower_bound = np.int(np.ceil((r - 1) * h / (2 * r)))
    upper_bound = np.int((r + 1) * h / (2 * r))
    mask[lower_bound : upper_bound, lower_bound : upper_bound] = 1   
    return np.where(mask > np.percentile(mask, (1 - sparsity) * 100) , 1, 0)


