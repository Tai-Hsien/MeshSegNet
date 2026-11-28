import torch
import numpy as np

def weighting_DSC(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    # all classes at once: sum over batch and cells, keep the class axis
    intersection = (one_hot * y_true).sum(dim=(0, 1))
    w = class_weights / class_weights.sum()
    return (w * (2. * intersection + smooth) / (one_hot.sum(dim=(0, 1)) + y_true.sum(dim=(0, 1)) + smooth)).sum()


def weighting_SEN(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    # all classes at once: sum over batch and cells, keep the class axis
    intersection = (one_hot * y_true).sum(dim=(0, 1))
    w = class_weights / class_weights.sum()
    return (w * (intersection + smooth) / (y_true.sum(dim=(0, 1)) + smooth)).sum()


def weighting_PPV(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    # all classes at once: sum over batch and cells, keep the class axis
    intersection = (one_hot * y_true).sum(dim=(0, 1))
    w = class_weights / class_weights.sum()
    return (w * (intersection + smooth) / (one_hot.sum(dim=(0, 1)) + smooth)).sum()

   
def Generalized_Dice_Loss(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.

    # all classes at once: sum over batch and cells, keep the class axis
    intersection = (y_pred * y_true).sum(dim=(0, 1))
    w = class_weights / class_weights.sum()
    return (w * (1 - (2. * intersection + smooth) /
                 (y_pred.sum(dim=(0, 1)) + y_true.sum(dim=(0, 1)) + smooth))).sum()


def DSC(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    dsc = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
        
    return dsc


def SEN(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    sen = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            sen.append(((intersection + smooth) / (true_flat.sum() + smooth)))
            
        sen = np.asarray(sen)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            sen.append(((intersection + smooth) / (true_flat.sum() + smooth)))
            
        sen = np.asarray(sen)
        
    return sen


def PPV(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    ppv = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            ppv.append(((intersection + smooth) / (pred_flat.sum() + smooth)))
            
        ppv = np.asarray(ppv)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            ppv.append(((intersection + smooth) / (pred_flat.sum() + smooth)))
            
        ppv = np.asarray(ppv)
        
    return ppv
