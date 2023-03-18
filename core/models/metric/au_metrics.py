import numpy as np
import torch
from sklearn.metrics import f1_score
from core.utils import METRIC_REGISTRY


@METRIC_REGISTRY.register()
class AUMetric:
    def __call__(self, pred, gt, sigmoid=True, threshold=0.5):
        """calc au metrics(with sigmoid)

        Args:
            pred (list(list())): pred list like [[-0.2, 0.1, 0.2, 0.3, ..], [...], ...]
            gt (list(list())): label list like [[0, 1, 0, ...], ...]
            threshold (float): threshold to determine whether active (default: 0.5)

        Returns:
            F1_mean (float): binary F1 score
            acc (float): accuracy
            F1 (list(float)): each au's F1 score
        """
        F1 = []
        if sigmoid:
            pred = torch.tensor(pred)
            pred = torch.sigmoid(pred)
        
        gt = np.array(gt)
        pred = np.array(pred)
        class_num = gt.shape[-1]

        index = [i for i in range(class_num)]

        for t in index:
            gt_ = gt[:, t]
            pred_ = pred[:, t]
            new_pred = ((pred_ >= threshold) * 1).flatten()
            F1.append(f1_score(gt_.flatten(), new_pred))

        F1_mean = np.mean(F1)

        counts = gt.shape[0]
        accs = 0
        for i in range(counts):
            pred_label = ((pred[i,:] >= threshold) * 1).flatten()
            gg = gt[i].flatten()
            j = 0
            for k in index:
                if int(gg[k]) == int(pred_label[k]):
                        j += 1
            if j == class_num:
                accs += 1

        acc = 1.0 * accs / counts

        return {'F1': F1_mean, 'ACC': acc, 'F1_list': F1}



