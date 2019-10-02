# _*_ coding: utf-8 _*_
"""
    Author: Kwong
    Create time: 2019/10/2 21:51 
"""
import numpy as np

"""
    confusion_matrix calculate
    from sklearn.metrics import confusion_matrix
"""
# code from https://tianws.github.io/skill/2018/10/30/miou/


class Measure:
    """
        input: pred_img, label
                [W * H]
        if there is a batch, change the dimension with function view()
        output:
                precesion, recall, mIoU, f1
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros([num_classes, num_classes])

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, pred, lab):
        pred = pred.detach().numpy().astype(np.uint8)
        lab = lab.detach().numpy().astype(np.uint8)
        self.hist = self._fast_hist(pred, lab)

    def evaluate(self):
        tp = np.diag(self.hist)
        fn = self.hist.sum(axis=0) - tp
        fp = self.hist.sum(axis=1) - tp
        precesion = tp.sum()/(tp + fp).sum()
        recall = np.mean(tp / (fn + tp))
        mIoU = np.mean(tp / (fn + fp + tp))
        # freq = self.hist.sum(axis=1) / self.hist.sum()
        # fwavacc = (freq[freq > 0] * miu[freq > 0]).sum()
        f1 = 2 * precesion * recall / (precesion + recall)
        return precesion, recall, mIoU, f1


if __name__ == "__main__":
    import torch
    a = torch.randint(low=0, high=2, size=[3, 3, 3]).view(-1)
    b = torch.randint(low=0, high=2, size=[3, 3, 3]).view(-1)
    print(a)
    print(b)
    miou = Measure(2)
    miou.add_batch(a, b)
    print(miou.evaluate())