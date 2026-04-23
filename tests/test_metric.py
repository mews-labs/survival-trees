import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

from survival_trees import metric
from survival_trees import plotting


def test_auc_():
    t1 = np.linspace(0, 1, num=100)
    t2 = np.linspace(1, 0, num=100)
    t3 = np.linspace(2, 0, num=100)
    t4 = np.linspace(0, 2, num=100)

    data = 1 - pd.DataFrame(np.array([t1, t2, t3, t4]))
    target = np.array([0, 1, 1, 0])
    time = np.array([90, 20, 90, 40])
    fig, ax = plot.subplots(ncols=2)
    plot.sca(ax[0])
    plotting.tagged_curves(data, label=target, time_event=time)

    metric.time_dependent_auc(data, target, time, method="harrell").plot(ax=ax[1], label="Harrel index", alpha=0.5)
    metric.time_dependent_auc(data, target, time, method="roc-cd").plot(ax=ax[1], label="$AUC^{C, D}$", alpha=0.5)
    (metric.concordance_index(-data, target, time) + 0.01).plot(ax=ax[1], label="Concordance index", alpha=0.5)
    metric.time_dependent_auc(data, target, time, method="roc-id").plot(ax=ax[1], label="$AUC^{I, D}$", marker=".", lw=0, alpha=0.5)
    ax[1].legend(loc=4)


def test_t_auc():
    t1 = np.linspace(0, 1, num=3)
    t2 = np.linspace(1, 0, num=3)
    t3 = np.linspace(2, 0, num=3)
    t4 = np.linspace(0, 2, num=3)

    data = 1 - pd.DataFrame(np.array([t1, t2, t3, t4]))
    target = np.array([0, 1, 1, 0])
    time = np.array([2, 1, 2, 1])

    harrel = metric.time_dependent_auc(data, target, time, method="harrell")
    roc_cd = metric.time_dependent_auc(data, target, time, method="roc-cd")
    roc_id = metric.time_dependent_auc(data, target, time, method="roc-id")

    assert (np.array(harrel) == np.array((0, 0.5, 1))).all(),   "Error in Harrel index computation"
    assert (np.array(roc_cd) == np.array((5/6, 1))).all(),      "Error in ROC CD computation"
    assert (np.array(roc_id) == np.array((5/6, 5/6))).all(),    "Error in ROC CID  index computation"
