import numpy as np


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mape(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    mask = np.abs(y) > 1e-9
    return (
        float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)
        if mask.any()
        else 0.0
    )


def wmae(y, yhat, holiday=None):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    weights = (
        np.where(np.asarray(holiday).astype(bool), 5.0, 1.0)
        if holiday is not None
        else np.ones_like(y)
    )
    return float(np.sum(weights * np.abs(y - yhat)) / np.sum(weights))
