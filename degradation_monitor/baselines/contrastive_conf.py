"""ContrastiveConf (Park, Sobolewski & Azizan, TPAMI 2026): Conf+ - lambda Conf- over a frozen DETR's queries, with lambda cross-fitted on labelled clean images."""
from __future__ import annotations

import warnings

import numpy as np
import uq_detr
from scipy.special import expit
from scipy.stats import ConstantInputWarning, pearsonr

from ..detector.postprocess import checked_outputs as _checked

THETA = 0.3  # uq-detr's threshold split between positive and negative queries
UQ_DETR_LAMBDA_GRID = (0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20)  # uq_detr.fit_lambda default


def query_detections(logits, boxes_cxcywh, image_size) -> uq_detr.Detections:
    """All queries of one image as uq-detr Detections (image_size is width, height)."""
    logits, boxes = _checked(logits, boxes_cxcywh)
    width, height = image_size
    return uq_detr.Detections.from_cxcywh(boxes, expit(logits), image_size=(height, width))


def contrastive_parts(query_sets, theta: float = 0.3):
    """Conf+ and Conf- from uq-detr's threshold split, recovered from two lambda values."""
    conf_pos = np.asarray(
        uq_detr.contrastive_conf(query_sets, method="threshold", param=theta, lambda_=0.0), float)
    difference = np.asarray(
        uq_detr.contrastive_conf(query_sets, method="threshold", param=theta, lambda_=1.0), float)
    return conf_pos, conf_pos - difference


def contrastive_degradation(conf_pos, conf_neg, lam: float) -> np.ndarray:
    """Higher means more likely degraded: minus ContrastiveConf."""
    return -(np.asarray(conf_pos, float) - lam * np.asarray(conf_neg, float))


def fit_lambda_from_parts(conf_pos, conf_neg, reliability, grid=UQ_DETR_LAMBDA_GRID):
    """uq_detr.fit_lambda on precomputed Conf+/Conf-: first lambda with the highest Pearson r."""
    conf_pos, conf_neg, reliability = (np.asarray(v, float) for v in (conf_pos, conf_neg, reliability))
    valid = ~np.isnan(reliability)
    if valid.sum() < 3:
        raise ValueError("need at least 3 images with a defined reliability value")
    best_lambda, best_pcc = 0.0, -np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for lam in grid:
            pcc = pearsonr(conf_pos[valid] - lam * conf_neg[valid], reliability[valid])[0]
            if pcc > best_pcc:
                best_lambda, best_pcc = float(lam), float(pcc)
    return best_lambda, best_pcc


def cross_fit_lambda(conf_pos, conf_neg, reliability, folds):
    """Fit lambda for each fold on the other folds only."""
    conf_pos, conf_neg, reliability, folds = (np.asarray(v) for v in (conf_pos, conf_neg, reliability, folds))
    per_fold = {}
    for fold in np.unique(folds):
        others = folds != fold
        per_fold[int(fold)] = fit_lambda_from_parts(conf_pos[others], conf_neg[others], reliability[others])[0]
    return np.array([per_fold[int(f)] for f in folds]), per_fold
