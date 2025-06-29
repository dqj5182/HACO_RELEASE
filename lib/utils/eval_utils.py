import torch
import numpy as np


def evaluation(outputs, targets_data, meta_info, mode='val', thres=0.5):
    eval_out = {}

    # GT
    mesh_valid = meta_info['mano_valid'] is not None

    # Pred
    contact_pred = outputs['contact_out'].sigmoid()[0].detach().cpu().numpy()

    # Error Calculate
    if mesh_valid:
        # Contact Metrics
        cont_pre, cont_rec, cont_f1 = compute_contact_metrics(targets_data['contact_data']['contact_h'][0].detach().cpu().numpy(), contact_pred, mesh_valid, thres=thres)
        eval_out['cont_pre'] = cont_pre
        eval_out['cont_rec'] = cont_rec
        eval_out['cont_f1'] = cont_f1

    return eval_out


def compute_contact_metrics(gt, pred, valid, thres=0.5):
    """
    Compute precision, recall, and f1 using NumPy
    """
    if valid:
        # True Positives
        tp_num = np.sum(gt[pred >= thres])

        # Denominators for precision and recall
        precision_denominator = np.sum(pred >= thres)
        recall_denominator = np.sum(gt)

        # Compute precision, recall, and F1 score
        precision_ = tp_num / precision_denominator if precision_denominator > 0 else None
        recall_ = tp_num / recall_denominator if recall_denominator > 0 else None
        if precision_ is not None and recall_ is not None and (precision_ + recall_) > 0:
            f1_ = 2 * precision_ * recall_ / (precision_ + recall_)
        else:
            f1_ = None
    else:
        # If not valid, return None for metrics
        precision_ = None
        recall_ = None
        f1_ = None

    return precision_, recall_, f1_