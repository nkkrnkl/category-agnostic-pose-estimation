import numpy as np

def compute_f1(quant_result_dict, metric_category):
    for metric in metric_category:
        prec = quant_result_dict[metric+'_prec']
        rec = quant_result_dict[metric+'_rec']
        f1 = 2*prec*rec/(prec+rec+1e-5)
        quant_result_dict[metric+'_f1'] = f1
    return quant_result_dict