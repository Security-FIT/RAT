from scipy.stats import norm
from sklearn.metrics import det_curve, DetCurveDisplay
import numpy as np
import matplotlib.pyplot as plt

def calculate_EER(labels, predictions, name=None, plot_det: bool = False, det_subtitle: str = "") -> float:
        """
        Calculate the Equal Error Rate (EER) from the labels and predictions
        """
        fpr, fnr, _ = det_curve(labels, predictions, pos_label=0)

        # eer from fpr and fnr can differ a bit (its an approximation), so we compute both and take the average
        eer_fpr = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer_fnr = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = (eer_fpr + eer_fnr) / 2

        # Display the DET curve
        if plot_det:
            # eer_fpr_probit = norm.ppf(eer_fpr)
            # eer_fnr_probit = norm.ppf(eer_fnr)
            eer_probit = norm.ppf(eer)

            DetCurveDisplay(fpr=fpr, fnr=fnr, pos_label=0).plot()
            # plt.plot(
            #     eer_fpr_probit,
            #     eer_fpr_probit,
            #     marker="o",
            #     markersize=5,
            #     label=f"EER from FPR: {eer:.2f}",
            #     color="blue",
            # )
            # plt.plot(
            #     eer_fnr_probit,
            #     eer_fnr_probit,
            #     marker="o",
            #     markersize=5,
            #     label=f"EER from FNR: {eer:.2f}",
            #     color="green",
            # )
            plt.plot(eer_probit, eer_probit, marker="o", markersize=4, label=f"EER: {eer:.2f}", color="red")
            plt.legend()
            plt.title(f"DET Curve {name} {det_subtitle}")
            plt.savefig(f"./{name}_{det_subtitle}_DET.png")

        return eer

def calculate_minDCF(labels, predictions, p_target=0.95, c_miss=1, c_fa=10) -> float:
    """
    Calculate the minimum Detection Cost Function (minDCF)
    """
    far, frr, thresholds = det_curve(labels, predictions, pos_label=0)

    c_det = c_miss * frr * p_target + c_fa * far * (1 - p_target)
    min_c_det = np.min(c_det)

    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    
    return min_dcf
