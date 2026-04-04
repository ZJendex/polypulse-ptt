"""Evaluation: peak detection accuracy and pulse-transit-time (PTT) error metrics."""

import torch
import numpy as np
from scipy.signal import find_peaks
from scipy import stats


class PulseEval:
    def __init__(self, peak_min_distance: int = 250, site=None):
        self.peak_min_distance = peak_min_distance
        self.site = site

    def peak_detection(self, target, pred, peak_min_height=0.25):
        """Detect predicted peaks and match each ground-truth peak to the nearest prediction.

        Unmatched GT peaks (no predicted peak within peak_min_distance//2) get None.
        Returns: (target_peaks, matched_peaks, signed_distances, peak_heights) per batch.
        """
        target_peaks_list, pred_peaks_list, pred_heights_list = [], [], []
        for i in range(target.shape[0]):
            peaks, props = find_peaks(pred[i].squeeze(), height=peak_min_height, distance=self.peak_min_distance)
            pred_peaks_list.append(torch.tensor(peaks))
            pred_heights_list.append(props['peak_heights'])
            target_peaks_list.append(np.where(target[i] == 1)[0])

        matched_peaks_list, signed_distances_list, peak_heights_list = [], [], []
        for i, (pred_peaks, target_peaks) in enumerate(zip(pred_peaks_list, target_peaks_list)):
            matched_peaks, signed_distances, peak_heights = [], [], []

            if len(pred_peaks) == 0:
                matched_peaks_list.append([None] * len(target_peaks))
                signed_distances_list.append([None] * len(target_peaks))
                peak_heights_list.append([0] * len(target_peaks))
                continue

            distances = np.expand_dims(pred_peaks, 1) - np.expand_dims(target_peaks, 0)
            abs_distances = np.abs(distances)
            min_idx = np.argmin(abs_distances, axis=0)

            for pi, min_dist in enumerate(np.min(abs_distances, axis=0)):
                if min_dist < self.peak_min_distance // 2:
                    matched_peaks.append(pred_peaks[min_idx[pi]])
                    signed_distances.append(distances[min_idx[pi], pi])
                    peak_heights.append(pred_heights_list[i][min_idx[pi]])
                else:
                    matched_peaks.append(None)
                    signed_distances.append(None)
                    peak_heights.append(0)

            matched_peaks_list.append(matched_peaks)
            signed_distances_list.append(signed_distances)
            peak_heights_list.append(peak_heights)

        return target_peaks_list, matched_peaks_list, signed_distances_list, peak_heights_list

    def peak_error(self, pred: torch.Tensor, target: torch.Tensor, heights: list = [], debug_fnames=None):
        """Compute count-error rate and distance errors across height thresholds.

        Single detection pass; each threshold filters by predicted peak amplitude.
        """
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        if len(heights) == 0:
            heights = np.arange(0.1, 0.96, 0.01)

        target_peaks_list, matched_peaks_list, signed_distances_list, peak_heights_list = \
            self.peak_detection(target_np, pred_np)

        avg_count_error_rates, distance_errors, signed_distance_errors = [], [], []
        for height in heights:
            matched_all, signed_all, total_peaks = [], [], 0
            for target_peaks, _, signed_distances, peak_heights in zip(
                target_peaks_list, matched_peaks_list, signed_distances_list, peak_heights_list
            ):
                valid = np.where(np.array(peak_heights) > height)[0]
                matched_all.extend([abs(d) for i, d in enumerate(signed_distances) if i in valid and d is not None])
                signed_all.extend([d for i, d in enumerate(signed_distances) if i in valid and d is not None])
                total_peaks += len(target_peaks)

            avg_count_error_rates.append((total_peaks - len(matched_all)) / total_peaks)
            distance_errors.append(matched_all)
            signed_distance_errors.append(signed_all)

        return heights, avg_count_error_rates, distance_errors, signed_distance_errors

    def pulse_transit_time(self, peaks_list_1, peaks_list_2, min_dist, max_dist,
                           height_list_1=None, height_list_2=None):
        """Per-beat PTT: for each site-1 peak find closest site-2 peak within [min_dist, max_dist]."""
        ptts_list, all_peaks, all_heights = [], [], []
        for i, (p1, p2) in enumerate(zip(peaks_list_1, peaks_list_2)):
            p1c = np.array([float('inf') if x is None else x for x in p1])
            p2c = np.array([float('inf') if x is None else x for x in p2])
            dists = np.expand_dims(p1c, 1) - np.expand_dims(p2c, 0)
            valid = (dists > min_dist) & (dists < max_dist)

            for row in range(valid.shape[0]):
                vi = np.nonzero(valid[row])[0]
                ptts_list.append(dists[row, vi[0]] if len(vi) > 0 else None)
                if height_list_1 is not None and height_list_2 is not None:
                    all_heights.append(
                        height_list_1[i][row] + height_list_2[i][vi[0]] if len(vi) > 0 else None
                    )
            all_peaks.extend(p1)

        if height_list_1 is not None and height_list_2 is not None:
            return np.array(ptts_list), np.array(all_peaks), np.array(all_heights)
        return np.array(ptts_list), np.array(all_peaks)

    def remove_outliers(self, seq1, seq2):
        """Drop paired samples where either has |z-score| >= 2."""
        z1 = stats.zscore(np.array(seq1, dtype=np.float64))
        z2 = stats.zscore(np.array(seq2, dtype=np.float64))
        mask = (np.abs(z1) < 2) & (np.abs(z2) < 2)
        return seq1[mask], seq2[mask]

    def smooth_sequence(self, seq):
        return np.convolve(seq, np.ones(5) / 5, mode='valid')

    def ptt_error(self, pred, target, ptt_queries, height_thrs, topn=20):
        """Evaluate PTT accuracy for each (site1, site2, min_dist, max_dist) query."""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        assert pred_np.shape[1] == target_np.shape[1] == len(height_thrs)

        target_peaks_sites, pred_peaks_sites, pred_heights_sites = [], [], []
        for i in range(pred_np.shape[1]):
            tp, mp, _, ph = self.peak_detection(target_np[:, i, :], pred_np[:, i, :], peak_min_height=height_thrs[i])
            target_peaks_sites.append(tp)
            pred_peaks_sites.append(mp)
            pred_heights_sites.append(ph)

        metrics, full_data = [], []
        for s1, s2, lo, hi in ptt_queries:
            gt_ptts, gt_peaks = self.pulse_transit_time(
                target_peaks_sites[s1], target_peaks_sites[s2], lo, hi)
            pred_ptts, pred_peaks, pred_heights = self.pulse_transit_time(
                pred_peaks_sites[s1], pred_peaks_sites[s2], lo, hi,
                pred_heights_sites[s1], pred_heights_sites[s2])
            assert len(gt_ptts) == len(pred_ptts)

            gt_valid = np.where(gt_ptts != None)[0]
            gt_rate = len(gt_valid) / len(gt_ptts)
            both_valid = np.where((pred_ptts != None) & (gt_ptts != None))[0]
            pred_rate = len(both_valid) / len(pred_ptts)

            gv, pv = gt_ptts[both_valid], pred_ptts[both_valid]
            gv, pv = self.remove_outliers(gv, pv)
            if len(gv) > 0:
                gv, pv = self.smooth_sequence(gv), self.smooth_sequence(pv)

            metrics.append({'gt_ptt_rate': gt_rate, 'pred_ptt_rate': pred_rate,
                            'ptt_err': np.median(np.abs(gv - pv))})
            full_data.append({'gt_ptt': gv, 'pred_ptt': pv})

        return metrics, full_data
