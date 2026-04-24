# model.py

"""
Contains all data structures and pure computational logic, independent of the user interface. This file never uses Tkinter or matplotlib figure handling. It includes:
- DICOMCache: Manages loading and caching of DICOM images and their calibration data.
- LCS: Implements the Low Contrast Separation algorithm for ultrasound image quality analysis.
- LCPCalculator: Implements the Low Contrast Penetration calculations using two images and a rectangular ROI.
- PixelSpacingHelper: Utility class to extract pixel spacing from DICOM metadata or user calibration.
"""
import os
import numpy as np
import pydicom
import cv2
from scipy.signal import medfilt


class DICOMCache:
    def __init__(self, cache_window=8):
        self.cache_window = cache_window
        self.files = []
        self._cache = {}          # index -> (image_array, mm_per_pix)

    def set_file_list(self, file_list):
        """Set the list of DICOM file paths (must be sorted by InstanceNumber)."""
        self.files = file_list
        self._cache.clear()
        self._current_index = 0

    def _load_one(self, index):
        ds = pydicom.dcmread(self.files[index])
        img = ds.pixel_array

        # Convert to 2D grayscale
        if img.ndim == 3:
            # Could be multi-frame OR colour
            if img.shape[-1] == 3:   # RGB or YBR
                # Convert colour to grayscale using luminosity weights
                img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
            else:
                # Assume multi-frame: take the first frame
                img = img[0]
        elif img.ndim > 3:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}")

        # Ensure we have a 2D array
        if img.ndim != 2:
            raise ValueError(f"After processing, image is not 2D. Shape: {img.shape}")

        # Convert to float32 and normalise to [0,1]
        img = img.astype(np.float32)
        max_val = img.max()
        if max_val > 0:
            img = img / max_val

        # Optional: invert if MONOCHROME1 (white = background)
        if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == 'MONOCHROME1':
            img = 1.0 - img

        mm = PixelSpacingHelper.from_dicom(ds)
        return img, mm

    def _update_cache(self, index):
        total = len(self.files)
        if total == 0:
            return
        low = max(0, index - self.cache_window)
        high = min(total - 1, index + self.cache_window)
        for idx in range(low, high + 1):
            if idx not in self._cache:
                self._cache[idx] = self._load_one(idx)
        to_remove = [idx for idx in self._cache if idx < low or idx > high]
        for idx in to_remove:
            del self._cache[idx]

    def get_image(self, index):
        """Return only the image array (2D float32)."""
        self._update_cache(index)
        img, _ = self._cache[index]
        return img

    def get_calibration(self, index):
        """Return mm_per_pix for the given image index."""
        self._update_cache(index)
        _, mm = self._cache[index]
        return mm

    def get_image_and_calib(self, index):
        """Return (image, mm_per_pix) tuple."""
        self._update_cache(index)
        return self._cache[index]
    
    def get_current_index(self):
        return self._current_index

    def total_count(self):
        return len(self.files)


class LCS:
    """
    Low Contrast Separation (LCS) processing class.
    Ported from Java to Python for analysing ultrasound image quality.
    See 'A COMPUTERISED QUALITY CONTROL TESTING SYSTEM FOR B-MODE ULTRASOUND (2001), Gibson et al'
    """
    @staticmethod
    def process_circle(img, inr=0.7, outr=1.35):
        user_roi = (0, 0, img.shape[1], img.shape[0])
        in_roi = LCS.expand_circle(user_roi, inr)
        out_roi = LCS.expand_circle(user_roi, outr)
        in_pixels = LCS.get_pixels_on_circle(in_roi, 0.05, img)
        out_pixels = LCS.get_pixels_on_circle(out_roi, 0.05, img)
        if len(in_pixels) == 0 or len(out_pixels) == 0:
            return None
        return LCS.mean(in_pixels) / LCS.mean(out_pixels)

    @staticmethod
    def get_pixels_on_circle(roi_bounds, tol, img):
        x, y, width, height = roi_bounds
        x0 = x + width / 2
        y0 = y + height / 2
        x_semi_axis = width / 2
        y_semi_axis = height / 2
        result = []
        for px in np.arange(x0 - x_semi_axis - 20, x0 + x_semi_axis + 20, 1):
            for py in np.arange(y0 - y_semi_axis - 20, y0 + y_semi_axis + 20, 1):
                d = ((px - x0) / x_semi_axis) ** 2 + ((py - y0) / y_semi_axis) ** 2
                if (1 - tol) <= d <= (1 + tol):
                    px_int = int(np.round(px))
                    py_int = int(np.round(py))
                    if 0 <= px_int < img.shape[1] and 0 <= py_int < img.shape[0]:
                        result.append(img[py_int, px_int])
        return np.array(result, dtype=np.float64)

    @staticmethod
    def expand_circle(rect, factor):
        x, y, width, height = rect
        new_width = width * factor
        new_height = height * factor
        new_x = x - (new_width - width) / 2
        new_y = y - (new_height - height) / 2
        return (new_x, new_y, new_width, new_height)

    @staticmethod
    def mean(data):
        if len(data) == 0:
            return 0.0
        return np.sum(data) / float(len(data))

    @staticmethod
    def sd(data):
        if len(data) <= 1:
            return 0.0
        mean_val = LCS.mean(data)
        variance = np.sum((data - mean_val) ** 2) / (len(data) - 1.0)
        return np.sqrt(variance)

    @staticmethod
    def calculate_contrast_index(image, circle_x, circle_y, radius, inr=0.7, outr=1.35):
        x_min = max(0, int(circle_x - radius))
        y_min = max(0, int(circle_y - radius))
        x_max = min(image.shape[1], int(circle_x + radius))
        y_max = min(image.shape[0], int(circle_y + radius))
        roi = image[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None
        roi_width = x_max - x_min
        roi_height = y_max - y_min
        roi_bounds = (0, 0, roi_width, roi_height)
        in_roi = LCS.expand_circle(roi_bounds, 0.7)
        out_roi = LCS.expand_circle(roi_bounds, 1.35)
        in_pixels = LCS.get_pixels_on_circle(in_roi, 0.05, roi)
        out_pixels = LCS.get_pixels_on_circle(out_roi, 0.05, roi)
        if len(in_pixels) == 0 or len(out_pixels) == 0:
            return None
        mean_in = LCS.mean(in_pixels)
        mean_out = LCS.mean(out_pixels)
        sd_in = LCS.sd(in_pixels)
        sd_out = LCS.sd(out_pixels)
        contrast_index = mean_in / mean_out
        se = np.sqrt((sd_in**2 / len(in_pixels)) + (sd_out**2 / len(out_pixels)))
        return {
            'contrast_index': contrast_index,
            'mean_inner': mean_in,
            'mean_outer': mean_out,
            'sd_inner': sd_in,
            'sd_outer': sd_out,
            'standard_error': se,
            'num_pixels_inner': len(in_pixels),
            'num_pixels_outer': len(out_pixels)
        }


class LCPCalculator:
    """
    Low Contrast Penetration (LCP) calculations.
    Uses two images (pre‑ and post‑frame) and a rectangular ROI.
    """
    @staticmethod
    def crop_image(img_array, roi):
        """Return cropped 2D region from a normalised image array."""
        x1, y1, x2, y2 = roi
        y_min = int(min(y1, y2))
        y_max = int(max(y1, y2))
        x_min = int(min(x1, x2))
        x_max = int(max(x1, x2))
        return img_array[y_min:y_max, x_min:x_max]

    @staticmethod
    def calculate_sum_diff_images(roi1, roi2):
        """Compute sum and difference images (both 2D)."""
        sum_img = roi1 + roi2
        diff_img = roi1 - roi2
        return sum_img, diff_img

    @staticmethod
    def calculate_std_dev(sum_img, diff_img, window_size=3):
        number_of_steps = (sum_img.shape[0] - 10) // window_size + 1
        noise_std = []
        signal_std = []
        snr = []
        for i in range(number_of_steps):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, sum_img.shape[0])
            sum_slice = sum_img[start_idx:end_idx, :]
            diff_slice = diff_img[start_idx:end_idx, :]
            noise = np.std(diff_slice) * (2 ** -0.5)
            noise_std.append(noise)
            var_sum = np.std(sum_slice) ** 2
            var_diff = np.std(diff_slice) ** 2
            signal = 0.5 * np.sqrt(max(0, var_sum - var_diff))
            signal_std.append(signal)
            if noise > 1e-6:
                snr.append(signal / noise)
            else:
                snr.append(0.0)
        depth_pixels = list(range(0, number_of_steps * window_size, window_size))

        # Apply 7‑point median filter
        noise_std = medfilt(noise_std, 7)
        signal_std = medfilt(signal_std, 7)

        # Apply 7‑point moving average
        kernel = np.ones(7) / 7
        noise_std = np.convolve(noise_std, kernel, mode='same')
        signal_std = np.convolve(signal_std, kernel, mode='same')

        return np.array(signal_std), np.array(noise_std), np.array(snr), np.array(depth_pixels)

    @staticmethod
    def determine_lcp_depth(snr, depth_pixels, mm_per_pix, threshold=2.0, skip_initial_fraction=0.5):
        """
        Find the first depth where SNR falls below threshold, ignoring the initial part.
        
        Parameters:
            snr : array of SNR values
            depth_pixels : array of depths in pixels
            mm_per_pix : conversion factor
            threshold : SNR threshold (default 2.0)
            skip_initial_fraction : fraction of total depth to skip before checking (default 0.5)
        Returns:
            LCP depth in mm, or None if never below threshold after the skipped region.
        """
        if len(depth_pixels) == 0 or len(snr) == 0:
            return None
        
        total_depth_mm = depth_pixels[-1] * mm_per_pix
        skip_depth_mm = total_depth_mm * skip_initial_fraction
        
        # Find first index where depth >= skip_depth_mm
        start_idx = 0
        for i, d_pix in enumerate(depth_pixels):
            if d_pix * mm_per_pix >= skip_depth_mm:
                start_idx = i
                break
        
        for i in range(start_idx, len(snr)):
            if snr[i] < threshold:
                return depth_pixels[i] * mm_per_pix
        return None

    def compute_lcp(self, img1_array, img2_array, roi, mm_per_pix):
        """
        Full LCP computation.
        Returns a dict with keys: lcp_depth_mm, signal_std, noise_std, snr, depth_mm.
        If an error occurs, the dict contains an 'error' key.
        """
        try:
            roi1 = self.crop_image(img1_array, roi)
            roi2 = self.crop_image(img2_array, roi)

            if roi1.size == 0 or roi2.size == 0:
                return {
                    'lcp_depth_mm': None,
                    'signal_std': np.array([]),
                    'noise_std': np.array([]),
                    'snr': np.array([]),
                    'depth_mm': np.array([]),
                    'error': 'ROI is empty or out of bounds.'
                }

            # Check if the two images are virtually identical
            max_diff = np.max(np.abs(roi1 - roi2))
            if max_diff < 1e-6:
                return {
                    'lcp_depth_mm': None,
                    'signal_std': np.array([]),
                    'noise_std': np.array([]),
                    'snr': np.array([]),
                    'depth_mm': np.array([]),
                    'error': 'The two images are identical – cannot compute LCP. Store two different images.'
                }

            sum_img, diff_img = self.calculate_sum_diff_images(roi1, roi2)
            signal_std, noise_std, snr, depth_pix = self.calculate_std_dev(sum_img, diff_img)
            lcp_mm = self.determine_lcp_depth(snr, depth_pix, mm_per_pix)
            depth_mm = depth_pix * mm_per_pix

            return {
                'lcp_depth_mm': lcp_mm,
                'signal_std': signal_std,
                'noise_std': noise_std,
                'snr': snr,
                'depth_mm': depth_mm
            }
        except Exception as e:
            return {
                'lcp_depth_mm': None,
                'signal_std': np.array([]),
                'noise_std': np.array([]),
                'snr': np.array([]),
                'depth_mm': np.array([]),
                'error': f'LCP computation failed: {str(e)}'
            }
            

class PixelSpacingHelper:
    """Extract mm_per_pix from DICOM metadata or user calibration."""
    @staticmethod
    def from_dicom(ds):
        """
        Try to get pixel spacing (in mm/pixel) from DICOM dataset.
        Returns None if not found.
        """
        # Try (0028,0030) Pixel Spacing
        pixel_spacing = ds.get("0028,0030")
        if pixel_spacing is not None:
            try:
                spacing = np.asarray(pixel_spacing, dtype=float)
                return spacing[0] * 10.0   # convert cm to mm
            except (ValueError, IndexError):
                pass
        # Try US Region Sequence (0018,6011)
        us_seq = ds.get((0x0018, 0x6011))
        if us_seq is not None:
            # us_seq is a DataElement with VR 'SQ' – its value is a list
            seq_value = us_seq.value
            if seq_value and len(seq_value) > 0:
                spacing = seq_value[0].get((0x0018, 0x602E))
                if spacing is not None:
                    return spacing.value * 10.0
        return None

    @staticmethod
    def from_user_line(pixel_length):
        """
        Calculate mm_per_pix from a line of known length (10mm).
        pixel_length : length in pixels of the drawn 10mm line.
        Returns mm_per_pix.
        """
        if pixel_length <= 0:
            raise ValueError("Pixel length must be positive")
        return 10.0 / pixel_length