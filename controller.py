# controller.py

"""
Contains the Controller class which acts as the intermediary between the Model and the View. It handles all user interactions, updates the model based on user actions, and updates the view accordingly.
The Controller class manages the application state, responds to button clicks and other events from the view, and performs the necessary computations using the model. 
It also handles the image viewer window, including navigation through images, ROI selection, circle detection, and measurement tools. 
The controller ensures that the model and view remain synchronized as the user interacts with the application.
"""
import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import RectangleSelector
from tkinter import filedialog, messagebox

from model import DICOMCache, LCS, LCPCalculator, PixelSpacingHelper


class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view          # MainWindow (control panel)
        self.cache = model['dicom_cache']
        self.lcs = model['lcs']
        self.lcp_calc = model['lcp']
        self.stored_images = model['stored_images']
        self.roi_coords = None
        self.mm_per_pix = None
        self.detected_circles = []
        self.circle_artists = []
        self.selected_circle = None
        self.dragging = False
        self.current_index = 0
        self._copied_roi = None

        # Image viewer window (will be created when first folder opened)
        self.image_viewer = None
        self.fig = None
        self.ax = None
        self.rect_selector = None

    # ------------------------------------------------------------------
    #  Image viewer window management
    # ------------------------------------------------------------------
    def _create_image_viewer(self):
        try:
            from view import ImageViewerWindow
            total = self.cache.total_count()
            self.image_viewer = ImageViewerWindow(self.view.root, total)
            self.image_viewer.set_slider_callback(self.on_slider_jump)
            self.setup_figure_callbacks()
            self.view.insert_output("Image viewer window opened.\n")
        except Exception as e:
            self.view.insert_output(f"Failed to create image viewer: {e}\n")
            import traceback
            self.view.insert_output(traceback.format_exc())

    def setup_figure_callbacks(self):
        """Connect matplotlib events and add RectangleSelector."""
        if self.image_viewer is None:
            return
        self.fig = self.image_viewer.get_canvas().figure
        self.ax = self.image_viewer.get_ax()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        # RectangleSelector for ROI (using updated parameter name 'props')
        self.rect_selector = RectangleSelector(
            self.ax, self.on_roi_selected,
            useblit=True, interactive=True,
            props=dict(facecolor='none', edgecolor='pink', linewidth=0.5)
        )

    def update_display(self):
        if self.image_viewer is None:
            return
        try:
            img = self.cache.get_image(self.current_index)
            self.mm_per_pix = self.cache.get_calibration(self.current_index)
            self.image_viewer.update_image(img)
            self.image_viewer.set_title(f"DICOM Image {self.current_index+1} / {self.cache.total_count()}")
            self.image_viewer.update_slider(self.current_index)   # <-- crucial
        except Exception as e:
            self.view.insert_output(f"ERROR in update_display: {e}\n")

    # ------------------------------------------------------------------
    #  Button callbacks
    # ------------------------------------------------------------------
    def on_open_folder(self):
        folder = filedialog.askdirectory(title="Select DICOM Folder")
        if not folder:
            return
        files = os.listdir(folder)
        dicom_files = self._check_dicom_files(files, folder)
        if not dicom_files:
            messagebox.showinfo("No DICOM Files", "No valid DICOM files found.")
            return
        self.cache.set_file_list(dicom_files)
        self.current_index = 0
        self.mm_per_pix = None
        self.stored_images = {'img1': None, 'img2': None}
        self.detected_circles.clear()
        self.circle_artists.clear()
        self.roi_coords = None

        # Try to read mm_per_pix from DICOM header of first image
        ds = pydicom.dcmread(dicom_files[0])
        self.mm_per_pix = PixelSpacingHelper.from_dicom(ds)

        # Create image viewer window if not already
        if self.image_viewer is None:
            self._create_image_viewer()
            self.image_viewer.set_total_images(len(dicom_files))
            self.image_viewer.update_slider(0)
        self.update_display()
        self.view.insert_output(f"Loaded {len(dicom_files)} DICOM files.\n")
        if self.mm_per_pix:
            self.view.insert_output(f"Pixel spacing: {self.mm_per_pix:.4f} mm/pixel (from DICOM).\n")
        else:
            self.view.insert_output("No pixel spacing found. Please calibrate using a 10mm line.\n")

    def on_add_image(self):
        self.view.insert_output("Add Image: not yet implemented.\n")

    def on_remove_image(self):
        self.view.insert_output("Remove Image: not yet implemented.\n")

    def on_sort_images(self):
        self.view.insert_output("Sort images by date: not yet implemented.\n")

    def on_store_image1(self):
        if self.cache.total_count() == 0:
            return
        img_array, mm = self.cache.get_image_and_calib(self.current_index)
        self.stored_images['img1'] = {
            'array': img_array,
            'mm_per_pix': mm,
            'ds': pydicom.dcmread(self.cache.files[self.current_index])
        }
        self.view.insert_output(f"Image 1 stored (calibration: {mm:.4f} mm/pixel)\n")

    def on_store_image2(self):
        if self.cache.total_count() == 0:
            return
        img_array, mm = self.cache.get_image_and_calib(self.current_index)
        self.stored_images['img2'] = {
            'array': img_array,
            'mm_per_pix': mm,
            'ds': pydicom.dcmread(self.cache.files[self.current_index])
        }
        self.view.insert_output(f"Image 2 stored (calibration: {mm:.4f} mm/pixel)\n")


    def on_curved_recon(self):
        self.view.insert_output("Curved reconstruction: not yet implemented.\n")

    def on_uniformity(self):
        self.view.insert_output("Uniformity measurements: not yet implemented.\n")

    def on_run_lcs(self):
        if not self.detected_circles:
            messagebox.showinfo("No Circles", "Please detect circles first (Run Circle Detection).")
            return
        img = self.cache.get_image(self.current_index)
        results = []
        for (x, y, r) in self.detected_circles:
            data = LCS.calculate_contrast_index(img, x, y, r)
            if data:
                results.append({'circle': (x, y, r), 'data': data})
                self.view.insert_output(
                    f"Circle ({x}, {y}) radius={r}: Contrast Index = {data['contrast_index']:.4f}\n"
                )
        if not results:
            self.view.insert_output("No valid LCS data for detected circles.\n")
        else:
            ci_vals = [d['data']['contrast_index'] for d in results]
            self.view.insert_output(f"Mean Contrast Index: {np.mean(ci_vals):.4f}\n")
            self.view.insert_output(f"Std Dev: {np.std(ci_vals):.4f}\n")

    def on_hcs_tool(self):
        self.view.insert_output("HCS Tool: not yet implemented.\n")

    def on_run_lcp(self):
        if self.stored_images['img1'] is None or self.stored_images['img2'] is None:
            messagebox.showinfo("Missing Images", "Store both images first.")
            return
        if self.roi_coords is None:
            messagebox.showinfo("No ROI", "Draw an ROI first.")
            return

        # Get calibrations
        mm1 = self.stored_images['img1'].get('mm_per_pix')
        mm2 = self.stored_images['img2'].get('mm_per_pix')
        if mm1 is None or mm2 is None:
            self.view.insert_output("Calibration missing for one or both images. Please calibrate manually.\n")
            # Optionally prompt user to draw 10mm line here
            return

        # If calibrations differ, warn and use the first (or average)
        if abs(mm1 - mm2) > 0.01:
            self.view.insert_output(f"Warning: Image calibrations differ ({mm1:.4f} vs {mm2:.4f}). Using first image's calibration.\n")
        mm_per_pix = mm1   # or (mm1+mm2)/2

        img1_arr = self.stored_images['img1']['array']
        img2_arr = self.stored_images['img2']['array']
        result = self.lcp_calc.compute_lcp(img1_arr, img2_arr, self.roi_coords, mm_per_pix)

        # Check for errors
        if result is None:
            self.view.insert_output("LCP calculation failed: returned None.\n")
            return
        if 'error' in result:
            self.view.insert_output(f"LCP Error: {result['error']}\n")
            return

        # Valid result
        if result['lcp_depth_mm'] is None:
            self.view.insert_output("LCP: SNR never dropped below threshold.\n")
        else:
            self.view.insert_output(f"LCP Depth: {result['lcp_depth_mm']:.1f} mm\n")

        # Plot only if we have valid data
        if len(result['depth_mm']) > 0:
            fig, ax1 = plt.subplots(num="LCP Depth Profile")
            ax1.plot(result['depth_mm'], result['noise_std'], 'r', label='Noise Std')
            ax1.plot(result['depth_mm'], result['signal_std'], 'orange', label='Signal Std')
            ax1.set_xlabel('Depth (mm)')
            ax1.set_ylabel('Standard Deviation')
            ax1.legend(loc='upper right')
            ax2 = ax1.twinx()
            ax2.plot(result['depth_mm'], result['snr'], 'b', label='SNR')
            ax2.set_ylabel('SNR')
            ax2.legend(loc='upper left')
            plt.title('LCP Depth Profile')
            plt.tight_layout()
            plt.show(block=False)
        else:
            self.view.insert_output("No depth profile data to plot.\n")

    def on_save_roi(self):
        if self.roi_coords is None:
            self.view.insert_output("No ROI to save.\n")
            return
        try:
            import json
            with open("last_roi.json", "w") as f:
                json.dump(self.roi_coords, f)
            self.view.insert_output(f"ROI saved: {self.roi_coords}\n")
        except Exception as e:
            self.view.insert_output(f"Error saving ROI: {e}\n")

    def on_load_roi(self):
        try:
            import json
            with open("last_roi.json", "r") as f:
                coords = json.load(f)
            self.roi_coords = tuple(coords)
            self.view.insert_output(f"ROI loaded: {self.roi_coords}\n")
            # Draw rectangle on the image
            if self.ax is not None:
                x1, y1, x2, y2 = self.roi_coords
                rect = Rectangle((x1, y1), x2-x1, y2-y1,
                                 edgecolor='pink', facecolor='none', linewidth=0.5)
                self.ax.add_patch(rect)
                self.fig.canvas.draw()
        except FileNotFoundError:
            self.view.insert_output("No saved ROI found.\n")
        except Exception as e:
            self.view.insert_output(f"Error loading ROI: {e}\n")

    def on_copy_roi(self):
        if self.roi_coords is None:
            self.view.insert_output("No ROI to copy.\n")
            return
        self._copied_roi = self.roi_coords
        self.view.insert_output(f"ROI copied: {self.roi_coords}\n")

    def on_paste_roi(self):
        if hasattr(self, '_copied_roi') and self._copied_roi:
            self.roi_coords = self._copied_roi
            self.view.insert_output(f"ROI pasted: {self.roi_coords}\n")
            if self.ax is not None:
                x1, y1, x2, y2 = self.roi_coords
                rect = Rectangle((x1, y1), x2-x1, y2-y1,
                                 edgecolor='pink', facecolor='none', linewidth=0.5)
                self.ax.add_patch(rect)
                self.fig.canvas.draw()
        else:
            self.view.insert_output("No ROI in clipboard.\n")

    def on_print_dicom(self):
        if self.cache.total_count() == 0:
            self.view.insert_output("No DICOM loaded.\n")
            return
        ds = pydicom.dcmread(self.cache.files[self.current_index])
        output = f"DICOM info for file {self.current_index+1}:\n"
        output += f"Patient Name: {ds.get('PatientName', 'N/A')}\n"
        output += f"Modality: {ds.get('Modality', 'N/A')}\n"
        output += f"Study Date: {ds.get('StudyDate', 'N/A')}\n"
        output += f"Image Dimensions: {ds.Rows} x {ds.Columns}\n"
        self.view.insert_output(output)

    def on_measure(self):
        if self.image_viewer is None:
            self.view.insert_output("Open a DICOM folder first.\n")
            return
        self.image_viewer.set_title("Click first point, then second point")
        self.fig.canvas.draw()
        points = plt.ginput(2, timeout=-1)
        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            dist_px = np.hypot(x2-x1, y2-y1)
            if self.mm_per_pix:
                dist_mm = dist_px * self.mm_per_pix
                self.view.insert_output(f"Distance: {dist_px:.1f} px = {dist_mm:.2f} mm\n")
            else:
                self.view.insert_output(f"Distance: {dist_px:.1f} pixels (calibrate for mm)\n")
        self.image_viewer.set_title("DICOM Viewer")
        self.fig.canvas.draw()

    def on_clear_output(self):
        self.view.clear_output()

    def on_print_roi(self):
        if self.roi_coords is None:
            self.view.insert_output("No ROI defined.\n")
        # else:
            # self.view.insert_output(f"Current ROI: {self.roi_coords}\n")

    def on_clear_overlay(self):
        for artist in self.circle_artists:
            artist.remove()
        self.circle_artists.clear()
        self.detected_circles.clear()
        self.fig.canvas.draw()
        self.view.insert_output("Overlays cleared.\n")

    def on_fwhm(self):
        self.view.insert_output("FWHM tool: not yet implemented.\n")

    def on_calibrate_mm(self):
        if self.image_viewer is None:
            self.view.insert_output("Open a DICOM folder first.\n")
            return
        self.image_viewer.set_title("Click two points to draw a 10mm line for calibration")
        self.fig.canvas.draw()
        points = plt.ginput(2, timeout=-1)
        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            pixel_len = np.hypot(x2-x1, y2-y1)
            if pixel_len > 0:
                self.mm_per_pix = 10.0 / pixel_len
                self.view.insert_output(f"Calibration set: {self.mm_per_pix:.4f} mm/pixel.\n")
                self.image_viewer.set_title("DICOM Viewer")
            else:
                messagebox.showinfo("Error", "Line length is zero.")
        else:
            messagebox.showinfo("Cancelled", "Calibration cancelled.")
        self.image_viewer.set_title("DICOM Viewer")
        self.fig.canvas.draw()

    def on_detect_circles(self):
        if self.roi_coords is None:
            messagebox.showinfo("No ROI", "Please draw an ROI rectangle first.")
            return
        if self.mm_per_pix is None:
            self.on_calibrate_mm()
            if self.mm_per_pix is None:
                return

        img_float = self.cache.get_image(self.current_index)
        img_uint8 = (img_float * 255).astype(np.uint8)
        x1, y1, x2, y2 = self.roi_coords
        xi1, xi2 = int(min(x1, x2)), int(max(x1, x2))
        yi1, yi2 = int(min(y1, y2)), int(max(y1, y2))
        roi = img_uint8[yi1:yi2, xi1:xi2]
        if roi.size == 0:
            self.view.insert_output("Invalid ROI.\n")
            return

        import cv2
        def unsharp_mask(image, sigma=1.0, strength=1.5):
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            return cv2.addWeighted(image, 1+strength, blurred, -strength, 0)

        sharp = unsharp_mask(roi)
        denoised = cv2.medianBlur(sharp, 9)
        sharp2 = unsharp_mask(denoised)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharp2)
        sharp3 = unsharp_mask(enhanced)
        blurred = cv2.GaussianBlur(sharp3, (9,9), 0)
        edges = cv2.Canny(blurred, 50, 90, apertureSize=3, L2gradient=True)

        radius_px = 4.0 / self.mm_per_pix
        min_r = int(radius_px * 0.95)
        max_r = int(radius_px * 1.05)
        min_dist_px = 12.0 / self.mm_per_pix

        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,
                                   dp=1.5, minDist=min_dist_px,
                                   param1=250, param2=6,
                                   minRadius=min_r, maxRadius=max_r)

        # Clear previous circles
        for artist in self.circle_artists:
            artist.remove()
        self.circle_artists.clear()
        self.detected_circles.clear()

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                x_full = cx + xi1
                y_full = cy + yi1
                if (x_full - r >= xi1 and y_full - r >= yi1 and
                    x_full + r <= xi2 and y_full + r <= yi2):
                    self.detected_circles.append((x_full, y_full, r))
                    circ = Circle((x_full, y_full), r, edgecolor='yellow', fill=False, lw=2)
                    self.ax.add_artist(circ)
                    self.circle_artists.append(circ)
            self.view.insert_output(f"Detected {len(self.detected_circles)} circles.\n")
            self.fig.canvas.draw()
        else:
            self.view.insert_output("No circles detected.\n")
    
    def on_slider_jump(self, new_index):
        if 0 <= new_index < self.cache.total_count():
            self.current_index = new_index
            self.update_display()
    # ------------------------------------------------------------------
    #  ROI selection callback
    # ------------------------------------------------------------------
    def on_roi_selected(self, eclick, erelease):
        self.roi_coords = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)
        # self.view.insert_output(f"ROI selected: {self.roi_coords}\n")

    # ------------------------------------------------------------------
    #  Keyboard and scroll navigation
    # ------------------------------------------------------------------
    def on_key_press(self, event):
        if event.key == 'left':
            self.current_index = max(0, self.current_index - 1)
            self.update_display()
        elif event.key == 'right':
            self.current_index = min(self.cache.total_count() - 1, self.current_index + 1)
            self.update_display()

    def on_scroll(self, event):
        if event.button == 'down':
            self.current_index = min(self.cache.total_count() - 1, self.current_index + 1)
        else:
            self.current_index = max(0, self.current_index - 1)
        self.update_display()

    # ------------------------------------------------------------------
    #  Circle dragging
    # ------------------------------------------------------------------
    def on_click(self, event):
        if event.inaxes != self.ax or not self.detected_circles:
            return
        for i, (x, y, r) in enumerate(self.detected_circles):
            if (x - event.xdata)**2 + (y - event.ydata)**2 <= r**2:
                self.selected_circle = self.circle_artists[i]
                self.dragging = True
                break

    def on_drag(self, event):
        if self.dragging and self.selected_circle is not None and event.inaxes == self.ax:
            self.selected_circle.center = (event.xdata, event.ydata)
            self.fig.canvas.draw()

    def on_release(self, event):
        if self.dragging and self.selected_circle is not None:
            idx = self.circle_artists.index(self.selected_circle)
            x, y = self.selected_circle.center
            r = self.detected_circles[idx][2]
            self.detected_circles[idx] = (x, y, r)
        self.dragging = False
        self.selected_circle = None

    # ------------------------------------------------------------------
    #  Helper: validate DICOM files
    # ------------------------------------------------------------------
    def _read_bytes(self, file_path, start, end):
        with open(file_path, "rb") as f:
            f.seek(start)
            return f.read(end - start)

    def _check_dicom_files(self, files, dicom_dir):
        valid = []
        for f in files:
            full = os.path.join(dicom_dir, f)
            try:
                with open(full, "rb") as fp:
                    fp.seek(128)
                    header = fp.read(4)
                if header == b"DICM":
                    valid.append(full)
            except:
                continue
        def get_inst_num(path):
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                return getattr(ds, 'InstanceNumber', 0)
            except:
                return 0
        return sorted(valid, key=get_inst_num)