# viewer.py

import math
import os
from matplotlib.patches import Rectangle
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import convolve
from matplotlib.widgets import RectangleSelector
from tkinter import Tk, filedialog, messagebox, TclError

"""
class SuperClass:
    def calculate_mm_per_pix(self):
        if self.ds1 is not None:
            # Retrieve DICOM header information for y1 and y2
            pixel_spacing = self.ds1.get("0028,0030")
            if pixel_spacing is not None:
                mm_per_pix = np.asarray(pixel_spacing) * 10
            else:
                us_regions_seq = self.ds1.get((0x0018, 0x6011))
                pixel_spacing = us_regions_seq[0].get((0x0018, 0x602E)).value
                mm_per_pix = pixel_spacing * 10
            return mm_per_pix
        else:
            return None
"""   


class DCMViewer():
    def __init__(self):
        self.current_index = 0
        self.dicom_files = []
        self.image = None
        self.ds1 = None
        self.ds2 = None
        self.fig = None
        self.ax = None
        self.rect_selector = None
        self.x1, self.x2, self.y1, self.y2 = 0, 0, 0, 0
        self.txtfld2 = None
        self.top_left = None
        self.bottom_right = None
        self.depth = 0
        self.noise_std_dev = 0
        self.signal_std_dev = 0
        self.snr = 0
        self.filtered_signal_std_dev, self.filtered_noise_std_dev = 0, 0
        self.circle_radius = 0
        
    def load_images_progressively(self):
        for file_path in self.dicom_files:
            ds = pydicom.dcmread(file_path)
            image = ds.pixel_array.astype(np.uint8) / ds.pixel_array.max()
            yield image

    def _open_folder_dialog(self):
        root = Tk()
        root.withdraw()
        dicom_dir = filedialog.askdirectory()
        return dicom_dir

    @staticmethod
    def _read_bytes(file_path, start_position, end_position):
        with open(file_path, "rb") as f:
            f.seek(start_position)
            bytes_read = f.read(end_position - start_position)
        return bytes_read

    @staticmethod   
    def _check_dicom_files(files, dicom_dir):
        dicom_files = []
        for f in files:
            bytes_read = DCMViewer._read_bytes(os.path.join(dicom_dir, f), 128, 132)
            if bytes_read == b"DICM":
                dicom_files.append(os.path.join(dicom_dir, f))
            else:
                print(f"The file {f} is not a DICOM file.")
        return sorted(dicom_files, key=lambda f: pydicom.dcmread(f).InstanceNumber)

    def _load_and_display_images(self):
        for image in self.load_images_progressively():
            self.image = image
            if self.ax is not None:
                self.ax.imshow(self.image, cmap='gray')
            plt.pause(0.1)
            
    def _create_plot(self):
        self.fig, self.ax = plt.subplots()
        im = self.ax.imshow(self.image, cmap='gray')

    def _configure_plot(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.rect_selector = RectangleSelector(self.ax, self.on_select, useblit=True, interactive=True)
        self.rect_selector.rectprops = dict(facecolor='none', edgecolor='pink', linewidth=0.5)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.spines['bottom'].set_color('black')
        self.ax.spines['top'].set_color('black')
        self.ax.spines['left'].set_color('black')
        self.ax.spines['right'].set_color('black')
        self.ax.axis('off')
        self.ax.set_title('DICOM Viewer')
        plt.style.use('dark_background')
        plt.show()

    def dcm_view(self):
        plt.style.use('dark_background')
        while True:
            try:
                dicom_dir = self._open_folder_dialog()

                if not dicom_dir:
                    break

                files = os.listdir(dicom_dir)
                dicom_files = self._check_dicom_files(files, dicom_dir)

                if not dicom_files:
                    messagebox.showinfo("No DICOM files", "The selected folder does not contain any DICOM files.")
                    continue

                self.dicom_files = dicom_files
                self._load_and_display_images()
                self.current_index = 0
                self._create_plot()
                self._configure_plot()
                break

            except TclError:
                break
            except Exception as e:
                messagebox.showinfo("Error", f"An error occurred: {str(e)}")
    
    current_selector = "rectangle"  # Default to rectangle selector
    circle_selector = None  # Variable to hold the circle selector
    
    def switch_selector(self):
        global current_selector, circle_selector
        rect_selector = RectangleSelector(ax, viewer.on_select, useblit=False, interactive=True)
        rect_selector.rectprops = dict(facecolor='none', edgecolor='pink', linewidth=0.5)
        rect_selector.set_active(True)
        if self.current_selector == "rectangle":
            current_selector = "circle"
            rect_selector.set_active(False)
            rect_selector.set_visible(False)

            # Create a new circle selector
            circle_selector = plt.Circle((0, 0), 0, edgecolor='green', facecolor='none', linewidth=2)
            self.ax.add_patch(circle_selector)

        else:
            current_selector = "rectangle"
            if self.circle_selector:
                self.circle_selector.remove()  # Remove the existing circle selector
                self.circle_selector = None

            rect_selector.set_active(True)
            rect_selector.set_visible(True)


    
    def on_circle_select(self, event):
        if event.key == 'c':
            # Set the circle radius based on user mouse click
            self.circle_radius = max(abs(self.x2 - self.x1), abs(self.y2 - self.y1)) / 2
            # Draw the circle with the updated radius
            self.circle_selector.set_radius(self.circle_radius)
            self.circle_selector.center = ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
            self.ax.add_patch(self.circle_selector)
            plt.draw()
            
    def on_key_press(self, event):
        if event.key == 'left':
            self.current_index = max(0, self.current_index - 1)
        elif event.key == 'right':
            self.current_index = min(len(self.dicom_files) - 1, self.current_index + 1)

        ds = pydicom.dcmread(self.dicom_files[self.current_index])
        self.image = ds.pixel_array
        if self.ax is not None:
            self.ax.imshow(self.image, cmap='gray')
        plt.draw()
    
    def on_select(self, eclick, erelease):
        """
        Callback function to draw a rectangle and update the coordinates
        eclick and erelease are the press and release events
        """
        
        self.x1, self.y1 = eclick.xdata, eclick.ydata
        self.x2, self.y2 = erelease.xdata, erelease.ydata
        ds = pydicom.dcmread(self.dicom_files[self.current_index])
        # Extract y_min and y_max from DICOM metadata of the current image
        us_regions_seq = ds[0x0018, 0x6011]
        if us_regions_seq:
            y1 = us_regions_seq[0][0x0018, 0x601a].value
            # print(y1)
            y2 = us_regions_seq[0][0x0018, 0x601e].value
            # print(y2)
            self.x1 = min(eclick.xdata, erelease.xdata)
            # print(self.x1)
            self.x2 = max(eclick.xdata, erelease.xdata)
            # print(self.x2)
            self.top_left = (self.x1, y1)
            self.bottom_right = (self.x2, y2)

        # Draw the rectangle on the plot
        self.ax.add_patch(Rectangle((self.x1, self.y1), self.x2 - self.x1, self.y2 - self.y1, 
                                    edgecolor='none', facecolor='none', fill="false", linewidth=0))
        # Redraw the image and rectangle
        self.fig.canvas.draw()

    def calc_mm(self):
        ds = pydicom.dcmread(self.dicom_files[self.current_index])
        if ds is not None:
            pixel_spacing = ds.get("0028,0030")
            if pixel_spacing is not None:
                try:
                    self.mm_per_pix = np.asarray(pixel_spacing) * 10
                    # print(f'Pixel spacing from first image: {self.mm_per_pix}')
                except ValueError:
                    print(f'Error: Pixel spacing {pixel_spacing} cannot be converted to float.')
            else:
                us_regions_seq = ds.get((0x0018, 0x6011))
                pixel_spacing = us_regions_seq[0].get((0x0018, 0x602E)).value
                self.mm_per_pix = pixel_spacing * 10
                # print(f'Pixel spacing from first image: {self.mm_per_pix}')
            return self.mm_per_pix
    """

        # Instantiate the LCP class
        lcp_instance = LCP()
        # Calculate mm_per_pix here
        mm_per_pix = lcp_instance.calculate_mm_per_pix()

        if mm_per_pix is not None:
            # Pass mm_per_pix to calc_lcp
            lcp_instance.calc_lcp(self.x1, self.y1, self.x2, self.y2, ds1=self.ds1, mm_per_pix=mm_per_pix)
        else:
            print("mm_per_pix is None. Cannot calculate LCP.")
            
        return self.x1, self.x2, self.y1, self.y2
    """
    def update_image(self, index, ax):
        ds = pydicom.dcmread(self.dicom_files[index])
        ax.clear()
        ax.imshow(ds.pixel_array, cmap=plt.cm.gray, aspect='equal')
        ax.set_title('DICOM Image %d' % ds.InstanceNumber)
        self.fig.canvas.draw()

    def on_scroll(self, event):
        if event.button == 'down':
            if self.current_index < len(self.dicom_files) - 1:
                self.current_index += 1
        else:
            if self.current_index > 0:
                self.current_index -= 1
        self.update_image(self.current_index, self.ax)

    def store_image_1(self):
        if self.image is not None:
            self.img1 = pydicom.dcmread(self.dicom_files[self.current_index])
            self.txtfld2.insert("end", f"Image 1 stored.\n")
        else:
            messagebox.showinfo("No Image", "No image to store. Load a DICOM image first.")
        return self.img1

    def store_image_2(self):
        if self.image is not None:
            self.img2 = pydicom.dcmread(self.dicom_files[self.current_index])
            self.txtfld2.insert("end", f"Image 2 stored.\n")
        else:
            messagebox.showinfo("No Image", "No image to store. Load a DICOM image first.")  
        return self.img2     
    
    def calc_lcp(self, x1, y1, x2, y2):
        roi = (x1, y1, x2, y2)
        mm_per_pix = self.calc_mm()
        # Crop the images
        roi1 = self.crop_image1(self.img1, roi)

        roi2 = self.crop_image2(self.img2, roi)

        # Perform the LCP calculations
        self.sum_img, self.diff_img = self.calculate_sum_diff_images(roi1, roi2)
        self.signal_std_dev, self.noise_std_dev, self.snr, self.depth = self.calculate_std_dev(self.sum_img, self.diff_img)
        
        lcp = self.determine_lcp_depths(self.snr, self.depth, mm_per_pix)
        # print(f'LCP: {lcp}')
        
        # Print the LCP value in txtfld2
        self.txtfld2.insert("end", f"LCP: {lcp}\n")
        
        fig, ax1 = plt.subplots()
        ax1.plot([d * mm_per_pix for d in self.depth], self.noise_std_dev, 'r', label='Noise Standard Deviation')
        ax1.plot([d * mm_per_pix for d in self.depth], self.signal_std_dev, 'orange', label='Signal Standard Deviation')
        ax1.set_xlabel('Depth (mm)')
        ax1.set_ylabel('Standard Deviation')
        ax1.legend(loc='upper right')
        
        ax2 = ax1.twinx()
        ax2.plot([d * mm_per_pix for d in self.depth], self.snr, 'b', label='SNR')
        ax2.set_ylabel('SNR')
        ax2.legend(loc='upper left')
        plt.title('LCP Depth Profile')
        plt.tight_layout()
        plt.show()
        return self.sum_img, self.diff_img
        
    def crop_image1(self, img1, roi):
        x1, y1, x2, y2 = roi
        array1 = img1.pixel_array.astype(np.uint8) / img1.pixel_array.max()
        cropped_img1 = array1[int(min(y1, y2)):int(max(y1, y2)), int(min(x1, x2)):int(max(x1, x2))]
        return cropped_img1
    
    def crop_image2(self, img2, roi):
        x1, y1, x2, y2 = roi
        array2 = img2.pixel_array.astype(np.uint8) / img2.pixel_array.max()
        cropped_img2 = array2[int(min(y1, y2)):int(max(y1, y2)), int(min(x1, x2)):int(max(x1, x2))]
        return cropped_img2
    
    def calculate_sum_diff_images(self, roi1, roi2):
        sum_img = np.zeros(roi1.shape, dtype=np.uint8)
        sum_img = np.add(roi1, roi2)
        
        if np.array_equal(roi1, roi2):
            print('images are the same')
            
        else:
            diff_img = np.zeros(roi1.shape, dtype=np.uint8)
            diff_img = np.subtract(roi1, roi2)
            
        return sum_img, diff_img 
    
    def calculate_std_dev(self, sum_img, diff_img):
        window_size = 3
        number_of_steps = (sum_img.shape[0] - 10) // window_size + 1  # Assuming `sum_img` is a numpy array
        
        self.noise_std_dev = []
        self.signal_std_dev = []
        self.snr = []
        
        for i in range(number_of_steps):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size

            # Ensure the end index does not go beyond the image shape
            end_idx = min(end_idx, sum_img.shape[0])

            # Get a slice of the sum and diff images
            sum_slice = sum_img[start_idx:end_idx, :]
            diff_slice = diff_img[start_idx:end_idx, :]

            # Calculate the standard deviation for noise as 2^-0.5 * (standard deviation for diff)
            self.noise_std_dev.append(np.std(diff_slice) * 2 ** -0.5)

            # Calculate the standard deviation for signal as 0.5 * ((standard deviation for sum)^2 - (standard deviation
            # for sum)^2)^0.5
            self.signal_std_dev.append(0.5 * ((np.std(sum_slice) ** 2 - np.std(diff_slice) ** 2) ** 0.5))

            # Calculate the signal-to-noise ratio per slice as the signal standard deviation of the slice divided by the
            # noise standard deviation of the slice
            if self.noise_std_dev[-1] > 0:
                self.snr.append(self.signal_std_dev[-1] / self.noise_std_dev[-1])
            else:
                self.snr.append(0.0)

        # Update the depth calculation
        depth = list(range(10, 10 + number_of_steps * window_size, window_size))
        self.noise_std_dev = medfilt(self.noise_std_dev, 7)
        self.signal_std_dev = medfilt(self.signal_std_dev, 7)    

        return self.signal_std_dev, self.noise_std_dev, self.snr, depth

    def determine_lcp_depths(self, snr, depth, mm_per_pix):
        # determine LCP depth
        threshold = 2  # or whatever your threshold is
        for i in range(len(snr)):
            if snr[i] < threshold:
                lcp_depth = depth[i]
                break
        else:
            print("No elements in snr are less than the threshold.")
            lcp_depth = None  # or some other default value

        lcp_depth_mm = round(lcp_depth * mm_per_pix, 1) if lcp_depth is not None else None
        return lcp_depth_mm

    # LCS part of the code
class LowContrastSensitivity:
    def __init__(self, image, contrast_regions, radius_ratio_1=0.7, radius_ratio_2=1.35, significance_threshold=3.3):
        self.image = image
        self.contrast_regions = contrast_regions
        self.radius_ratio_1 = radius_ratio_1
        self.radius_ratio_2 = radius_ratio_2
        self.significance_threshold = significance_threshold
        self.contrast_indices = []

    def calculate_contrast_indices(self):
        for region in self.contrast_regions:
            contrast_index = self._calculate_contrast_index(region)
            self.contrast_indices.append(contrast_index)
        return self.contrast_indices

    def _calculate_contrast_index(self, region):
        x1, y1, x2, y2 = region  # Coordinates of the circular region
        radius_inner = (x2 - x1) * self.radius_ratio_1
        radius_outer = (x2 - x1) * self.radius_ratio_2

        # Extract pixel values around two circles
        pixels_inner = self._extract_pixels_in_circle(x1, y1, radius_inner)
        pixels_outer = self._extract_pixels_in_circle(x1, y1, radius_outer)

        # Calculate mean and SD of pixel values
        mean_inner = np.mean(pixels_inner)
        mean_outer = np.mean(pixels_outer)
        sd_difference = np.std(np.concatenate([pixels_inner, pixels_outer]))

        # Calculate contrast index using eqn (5)
        contrast_index = mean_inner / mean_outer

        # Check for significance based on threshold
        if sd_difference * self.significance_threshold < np.abs(mean_inner - mean_outer):
            return contrast_index
        else:
            return None

    def _extract_pixels_in_circle(self, x_center, y_center, radius):
        y, x = np.ogrid[:self.image.shape[0], :self.image.shape[1]]
        distance = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        mask = distance <= radius
        return self.image[mask]

