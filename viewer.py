# viewer.py

import cv2
import os
from matplotlib.patches import Rectangle, Circle
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
        self.detected_circles = []  # List to store detected circles
        self.selected_circle = None  # To store the selected circle for dragging
        self.dragging = False  # Track if dragging is in progress
        self.circle_artists = []  # List to store circle artists (graphical objects)
        self.txtfld2 = None
        self.top_left = None
        self.bottom_right = None
        self.depth = 0
        self.noise_std_dev = 0
        self.signal_std_dev = 0
        self.snr = 0
        self.filtered_signal_std_dev, self.filtered_noise_std_dev = 0, 0


        
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
                self.ax.set_title('DICOM Image {}'.format(self.current_index + 1))
                self.fig.canvas.draw()
                plt.pause(0.1)
            
    def _create_plot(self):
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image, cmap='gray')
        self.ax.axis('off')
        self.ax.set_title('DICOM Viewer')
        plt.style.use('dark_background')

    def _configure_plot(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.rect_selector = RectangleSelector(self.ax, self.on_select, useblit=True, interactive=True)
        self.rect_selector.rectprops = dict(facecolor='none', edgecolor='pink', linewidth=0.5)
        self.ax.spines['bottom'].set_color('black')
        self.ax.spines['top'].set_color('black')
        self.ax.spines['left'].set_color('black')
        self.ax.spines['right'].set_color('black')
        self.ax.axis('off')
        self.ax.set_title('DICOM Viewer')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.style.use('dark_background')
        
    def show_image(self, title, img):
        """
        Displays an image in a new figure with a title.
        """
        plt.figure()  # Create a new figure for each image
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')  # Turn off the axis for better visualization
        plt.show()  # Display the image
        
    def enable_circle_dragging(self):
        # Connect mouse events to the canvas
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
    
    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Find the closest circle to the click point
        for i, (x, y, r) in enumerate(self.detected_circles):
            # Check if the click is inside the circle's area
            if (x - event.xdata) ** 2 + (y - event.ydata) ** 2 <= r ** 2:
                self.selected_circle = self.circle_artists[i]
                self.dragging = True
                print(f"Circle selected at ({x}, {y}) with radius {r}")
                break

    def on_drag(self, event):
        if not self.dragging or self.selected_circle is None:
            return

        # Update the circle's position while dragging
        if event.inaxes != self.ax:
            return

        # Update the circle's center to the new mouse position
        self.selected_circle.center = (event.xdata, event.ydata)
        self.fig.canvas.draw()

    def on_release(self, event):
        self.dragging = False
        self.selected_circle = None
                
    def unsharp_mask(self, image, sigma=1.0, strength=1.5):
        """
        Apply unsharp mask to sharpen the image.
        
        :param image: Input image
        :param sigma: Standard deviation for Gaussian blur
        :param strength: Strength of the sharpening
        :return: Sharpened image
        """
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
        return sharpened   
    
    def detect_circle_in_roi(self, image, x1, y1, x2, y2):
        """
        Detects circles in the selected ROI using Hough Circle Transform with the key preprocessing steps.
        """
        # Calculate mm_per_pix (assuming `calculate_mm_per_pix` exists in your superclass or current class)
        mm_per_pix = self.calc_mm()

        # If mm_per_pix is None, show an error
        if mm_per_pix is None:
            print("mm_per_pix could not be determined.")
            return

        # Convert the known 8mm radius to pixels
        radius_in_pixels = 4 / mm_per_pix
        min_radius = int(radius_in_pixels * 0.95)  # Allow some tolerance
        max_radius = int(radius_in_pixels * 1.05)
        
        # Extract the ROI from the image
        roi = image[int(min(y1, y2)):int(max(y1, y2)), int(min(x1, x2)):int(max(x1, x2))]

        # Convert to grayscale if not already
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi
            
        # Apply unsharp mask to sharpen the image
        sharpened_roi = self.unsharp_mask(gray_roi)
        # self.show_image("Sharpened ROI (Unsharp Mask)", sharpened_roi)
        
        # Apply median filtering to reduce speckle noise
        denoised_roi = cv2.medianBlur(sharpened_roi, 9)
        self.show_image("Denoised ROI (Median Blur)", denoised_roi)
        
        # Apply unsharp mask to sharpen the image
        sharpeneded_roi = self.unsharp_mask(denoised_roi)
        # self.show_image("Sharpened ROI (Unsharp Mask)", sharpeneded_roi)

        # Apply CLAHE to enhance local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_roi = clahe.apply(sharpeneded_roi)
        # self.show_image("Enhanced ROI (CLAHE)", enhanced_roi)

        # Apply unsharp mask to sharpen the image
        sharpenededed_roi = self.unsharp_mask(enhanced_roi)
        # self.show_image("Sharpened ROI (Unsharp Mask)", sharpenededed_roi)
        
        # Apply Gaussian blur to smooth out the image
        blurred_roi = cv2.GaussianBlur(sharpenededed_roi, (9, 9), 0)
        # self.show_image("Blurred ROI (Gaussian Blur)", blurred_roi)
        
        # Optionally, apply Canny edge detection
        edges = cv2.Canny(blurred_roi, 50, 90, apertureSize=3, L2gradient=True)
        # self.show_image("Edges (Canny)", edges)
        
        # Convert 12mm of the distance between the centres into pixels
        min_center_dist_in_pixels = 12 / mm_per_pix

        # Now use Hough Circle Transform directly on the blurred image
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,
                                dp=1.5,
                                minDist=min_center_dist_in_pixels,
                                param1=250,
                                param2=6,
                                minRadius=min_radius,
                                maxRadius=max_radius)

        detected_circles = []

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Adjust coordinates to the original image
                adjusted_x = x + int(min(x1, x2))
                adjusted_y = y + int(min(y1, y2))

                # Check if the circle is fully contained within the ROI
                if (adjusted_x - r >= x1 and adjusted_y - r >= y1 and
                    adjusted_x + r <= x2 and adjusted_y + r <= y2):
                    
                    # If the circle fits in the ROI, draw and store it
                    circle_artist = Circle((adjusted_x, adjusted_y), r, edgecolor='yellow', fill=False, lw=2)
                    self.ax.add_artist(circle_artist)
                    self.circle_artists.append(circle_artist)
                    self.detected_circles.append((adjusted_x, adjusted_y, r))
                    
                    # Store the detected circle's center and radius
                    detected_circles.append((adjusted_x, adjusted_y, r))
            # Now filter circles based on distance between centers
            filtered_circles = []
            for i, (cx1, cy1, r1) in enumerate(detected_circles):
                is_valid = True
                for j, (cx2, cy2, r2) in enumerate(detected_circles):
                    if i != j:
                        # Calculate Euclidean distance between two circle centers
                        dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                        # Convert 12mm distance to pixels
                        expected_dist_in_pixels = 12 / mm_per_pix
                        # Check if the distance is within a reasonable range (e.g., 10% tolerance)
                        if not (0.9 * expected_dist_in_pixels <= dist <= 1.1 * expected_dist_in_pixels):
                            is_valid = False
                            break
                if is_valid:
                    filtered_circles.append((cx1, cy1, r1))
                    
                # Draw and store valid circles
                for (x, y, r) in filtered_circles:
                    circle_artist = Circle((x, y), r, edgecolor='yellow', fill=False, lw=2)
                    self.ax.add_artist(circle_artist)
                    self.circle_artists.append(circle_artist)
                    self.detected_circles.append((x, y, r))
                print(f"Detected circle at (x={x}, y={y}) with radius={r}")
                
            # Redraw the figure to show the circles
            self.fig.canvas.draw()
        else:
            print("No circles detected in the ROI.")

    
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
                self.current_index = 0
                self.image = self.load_images_progressively().__next__()  # Load the first image
                self._create_plot()
                self._configure_plot()
                plt.show()
                break

            except TclError:
                break
            except Exception as e:
                messagebox.showinfo("Error", f"An error occurred: {str(e)}")

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
        self.ax.add_patch(Rectangle((self.x1, self.y1), self.x2 - self.x1, self.y2 - self.y1,
                                    edgecolor='none', facecolor='none', fill=False, linewidth=0))

        # Read the DICOM image
        ds = pydicom.dcmread(self.dicom_files[self.current_index])
        self.image = ds.pixel_array

        # Call the circle detection function
        self.detect_circle_in_roi(self.image, self.x1, self.y1, self.x2, self.y2)

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

    def calculate_single_lcs(self, image, circle):
        x, y, radius = circle[0]
        
        # Coordinates of the circular region
        x1, y1, x2, y2 = int(x - radius), int(y - radius), int(x + radius), int(y + radius)

        # Extract pixel values around two circles
        pixels_inner = self._extract_pixels_in_circle(x, y, radius * 0.7, image)
        pixels_outer = self._extract_pixels_in_circle(x, y, radius * 1.35, image)

        # Calculate mean and SD of pixel values
        mean_inner = np.mean(pixels_inner)
        mean_outer = np.mean(pixels_outer)
        sd_difference = np.std(np.concatenate([pixels_inner, pixels_outer]))

        # Calculate t-value and degrees of freedom
        t_value = (mean_inner - mean_outer) / (sd_difference / np.sqrt(len(pixels_inner)))
        degrees_of_freedom = len(pixels_inner) + len(pixels_outer) - 2

        # Calculate p-value
        p_value = 2 * (1 - t.cdf(np.abs(t_value), degrees_of_freedom))

        # Check for significance based on threshold (3.3 SE)
        if p_value < 0.001:
            # Calculate index of contrast using eqn (5)
            index_of_contrast = mean_inner / mean_outer
            return index_of_contrast
        else:
            return None

    def _extract_pixels_in_circle(self, x_center, y_center, radius, image):
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
        mask = distance <= radius
        return image[mask]