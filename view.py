# view.py

"""
Contains the MainWindow and ImageViewerWindow classes which define the graphical user interface of the application using Tkinter.
The MainWindow class creates the main control panel with all the buttons and a text output area. It provides methods to set callbacks for each button, insert text into the output area, and show informational or error messages.
The ImageViewerWindow class creates a separate window for displaying DICOM images using a matplotlib figure embedded in Tkinter. 
It includes a slider for navigating through multiple images and methods to update the displayed image and set callbacks for slider changes.
"""
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class MainWindow:
    """Control panel: buttons on left, text output on right."""
    def __init__(self, root):
        self.root = root
        self.root.title("Ultrasound QC Toolbox - Control Panel")
        self.root.geometry("610x700")

        # Left frame for buttons – will resize to fit content
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(2,0), pady=5)
        left_frame.pack_propagate(False)   # we'll set width after buttons are created

        # Right frame for text output
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0,2), pady=5)

        # ---- Scrollable button area ----
        canvas = tk.Canvas(left_frame, highlightthickness=0)
        scrollbar = tk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ---- Create all buttons ----
        self._create_buttons(scrollable_frame)

        # Resize left frame to exactly fit buttons + scrollbar
        self.root.update_idletasks()
        total_width = scrollable_frame.winfo_reqwidth() + scrollbar.winfo_reqwidth() + 10
        left_frame.config(width=total_width)

        # ---- Text output widget (right side) ----
        text_frame = tk.Frame(right_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_text = tk.Scrollbar(text_frame, command=self.output_text.yview)
        scroll_text.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scroll_text.set)

        self.output_text.insert(tk.END, "Ready. Open a DICOM folder to begin.\n")

        self.callbacks = {}

    def _create_buttons(self, parent):
        # Group headers and buttons (same as before)
        tk.Label(parent, text="=== IMAGES ===", font=("Arial", 10, "bold")).pack(anchor="w", pady=(5,0))
        self.btn_open = tk.Button(parent, text="Open Folder", width=25)
        self.btn_open.pack(pady=2)
        self.btn_add = tk.Button(parent, text="Add Image", width=25)
        self.btn_add.pack(pady=2)
        self.btn_remove = tk.Button(parent, text="Remove Image", width=25)
        self.btn_remove.pack(pady=2)
        self.btn_sort = tk.Button(parent, text="Sort images by date", width=25)
        self.btn_sort.pack(pady=2)

        tk.Label(parent, text="=== STORAGE ===", font=("Arial", 10, "bold")).pack(anchor="w", pady=(5,0))
        self.btn_img1 = tk.Button(parent, text="Store Image 1", width=25)
        self.btn_img1.pack(pady=2)
        self.btn_img2 = tk.Button(parent, text="Store Image 2", width=25)
        self.btn_img2.pack(pady=2)

        tk.Label(parent, text="=== PROCESSING ===", font=("Arial", 10, "bold")).pack(anchor="w", pady=(5,0))
        self.btn_curved = tk.Button(parent, text="Curved reconstruction", width=25)
        self.btn_curved.pack(pady=2)
        self.btn_uniformity = tk.Button(parent, text="Uniformity measurements", width=25)
        self.btn_uniformity.pack(pady=2)
        self.btn_lcs = tk.Button(parent, text="Run LCS", width=25)
        self.btn_lcs.pack(pady=2)
        self.btn_hcs = tk.Button(parent, text="HCS Tool", width=25)
        self.btn_hcs.pack(pady=2)
        self.btn_lcp = tk.Button(parent, text="Run LCP", width=25)
        self.btn_lcp.pack(pady=2)

        tk.Label(parent, text="=== ROI ===", font=("Arial", 10, "bold")).pack(anchor="w", pady=(5,0))
        self.btn_save_roi = tk.Button(parent, text="Save ROI", width=25)
        self.btn_save_roi.pack(pady=2)
        self.btn_load_roi = tk.Button(parent, text="Load ROI", width=25)
        self.btn_load_roi.pack(pady=2)
        self.btn_copy_roi = tk.Button(parent, text="Copy ROI", width=25)
        self.btn_copy_roi.pack(pady=2)
        self.btn_paste_roi = tk.Button(parent, text="Paste ROI", width=25)
        self.btn_paste_roi.pack(pady=2)

        tk.Label(parent, text="=== UTILITIES ===", font=("Arial", 10, "bold")).pack(anchor="w", pady=(5,0))
        self.btn_print_dicom = tk.Button(parent, text="Print DICOM", width=25)
        self.btn_print_dicom.pack(pady=2)
        self.btn_measure = tk.Button(parent, text="Measure", width=25)
        self.btn_measure.pack(pady=2)
        self.btn_clear_output = tk.Button(parent, text="Clear Output", width=25)
        self.btn_clear_output.pack(pady=2)
        self.btn_print_roi = tk.Button(parent, text="Print ROI", width=25)
        self.btn_print_roi.pack(pady=2)
        self.btn_clear_overlay = tk.Button(parent, text="Clear Overlay", width=25)
        self.btn_clear_overlay.pack(pady=2)
        self.btn_fwhm = tk.Button(parent, text="FWHM", width=25)
        self.btn_fwhm.pack(pady=2)
        self.btn_calibrate = tk.Button(parent, text="Calibrate (10mm)", width=25)
        self.btn_calibrate.pack(pady=2)
        self.btn_detect_circles = tk.Button(parent, text="Detect Circles", width=25)
        self.btn_detect_circles.pack(pady=2)

    def set_callbacks(self, **callbacks):
        self.callbacks = callbacks
        mapping = {
            'on_open_folder': self.btn_open,
            'on_add_image': self.btn_add,
            'on_remove_image': self.btn_remove,
            'on_sort_images': self.btn_sort,
            'on_store_image1': self.btn_img1,
            'on_store_image2': self.btn_img2,
            'on_curved_recon': self.btn_curved,
            'on_uniformity': self.btn_uniformity,
            'on_run_lcs': self.btn_lcs,
            'on_hcs_tool': self.btn_hcs,
            'on_run_lcp': self.btn_lcp,
            'on_save_roi': self.btn_save_roi,
            'on_load_roi': self.btn_load_roi,
            'on_copy_roi': self.btn_copy_roi,
            'on_paste_roi': self.btn_paste_roi,
            'on_print_dicom': self.btn_print_dicom,
            'on_measure': self.btn_measure,
            'on_clear_output': self.btn_clear_output,
            'on_print_roi': self.btn_print_roi,
            'on_clear_overlay': self.btn_clear_overlay,
            'on_fwhm': self.btn_fwhm,
            'on_calibrate_mm': self.btn_calibrate,
            'on_detect_circles': self.btn_detect_circles,
        }
        for key, btn in mapping.items():
            if key in callbacks:
                btn.config(command=callbacks[key])

    def insert_output(self, text):
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.root.update_idletasks()

    def clear_output(self):
        self.output_text.delete(1.0, tk.END)

    def show_info(self, title, message):
        from tkinter import messagebox
        messagebox.showinfo(title, message)

    def show_error(self, title, message):
        from tkinter import messagebox
        messagebox.showerror(title, message)


class ImageViewerWindow:
    """Separate window with image display and a navigation slider."""
    def __init__(self, parent_root, total_images=0):
        self.window = tk.Toplevel(parent_root)
        self.window.title("DICOM Image Viewer")
        self.window.geometry("800x700")
        self.window.lift()
        self.window.focus_force()

        # Use grid layout: row0 = canvas (expands), row1 = slider (fixed height)
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=0)
        self.window.grid_columnconfigure(0, weight=1)

        # Matplotlib figure
        self.fig = Figure(figsize=(7, 7), dpi=100, facecolor='black')
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.ax.set_position([0, 0, 1, 1])
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Slider frame (row 1)
        slider_frame = tk.Frame(self.window)
        slider_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        self.slider_label = tk.Label(slider_frame, text="Image:")
        self.slider_label.pack(side=tk.LEFT, padx=5)

        self.slider = tk.Scale(slider_frame, from_=1, to=max(1, total_images),
                               orient=tk.HORIZONTAL, length=500,
                               showvalue=0, tickinterval=0)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.index_label = tk.Label(slider_frame, text="0/0", width=10)
        self.index_label.pack(side=tk.LEFT, padx=5)

        # Slider callback
        self.on_slider_changed = None
        self.slider.bind("<ButtonRelease-1>", self._slider_released)

    def _slider_released(self, event):
        if self.on_slider_changed:
            new_index = self.slider.get() - 1
            self.on_slider_changed(new_index)

    def set_total_images(self, total):
        self.slider.config(to=max(1, total))
        self.update_slider(0)

    def update_slider(self, current_index):
        self.slider.set(current_index + 1)
        self.index_label.config(text=f"{current_index+1}/{self.slider.cget('to')}")

    def set_slider_callback(self, callback):
        self.on_slider_changed = callback

    def get_canvas(self):
        return self.canvas

    def get_ax(self):
        return self.ax

    def update_image(self, image_array):
        self.ax.clear()
        self.ax.imshow(image_array, cmap='gray', aspect='auto')
        self.ax.axis('off')
        self.ax.set_position([0, 0, 1, 1])
        self.canvas.draw()

    def set_title(self, title):
        self.ax.set_title(title, color='white', fontsize=10)
        self.canvas.draw()