# main.py

"""
Initialises the entire application, creates the Tkinter root window, instantiates the model, view, and controller, and starts the GUI event loop. 
The main function sets up the MVC architecture by creating instances of the model (DICOMCache, LCS, LCPCalculator), the view (MainWindow), and the controller (Controller). 
It also connects all the button callbacks in the view to the corresponding methods in the controller. Finally, it starts the Tkinter main loop to run the application.
"""
import tkinter as tk
from model import DICOMCache, LCS, LCPCalculator
from view import MainWindow
from controller import Controller


def main():
    root = tk.Tk()
    view = MainWindow(root)

    # Model
    cache = DICOMCache(cache_window=8)
    lcs = LCS()
    lcp = LCPCalculator()
    model = {
        'dicom_cache': cache,
        'lcs': lcs,
        'lcp': lcp,
        'stored_images': {'img1': None, 'img2': None},
        'current_index': 0,
        'roi_coords': None,
        'mm_per_pix': None,
        'detected_circles': [],
        'circle_contrast_data': []
    }

    controller = Controller(model, view)

    # Connect all buttonsf
    view.set_callbacks(
        on_open_folder=controller.on_open_folder,
        on_add_image=controller.on_add_image,
        on_remove_image=controller.on_remove_image,
        on_sort_images=controller.on_sort_images,
        on_store_image1=controller.on_store_image1,
        on_store_image2=controller.on_store_image2,
        on_curved_recon=controller.on_curved_recon,
        on_uniformity=controller.on_uniformity,
        on_run_lcs=controller.on_run_lcs,
        on_hcs_tool=controller.on_hcs_tool,
        on_run_lcp=controller.on_run_lcp,
        on_save_roi=controller.on_save_roi,
        on_load_roi=controller.on_load_roi,
        on_copy_roi=controller.on_copy_roi,
        on_paste_roi=controller.on_paste_roi,
        on_print_dicom=controller.on_print_dicom,
        on_measure=controller.on_measure,
        on_clear_output=controller.on_clear_output,
        on_print_roi=controller.on_print_roi,
        on_clear_overlay=controller.on_clear_overlay,
        on_fwhm=controller.on_fwhm,
        on_calibrate_mm=controller.on_calibrate_mm,
        on_detect_circles=controller.on_detect_circles,
    )

    root.mainloop()


if __name__ == "__main__":
    main() 