
import os
import numpy as np
import pydicom  # , gdcm
import matplotlib.pyplot as plt
import sys
from viewer import DCMViewer #, LCP
from matplotlib.widgets import Slider, RectangleSelector
from tkinter import Tk, filedialog, messagebox, Label, Text, Button, TclError
from scipy.signal import medfilt

# Add the directory containing your module to the Python path
sys.path.insert(0, 'C:\\DEV\\USTool\\venv\\USTool\\gui')

ds = None
def on_close():
    print("Window closed")
    window.destroy()  # This will terminate the application.
    sys.exit()  # This will terminate the Python process.



if __name__ == '__main__':
    fig, ax = plt.subplots()
    viewer = DCMViewer()
    window = Tk()
    window.protocol("WM_DELETE_WINDOW", on_close)  # Assign the handler function for the window close event.

    # Button to switch between rectangle and circular selectors
    btn_switch_selector = Button(window, text="Rect / Circle", fg='black', width=20, command=viewer.switch_selector)
    btn_switch_selector.place(x=240, y=270)
    
    # Declaring buttons for all widgets
    btn = Button(window, text="Open new", fg="black", width=20)
    btn.place(x=80, y=90)

    # Binds the button press with starting the image viewer
    btn.config(command=lambda: viewer.dcm_view())

    btn1 = Button(window, text="Add image", fg='black', width=20)
    btn1.place(x=80, y=120)

    # Binds the button press with starting

    btn2 = Button(window, text="Remove image", fg='black', width=20)
    btn2.place(x=80, y=150)

    # Binds the button press with starting

    btn3 = Button(window, text="Sort images by date", fg='black', width=20)
    btn3.place(x=80, y=180)

    # Binds the button press with starting


    btn6 = Button(window, text="Curved reconstruction", fg='blue', width=20)
    btn6.place(x=240, y=120)

    btn7 = Button(window, text="Uniformity measurements", fg='blue', width=20)
    btn7.place(x=240, y=150)
    # Binds the uniformity button press with starting Horssen measurements on the image with the ROI

    # btn7.bind('<Button-1>', lambda event: Horssen())

    # Binds the button press with starting

    btn8 = Button(window, text="Run LCS", fg='blue', width=20)
    btn8.place(x=240, y=180)

    # Binds the button press with starting

    btn9 = Button(window, text="HCS Tool", fg='blue', width=20)
    btn9.place(x=240, y=210)

    # Binds the button press with starting LCP

    btn10 = Button(window, text="Run LCP", fg='blue', width=20)
    btn10.place(x=80, y=240)
    btn10.config(command=lambda: viewer.calc_lcp(viewer.x1, viewer.y1, viewer.x2, viewer.y2))

    btn101 = Button(window, text="image 1", fg='blue', width=16)
    btn101.place(x=108, y=270)
    btn101.config(command=viewer.store_image_1)

    btn102 = Button(window, text="image 2", fg='blue', width=16)
    btn102.place(x=108, y=300)
    btn102.config(command=viewer.store_image_2)

    # Binds the button press with starting


    btn11 = Button(window, text="Save ROI", fg='red', width=20)
    btn11.place(x=400, y=90)

    # Binds the button press with starting

    btn12 = Button(window, text="Load ROI", fg='red', width=20)
    btn12.place(x=400, y=120)

    # Binds the button press with starting

    btn13 = Button(window, text="Copy ROI", fg='red', width=20)
    btn13.place(x=400, y=150)

    # Binds the button press with starting

    btn14 = Button(window, text="Paste ROI", fg='red', width=20)
    btn14.place(x=400, y=180)

    # Binds the button press with starting


    btn15 = Button(window, text="Print DICOM", fg='black')
    btn15.place(x=20, y=390)

    # Binds the button press with starting

    btn16 = Button(window, text="Measure", fg='black')
    btn16.place(x=100, y=390)

    # Binds the button press with starting

    btn17 = Button(window, text="Clear Output", fg='black')
    btn17.place(x=158, y=390)

    # Binds the button press with starting

    btn18 = Button(window, text="Print ROI", fg='black')
    btn18.place(x=239, y=390)

    # Binds the button press with starting

    btn19 = Button(window, text="Clear Overlay", fg='black')
    btn19.place(x=300, y=390)

    # Binds the button press with starting

    btn20 = Button(window, text="FWHM", fg='black')
    btn20.place(x=383, y=390)

    # Binds the button press with starting

    # buttons = ["Open new", "Add Image", "Remove Image", "Sort images by date", "Curved reconstruction",
    # "Uniformity measurements", "Run LCS", "HCS Tool", "Run LCP", "image 1", "image 2",
    # "Save ROI", "Load ROI", "Copy ROI", "Paste ROI", "Print DICOM", "Measure", "Clear Output",
    # "Print ROI", "Clear Overlay", "FWHM"]
    # Declaring text field for measurements output
    txtfld2 = Text(window, width=60)
    txtfld2.place(x=20, y=420, width=550, height=120)
    # Create the txtfld2 Text widget and store its reference in the viewer instance
    viewer.txtfld2 = txtfld2

    # Label for Images group of widgets
    lbl1 = Label(window, text="Images", fg='black')
    lbl1.place(x=120, y=40)

    # Label for Scripts group of widgets
    lbl2 = Label(window, text="Scripts", fg='black')
    lbl2.place(x=280, y=40)

    # Label for ROI group of widgets
    lbl3 = Label(window, text="ROI", fg='black')
    lbl3.place(x=460, y=40)

    window.title('USToolbox')
    window.geometry("600x600+10+10")
    window.mainloop()
    

