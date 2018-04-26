import utils as ut
import tkinter as tk
import matplotlib as mpl
import scipy.misc as smp
import neural_network as nn
import matplotlib.animation as animation

mpl.use("TkAgg")

from matplotlib import style
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg


# Constants
LARGE_FONT = ("Verdana", 12)
style.use("ggplot")

# Variables


class MainGUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.iconbitmap(self, default="./png/ai-icon.ico")
        tk.Tk.wm_title(self, "MultiLayered Perceptron")

        main_container = tk.Frame(self)

        main_container.pack(side="top", fill="both", expand=True)

        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (TrainPage, TestPage):
            frame = F(main_container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(TrainPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class TrainPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        img = smp.toimage(255 - nn.train_imgs[60].reshape((28, 28)))  # Create a PIL image
        img.save('./tmp/tmp_test_img.bmp')

        img_pil = Image.open("./tmp/tmp_test_img.bmp")
        img_pil = img_pil.resize((80, 80), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img_pil)
        label1 = tk.Label(self, image=img_tk)
        label1.image = img_tk
        label1.pack(pady=10, padx=10)

        file_text_input = tk.Entry(self)

        image1 = tk.PhotoImage(file="./png/001-login.png")
        button = tk.Button(self, border=0, command=lambda: controller.show_frame(TestPage))
        button.image = image1
        button.config(image=image1)
        button.pack(pady=10, padx=10)


class TestPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Main Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        # file_text_input = tk.Entry(self, )

        image1 = tk.PhotoImage(file="./png/001-login.png")
        button = tk.Button(self, border=0, command=get_test_image)
        button.image = image1
        button.config(image=image1)
        button.pack(pady=10, padx=10)

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8], [5,4,8,9,6,1,1,3])

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def get_test_image():
    name = askopenfilename(initialdir="./",
                           filetypes=(("Image File", "*.png;*.jpg;*.jpeg;*.gif"), ("All Files", "*.*")),
                           title="Choose a file.")

    print(name)


app = MainGUI()
app.mainloop()
