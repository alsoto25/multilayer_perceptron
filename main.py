import utils as ut
import re
import os
import tkinter as tk
import matplotlib as mpl
from matplotlib import style
import matplotlib.pyplot as plt
# import scipy.misc as smp
import neural_network_3 as nn
import numpy as np
from mnist import MNIST
from matplotlib import style
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# MNIST Constants
mndata = MNIST('./MNIST')
# test_imgs, test_lbls = mndata.load_testing()
# train_imgs, train_lbls = mndata.load_training()
#
# train_imgs = np.concatenate((np.asarray(train_imgs), np.random.uniform(0, 256, size=(6000, 784))), axis=0)
# train_lbls = np.append(np.asarray(train_lbls), np.ones(6000, dtype=np.int8)*10)
#
# test_imgs = np.concatenate((np.asarray(test_imgs), np.random.uniform(0, 256, size=(1000, 784))), axis=0)
# test_lbls = np.append(np.asarray(test_lbls), np.ones(1000, dtype=np.int8)*10)


mpl.use('TkAgg')

# Constants
LARGE_FONT = ('Verdana', 12)
EXTRA_LARGE_FONT = ('Verdana', 18)
style.use('ggplot')

# Variables
network = nn.Network(np.asarray([[0, 1], [1, 0]]), [0, 1], 2, [4, 3], 2, np.asarray([[0, 1], [1, 0]]), [0, 1])


class MainGUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.iconbitmap(self, default='./png/ai-icon.ico')
        tk.Tk.wm_title(self, 'MultiLayered Perceptron')

        main_container = tk.Frame(self)

        main_container.pack(side='top', fill='both', expand=True)

        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (TrainPage, TestPage):
            frame = F(main_container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame(TrainPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class TrainPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        self.grid_columnconfigure(2, weight=1)

        title_label = tk.Label(self, text='Load Network', font=LARGE_FONT, justify='center')
        title_label.grid(row=0, sticky=tk.N, columnspan=2, pady=10, padx=10)

        # img = smp.toimage(255 - nn.train_imgs[30].reshape((28, 28)))  # Create a PIL image
        # img.save('./tmp/tmp_test_img.bmp')
        #
        # img_pil = Image.open('./tmp/tmp_test_img.bmp')
        # img_pil = img_pil.resize((80, 80), Image.ANTIALIAS)
        # img_tk = ImageTk.PhotoImage(img_pil)
        # label1 = tk.Label(self, image=img_tk)
        # label1.image = img_tk
        # label1.grid(row=1, column=0, pady=10, padx=10)

        layers_lbl = tk.Label(self, text='Hidden Layer Sizes:')
        acc_lbl = tk.Label(self, text='Accuracy:')
        drop_lbl = tk.Label(self, text='Dropout:')
        batch_lbl = tk.Label(self, text='Batch Size:')
        epochs_lbl = tk.Label(self, text='Epochs:')

        self.layers_txt = tk.Entry(self, state='disabled')
        self.acc_txt = tk.Entry(self, state='disabled')
        self.drop_txt = tk.Entry(self, state='disabled')
        self.batch_txt = tk.Entry(self, state='disabled')
        self.epochs_txt = tk.Entry(self, state='disabled')

        load_lbl = tk.Label(self, text='Choose your network:')
        self.load_txt = tk.Entry(self, state='disabled')
        load_img = tk.PhotoImage(file='./png/001-login.png')
        load_btn = tk.Button(self, border=0, command=lambda: self.get_network())
        load_btn.image = load_img
        load_btn.config(image=load_img)

        load_lbl.grid(row=1, column=0, pady=10, padx=10, sticky=tk.E)
        self.load_txt.grid(row=1, column=1, pady=10, padx=10, sticky=tk.W)
        load_btn.grid(row=1, column=2, pady=10, padx=3, sticky=tk.W)

        layers_lbl.grid(row=2, column=0, pady=3, padx=10, sticky=tk.E)
        self.layers_txt.grid(row=2, column=1, pady=3, padx=10, sticky=tk.W)

        acc_lbl.grid(row=3, column=0, pady=3, padx=10, sticky=tk.E)
        self.acc_txt.grid(row=3, column=1, pady=3, padx=10, sticky=tk.W)

        drop_lbl.grid(row=4, column=0, pady=3, padx=10, sticky=tk.E)
        self.drop_txt.grid(row=4, column=1, pady=3, padx=10, sticky=tk.W)

        batch_lbl.grid(row=5, column=0, pady=3, padx=10, sticky=tk.E)
        self.batch_txt.grid(row=5, column=1, pady=3, padx=10, sticky=tk.W)

        epochs_lbl.grid(row=6, column=0, pady=3, padx=10, sticky=tk.E)
        self.epochs_txt.grid(row=6, column=1, pady=3, padx=10, sticky=tk.W)

    def get_network(self):
        name = askopenfilename(initialdir='./pickles/',
                               title='Choose a network.')

        name = name.split('/')[-1]

        print(name)

        self.load_txt.config(state='normal')
        self.load_txt.delete(0, tk.END)
        self.load_txt.insert(0, name)
        self.load_txt.config(state='disabled')

        regex = re.findall(r'\D_(\d*)(?:_|$)', name, re.I)

        network.load(name)

        hidden_layers = str(network.n_in) + ', '
        for i in range(len(network.hidden_layer_sizes)):
            hidden_layers += str(network.hidden_layer_sizes[0]) + ', '
        hidden_layers += str(network.n_out)

        self.layers_txt.config(state='normal')
        self.layers_txt.delete(0, tk.END)
        self.layers_txt.insert(0, hidden_layers)
        self.layers_txt.config(state='disabled')

        self.acc_txt.config(state='normal')
        self.acc_txt.delete(0, tk.END)
        self.acc_txt.insert(0, regex[5][:2] + '.' + regex[5][2:] + '%')
        self.acc_txt.config(state='disabled')

        self.drop_txt.config(state='normal')
        drp = float(regex[3][:1] + '.' + regex[3][1:]) * 100
        self.drop_txt.delete(0, tk.END)
        self.drop_txt.insert(0, str(drp) + '%')
        self.drop_txt.config(state='disabled')

        self.batch_txt.config(state='normal')
        self.batch_txt.delete(0, tk.END)
        self.batch_txt.insert(0, regex[4])
        self.batch_txt.config(state='disabled')

        self.epochs_txt.config(state='normal')
        self.epochs_txt.delete(0, tk.END)
        self.epochs_txt.insert(0, regex[1])
        self.epochs_txt.config(state='disabled')

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.plot(range(len(network.accs)), network.accs)

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=7, column=0, columnspan=3)

        img_btn = tk.Button(self, text='Predict Image', command=lambda: self.controller.show_frame(TestPage))
        img_btn.grid(row=8, column=0, columnspan=3, pady=15)


class TestPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=3)

        title_label_1 = tk.Label(self, text='Load Image', font=LARGE_FONT)
        title_label_1.grid(row=0, column=0, columnspan=2, pady=10, padx=10)

        title_label_2 = tk.Label(self, text='Predicted Number', font=LARGE_FONT)
        title_label_2.grid(row=0, column=3, columnspan=2, pady=10, padx=10)

        load_lbl = tk.Label(self, text='Choose your image:')
        self.load_txt = tk.Entry(self, state='disabled')
        load_img = tk.PhotoImage(file='./png/001-login.png')
        load_btn = tk.Button(self, border=0, command=lambda: self.get_test_image())
        load_btn.image = load_img
        load_btn.config(image=load_img)

        back_btn = tk.Button(self, text='Back', command=lambda: controller.show_frame(TrainPage))

        self.img_lbl = tk.Label(self)
        self.pred_lbl = tk.Label(self, font=EXTRA_LARGE_FONT)

        load_lbl.grid(row=1, column=0, pady=10, padx=10, sticky=tk.E)
        self.load_txt.grid(row=1, column=1, pady=10, padx=10, sticky=tk.W)
        load_btn.grid(row=1, column=2, pady=10, padx=3, sticky=tk.W)

        self.img_lbl.grid(row=2, column=0, columnspan=3, padx=3, pady=25)
        self.pred_lbl.grid(row=2, column=3, columnspan=1, padx=3, pady=25)

        back_btn.grid(row=3, column=0, columnspan=4, pady=15)

    def get_test_image(self):
        name = askopenfilename(initialdir='./',
                               filetypes=(('Image File', '*.png;*.jpg;*.jpeg;*.gif;*.bmp'), ('All Files', '*.*')),
                               title='Choose a file.')

        self.load_txt.config(state='normal')
        self.load_txt.delete(0, tk.END)
        self.load_txt.insert(0, name)
        self.load_txt.config(state='disabled')

        img_pil = Image.open(name)
        img_array = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
        if len(img_array.shape) > 2:
            img_array = ut.greyscale_image(img_array)

        img_array = img_array.flatten()

        img_pil = img_pil.resize((80, 80), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.img_lbl.config(image=img_tk)
        self.img_lbl.image = img_tk

        pred_res = network.predict(img_array)
        if pred_res == 10:
            pred_res = network.predict(255 - img_array)
            if pred_res == 10:
                pred_res = 'Not a known number'
            else:
                str(pred_res)
        else:
            pred_res = str(pred_res)
        self.pred_lbl.config(text=pred_res)


# app = MainGUI()
# app.mainloop()

# network = nn.Network(np.asarray(mndata.load_training()[0]),
#                      np.asarray(mndata.load_training()[1]),
#                      784, [128, 1024], 11,
#                      np.asarray(mndata.load_testing()[0]),
#                      np.asarray(mndata.load_testing()[1]))
#
# network.train()
# network.load('im_nn_1024_128_ep_41_lr_001_dp_05_bs_50_acc_9880909090909091')
# print(len(netwk.costs))
# print(netwk.costs)
# print(network.test(nn.test_imgs, nn.test_lbls))

# for i in range(784):
#     network.print_img(network.hidden_layers[0], i, filename='hidden_layer_1/784_784/' + str(i))
#     network.print_img(network.hidden_layers[1], i, filename='hidden_layer_2/784_784/' + str(i))
#
# for i in range(10):
#     netwk.print_img(netwk.log_layer, i, filename='log_layer/' + str(i))


# network = nn.Network(nn.train_imgs, nn.train_lbls, nn.train_imgs.shape[1],
#                     [512, 256, 128], np.unique(nn.train_lbls).size)


# network.load('im_nn_1024_128_ep_41_lr_001_dp_05_bs_50_acc_9880909090909091')

directory = os.fsencode('./pickles/')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    network.load(filename)

    lay = ''
    for x in network.hidden_layer_sizes:
        lay += str(x) + '_'

    plt.plot(range(len(network.accs)), network.accs)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.savefig('./plots/' + lay + 'accs.png')
    plt.close()

    plt.plot(range(len(network.costs)), network.costs)
    plt.xlabel('Epochs')
    plt.ylabel('Test Cost')
    plt.savefig('./plots/' + lay + 'costs.png')
    plt.close()
