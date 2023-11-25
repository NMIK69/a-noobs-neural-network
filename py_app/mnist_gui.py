import tkinter.filedialog as tkfd
import tkinter as tk
import numpy as np
import math
import struct
from PIL import Image, ImageDraw


class FF_Net:
	
	def __init__(self):
		self.layers = []
		self.weights = [] 
		self.bias = [] 



	def load_from_file(self, filename):
		self.bias = []
		self.weights = []
		self.layers = []
		
		fp = open(filename, "rb")

		num_layers = struct.unpack('i', fp.read(4))[0]
		self.layers = [[] for x in range(num_layers)]

		layer_size = []
		for i in range(num_layers):
			val = struct.unpack('i', fp.read(4))[0]
			layer_size.append(val)

		bias_size = num_layers - 1
		weight_size = num_layers - 1

		for i in range(bias_size):
			self.bias.append([])
			for j in range(layer_size[i + 1]):
				val = struct.unpack('f', fp.read(4))[0]
				self.bias[i].append(val)


		for i in range(weight_size):
			self.weights.append([])
			H = layer_size[i + 1]
			W = layer_size[i]

			for j in range(H):
				self.weights[i].append([])
				for k in range(W):
					val = struct.unpack('f', fp.read(4))[0]
					self.weights[i][j].append(val)

		fp.close()

			
	def sigmoid(self, n):
		return 1.0 / (1.0 + np.exp(-n))
	
	def mmul(self, w, l, b):
		H = len(w)
		W = len(w[0])

		ret = [0 for _ in range(H)]

		for i in range(H):
			net = 0.0
			for j in range(W):
				net = net + (w[i][j] * l[j])
			ret[i] = self.sigmoid(net + b[i])

		return ret

	def fprop(self, input_data):
		self.layers[0] = input_data
		for i in range(1, len(self.layers)):
			self.layers[i] = self.mmul(self.weights[i-1],
						    self.layers[i-1], 
						       self.bias[i-1])


	def get_prediction(self):
		max_outval = -1
		res = -1

		outlayer = self.layers[-1]
		for i in range(len(outlayer)):
			if(outlayer[i] > max_outval):
				max_outval = outlayer[i]
				res = i

		return res
			
	

class GUI:

	def __init__(self):

		self.nn = FF_Net()

		self.img_size = 560
		self.img = Image.new("L", (self.img_size, self.img_size), 0)
		self.img_draw = ImageDraw.Draw(self.img)
		
		self.line_width = 40
		self.line_color = "#000000"

		self.root = tk.Tk()
		self.root.title("MNIST Test")
		self.root.geometry("720x800")
		

		self.load_nn_btn = tk.Button(self.root, text="Load Neural Network", 
							width=67, command=self.load_nn)
		self.load_nn_btn.pack(pady=8, ipady=8)

		
		self.dc_frame = tk.Frame(self.root,
				   	 highlightbackground="black", 
					 highlightthickness=2)
		self.dc_frame.pack()
		for i in range(5):
			self.dc_frame.columnconfigure(i, weight=1)
			self.dc_frame.rowconfigure(i, weight=1)

		self.dc_lbl = tk.Label(self.dc_frame, text="Drawing Area:",
					borderwidth=2, relief="ridge", bg="white")
		self.dc_lbl.grid(row=0, column=0, sticky=tk.W + tk.E)
		
		self.draw_canvas = tk.Canvas(self.dc_frame, width=self.img_size, 
						height=self.img_size, cursor="dot", bg="white")
		self.draw_canvas.grid(row=1, column=0)
		self.draw_canvas.bind("<B1-Motion>", self.paint)

		self.clear_btn = tk.Button(self.dc_frame, text="Clear",
						command=self.clear_canvas)
		self.clear_btn.grid(row=2, column=0, sticky=tk.W + tk.E)

		self.make_pred_btn = tk.Button(self.dc_frame, text="Make Prediction",
								command=self.predict)
		self.make_pred_btn.grid(row=3, column=0, sticky=tk.W + tk.E)

		self.prediction_lbl = tk.Label(self.dc_frame, text="Prediction:___ ")
		self.prediction_lbl.grid(row=4, column=0, sticky=tk.W + tk.E)

		self.root.mainloop()


	def paint(self, event):
		x1, y1 = (event.x - 1), (event.y - 1)
		x2, y2 = (event.x + 1), (event.y + 1)
		self.draw_canvas.create_oval(x1, y1, x2, y2,
					fill=self.line_color,
					width=self.line_width)

		lwh = self.line_width/2
		x1, y1 = (event.x - lwh), (event.y - lwh)
		x2, y2 = (event.x + lwh), (event.y + lwh)
		self.img_draw.ellipse([x1, y1, x2, y2], outline=255, 
					fill=255, width=self.line_width)

	
	def clear_canvas(self):
		self.draw_canvas.delete("all")
		self.img_draw.rectangle([0, 0, self.img_size, self.img_size], fill=0)


	def load_nn(self):
		fname = tkfd.askopenfilename(title="Select Neural Network file")
		self.nn.load_from_file(fname)

		
	def predict(self):
		temp = self.img.copy()
		temp.thumbnail((28,28), resample=Image.LANCZOS)
		#temp.save("temp.jpg")
		pd = list(temp.getdata())

		for i in range(len(pd)):
			if(pd[i] < 10):
				pd[i] = 0

		for i in range(len(pd)):
			pd[i] = pd[i] / 255

		self.nn.fprop(pd)
		res = self.nn.get_prediction()
		self.prediction_lbl.config(text="Prediction: " + str(res))

GUI()
