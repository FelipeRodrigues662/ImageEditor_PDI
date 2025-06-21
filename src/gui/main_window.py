import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import os

from ..utils import image_utils
from ..utils.history import HistoryManager
from ..processing import histogram, filters, fourier, morphology


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDI - Editor de Imagens")
        self.geometry("1200x700")
        self.configure(bg="#f5f5f5")
        self.iconbitmap(default=None)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Estado
        self.image = None
        self.image_path = None
        self.photo = None
        self.history = HistoryManager()
        
        # Variáveis de zoom
        self.zoom_level = 100  # Zoom inicial em 100%
        self.original_image = None  # Imagem original sem redimensionamento
        self.display_image = None  # Imagem atual para exibição

        # Layout
        self.create_menu()
        self.create_widgets()
        self.bind_shortcuts()

    def create_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Abrir Imagem...", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Salvar Imagem Como...", command=self.save_image, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self.on_close)
        menubar.add_cascade(label="Arquivo", menu=file_menu)
        self.config(menu=menubar)

    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Painel esquerdo: controles
        self.controls_frame = ttk.Frame(main_frame, width=250)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.create_controls(self.controls_frame)

        # Painel central: imagem
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame para controles de zoom
        zoom_frame = ttk.Frame(self.image_frame)
        zoom_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Controles de zoom
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)
        self.zoom_scale = ttk.Scale(zoom_frame, from_=10, to=300, orient=tk.HORIZONTAL, 
                                   command=self.on_zoom_scale_change)
        self.zoom_scale.set(100)
        self.zoom_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=6)
        self.zoom_label.pack(side=tk.RIGHT)
        
        ttk.Button(zoom_frame, text="Reset", command=self.reset_zoom).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Canvas com scrollbars para a imagem
        self.canvas_frame = ttk.Frame(self.image_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        
        # Canvas para a imagem
        self.image_canvas = tk.Canvas(self.canvas_frame, bg="#ddd",
                                     xscrollcommand=self.h_scrollbar.set,
                                     yscrollcommand=self.v_scrollbar.set)
        
        # Configurar scrollbars
        self.h_scrollbar.config(command=self.image_canvas.xview)
        self.v_scrollbar.config(command=self.image_canvas.yview)
        
        # Layout do canvas e scrollbars
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind do scroll do mouse
        self.image_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.image_canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux
        self.image_canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux
        
        # Bind para redimensionamento da janela
        self.bind("<Configure>", self.on_window_resize)

        # Painel direito: histograma
        self.hist_frame = ttk.Frame(main_frame, width=250)
        self.hist_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.hist_canvas = tk.Canvas(self.hist_frame, width=220, height=180, bg="#fff")
        self.hist_canvas.pack(pady=20)
        self.hist_label = ttk.Label(self.hist_frame, text="Histograma", font=("Arial", 12, "bold"))
        self.hist_label.pack()

    def create_controls(self, parent):
        # Botões de desfazer/refazer
        undo_btn = ttk.Button(parent, text="Desfazer (Ctrl+Z)", command=self.undo)
        undo_btn.pack(fill=tk.X, pady=2)
        redo_btn = ttk.Button(parent, text="Refazer (Ctrl+Y)", command=self.redo)
        redo_btn.pack(fill=tk.X, pady=2)
        ttk.Separator(parent).pack(fill=tk.X, pady=8)

        # Filtros
        ttk.Label(parent, text="Filtros", font=("Arial", 11, "bold")).pack(pady=2)
        ttk.Button(parent, text="Média", command=lambda: self.apply_filter('mean')).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="Mediana", command=lambda: self.apply_filter('median')).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="Gaussiano", command=lambda: self.apply_filter('gaussian')).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="Máximo", command=lambda: self.apply_filter('max')).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="Mínimo", command=lambda: self.apply_filter('min')).pack(fill=tk.X, pady=1)
        ttk.Separator(parent).pack(fill=tk.X, pady=8)
        ttk.Label(parent, text="Passa-Alta", font=("Arial", 11, "bold")).pack(pady=2)
        ttk.Button(parent, text="Laplaciano", command=lambda: self.apply_filter('laplacian')).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="Roberts", command=lambda: self.apply_filter('roberts')).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="Prewitt", command=lambda: self.apply_filter('prewitt')).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="Sobel", command=lambda: self.apply_filter('sobel')).pack(fill=tk.X, pady=1)
        ttk.Separator(parent).pack(fill=tk.X, pady=8)
        ttk.Label(parent, text="Outros", font=("Arial", 11, "bold")).pack(pady=2)
        ttk.Button(parent, text="Equalizar Histograma", command=lambda: self.apply_filter('equalize')).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="Alargar Contraste", command=lambda: self.apply_filter('stretch')).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="Espectro de Fourier", command=self.show_fourier).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="Erosão", command=lambda: self.apply_filter('erosion')).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="Dilatação", command=lambda: self.apply_filter('dilation')).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="Otsu (Segmentação)", command=lambda: self.apply_filter('otsu')).pack(fill=tk.X, pady=1)

    def bind_shortcuts(self):
        self.bind_all('<Control-o>', lambda e: self.open_image())
        self.bind_all('<Control-s>', lambda e: self.save_image())
        self.bind_all('<Control-z>', lambda e: self.undo())
        self.bind_all('<Control-y>', lambda e: self.redo())

    def on_mouse_wheel(self, event):
        """Manipula o scroll do mouse para zoom"""
        if self.image is None:
            return
            
        # Determina a direção do scroll
        if event.num == 4 or event.delta > 0:  # Scroll up
            self.zoom_level = min(300, self.zoom_level + 10)
        elif event.num == 5 or event.delta < 0:  # Scroll down
            self.zoom_level = max(10, self.zoom_level - 10)
            
        # Atualiza a escala e a imagem
        self.zoom_scale.set(self.zoom_level)
        self.update_zoom_display()
        self.update_image()

    def on_zoom_scale_change(self, value):
        """Manipula mudanças na escala de zoom"""
        self.zoom_level = int(float(value))
        self.update_zoom_display()
        self.update_image()

    def update_zoom_display(self):
        """Atualiza o label de zoom"""
        self.zoom_label.config(text=f"{self.zoom_level}%")

    def reset_zoom(self):
        """Reseta o zoom para 100%"""
        self.zoom_level = 100
        self.zoom_scale.set(100)
        self.update_zoom_display()
        self.update_image()

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if not file_path:
            return
        image = image_utils.load_image(file_path)
        if image is None:
            messagebox.showerror("Erro", "Não foi possível abrir a imagem.")
            return
        self.image = image_utils.convert_to_grayscale(image)
        self.original_image = self.image.copy()  # Guarda a imagem original
        self.image_path = file_path
        self.history.clear()
        self.history.add_action("Imagem aberta", self.image)
        self.reset_zoom()  # Reseta o zoom ao abrir nova imagem
        self.update_image()
        self.update_histogram()

    def save_image(self):
        if self.image is None:
            messagebox.showinfo("Salvar", "Nenhuma imagem para salvar.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp")])
        if not file_path:
            return
        image_utils.save_image(self.image, file_path)
        messagebox.showinfo("Salvar", f"Imagem salva em {file_path}")

    def update_image(self):
        if self.image is None:
            self.image_canvas.delete("all")
            return
            
        # Redimensiona a imagem baseado no zoom
        height, width = self.image.shape[:2]
        new_width = int(width * self.zoom_level / 100)
        new_height = int(height * self.zoom_level / 100)
        
        # Redimensiona a imagem
        import cv2
        resized_img = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Converte para PIL Image
        img = Image.fromarray(resized_img)
        self.photo = ImageTk.PhotoImage(img)
        
        # Limpa o canvas
        self.image_canvas.delete("all")
        
        # Obtém as dimensões do canvas
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # Se o canvas ainda não foi renderizado, usa valores padrão
        if canvas_width <= 1:
            canvas_width = 600
        if canvas_height <= 1:
            canvas_height = 400
        
        # Calcula a posição central
        x_center = (canvas_width - new_width) // 2
        y_center = (canvas_height - new_height) // 2
        
        # Garante que a posição não seja negativa
        x_center = max(0, x_center)
        y_center = max(0, y_center)
        
        # Adiciona a imagem centralizada
        self.image_canvas.create_image(x_center, y_center, anchor=tk.NW, image=self.photo)
        
        # Configura a região de scroll
        self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))
        
        # Centraliza a imagem após um pequeno delay para garantir que o canvas foi renderizado
        self.after(10, self.center_image)

    def center_image(self):
        """Centraliza a imagem no canvas"""
        if self.image is None or self.photo is None:
            return
            
        # Obtém as dimensões do canvas
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        # Obtém as dimensões da imagem
        img_width = self.photo.width()
        img_height = self.photo.height()
        
        # Calcula a posição central
        x_center = (canvas_width - img_width) // 2
        y_center = (canvas_height - img_height) // 2
        
        # Garante que a posição não seja negativa
        x_center = max(0, x_center)
        y_center = max(0, y_center)
        
        # Limpa o canvas e redesenha a imagem centralizada
        self.image_canvas.delete("all")
        self.image_canvas.create_image(x_center, y_center, anchor=tk.NW, image=self.photo)
        
        # Configura a região de scroll
        self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))

    def update_histogram(self):
        if self.image is None:
            self.hist_canvas.delete("all")
            return
        hist, _ = histogram.calculate_histogram(self.image)
        self.hist_canvas.delete("all")
        max_h = np.max(hist)
        for i in range(256):
            h = int(hist[i] / max_h * 150)
            self.hist_canvas.create_line(i*220/256, 180, i*220/256, 180-h, fill="#0077cc")

    def apply_filter(self, filter_name):
        if self.image is None:
            return
        img = self.image
        if filter_name == 'mean':
            img = filters.mean_filter(img)
        elif filter_name == 'median':
            img = filters.median_filter(img)
        elif filter_name == 'gaussian':
            img = filters.gaussian_filter(img)
        elif filter_name == 'max':
            img = filters.max_filter(img)
        elif filter_name == 'min':
            img = filters.min_filter(img)
        elif filter_name == 'laplacian':
            img = filters.laplacian_filter(img)
        elif filter_name == 'roberts':
            img = filters.roberts_filter(img)
        elif filter_name == 'prewitt':
            img = filters.prewitt_filter(img)
        elif filter_name == 'sobel':
            img = filters.sobel_filter(img)
        elif filter_name == 'equalize':
            img = histogram.equalize_histogram(img)
        elif filter_name == 'stretch':
            img = histogram.stretch_contrast(img)
        elif filter_name == 'erosion':
            img = morphology.erosion(img)
        elif filter_name == 'dilation':
            img = morphology.dilation(img)
        elif filter_name == 'otsu':
            img = self.apply_otsu(img)
        else:
            return
        self.image = img
        self.original_image = self.image.copy()  # Atualiza a imagem original
        self.history.add_action(f"Filtro: {filter_name}", self.image)
        self.update_image()
        self.update_histogram()

    def apply_otsu(self, img):
        # Segmentação por Otsu
        import cv2
        if len(img.shape) == 3:
            img = image_utils.convert_to_grayscale(img)
        _, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu_img

    def show_fourier(self):
        if self.image is None:
            return
        mag = fourier.get_magnitude_spectrum(fourier.compute_fft(self.image))
        mag = (mag / np.max(mag) * 255).astype(np.uint8)
        win = tk.Toplevel(self)
        win.title("Espectro de Fourier")
        img = image_utils.resize_image(mag, max_size=400)
        img = Image.fromarray(img)
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=photo)
        lbl.image = photo
        lbl.pack()

    def undo(self):
        action = self.history.undo()
        if action:
            self.image = action['image']
            self.original_image = self.image.copy()
            self.update_image()
            self.update_histogram()

    def redo(self):
        action = self.history.redo()
        if action:
            self.image = action['image']
            self.original_image = self.image.copy()
            self.update_image()
            self.update_histogram()

    def on_close(self):
        self.destroy()

    def on_window_resize(self, event):
        """Centraliza a imagem quando a janela é redimensionada"""
        if self.image is not None:
            # Usa um pequeno delay para garantir que o redimensionamento foi concluído
            self.after(50, self.center_image)


def run_app():
    app = MainWindow()
    app.mainloop() 