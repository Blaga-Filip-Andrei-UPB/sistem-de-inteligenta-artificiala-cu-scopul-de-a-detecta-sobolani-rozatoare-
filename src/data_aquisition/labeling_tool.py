import os
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import csv

# Extensii de fișiere imagine acceptate
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')


class LabelTool:
    def __init__(self, root):
        self.root = root
        root.title("Bounding box labeling tool")

        # Variabile pentru gestionarea imaginilor
        self.image_folder = None
        self.image_files = []
        self.idx = 0

        # Imaginea curentă în format PIL și Tkinter
        self.current_image = None
        self.current_tkimage = None

        # Dimensiunea maximă de afișare
        self.display_size = (1000, 800)

        # Factori de scalare între imaginea afișată și cea originală
        self.scale_x = 1.0
        self.scale_y = 1.0

        # Dicționar: nume_imagine -> listă de bounding box-uri
        self.boxes_by_image = {}

        self.create_widgets()
        self.bind_events()

    def create_widgets(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Open folder", command=self.open_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="<< Prev", command=self.prev_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Next >>", command=self.next_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=8)

        label_frame = ttk.Frame(toolbar)
        label_frame.pack(side=tk.RIGHT, padx=6)

        ttk.Label(label_frame, text="Label:").pack(side=tk.LEFT)

        self.label_var = tk.StringVar()
        self.label_combo = ttk.Combobox(
            label_frame,
            textvariable=self.label_var,
            values=["sobolan", "cap_sobo"],
            width=18
        )
        self.label_combo.pack(side=tk.LEFT, padx=4)
        self.label_combo.set("sobolan")

        ttk.Button(label_frame, text="Add label", command=self.add_label_dialog).pack(side=tk.LEFT, padx=4)

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # Canvas pentru afișarea imaginii și a bounding box-urilor
        self.canvas = tk.Canvas(main, bg="black", cursor="cross")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = ttk.Frame(main, width=300)
        side.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(side, text="Boxes in current image:").pack(anchor="nw", padx=4, pady=(6, 0))

        self.box_list = tk.Listbox(side)
        self.box_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        ttk.Button(side, text="Delete selected box", command=self.delete_selected_box).pack(
            fill=tk.X, padx=4, pady=2
        )

        self.status = ttk.Label(self.root, text="Open a folder to start", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())

    def open_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return

        self.image_folder = folder
        self.image_files = sorted(
            f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)
        )

        if not self.image_files:
            messagebox.showerror("Error", "No images found in folder")
            return

        self.idx = 0
        self.boxes_by_image.clear()
        self.load_image()

    def load_image(self):
        filename = self.image_files[self.idx]
        path = os.path.join(self.image_folder, filename)

        self.current_image = Image.open(path).convert("RGB")
        orig_w, orig_h = self.current_image.size

        max_w, max_h = self.display_size
        ratio = min(
            max_w / orig_w if orig_w > max_w else 1.0,
            max_h / orig_h if orig_h > max_h else 1.0
        )

        disp_w = int(orig_w * ratio)
        disp_h = int(orig_h * ratio)

        self.scale_x = orig_w / disp_w
        self.scale_y = orig_h / disp_h

        display_img = self.current_image.resize((disp_w, disp_h), Image.LANCZOS)
        self.current_tkimage = ImageTk.PhotoImage(display_img)

        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_tkimage)

        self.current_boxes = self.boxes_by_image.get(filename, [])
        self.redraw_boxes()
        self.update_box_list()
        self.update_status()

    def update_status(self):
        fname = self.image_files[self.idx]
        w, h = self.current_image.size
        count = len(self.current_boxes)
        self.status.config(text=f"[{self.idx+1}/{len(self.image_files)}] {fname} — {w}x{h} — {count} boxes")

    def save_current_boxes(self):
        fname = self.image_files[self.idx]
        self.boxes_by_image[fname] = list(self.current_boxes)

    def prev_image(self):
        if self.idx > 0:
            self.save_current_boxes()
            self.idx -= 1
            self.load_image()

    def next_image(self):
        if self.idx < len(self.image_files) - 1:
            self.save_current_boxes()
            self.idx += 1
            self.load_image()

    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.temp_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y,
            self.start_x, self.start_y,
            outline="red", width=2
        )

    def on_mouse_drag(self, event):
        self.canvas.coords(self.temp_rect, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        x0, y0 = self.start_x, self.start_y
        x1, y1 = event.x, event.y

        if abs(x1 - x0) < 3 or abs(y1 - y0) < 3:
            self.canvas.delete(self.temp_rect)
            return

        label = self.label_var.get().strip()
        if not label:
            label = simpledialog.askstring("Label", "Label name:")
            if not label:
                self.canvas.delete(self.temp_rect)
                return

        disp_x = min(x0, x1)
        disp_y = min(y0, y1)
        disp_w = abs(x1 - x0)
        disp_h = abs(y1 - y0)

        orig_x = int(disp_x * self.scale_x)
        orig_y = int(disp_y * self.scale_y)
        orig_w = int(disp_w * self.scale_x)
        orig_h = int(disp_h * self.scale_y)

        img_w, img_h = self.current_image.size
        orig_x = max(0, min(orig_x, img_w - 1))
        orig_y = max(0, min(orig_y, img_h - 1))
        orig_w = min(orig_w, img_w - orig_x)
        orig_h = min(orig_h, img_h - orig_y)

        self.current_boxes.append({
            "label": label,
            "x": orig_x,
            "y": orig_y,
            "w": orig_w,
            "h": orig_h,
            "img_w": img_w,
            "img_h": img_h
        })

        self.canvas.delete(self.temp_rect)
        self.redraw_boxes()
        self.update_box_list()

    def redraw_boxes(self):
        self.canvas.delete("box")
        for b in self.current_boxes:
            dx = int(b["x"] / self.scale_x)
            dy = int(b["y"] / self.scale_y)
            dw = int(b["w"] / self.scale_x)
            dh = int(b["h"] / self.scale_y)

            self.canvas.create_rectangle(
                dx, dy, dx + dw, dy + dh,
                outline="lime", width=2, tags="box"
            )
            self.canvas.create_text(
                dx + 4, dy + 4,
                anchor="nw", text=b["label"],
                fill="yellow", tags="box"
            )

    def update_box_list(self):
        self.box_list.delete(0, tk.END)
        for i, b in enumerate(self.current_boxes):
            self.box_list.insert(
                tk.END,
                f"{i+1}: {b['label']} ({b['x']},{b['y']}) {b['w']}x{b['h']}"
            )

    def delete_selected_box(self):
        sel = self.box_list.curselection()
        if sel:
            del self.current_boxes[sel[0]]
            self.redraw_boxes()
            self.update_box_list()

    def add_label_dialog(self):
        label = simpledialog.askstring("New label", "Label name:")
        if label:
            values = list(self.label_combo["values"])
            if label not in values:
                values.append(label)
                self.label_combo["values"] = values
            self.label_combo.set(label)

    def export_csv(self):
        self.save_current_boxes()
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return

        fields = [
            "label_name",
            "bbox_x",
            "bbox_y",
            "bbox_width",
            "bbox_height",
            "image_name",
            "image_width",
            "image_height"
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            writer.writeheader()

            for fname, boxes in self.boxes_by_image.items():
                for b in boxes:
                    writer.writerow({
                        "label_name": b["label"],
                        "bbox_x": b["x"],
                        "bbox_y": b["y"],
                        "bbox_width": b["w"],
                        "bbox_height": b["h"],
                        "image_name": fname,
                        "image_width": b["img_w"],
                        "image_height": b["img_h"]
                    })


if __name__ == "__main__":
    root = tk.Tk()
    LabelTool(root)
    root.mainloop()
