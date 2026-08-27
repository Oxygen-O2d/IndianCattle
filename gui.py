import os
import glob
import time
import threading
import shutil
import datetime
import torch
import customtkinter as ctk
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from fpdf import FPDF
from predict import load_models, predict_image
from nim_audit import run_nim_audit, is_online, get_breed_encyclopedia

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# --- ULTRA PREMIUM PALETTE ---
BG_COLOR = "#09090b"
CARD_COLOR = "#18181b"
BORDER_COLOR = "#27272a"
TEXT_MAIN = "#ffffff"
TEXT_SUB = "#a1a1aa"
ACCENT = "#6366f1"
ACCENT_HOVER = "#4f46e5"
WARN_COLOR = "#ef4444"
WARN_HOVER = "#dc2626"
SUCCESS_COLOR = "#10b981"
SUCCESS_HOVER = "#059669"

# ============================================================
# ANIMATION PRIMITIVES
# ============================================================

class SpinnerCanvas(tk.Canvas):
    """Silky 60fps rotating arc spinner drawn natively on Canvas."""
    def __init__(self, parent, size=72, **kwargs):
        bg = kwargs.pop('bg', BG_COLOR)
        super().__init__(parent, width=size, height=size,
                         bg=bg, highlightthickness=0, **kwargs)
        self.size = size
        self._angle = 0
        self._running = False

    def start(self):
        self._running = True
        self._tick()

    def stop(self):
        self._running = False
        self.delete("all")

    def _tick(self):
        if not self._running:
            return
        self.delete("all")
        pad = 10
        s, e = pad, self.size - pad
        # Track ring
        self.create_arc(s, pad, e, e, start=0, extent=359,
                        style="arc", outline=BORDER_COLOR, width=5)
        # Spinning gradient arc
        self.create_arc(s, pad, e, e, start=self._angle, extent=240,
                        style="arc", outline=ACCENT, width=5)
        # Inner glow dot at arc tip
        import math
        tip_rad = math.radians(self._angle)
        cx, cy = (s + e) / 2, (pad + e) / 2
        r = (e - s) / 2
        tx = cx + r * math.cos(tip_rad)
        ty = cy - r * math.sin(tip_rad)
        self.create_oval(tx-4, ty-4, tx+4, ty+4, fill=ACCENT, outline="")
        self._angle = (self._angle + 6) % 360
        self.after(16, self._tick)   # ~60 fps


class ConfidenceBar(tk.Canvas):
    """Animated horizontal fill bar sweeping from 0 -> target_pct."""
    def __init__(self, parent, width=280, height=5, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=CARD_COLOR, highlightthickness=0, **kwargs)
        self._bar_width = width
        self._target_px = 0
        self._current_px = 0
        self._fill_color = SUCCESS_COLOR
        # Background track
        self.create_rectangle(0, 0, width, height, fill=BORDER_COLOR, outline="", tags="track")
        self._bar_rect = self.create_rectangle(0, 0, 0, height, fill=SUCCESS_COLOR, outline="", tags="bar")

    def animate_to(self, pct: float, color: str):
        self._target_px = max(0, min(pct / 100.0, 1.0)) * self._bar_width
        self._fill_color = color
        self._current_px = 0
        self.itemconfig(self._bar_rect, fill=color)
        self._sweep()

    def _sweep(self):
        if self._current_px >= self._target_px:
            return
        self._current_px = min(self._current_px + (self._target_px / 25), self._target_px)
        self.coords(self._bar_rect, 0, 0, self._current_px, 5)
        self.after(16, self._sweep)


class PulsingLabel:
    """Makes a CTkLabel gently pulse between two colors."""
    def __init__(self, label, color_a: str, color_b: str, period_ms=1200):
        self._lbl = label
        self._a = color_a
        self._b = color_b
        self._period = period_ms
        self._step = 0
        self._running = False

    def start(self):
        self._running = True
        self._tick()

    def stop(self, final_color=None):
        self._running = False
        if final_color:
            self._lbl.configure(text_color=final_color)

    def _tick(self):
        if not self._running:
            return
        import math
        t = (math.sin(math.pi * self._step / (self._period / 16)) + 1) / 2
        r = lambda h: int(h[1:3], 16)
        g = lambda h: int(h[3:5], 16)
        b2 = lambda h: int(h[5:7], 16)
        def lerp(c1, c2, frac):
            return "#{:02x}{:02x}{:02x}".format(
                int(r(c1) + (r(c2) - r(c1)) * frac),
                int(g(c1) + (g(c2) - g(c1)) * frac),
                int(b2(c1) + (b2(c2) - b2(c1)) * frac))
        self._lbl.configure(text_color=lerp(self._a, self._b, t))
        self._step += 1
        self._lbl.after(16, self._tick)


class App(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        try: self.TkdndVersion = TkinterDnD._require(self)
        except Exception as e: print("DnD Fail", e)

        self.title("BovineAI Diagnostic OS")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        self.configure(fg_color=BG_COLOR)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_b0 = None
        self.model_cnx = None
        self.class_mapping = None
        self.current_image_path = None
        self.last_results = None
        self.last_heatmap = None
        
        # Grid config: 0 is Sidebar, 1 is Main Content
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self.setup_sidebar()
        self.setup_frames()
        self.after(100, self.init_model)

    def init_model(self):
        try:
            self.model_b0, self.model_cnx, self.class_mapping = load_models('best_model.pth', 'best_convnext.pth', 'class_mapping.json', self.device)
            if self.model_cnx is not None:
                status_txt = "Ready! [EfficientNet + ConvNeXt Ensemble]"
            else:
                status_txt = "Ready! [EfficientNet-B0]"
            self.status_indicator.configure(text=status_txt, text_color=SUCCESS_COLOR)
            # Start pulsing the sidebar status indicator
            self._status_pulse = PulsingLabel(self.status_indicator, SUCCESS_COLOR, "#064e3b")
            self._status_pulse.start()
        except Exception as e:
            self.status_indicator.configure(text=f"Engine Offline: {e}", text_color=WARN_COLOR)

    # ==========================
    # ROUTING & SIDEBAR
    # ==========================
    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color="#000000", border_width=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)
        
        # Logo Area
        logo_frm = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frm.pack(fill="x", pady=40, padx=30)
        ctk.CTkLabel(logo_frm, text="Bovine", font=ctk.CTkFont(family="Helvetica", size=28, weight="bold"), text_color=TEXT_MAIN).pack(side="left")
        ctk.CTkLabel(logo_frm, text="AI", font=ctk.CTkFont(family="Helvetica", size=28, weight="bold"), text_color=ACCENT).pack(side="left")
        
        # Nav Buttons
        self.nav_btns = {}
        nav_items = [("Diagnostic Studio", "page_diagnostic"), 
                     ("Mass Batch Ledger", "page_batch"), 
                     ("Analytics Matrix", "page_analytics")]
                     
        for text, page_name in nav_items:
            btn = ctk.CTkButton(self.sidebar, text=text, font=ctk.CTkFont(size=14, weight="bold"),
                                fg_color="transparent", text_color=TEXT_SUB, hover_color=CARD_COLOR,
                                anchor="w", width=220, height=45, corner_radius=8,
                                command=lambda n=page_name: self.select_page(n))
            btn.pack(pady=5, padx=20)
            self.nav_btns[page_name] = btn

        # Status
        self.status_indicator = ctk.CTkLabel(self.sidebar, text="Initializing OS...", text_color=ACCENT, font=ctk.CTkFont(size=12))
        self.status_indicator.pack(side="bottom", pady=30, padx=20, anchor="w")

    def select_page(self, page_name):
        # Update Nav colors
        for name, btn in self.nav_btns.items():
            if name == page_name:
                btn.configure(fg_color=CARD_COLOR, text_color=TEXT_MAIN)
            else:
                btn.configure(fg_color="transparent", text_color=TEXT_SUB)
                
        # Show mapping
        for name, frame in self.pages.items():
            if name == page_name: frame.grid(row=0, column=1, sticky="nsew", padx=30, pady=30)
            else: frame.grid_forget()

    def setup_frames(self):
        self.pages = {}
        self.pages["page_diagnostic"] = self.build_diagnostic_page()
        self.pages["page_batch"] = self.build_batch_page()
        self.pages["page_analytics"] = self.build_analytics_page()
        self.select_page("page_diagnostic")

    # ==========================
    # PAGE 1: DIAGNOSTIC STUDIO
    # ==========================
    def build_diagnostic_page(self):
        page = ctk.CTkFrame(self, fg_color="transparent")
        page.grid_rowconfigure(1, weight=1)   # image cards get all the middle space
        page.grid_rowconfigure(2, weight=1)   # scroll zone below
        page.grid_columnconfigure(0, weight=1)
        page.grid_columnconfigure(1, weight=1)

        # ---- HEADER ----
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        ctk.CTkLabel(header, text="Diagnostic Studio", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_MAIN).pack(side="left")
        self.btn_pdf = ctk.CTkButton(header, text="Export PDF Report", fg_color="transparent", border_width=1, border_color=BORDER_COLOR, hover_color=CARD_COLOR, text_color=TEXT_MAIN, state="disabled", command=self.export_pdf)
        self.btn_pdf.pack(side="right", padx=10)
        self.btn_encyclopedia = ctk.CTkButton(header, text="📖 Breed Encyclopedia", fg_color="transparent", border_width=1, border_color="#3730a3", hover_color=CARD_COLOR, text_color=ACCENT, state="disabled", command=self.trigger_encyclopedia)
        self.btn_encyclopedia.pack(side="right", padx=10)
        self.btn_analyze = ctk.CTkButton(header, text="Neural Inference", fg_color=ACCENT, hover_color=ACCENT_HOVER, state="disabled", command=self.start_analysis_thread)
        self.btn_analyze.pack(side="right", padx=10)
        ctk.CTkButton(header, text="Select Media", fg_color=CARD_COLOR, border_width=1, border_color=BORDER_COLOR, hover_color=BORDER_COLOR, command=self.load_image_event).pack(side="right", padx=10)

        # ---- IMAGE CARDS (fixed, not scrolled) ----
        def create_card(parent, title):
            c = ctk.CTkFrame(parent, fg_color=CARD_COLOR, border_width=1, border_color=BORDER_COLOR, corner_radius=16)
            c.grid_rowconfigure(1, weight=1)
            c.grid_columnconfigure(0, weight=1)
            hdr = ctk.CTkFrame(c, fg_color="transparent")
            hdr.grid(row=0, column=0, sticky="ew", padx=20, pady=15)
            ctk.CTkLabel(hdr, text=title, font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_SUB).pack(side="left")
            lbl = ctk.CTkLabel(c, text="Awaiting Source", text_color=BORDER_COLOR, font=ctk.CTkFont(size=16))
            lbl.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
            return c, lbl

        self.original_card, self.img_lbl = create_card(page, "Original Capture")
        self.original_card.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        try:
            self.original_card.drop_target_register(DND_FILES)
            self.original_card.dnd_bind('<<Drop>>', self.drop_image_event)
        except Exception: pass

        self.cam_card, self.cam_lbl = create_card(page, "Grad-CAM Matrix")
        self.cam_card.grid(row=1, column=1, sticky="nsew", padx=(10, 0))

        # ---- SCROLLABLE RESULTS ZONE ----
        self.diag_scroll = ctk.CTkScrollableFrame(page, fg_color="transparent", corner_radius=0)
        self.diag_scroll.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(20, 0))
        self.diag_scroll.grid_columnconfigure(0, weight=1)

        # Canvas Spinner (replaces flat progress bar)
        self.spinner = SpinnerCanvas(self.diag_scroll, size=72, bg=BG_COLOR)

        # OOD alert
        self.ood_alert = ctk.CTkLabel(self.diag_scroll, text="⚠️ OUT OF DISTRIBUTION", text_color=WARN_COLOR, font=ctk.CTkFont(size=14, weight="bold"))

        # Rank cards row — nested frame inside scroll
        self.res_dash = ctk.CTkFrame(self.diag_scroll, fg_color="transparent")
        self.res_dash.grid_columnconfigure((0, 1, 2), weight=1)

        self.result_cards = []
        for i in range(3):
            c = ctk.CTkFrame(self.res_dash, fg_color=CARD_COLOR, border_width=1, border_color=BORDER_COLOR, corner_radius=12)
            ctk.CTkLabel(c, text=f"RANK {i+1}", font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_SUB).pack(anchor="w", padx=20, pady=(15, 0))
            lbl_b = ctk.CTkLabel(c, text="--", font=ctk.CTkFont(size=20, weight="bold"), text_color=TEXT_MAIN)
            lbl_b.pack(anchor="w", padx=20)
            lbl_p = ctk.CTkLabel(c, text="0.0%", font=ctk.CTkFont(size=12), text_color=TEXT_SUB)
            lbl_p.pack(anchor="w", padx=20, pady=(0, 6))
            conf_bar = ConfidenceBar(c, width=220, height=5)
            conf_bar.pack(anchor="w", padx=20, pady=(0, 14))
            self.result_cards.append({"frame": c, "breed": lbl_b, "prob": lbl_p, "bar": conf_bar})

        # Feedback bar
        self.feedback_frame = ctk.CTkFrame(self.diag_scroll, fg_color=CARD_COLOR, border_width=1, border_color=BORDER_COLOR, corner_radius=12)
        ctk.CTkLabel(self.feedback_frame, text="Did the engine perform well?", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=20, pady=15)
        ctk.CTkButton(self.feedback_frame, text="Thumbs Up", fg_color="transparent", border_width=1, border_color=SUCCESS_COLOR, text_color=SUCCESS_COLOR, hover_color="#064e3b", command=lambda: self.feedback_frame.pack_forget()).pack(side="right", padx=(5, 20))
        ctk.CTkButton(self.feedback_frame, text="Thumbs Down", fg_color="transparent", border_width=1, border_color=WARN_COLOR, text_color=WARN_COLOR, hover_color="#7f1d1d", command=self.feedback_incorrect).pack(side="right", padx=5)

        # ---- NIM AI AUDIT PANEL ----
        self.nim_panel = ctk.CTkFrame(self.diag_scroll, fg_color=CARD_COLOR, border_width=1, border_color="#3730a3", corner_radius=12)
        nim_header = ctk.CTkFrame(self.nim_panel, fg_color="transparent")
        nim_header.pack(fill="x", padx=20, pady=(15, 5))
        ctk.CTkLabel(nim_header, text="🧠", font=ctk.CTkFont(size=18)).pack(side="left")
        ctk.CTkLabel(nim_header, text="NVIDIA NIM AI Audit", font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_MAIN).pack(side="left", padx=8)
        self.nim_status_dot = ctk.CTkLabel(nim_header, text="● Connecting...", font=ctk.CTkFont(size=11), text_color=ACCENT)
        self.nim_status_dot.pack(side="right")
        self.nim_verdict_lbl = ctk.CTkLabel(self.nim_panel, text="--", font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_MAIN)
        self.nim_verdict_lbl.pack(anchor="w", padx=20)
        self.nim_reason_lbl = ctk.CTkLabel(self.nim_panel, text="Awaiting neural inference...", font=ctk.CTkFont(size=13), text_color=TEXT_SUB, wraplength=700, justify="left")
        self.nim_reason_lbl.pack(anchor="w", padx=20, pady=(2, 5))
        self.nim_alert_lbl = ctk.CTkLabel(self.nim_panel, text="", font=ctk.CTkFont(size=12, weight="bold"), text_color=WARN_COLOR)
        self.nim_alert_lbl.pack(anchor="w", padx=20, pady=(0, 15))

        # ---- BREED ENCYCLOPEDIA PANEL ----
        self.enc_panel = ctk.CTkFrame(self.diag_scroll, fg_color=CARD_COLOR, border_width=1, border_color="#065f46", corner_radius=12)
        enc_header_row = ctk.CTkFrame(self.enc_panel, fg_color="transparent")
        enc_header_row.pack(fill="x", padx=20, pady=(15, 5))
        ctk.CTkLabel(enc_header_row, text="📖", font=ctk.CTkFont(size=18)).pack(side="left")
        ctk.CTkLabel(enc_header_row, text="Breed Encyclopedia", font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_MAIN).pack(side="left", padx=8)
        self.enc_loading_lbl = ctk.CTkLabel(enc_header_row, text="● Querying NIM...", font=ctk.CTkFont(size=11), text_color=SUCCESS_COLOR)
        self.enc_loading_lbl.pack(side="right")
        enc_grid = ctk.CTkFrame(self.enc_panel, fg_color="transparent")
        enc_grid.pack(fill="x", padx=20, pady=(5, 15))
        enc_grid.grid_columnconfigure(1, weight=1)
        self.enc_fields = {}
        for row_idx, (key, label) in enumerate([
            ("origin", "🌍 Origin"), ("milk_yield", "🥛 Milk Yield"),
            ("heat_tolerance", "🌡 Heat Tolerance"), ("economic_value", "💰 Economic Value"),
            ("physical_traits", "🐄 Physical Traits"), ("fun_fact", "⭐ Fun Fact"),
        ]):
            ctk.CTkLabel(enc_grid, text=label, font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_SUB, anchor="w").grid(row=row_idx, column=0, sticky="nw", pady=5, padx=(0, 15))
            val_lbl = ctk.CTkLabel(enc_grid, text="--", font=ctk.CTkFont(size=12), text_color=TEXT_MAIN, anchor="w", wraplength=600, justify="left")
            val_lbl.grid(row=row_idx, column=1, sticky="ew", pady=5)
            self.enc_fields[key] = val_lbl

        return page

    # ==========================
    # PAGE 2: MASS BATCH LEDGER
    # ==========================
    def build_batch_page(self):
        page = ctk.CTkFrame(self, fg_color="transparent")
        page.grid_rowconfigure(1, weight=1)
        page.grid_columnconfigure(0, weight=1)
        
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        ctk.CTkLabel(header, text="Mass Batch processing", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_MAIN).pack(side="left")
        
        self.btn_batch_pdf = ctk.CTkButton(header, text="Export PDF", fg_color="transparent", border_width=1, border_color=BORDER_COLOR, text_color=TEXT_MAIN, hover_color=CARD_COLOR, state="disabled", command=self.export_batch_pdf)
        self.btn_batch_pdf.pack(side="right", padx=10)
        ctk.CTkButton(header, text="Select Directory", fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.select_batch_dir).pack(side="right", padx=10)
        
        self.lbl_batch_status = ctk.CTkLabel(header, text="Ready", text_color=TEXT_SUB, font=ctk.CTkFont(size=13))
        self.lbl_batch_status.pack(side="right", padx=20)
        
        self.batch_grid = ctk.CTkScrollableFrame(page, fg_color="transparent")
        self.batch_grid.grid(row=1, column=0, sticky="nsew")
        self.batch_ledger = []
        return page

    # ==========================
    # PAGE 3: ANALYTICS MATRIX
    # ==========================
    def build_analytics_page(self):
        page = ctk.CTkFrame(self, fg_color="transparent")
        page.grid_rowconfigure(1, weight=1)
        page.grid_columnconfigure(0, weight=1)
        
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        ctk.CTkLabel(header, text="Heatmap Analytics", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_MAIN).pack(side="left")
        
        self.zoom_level = 1.0
        self.cm_original_img = None
        
        if os.path.exists('confusion_matrix.png'):
            ctk.CTkButton(header, text="Zoom In", width=80, fg_color=CARD_COLOR, border_width=1, border_color=BORDER_COLOR, hover_color=BORDER_COLOR, command=self.zoom_in_cm).pack(side="right", padx=5)
            self.lbl_zoom = ctk.CTkLabel(header, text="100%", width=40, font=ctk.CTkFont(weight="bold"))
            self.lbl_zoom.pack(side="right", padx=10)
            ctk.CTkButton(header, text="Zoom Out", width=80, fg_color=CARD_COLOR, border_width=1, border_color=BORDER_COLOR, hover_color=BORDER_COLOR, command=self.zoom_out_cm).pack(side="right", padx=5)
            
        self.cm_scroll = ctk.CTkScrollableFrame(page, fg_color=CARD_COLOR, border_width=1, border_color=BORDER_COLOR, corner_radius=16)
        self.cm_scroll.grid(row=1, column=0, sticky="nsew")
        self.cm_lbl = ctk.CTkLabel(self.cm_scroll, text="No model diagnostics found.", text_color=TEXT_SUB)
        self.cm_lbl.pack(expand=True, fill="both", pady=20, padx=20)
        
        if os.path.exists('confusion_matrix.png'):
            try:
                self.cm_original_img = Image.open('confusion_matrix.png').convert("RGB")
                self.update_cm_zoom()
            except Exception as e: print(e)
            
        return page

    # ==========================
    # LOGIC ALGORITHMS
    # ==========================
    def process_selected_image(self, filepath):
        self.current_image_path = filepath
        self.display_pil_image(Image.open(filepath), self.img_lbl)
        self.cam_lbl.configure(image=None, text="Awaiting Neural Pass")
        self.feedback_frame.pack_forget()
        self.ood_alert.pack_forget()
        self.nim_panel.pack_forget()
        self.enc_panel.pack_forget()
        self.res_dash.pack_forget()
        self.btn_analyze.configure(state="normal")
        self.btn_pdf.configure(state="disabled")
        self.btn_encyclopedia.configure(state="disabled")
        for card in self.result_cards: card["frame"].grid_forget()
            
    def display_pil_image(self, img, target_lbl):
        w, h = img.size
        # Scaled to maintain aspect ratio dynamically
        target_w = 400
        target_h = int(target_w * (h / w))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(target_w, target_h))
        target_lbl.configure(image=ctk_img, text="")
        target_lbl.image = ctk_img

    def start_analysis_thread(self):
        self.btn_analyze.configure(state="disabled", text="Processing...")
        self.ood_alert.pack_forget()
        self.feedback_frame.pack_forget()
        self.enc_panel.pack_forget()
        self.nim_panel.pack_forget()
        self.res_dash.pack_forget()
        for card in self.result_cards: card["frame"].grid_forget()
        self.spinner.pack(pady=20)
        self.spinner.start()
        threading.Thread(target=self.run_prediction).start()

    def load_image_event(self):
        filepath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.webp")])
        if filepath: self.process_selected_image(filepath)
        
    def drop_image_event(self, event):
        filepath = event.data
        if filepath.startswith('{') and filepath.endswith('}'): filepath = filepath[1:-1]
        self.process_selected_image(filepath)

    def run_prediction(self):
        time.sleep(0.5)
        res, heatmap, ood = predict_image(self.current_image_path, self.model_b0, self.model_cnx, self.class_mapping, self.device)
        self.after(0, lambda: self.update_ui(res, heatmap, ood))

    def update_ui(self, results, heatmap, ood):
        self.spinner.stop()
        self.spinner.pack_forget()
        self.btn_analyze.configure(state="normal", text="Run Neural Inference")
        self.btn_pdf.configure(state="normal")
        self.btn_encyclopedia.configure(state="normal")
        self.last_results, self.last_heatmap = results, heatmap
        
        if ood: self.ood_alert.pack(pady=(0, 10))
        if heatmap: self.display_pil_image(heatmap, self.cam_lbl)
        if results:
            self.last_prediction_breed = results[0]['breed']
            self.res_dash.pack(fill="x", pady=(10, 0), padx=5)
            for i, res in enumerate(results[:3]):
                card = self.result_cards[i]
                prob = res['probability']
                card['breed'].configure(text=res['breed'].replace("_", " "))
                bar_color = SUCCESS_COLOR if (i == 0 and prob > 80) else (WARN_COLOR if i == 0 else TEXT_SUB)
                card["frame"].configure(border_color=bar_color if i == 0 else BORDER_COLOR)
                card['prob'].configure(text=f"{prob:.2f}% Match")
                card["frame"].grid(row=0, column=i, sticky="nsew", padx=10)
                # Animate confidence bar after slight stagger
                self.after(i * 100, lambda c=card, p=prob, clr=bar_color: c['bar'].animate_to(p, clr))
            self.feedback_frame.pack(fill="x", pady=10, padx=5)
            # Fire NIM audit asynchronously
            self.nim_panel.pack(fill="x", pady=(5, 10), padx=5)
            self.nim_verdict_lbl.configure(text="Consulting NIM...", text_color=ACCENT)
            self.nim_reason_lbl.configure(text="")
            self.nim_alert_lbl.configure(text="")
            self.nim_status_dot.configure(text="● Querying...", text_color=ACCENT)
            threading.Thread(target=self.run_nim_thread, args=(results,), daemon=True).start()

    def run_nim_thread(self, results):
        audit = run_nim_audit(results)
        self.after(0, lambda: self.render_nim_audit(audit))

    def render_nim_audit(self, audit):
        verdict = audit.get('verdict', 'UNKNOWN')
        reason  = audit.get('reason', '')
        alert   = audit.get('alert', 'NO')

        verdict_colors = {
            'TRUSTED': SUCCESS_COLOR, 'REVIEW': '#f59e0b',
            'REJECT': WARN_COLOR, 'OFFLINE': TEXT_SUB,
            'ERROR': WARN_COLOR, 'NO KEY': TEXT_SUB, 'UNKNOWN': TEXT_SUB
        }
        status_texts = {
            'TRUSTED': '\u25cf Online', 'REVIEW': '\u25cf Online', 'REJECT': '\u25cf Online',
            'OFFLINE': '\u25cf Offline', 'ERROR': '\u25cf Error', 'NO KEY': '\u25cf No Key', 'UNKNOWN': '\u25cf Unknown'
        }
        color = verdict_colors.get(verdict, TEXT_SUB)
        self.nim_verdict_lbl.configure(text=f"Verdict: {verdict}", text_color=color)
        self.nim_status_dot.configure(text=status_texts.get(verdict, '\u25cf Done'), text_color=color)
        if 'YES' in str(alert).upper():
            self.nim_alert_lbl.configure(text=f"\u26a0\ufe0f  {alert}")
        else:
            self.nim_alert_lbl.configure(text="")
        # Typewriter reveal for the reason text
        self._typewrite(reason, 0)

    def _typewrite(self, full_text: str, idx: int):
        """Reveal text one character at a time for a premium typewriter effect."""
        self.nim_reason_lbl.configure(text=full_text[:idx])
        if idx < len(full_text):
            self.after(18, self._typewrite, full_text, idx + 1)

    def trigger_encyclopedia(self):
        """Show the panel immediately in loading state, then fire async NIM query."""
        self.enc_panel.pack(fill="x", pady=(5, 20), padx=5)
        self.enc_loading_lbl.configure(text="● Querying NIM...", text_color=ACCENT)
        for lbl in self.enc_fields.values():
            lbl.configure(text="Loading...")
        breed = self.last_prediction_breed if hasattr(self, 'last_prediction_breed') else "Unknown"
        threading.Thread(target=self.run_encyclopedia_thread, args=(breed,), daemon=True).start()

    def run_encyclopedia_thread(self, breed):
        data = get_breed_encyclopedia(breed)
        self.after(0, lambda: self.render_encyclopedia(data))

    def render_encyclopedia(self, data):
        if data.get('error'):
            self.enc_loading_lbl.configure(text="● Error", text_color=WARN_COLOR)
            if self.enc_fields.get('origin'):
                self.enc_fields['origin'].configure(text=data['error'])
            return
        self.enc_loading_lbl.configure(text="● Ready", text_color=SUCCESS_COLOR)
        for key in ['origin', 'milk_yield', 'heat_tolerance', 'economic_value', 'physical_traits', 'fun_fact']:
            val = data.get(key, '--')
            if self.enc_fields.get(key):
                self.enc_fields[key].configure(text=val if val else '--')
            
    def feedback_incorrect(self):
        if self.current_image_path and self.last_prediction_breed:
            out_dir = os.path.join("hard_negatives", self.last_prediction_breed)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy(self.current_image_path, os.path.join(out_dir, f"flagged_{int(time.time())}.jpg"))
            messagebox.showinfo("Active Learning", f"Anomaly Flagged\nSaved to {out_dir}")
        self.feedback_frame.grid_forget()

    # BATCH ENGINE
    def select_batch_dir(self):
        folder = filedialog.askdirectory()
        if not folder: return
        files = []
        for e in ["*.jpg", "*.jpeg", "*.png"]: files.extend(glob.glob(os.path.join(folder, e)))
        if not files: return
        for widget in self.batch_grid.winfo_children(): widget.destroy()
        self.lbl_batch_status.configure(text=f"Engaging {len(files)} items...", text_color=ACCENT)
        self.batch_ledger = []
        threading.Thread(target=self.run_batch, args=(files,)).start()

    def run_batch(self, files):
        row, col = 0, 0
        for i, filepath in enumerate(files):
            res, _, _ = predict_image(filepath, self.model_b0, self.model_cnx, self.class_mapping, self.device, enable_cam=False)
            if res:
                self.batch_ledger.append({"file": os.path.basename(filepath), "breed": res[0]['breed'].replace('_', ' '), "prob": res[0]['probability']})
                self.after(0, self.add_batch_item, filepath, res[0], row, col)
            col += 1
            if col > 3: col, row = 0, row + 1
            self.after(0, lambda idx=i: self.lbl_batch_status.configure(text=f"Processed: {idx+1}/{len(files)}"))
        self.after(0, lambda: self.lbl_batch_status.configure(text="Pipeline Complete", text_color=SUCCESS_COLOR))
        self.after(0, lambda: self.btn_batch_pdf.configure(state="normal"))

    def add_batch_item(self, filepath, top_res, row, col):
        card = ctk.CTkFrame(self.batch_grid, width=220, height=270, fg_color=CARD_COLOR, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        card.grid(row=row, column=col, padx=15, pady=15)
        try:
            img = Image.open(filepath).convert('RGB')
            w, h = img.size
            if w > h: img = img.crop(((w-h)//2, 0, (w+h)//2, h))
            else: img = img.crop((0, (h-w)//2, w, (h+w)//2))
            ctk_img = ctk.CTkImage(light_image=img, size=(160, 160))
            ctk.CTkLabel(card, image=ctk_img, text="").pack(pady=(15,5))
        except: pass
        color = SUCCESS_COLOR if top_res['probability'] > 80 else WARN_COLOR
        ctk.CTkLabel(card, text=f"{top_res['breed'].replace('_', ' ')}", font=ctk.CTkFont(size=15, weight="bold"), text_color=TEXT_MAIN).pack()
        ctk.CTkLabel(card, text=f"{top_res['probability']:.1f}% Match", font=ctk.CTkFont(size=12), text_color=color).pack()
        ctk.CTkButton(card, text="Inspect Layer", width=120, height=28, fg_color="transparent", hover_color=BORDER_COLOR, border_width=1, command=lambda f=filepath: self.inspect(f)).pack(pady=(10,15))

    def inspect(self, filepath):
        self.select_page("page_diagnostic")
        self.process_selected_image(filepath)
        self.start_analysis_thread()

    # ZOOM ALGO
    def zoom_in_cm(self):
        if self.zoom_level < 5.0: self.zoom_level += 0.25; self.update_cm_zoom()
    def zoom_out_cm(self):
        if self.zoom_level > 0.25: self.zoom_level -= 0.25; self.update_cm_zoom()
    def update_cm_zoom(self):
        if not self.cm_original_img: return
        w, h = self.cm_original_img.size
        # dynamically scale
        current_scale = (650.0 / h) * self.zoom_level
        ctk_img = ctk.CTkImage(light_image=self.cm_original_img, size=(int(w * current_scale), int(h * current_scale)))
        self.cm_lbl.configure(image=ctk_img, text="")
        self.cm_lbl.image = ctk_img
        self.lbl_zoom.configure(text=f"{int(self.zoom_level * 100)}%")

    # PDF ENGINES
    def export_pdf(self):
        if not self.last_results: return
        pdf = FPDF()
        pdf.add_page()
        pdf.set_fill_color(33, 33, 36)
        pdf.rect(0, 0, 210, 40, style="F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", size=26, style="B")
        pdf.set_xy(10, 10)
        pdf.cell(190, 15, text="BovineAI Diagnostic Report", align='C')
        pdf.set_font("Helvetica", size=11, style="I")
        pdf.set_xy(10, 25)
        pdf.cell(190, 10, text=f"Generated on {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}", align='C')
        
        pdf.set_text_color(40, 40, 40)
        pdf.set_xy(15, 45)
        pdf.set_font("Helvetica", size=16, style="B")
        pdf.cell(100, 10, text="Scan Overview", new_x="LMARGIN", new_y="NEXT")
        pdf.set_line_width(0.5); pdf.set_draw_color(200, 200, 200); pdf.line(15, 55, 195, 55)
        pdf.set_y(60)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(180, 8, text=f"Source Identity: {os.path.basename(self.current_image_path)}\nModel Backbone: EfficientNet-B0 + ConvNeXt-Tiny Ensemble\nTaxonomy Domain: 41 Indigenous Indian Breeds")
        
        pdf.set_xy(15, 95)
        pdf.set_font("Helvetica", size=16, style="B")
        pdf.cell(100, 10, text="Predictive Analysis", new_x="LMARGIN", new_y="NEXT")
        pdf.line(15, 105, 195, 105)
        pdf.set_y(110)
        for i, res in enumerate(self.last_results):
            if i == 0: pdf.set_font("Helvetica", size=14, style="B"); pdf.set_text_color(99, 102, 241)
            else: pdf.set_font("Helvetica", size=12); pdf.set_text_color(100, 100, 100)
            pdf.cell(100, 10, text=f"Rank {i+1}: {res['breed'].replace('_', ' ')}", align='L')
            pdf.cell(80, 10, text=f"{res['probability']:.2f}% Confidence", new_x="LMARGIN", new_y="NEXT", align='R')
            
        pdf.set_xy(15, 155)
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("Helvetica", size=16, style="B")
        pdf.cell(100, 10, text="Visual Diagnostics", new_x="LMARGIN", new_y="NEXT")
        pdf.line(15, 165, 195, 165)
        pdf.set_y(170)
        pdf.set_font("Helvetica", size=12, style="B")
        pdf.cell(90, 10, text="Original Capture", align='C')
        pdf.cell(90, 10, text="Grad-CAM Anomalies", new_x="LMARGIN", new_y="NEXT", align='C')
        
        try:
            if self.last_heatmap: self.last_heatmap.save("temp.jpg")
            pdf.set_draw_color(90, 90, 90)
            pdf.rect(19, 179, 82, 82); pdf.rect(109, 179, 82, 82)
            pdf.image(self.current_image_path, x=20, y=180, w=80)
            if self.last_heatmap: pdf.image("temp.jpg", x=110, y=180, w=80)
        except Exception: pass
        if os.path.exists("temp.jpg"): os.remove("temp.jpg")
        
        pdf.set_y(275); pdf.set_font("Helvetica", size=9, style="I"); pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 10, text="BovineAI Core Engine - Generated computationally via computer vision intelligence.", align='C')
        try:
             name = f"BovineAI_{int(time.time())}.pdf"
             pdf.output(name); messagebox.showinfo("Export Successful", f"Saved: {name}")
        except Exception as e: messagebox.showerror("Export Failed", f"{e}")

    def export_batch_pdf(self):
        if not self.batch_ledger: return
        pdf = FPDF()
        pdf.add_page()
        pdf.set_fill_color(33, 33, 36)
        pdf.rect(0, 0, 210, 30, style="F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", size=22, style="B")
        pdf.set_xy(10, 8)
        pdf.cell(190, 15, text="BovineAI Mass Batch Ledger", align='C')
        
        pdf.set_y(40)
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("Helvetica", size=12, style="B")
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(90, 10, text="Source File", border=1, fill=True)
        pdf.cell(70, 10, text="Primary Diagnosis", border=1, fill=True)
        pdf.cell(30, 10, text="Confidence", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", size=10)
        fill = False
        for i in self.batch_ledger:
            pdf.set_fill_color(240, 240, 240) if fill else pdf.set_fill_color(255, 255, 255)
            if i['prob'] < 45.0: pdf.set_text_color(200, 50, 50)
            else: pdf.set_text_color(40, 40, 40)
            pdf.cell(90, 8, text=i['file'][:45], border=1, fill=fill)
            pdf.cell(70, 8, text=i['breed'], border=1, fill=fill)
            pdf.cell(30, 8, text=f"{i['prob']:.1f}%", border=1, fill=fill, new_x="LMARGIN", new_y="NEXT")
            fill = not fill
            
        try:
             name = f"Batch_Ledger_{int(time.time())}.pdf"
             pdf.output(name); messagebox.showinfo("Export Successful", f"Saved: {name}")
        except Exception as e: messagebox.showerror("Export Failed", f"{e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
