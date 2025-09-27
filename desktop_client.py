# desktop_client.py
import base64, io, os, threading, requests
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

API_BASE = "http://127.0.0.1:8000"

class WoundDesktop:
    def __init__(self, root):
        self.root = root
        self.root.title("Wound Whisperer Desktop (API Client)")
        self.root.geometry("900x650")

        self.image_path = tk.StringVar()
        self.status = tk.StringVar(value="Ready")
        self.backend = tk.StringVar(value="unknown")

        self.result_vars = {
            "severity": tk.StringVar(),
            "healing": tk.StringVar(),
            "area_px": tk.StringVar(),
            "pct": tk.StringVar(),
            "perim": tk.StringVar(),
        }
        self.preview_label = None

        self._build_ui()
        self.check_ready()

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=16)
        frm.pack(fill="both", expand=True)

        # Top controls
        hdr = ttk.Label(frm, text="Wound Whisperer Desktop", font=("Arial", 16, "bold"))
        hdr.grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Label(frm, text="Image:").grid(row=1, column=0, sticky="w", pady=(12,6))
        ttk.Entry(frm, textvariable=self.image_path, width=60).grid(row=1, column=1, sticky="we", pady=(12,6), padx=(8,8))
        ttk.Button(frm, text="Browse…", command=self.browse).grid(row=1, column=2, sticky="e", pady=(12,6))

        btn = ttk.Button(frm, text="Analyze", command=self.analyze)
        btn.grid(row=2, column=0, columnspan=3, pady=(6,14))

        # Status + backend
        ttk.Label(frm, text="Status:").grid(row=3, column=0, sticky="w")
        ttk.Label(frm, textvariable=self.status).grid(row=3, column=1, sticky="w")
        ttk.Label(frm, text="Backend:").grid(row=3, column=2, sticky="e")
        ttk.Label(frm, textvariable=self.backend).grid(row=3, column=2, sticky="w", padx=(8,0))

        sep = ttk.Separator(frm)
        sep.grid(row=4, column=0, columnspan=3, sticky="we", pady=10)

        # Results grid
        results = ttk.LabelFrame(frm, text="Results", padding=12)
        results.grid(row=5, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(5, weight=1)
        frm.columnconfigure(1, weight=1)

        r = 0
        for label, key in [
            ("Severity", "severity"),
            ("Healing Potential", "healing"),
            ("Mask Area (px)", "area_px"),
            ("Wound % (128x128)", "pct"),
            ("Perimeter (px)", "perim"),
        ]:
            ttk.Label(results, text=f"{label}:").grid(row=r, column=0, sticky="w", pady=4)
            ttk.Label(results, textvariable=self.result_vars[key]).grid(row=r, column=1, sticky="w", pady=4)
            r += 1

        # Mask preview
        ttk.Label(results, text="Mask Preview:").grid(row=r, column=0, sticky="nw", pady=(8,4))
        self.preview_label = ttk.Label(results)
        self.preview_label.grid(row=r, column=1, sticky="w", pady=(8,4))

        # Stretch
        for c in range(3):
            results.columnconfigure(c, weight=1)

    def browse(self):
        f = filedialog.askopenfilename(
            title="Select wound image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if f:
            self.image_path.set(f)

    def check_ready(self):
        try:
            r = requests.get(f"{API_BASE}/readyz", timeout=5)
            j = r.json()
            if j.get("ready"):
                self.status.set("API ready")
            else:
                self.status.set("API not ready")
            self.backend.set(str(j.get("backend", "unknown")))
        except Exception as e:
            self.status.set(f"API unreachable: {e}")

    def analyze(self):
        path = self.image_path.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please choose a valid image file.")
            return
        self.status.set("Analyzing…")
        # Run in a thread to keep UI responsive
        t = threading.Thread(target=self._do_analyze, args=(path,), daemon=True)
        t.start()

    def _do_analyze(self, path):
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            payload = {"image_b64": b64}
            r = requests.post(f"{API_BASE}/analyze", json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            report_id = data["id"]
            # Fetch report to be consistent with web flow
            rep = requests.get(f"{API_BASE}/report/{report_id}", timeout=10).json()
            self._update_results(data.get("result", {}), rep)
            self.status.set("Done")
        except Exception as e:
            self.status.set(f"Error: {e}")
            messagebox.showerror("Analysis failed", str(e))

    def _update_results(self, result, report):
        # Numbers
        self.result_vars["severity"].set(result.get("severity", ""))
        self.result_vars["healing"].set(result.get("healing_potential", ""))
        self.result_vars["area_px"].set(str(result.get("mask_area_px", "")))
        self.result_vars["pct"].set(str(result.get("wound_percentage", "")))
        self.result_vars["perim"].set(str(result.get("perimeter_px", "")))

        # Mask preview (data URI)
        uri = result.get("mask_uri")
        if uri and uri.startswith("data:image"):
            try:
                b64 = uri.split(",", 1)[1]
                im = Image.open(io.BytesIO(base64.b64decode(b64)))
                im = im.resize((256, 256))
                self._photo = ImageTk.PhotoImage(im)  # keep reference
                self.preview_label.configure(image=self._photo)
            except Exception:
                self.preview_label.configure(text="(preview unavailable)")
        else:
            self.preview_label.configure(text="(no preview)")

def main():
    root = tk.Tk()
    app = WoundDesktop(root)
    root.mainloop()

if __name__ == "__main__":
    main()
