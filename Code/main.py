import tkinter as tk
from gui import SignatureApp

if __name__ == "__main__":
    root = tk.Tk()
    app = SignatureApp(root)  # dataset path optional
    root.mainloop()
