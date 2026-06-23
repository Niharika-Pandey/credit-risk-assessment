import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Color Palette Definitions (Amex-like consulting style: Slate, Teal, Deep Blue, Red-Orange for default)
PRIMARY_COLOR = "#0F2942"   # Deep Navy
SECONDARY_COLOR = "#1A8C8C" # Teal
ACCENT_COLOR = "#D9534F"    # Soft Red/Coral (for defaults/bad status)
LIGHT_COLOR = "#F4F6F8"     # Light Grey
DARK_COLOR = "#2C3E50"      # Slate Grey
SUCCESS_COLOR = "#2ECC71"   # Green (for creditworthy/good status)

def setup_logging():
    """Sets up standard logging configuration for console and log file."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "project.log")),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized.")

def create_directories():
    """Ensures all required project directories exist."""
    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "outputs",
        "logs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logging.info("Required directories verified/created.")

def apply_plot_theme():
    """Applies a consistent, professional plotting theme using matplotlib and seaborn."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#CCCCCC",
        "grid.color": "#EAEAEA",
    })
    logging.info("Visual plotting theme applied.")

def get_palette():
    """Returns standard risk assessment color palette."""
    return {
        "primary": PRIMARY_COLOR,
        "secondary": SECONDARY_COLOR,
        "accent": ACCENT_COLOR,
        "success": SUCCESS_COLOR,
        "light": LIGHT_COLOR,
        "dark": DARK_COLOR,
        "binary": [SUCCESS_COLOR, ACCENT_COLOR] # 0 = Good, 1 = Bad
    }
