import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def set_publication_style():
    """Set matplotlib parameters for publication-quality plots."""
    # Use LaTeX for text rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 12
    })
    
    # Set figure size to common publication widths
    # Nature: 89mm (single) or 183mm (double)
    # Science: 5.5in (single) or 7.2in (double)
    plt.rcParams["figure.figsize"] = (7.2, 5)  # inches
    plt.rcParams["figure.dpi"] = 300
    
    # Set color scheme
    sns.set_palette("colorblind")
    
    # Set other style parameters
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

def format_axis_ticks(ax, x_scientific=False, y_scientific=False):
    """Format axis ticks for publication quality.
    
    Args:
        ax: matplotlib axis object
        x_scientific (bool): Use scientific notation for x-axis
        y_scientific (bool): Use scientific notation for y-axis
    """
    if x_scientific:
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if y_scientific:
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Adjust tick parameters
    ax.tick_params(direction='out', length=6, width=0.8)
    ax.tick_params(which='minor', direction='out', length=4, width=0.6)

def save_publication_figure(fig, filename, tight=True):
    """Save figure in publication-ready formats.
    
    Args:
        fig: matplotlib figure object
        filename (str): Base filename without extension
        tight (bool): Whether to use tight_layout
    """
    if tight:
        plt.tight_layout()
    
    # Save in multiple formats
    fig.savefig(f"{filename}.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches='tight', dpi=300)
    fig.savefig(f"{filename}.svg", bbox_inches='tight', format='svg')