import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import textwrap

class PlayPauseButton:
    def __init__(self, ax, callback):
        self.button = Rectangle((0, 0), 1, 1, facecolor='#f0f0f0', edgecolor='#d0d0d0')
        ax.add_patch(self.button)
        self.callback = callback
        self.ax = ax
        self.playing = False
        self.text = ax.text(0.5, 0.5, '>', ha='center', va='center', fontsize=15, fontweight='bold')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_frame_on(True)
        self.connect()

    def connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidhover = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.cidleave = self.ax.figure.canvas.mpl_connect('axes_leave_event', lambda event: self.on_hover(event))

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.playing = not self.playing
        self.text.set_text('||' if self.playing else '>')
        self.callback(event)
        self.ax.figure.canvas.draw()

    def on_hover(self, event):
        if event.inaxes == self.ax:
            self.button.set_facecolor('#e0e0e0')
        else:
            self.button.set_facecolor('#f0f0f0')
        self.ax.figure.canvas.draw()

class NavigationButton:
    BUTTON_STYLES = {
        'prev': {'symbol': '<<', 'tooltip': 'Previous'},
        'next': {'symbol': '>>', 'tooltip': 'Next'},
        'reset': {'symbol': 'O', 'tooltip': 'Reset'}
    }

    def __init__(self, ax, button_type, callback):
        self.button = Rectangle((0, 0), 1, 1, facecolor='#f0f0f0', edgecolor='#d0d0d0')
        ax.add_patch(self.button)
        self.callback = callback
        self.ax = ax
        style = self.BUTTON_STYLES.get(button_type, {'symbol': '⏺', 'tooltip': 'Button'})
        self.text = ax.text(0.5, 0.5, style['symbol'], ha='center', va='center', fontsize=15, fontweight='bold')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_frame_on(True)
        self.connect()

    def connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidhover = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.cidleave = self.ax.figure.canvas.mpl_connect('axes_leave_event', lambda event: self.on_hover(event))

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.callback(event)
        self.ax.figure.canvas.draw()

    def on_hover(self, event):
        if event.inaxes == self.ax:
            self.button.set_facecolor('#e0e0e0')
        else:
            self.button.set_facecolor('#f0f0f0')
        self.ax.figure.canvas.draw()

def calculate_coverage(probabilities, threshold=0.001):
    """Calculate the percentage of grid points with non-zero probability."""
    total_points = len(probabilities)
    visited_points = sum(1 for prob in probabilities.values() if prob > threshold)
    return (visited_points / total_points) * 100

def animate_quantum_walk(qw, cw, steps, grid_size, interval=500, stop_at_completion=True):
    """Animate both quantum and classical walks side by side."""
    # Create figure with space for both plots, explanation, and coverage graph
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(4, 40, height_ratios=[2, 0.8, 1, 0.2], hspace=0.4)
    
    # Quantum walk plot and colorbar
    ax_quantum = fig.add_subplot(gs[0, :18])
    cax_quantum = fig.add_subplot(gs[0, 18])
    
    # Classical walk plot and colorbar
    ax_classical = fig.add_subplot(gs[0, 21:39])
    cax_classical = fig.add_subplot(gs[0, 39])
    
    # Coverage comparison graph
    ax_coverage = fig.add_subplot(gs[2, 2:38])
    ax_coverage.set_xlabel('Étapes', fontsize=8)
    ax_coverage.set_ylabel('Couverture (%)', fontsize=8)
    ax_coverage.set_title('Comparaison de la Vitesse d\'Exploration', fontsize=9)
    ax_coverage.grid(True, alpha=0.3)
    ax_coverage.tick_params(labelsize=7)
    
    # Add text explanations directly to the figure
    quantum_text = fig.text(0.25, 0.45, "", fontsize=8, fontfamily='DejaVu Sans', ha='center', va='center', 
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5),
                          linespacing=1.2, wrap=True, transform=fig.transFigure)
    classical_text = fig.text(0.75, 0.45, "", fontsize=8, fontfamily='DejaVu Sans', ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5),
                            linespacing=1.2, wrap=True, transform=fig.transFigure)
    
    # Buttons area
    btn_prev_ax = plt.axes([0.35, 0.03, 0.08, 0.04])
    btn_play_ax = plt.axes([0.45, 0.03, 0.08, 0.04])
    btn_next_ax = plt.axes([0.55, 0.03, 0.08, 0.04])
    btn_reset_ax = plt.axes([0.65, 0.03, 0.08, 0.04])
    
    _setup_plot(ax_quantum, grid_size)
    _setup_plot(ax_classical, grid_size)
    
    # Create custom colormap
    colors = [(1, 1, 1), (1, 0, 0)]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom_red", colors, N=n_bins)
    
    # Store distributions and coverage data
    quantum_distributions = []
    classical_distributions = []
    quantum_coverage = []
    classical_coverage = []
    max_prob = 0
    
    # Track completion steps
    quantum_completion_step = None
    classical_completion_step = None
    
    # Calculate distributions and coverage for all steps
    for step in range(steps + 1):
        q_dist = qw.simulate(step)
        c_dist = cw.simulate(step)
        quantum_distributions.append(q_dist)
        classical_distributions.append(c_dist)
        max_prob = max(max_prob, max(max(q_dist.values()), max(c_dist.values())))
        
        # Calculate coverage
        q_cov = calculate_coverage(q_dist)
        c_cov = calculate_coverage(c_dist)
        quantum_coverage.append(q_cov)
        classical_coverage.append(c_cov)
        
        # Track first completion
        if q_cov >= 99.9 and quantum_completion_step is None:
            quantum_completion_step = step
        if c_cov >= 99.9 and classical_completion_step is None:
            classical_completion_step = step
    
    # Create colorbars
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_prob))
    plt.colorbar(sm, cax=cax_quantum).set_label('Probabilité', fontsize=8, fontweight='bold')
    plt.colorbar(sm, cax=cax_classical).set_label('Probabilité', fontsize=8, fontweight='bold')
    
    # Explanations
    quantum_explanation = """Marche Quantique:
La particule quantique explore l'espace en empruntant simultanément tous les chemins possibles.
Ces chemins interfèrent entre eux, créant des motifs d'interférence constructive et destructive.
Cette superposition cohérente permet une exploration quadratiquement plus rapide qu'une marche classique.
Couverture: {:.1f}%"""

    classical_explanation = """Marche Classique:
La particule classique suit un unique chemin aléatoire à chaque étape.
Sans superposition ni interférence, elle doit explorer l'espace séquentiellement.
Cette exploration linéaire est fondamentalement plus lente que la marche quantique.
Couverture: {:.1f}%"""
    
    def update_plot(frame):
        ax_quantum.clear()
        ax_classical.clear()
        ax_coverage.clear()
        
        _setup_plot(ax_quantum, grid_size)
        _setup_plot(ax_classical, grid_size)
        
        # Plot distributions
        _plot_probabilities(ax_quantum, quantum_distributions[frame], max_prob, cmap)
        _plot_probabilities(ax_classical, classical_distributions[frame], max_prob, cmap)
        
        # Check if full coverage is achieved
        quantum_full = quantum_coverage[frame] >= 99.9
        classical_full = classical_coverage[frame] >= 99.9
        
        # Update titles with coverage and full exploration indicator
        quantum_title = f'Marche Quantique - Étape {frame}/{steps}\n{quantum_coverage[frame]:.1f}%'
        if quantum_full:
            quantum_title += '\n✓ Exploration Complète!'
        ax_quantum.set_title(quantum_title, fontsize=9, fontweight='bold', pad=5)
        
        classical_title = f'Marche Classique - Étape {frame}/{steps}\n{classical_coverage[frame]:.1f}%'
        if classical_full:
            classical_title += '\n✓ Exploration Complète!'
        ax_classical.set_title(classical_title, fontsize=9, fontweight='bold', pad=5)
        
        # Update explanations with coverage
        quantum_explanation_text = quantum_explanation.format(quantum_coverage[frame])
        classical_explanation_text = classical_explanation.format(classical_coverage[frame])
        
        quantum_text.set_text(quantum_explanation_text)
        classical_text.set_text(classical_explanation_text)
        
        # Update coverage graph
        ax_coverage.set_xlabel('Étapes', fontsize=8)
        ax_coverage.set_ylabel('Couverture (%)', fontsize=8)
        ax_coverage.set_title('Vitesse d\'Exploration', fontsize=9)
        ax_coverage.tick_params(labelsize=7)
        
        # Plot coverage lines up to completion or current frame
        quantum_data = quantum_coverage[:frame + 1]
        classical_data = classical_coverage[:frame + 1]
        frame_range = range(frame + 1)
        
        if quantum_completion_step is not None:
            if frame > quantum_completion_step:
                # Only plot up to completion step
                quantum_data = quantum_coverage[:quantum_completion_step + 1]
                frame_range_q = range(quantum_completion_step + 1)
                quantum_line = ax_coverage.plot(frame_range_q, quantum_data, 
                                           'r-', label='Quantique', linewidth=2)[0]
            else:
                quantum_line = ax_coverage.plot(frame_range, quantum_data, 
                                           'r-', label='Quantique', linewidth=2)[0]
        else:
            quantum_line = ax_coverage.plot(frame_range, quantum_data, 
                                       'r-', label='Quantique', linewidth=2)[0]
        
        classical_line = ax_coverage.plot(frame_range, classical_data, 
                                        'b-', label='Classique', linewidth=2)[0]
        
        # Add markers and vertical lines for full coverage points
        if quantum_completion_step is not None and frame >= quantum_completion_step:
            # Add star marker
            ax_coverage.plot(quantum_completion_step, quantum_coverage[quantum_completion_step], 
                           'r*', markersize=15, label='Quantique Complet')
            # Add vertical line
            ax_coverage.axvline(x=quantum_completion_step, color='red', 
                              linestyle='--', alpha=0.3, label='_nolegend_')
        
        if classical_completion_step is not None and frame >= classical_completion_step:
            # Add star marker
            ax_coverage.plot(classical_completion_step, classical_coverage[classical_completion_step], 
                           'b*', markersize=15, label='Classique Complet')
            # Add vertical line
            ax_coverage.axvline(x=classical_completion_step, color='blue', 
                              linestyle='--', alpha=0.3, label='_nolegend_')
        
        # Current frame indicator
        ax_coverage.axvline(x=frame, color='gray', linestyle=':', alpha=0.5)
        
        # Stop animation if both walks are complete and stop_at_completion is True
        if stop_at_completion and quantum_completion_step is not None and classical_completion_step is not None:
            if frame > max(quantum_completion_step, classical_completion_step):
                animation_running[0] = False
                ani.pause()
        
        ax_coverage.set_xlim(0, steps)
        ax_coverage.set_ylim(0, 100)
        ax_coverage.legend(fontsize=7, loc='lower right')
        
        # Add grid to coverage graph
        ax_coverage.grid(True, linestyle='--', alpha=0.3)
        
        return ax_quantum, ax_classical, ax_coverage
    
    frame_number = [0]
    animation_running = [True]
    
    def update(frame):
        update_plot(frame)
        return ax_quantum, ax_classical
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(quantum_distributions),
        interval=interval, blit=False, repeat=True
    )
    ani.pause()
    animation_running[0] = False
    
    # Button callbacks
    def on_prev(event):
        animation_running[0] = False
        ani.pause()
        frame_number[0] = (frame_number[0] - 1) % len(quantum_distributions)
        update_plot(frame_number[0])
        plt.draw()
    
    def on_next(event):
        animation_running[0] = False
        ani.pause()
        frame_number[0] = (frame_number[0] + 1) % len(quantum_distributions)
        update_plot(frame_number[0])
        plt.draw()
    
    def on_play_pause(event):
        animation_running[0] = not animation_running[0]
        if animation_running[0]:
            ani.resume()
        else:
            ani.pause()
        plt.draw()
    
    def on_reset(event):
        animation_running[0] = False
        ani.pause()
        frame_number[0] = 0
        update_plot(0)
        plt.draw()
    
    # Create custom buttons
    btn_prev = NavigationButton(btn_prev_ax, 'prev', on_prev)
    btn_play = PlayPauseButton(btn_play_ax, on_play_pause)
    btn_next = NavigationButton(btn_next_ax, 'next', on_next)
    btn_reset = NavigationButton(btn_reset_ax, 'reset', on_reset)
    
    plt.tight_layout()
    # Adjust layout to prevent text overlap
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9)
    plt.show()
    return ani

def _setup_plot(ax, grid_size):
    """Setup the plot axes and grid."""
    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Y', fontsize=8, fontweight='bold')
    ax.set_ylabel('X', fontsize=8, fontweight='bold')
    ax.tick_params(labelsize=7)

def _plot_probabilities(ax, probabilities, max_prob=None, cmap=None):
    """Plot probabilities as circles with text."""
    if max_prob is None:
        max_prob = max(probabilities.values())
    
    if cmap is None:
        colors = [(1, 1, 1), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list("custom_red", colors, N=100)
    
    for (x, y), prob in probabilities.items():
        if prob > 0:
            size = (prob / max_prob) * 800
            color = cmap(prob / max_prob)
            circle = Circle((y, x), np.sqrt(size)/50, 
                          color=color, alpha=0.7)
            ax.add_patch(circle)
            ax.text(y, x, f'{prob:.2f}', ha='center', va='center',
                   fontsize=7, color='black', fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, 
                           edgecolor='none', pad=0.5))