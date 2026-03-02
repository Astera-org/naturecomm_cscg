#!/usr/bin/env python3
"""Visualizations for the CSCG+SR blog post, inspired by de Cothi et al. (2022).

This script generates publication-quality figures that:
  1. Show the 10×10 arena and all 25 barrier configurations
  2. Show performance across trials (à la Cothi Figure 2)
  3. Compare our CSCG+SR agents against the blog's critique
  4. Visualize SR matrices (obs-space vs CSCG latent space vs ground truth)

Usage:
  python plot_cothi_blog.py                     # Just visualize arenas (no experiment)
  python plot_cothi_blog.py --run-experiment     # Run experiment + all plots
  python plot_cothi_blog.py --load results.npz   # Load saved results + plot
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cscg_sr_cothi import (
    make_cothi_arena,
    generate_cothi_barrier_configs,
    load_cothi_mazes,
    run_cothi_experiment,
    train_cscg,
    bfs_shortest_path,
    bfs_distances,
    ModelBasedAgent,
    ObsSpaceSRAgent,
    CSCGVIAgent,
    CSCGSRMatrixAgent,
)
from environment import GridEnv


# ════════════════════════════════════════════════════════════════════
#  Figure 1: Arena + Goal Visualization
# ════════════════════════════════════════════════════════════════════

def plot_arena(room, goal, outdir="figures"):
    """Plot the open 10×10 arena with observation coloring and goal marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    os.makedirs(outdir, exist_ok=True)
    H, W = room.shape
    n_obs = int(room.max()) + 1

    # Color map for observations
    obs_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    cmap = ListedColormap(obs_colors[:n_obs])

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw cells colored by observation
    for r in range(H):
        for c in range(W):
            obs = room[r, c]
            color = obs_colors[obs % 10]
            rect = plt.Rectangle((c, H - 1 - r), 1, 1,
                                 facecolor=color, edgecolor='#333333',
                                 linewidth=0.5, alpha=0.7)
            ax.add_patch(rect)
            ax.text(c + 0.5, H - 1 - r + 0.5, str(obs),
                    ha='center', va='center', fontsize=7,
                    fontweight='bold', color='white')

    # Mark goal
    gr, gc = goal // W, goal % W
    goal_rect = plt.Rectangle((gc, H - 1 - gr), 1, 1,
                               facecolor='none', edgecolor='red',
                               linewidth=3)
    ax.add_patch(goal_rect)
    ax.plot(gc + 0.5, H - 1 - gr + 0.5, '*', color='red',
            markersize=20, markeredgecolor='darkred', markeredgewidth=1)
    ax.text(gc + 0.5, H - 1 - gr + 0.15, 'GOAL',
            ha='center', va='bottom', fontsize=7,
            fontweight='bold', color='darkred')

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, W + 1))
    ax.set_yticks(np.arange(0, H + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linewidth=0.3, color='gray', alpha=0.5)
    ax.set_title("10×10 Arena (de Cothi et al. 2022 style)\n"
                 "Colors = observation types (10 types, each appears 10×)\n"
                 f"Goal at state {goal} (row {gr}, col {gc})",
                 fontsize=11, fontweight='bold')

    # Legend for observations
    patches = [mpatches.Patch(facecolor=obs_colors[i], edgecolor='gray',
                               label=f'Obs {i}') for i in range(n_obs)]
    ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1),
              fontsize=7, title='Observations')

    fig.tight_layout()
    outpath = os.path.join(outdir, "cothi_arena.png")
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"  Arena figure: {outpath}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
#  Figure 2: All 25 Barrier Configurations (5×5 Grid)
# ════════════════════════════════════════════════════════════════════

def _draw_grid_with_walls(ax, H, W, walls, goal, title="", show_obs=False,
                          room=None, blocked=None, starts=None):
    """Draw a grid with blocked cells shown as filled dark squares.

    If ``blocked`` is provided, blocked cells are drawn as dark squares.
    Otherwise, walls are drawn as thick red lines between cells (legacy).
    Start positions are shown as numbered blue circles if provided.
    """
    import matplotlib.pyplot as plt

    # Compute blocked cells from walls if not given explicitly
    if blocked is None:
        blocked = set()

    # Draw cells
    for r in range(H):
        for c in range(W):
            s = r * W + c
            if s in blocked:
                color = '#37474f'  # dark blue-grey for blocked
                rect = plt.Rectangle((c, H - 1 - r), 1, 1,
                                     facecolor=color, edgecolor='#263238',
                                     linewidth=0.3)
            else:
                color = '#f5f5f5'
                rect = plt.Rectangle((c, H - 1 - r), 1, 1,
                                     facecolor=color, edgecolor='#cccccc',
                                     linewidth=0.3)
            ax.add_patch(rect)

    # Mark goal
    gr, gc = goal // W, goal % W
    if goal not in blocked:
        goal_rect = plt.Rectangle((gc, H - 1 - gr), 1, 1,
                                   facecolor='#ffcdd2', edgecolor='red',
                                   linewidth=1.5)
        ax.add_patch(goal_rect)
        ax.plot(gc + 0.5, H - 1 - gr + 0.5, '*', color='red',
                markersize=8, markeredgecolor='darkred', markeredgewidth=0.5)

    # Draw start positions if provided
    if starts:
        for trial, start_state in sorted(starts.items()):
            sr_, sc_ = start_state // W, start_state % W
            sy_ = H - 1 - sr_ + 0.5
            sx_ = sc_ + 0.5
            ax.plot(sx_, sy_, 'o', color='#1565C0', markersize=4,
                    markeredgecolor='white', markeredgewidth=0.3, zorder=5)

    # If no blocked cells provided, fall back to wall-edge drawing
    if not blocked:
        drawn = set()
        for s, a, nb in walls:
            wall_key = tuple(sorted([s, nb]))
            if wall_key in drawn:
                continue
            drawn.add(wall_key)

            sr, sc = s // W, s % W
            sy = H - 1 - sr

            if a == 1:  # right
                ax.plot([sc + 1, sc + 1], [sy, sy + 1], '-',
                        color='#d32f2f', linewidth=2.5, solid_capstyle='round')
            elif a == 0:  # left
                ax.plot([sc, sc], [sy, sy + 1], '-',
                        color='#d32f2f', linewidth=2.5, solid_capstyle='round')
            elif a == 3:  # down
                ax.plot([sc, sc + 1], [sy, sy], '-',
                        color='#d32f2f', linewidth=2.5, solid_capstyle='round')
            elif a == 2:  # up
                ax.plot([sc, sc + 1], [sy + 1, sy + 1], '-',
                        color='#d32f2f', linewidth=2.5, solid_capstyle='round')

    # Outer boundary
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], '-k', linewidth=1.5)

    ax.set_xlim(-0.1, W + 0.1)
    ax.set_ylim(-0.1, H + 0.1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8, fontweight='bold')


def plot_all_barrier_configs(outdir="figures"):
    """Plot all 25 barrier configurations in a 5×5 grid.

    Uses the exact configs from ``mazes.mat``.  Blocked cells are drawn
    as dark filled squares, start positions as blue dots, goal as red star.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    room, goal, _ = make_cothi_arena()
    H, W = room.shape

    # Load the real paper configs (walls, starts, blocked)
    cothi_mazes = load_cothi_mazes()

    fig, axes = plt.subplots(5, 5, figsize=(18, 18))
    fig.suptitle(
        "25 Barrier Configurations (exact from de Cothi et al. 2022)\n"
        "Dark squares = blocked cells · Blue dots = trial start positions "
        "· ★ = goal",
        fontsize=14, fontweight='bold', y=0.98)

    for i, ax in enumerate(axes.flat):
        if i >= len(cothi_mazes):
            ax.axis('off')
            continue
        walls, starts, blocked = cothi_mazes[i]
        n_blocked = len(blocked)
        _draw_grid_with_walls(ax, H, W, walls, goal,
                              title=f"Config {i+1} ({n_blocked} blocked)",
                              blocked=blocked, starts=starts)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = os.path.join(outdir, "cothi_25_barrier_configs.png")
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"  25 barrier configs: {outpath}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
#  Figure 3: Blog-style Performance Across Trials
# ════════════════════════════════════════════════════════════════════

def plot_blog_style_results(results: dict, outdir="figures"):
    """Create blog-style performance plots matching Cothi et al. Fig 2.

    The blog shows:
      Panel A: Human performance vs MB/SR agents
      Panel B: Rat performance vs MB/SR agents

    We show:
      Panel A: Performance across trials (all configs pooled) — the key plot
      Panel B: Performance by barrier config
      Panel C: Open arena performance (bar chart)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(outdir, exist_ok=True)

    agent_names = results["agent_names"]
    open_steps = results["open_steps"]
    all_steps = results["all_steps"]
    n_configs = results["n_barrier_configs"]
    n_trials = results["trials_per_config"]
    n_seeds = results["n_seeds"]

    colors = {
        "Model-based (perfect)": "#1976D2",   # strong blue
        "SR (obs-space)": "#FF6F00",           # amber/orange
        "CSCG + VI": "#2E7D32",               # dark green
        "CSCG + SR": "#C2185B",               # dark pink (legacy)
        "CSCG + SR (stale)": "#9E9E9E",       # grey
        "CSCG + SR (explore)": "#C2185B",     # dark pink
    }
    markers = {
        "Model-based (perfect)": "s",
        "SR (obs-space)": "^",
        "CSCG + VI": "o",
        "CSCG + SR": "D",
        "CSCG + SR (stale)": "v",
        "CSCG + SR (explore)": "D",
    }

    # ═══════════════════════════════════════════════════════════
    # Main figure: 2-panel layout matching blog style
    # ═══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel A: Learning curve across trials (KEY PLOT) ──
    ax = axes[0]
    for name in agent_names:
        per_trial = all_steps[name].mean(axis=1)  # (n_seeds, n_trials)
        means = per_trial.mean(axis=0)
        sems = per_trial.std(axis=0) / max(1, np.sqrt(n_seeds))
        x = np.arange(1, n_trials + 1)
        ax.plot(x, means, '-', color=colors[name], label=name,
                lw=2.5, marker=markers[name], ms=7, markeredgecolor='white',
                markeredgewidth=0.8)
        ax.fill_between(x, means - sems, means + sems,
                        color=colors[name], alpha=0.15)

    ax.set_xlabel("Trial number", fontsize=12, fontweight='bold')
    ax.set_ylabel("Steps to goal (avg over configs)", fontsize=12,
                  fontweight='bold')
    ax.set_title("A. Performance Across Trials\n"
                 "(cf. de Cothi et al. 2022, Fig 2)",
                 fontsize=12, fontweight='bold')
    ax.set_xticks(np.arange(1, n_trials + 1))
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    # ── Panel B: Summary bar chart (open vs barrier) ──
    ax = axes[1]
    x_pos = np.arange(len(agent_names))
    width = 0.35

    open_means = []
    open_sems = []
    barrier_means = []
    barrier_sems = []

    for name in agent_names:
        om = open_steps[name].mean()
        os_val = open_steps[name].mean(axis=1).std() / max(1, np.sqrt(n_seeds))
        bm = all_steps[name].mean()
        bs = all_steps[name].mean(axis=(1, 2)).std() / max(1, np.sqrt(n_seeds))
        open_means.append(om)
        open_sems.append(os_val)
        barrier_means.append(bm)
        barrier_sems.append(bs)

    bars1 = ax.bar(x_pos - width/2, open_means, width, yerr=open_sems,
                   color=[colors[n] for n in agent_names],
                   alpha=0.5, capsize=4, edgecolor='white',
                   label='Open arena')
    bars2 = ax.bar(x_pos + width/2, barrier_means, width, yerr=barrier_sems,
                   color=[colors[n] for n in agent_names],
                   alpha=0.9, capsize=4, edgecolor='white',
                   label='With barriers')

    # Add hatching to open arena bars
    for bar in bars1:
        bar.set_hatch('///')

    ax.set_ylabel("Avg steps to goal", fontsize=12, fontweight='bold')
    ax.set_title("B. Open Arena vs Barriers\n"
                 "(comparing adaptation to barriers)",
                 fontsize=12, fontweight='bold')
    short_names = []
    for n in agent_names:
        if "explore" in n:
            short_names.append("CSCG+SR\n(explore)")
        elif "stale" in n:
            short_names.append("CSCG+SR\n(stale)")
        elif "(" in n:
            short_names.append(n.split("(")[0].strip())
        else:
            short_names.append(n)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, rotation=15, fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color='gray', lw=8, alpha=0.5, label='Open arena'),
        Line2D([0], [0], color='gray', lw=8, alpha=0.9, label='With barriers'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

    fig.suptitle(
        "CSCG + SR vs Model-Based vs Obs-Space SR — de Cothi et al. (2022) Setup\n"
        "Blog: blog.dileeplearning.com/p/a-critique-of-successor-representations",
        fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    outpath = os.path.join(outdir, "cothi_blog_performance.png")
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"  Blog-style performance: {outpath}")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # Supplementary: Per-config breakdown (sorted by difficulty)
    # ═══════════════════════════════════════════════════════════
    # Sort configs by model-based optimal steps so x-axis is meaningful
    mb_name = "Model-based (perfect)"
    mb_per_config = all_steps[mb_name].mean(axis=2).mean(axis=0)  # (n_configs,)
    sort_idx = np.argsort(mb_per_config)

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    for name in agent_names:
        per_config = all_steps[name].mean(axis=2)  # (n_seeds, n_configs)
        means = per_config.mean(axis=0)[sort_idx]
        sems = (per_config.std(axis=0) / max(1, np.sqrt(n_seeds)))[sort_idx]
        x = np.arange(1, n_configs + 1)
        ax2.plot(x, means, '-', color=colors[name], label=name,
                 lw=2, marker=markers[name], ms=5, markeredgecolor='white',
                 markeredgewidth=0.5)
        ax2.fill_between(x, means - sems, means + sems,
                         color=colors[name], alpha=0.15)

    ax2.set_xlabel("Barrier configuration (sorted by MB-optimal difficulty →)",
                   fontsize=12, fontweight='bold')
    ax2.set_ylabel("Avg steps to goal (over trials)", fontsize=12,
                   fontweight='bold')
    ax2.set_title("Performance by Barrier Configuration",
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc="upper left")
    ax2.set_xticks(np.arange(1, n_configs + 1))
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)

    fig2.tight_layout()
    outpath2 = os.path.join(outdir, "cothi_per_config.png")
    fig2.savefig(outpath2, dpi=200, bbox_inches='tight')
    print(f"  Per-config breakdown: {outpath2}")
    plt.close(fig2)


# ════════════════════════════════════════════════════════════════════
#  Figure 4: SR Matrix Comparison (obs-space vs ground truth vs CSCG)
# ════════════════════════════════════════════════════════════════════

def plot_sr_matrices_blog(outdir="figures", gamma=0.95):
    """Create SR matrix comparison figure.

    Shows why obs-space SR is lossy — only 10×10 matrix for 100 states.
    CSCG latent SR recovers ~100×100 structure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    room, goal, env_kw = make_cothi_arena()
    H, W = room.shape
    n_states = H * W
    n_obs = int(room.max()) + 1

    # Train CSCG (more clones & exploration for heavy aliasing)
    spo = (n_states + n_obs - 1) // n_obs
    nc = spo + spo  # 20 clones/obs for 10-way aliasing
    chmm, env, state_to_clone = train_cscg(
        room, env_kw, n_clones_per_obs=nc,
        explore_steps=60000, seed=42)
    env.set_goal(goal)
    n_cs = int(chmm.n_clones.sum())

    # Ground truth SR
    T_true = np.zeros((4, n_states, n_states))
    for s in range(n_states):
        for a in range(4):
            T_true[a, s, env.adj[s][a]] += 1.0
    T_true_avg = T_true.mean(axis=0)
    M_true = np.linalg.inv(np.eye(n_states) - gamma * T_true_avg)

    # Obs-space SR
    T_obs = np.zeros((4, n_obs, n_obs))
    counts = np.zeros((4, n_obs, n_obs))
    for s in range(n_states):
        o = env.obs_map[s]
        for a in range(4):
            nb = env.adj[s][a]
            counts[a, o, env.obs_map[nb]] += 1
    for a in range(4):
        rs = counts[a].sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        T_obs[a] = counts[a] / rs
    T_obs_avg = T_obs.mean(axis=0)
    M_obs = np.linalg.inv(np.eye(n_obs) - gamma * T_obs_avg)

    # CSCG latent SR
    T = chmm.T.astype(np.float64) + 1e-8
    T_norm = T / T.sum(axis=2, keepdims=True)
    T_cscg_avg = T_norm.mean(axis=0)
    M_cscg = np.linalg.inv(np.eye(n_cs) - gamma * T_cscg_avg)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(M_obs, aspect='auto', cmap='magma')
    axes[0].set_title(f"SR (observation space)\n{n_obs}×{n_obs}\n"
                      f"Lossy: 10 obs for 100 states",
                      fontsize=10, fontweight='bold', color='#FF6F00')
    axes[0].set_xlabel("Observation")
    axes[0].set_ylabel("Observation")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(M_true, aspect='auto', cmap='magma')
    axes[1].set_title(f"SR (ground-truth states)\n{n_states}×{n_states}\n"
                      f"Requires perfect location info",
                      fontsize=10, fontweight='bold', color='#1976D2')
    axes[1].set_xlabel("State")
    axes[1].set_ylabel("State")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(M_cscg, aspect='auto', cmap='magma')
    axes[2].set_title(f"SR (CSCG latent states)\n{n_cs}×{n_cs}\n"
                      f"CSCG recovers ~unique locations",
                      fontsize=10, fontweight='bold', color='#C2185B')
    axes[2].set_xlabel("Clone")
    axes[2].set_ylabel("Clone")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle(
        "Successor Representation Matrices — Why Obs-Space SR Fails\n"
        "10 observation types for 100 locations → massive aliasing → "
        "SR is meaningless on observations",
        fontsize=11, fontweight='bold', y=1.04)
    fig.tight_layout()
    outpath = os.path.join(outdir, "cothi_sr_matrices.png")
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"  SR matrices: {outpath}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
#  Figure 5: Value Function Comparison (open arena + one barrier config)
# ════════════════════════════════════════════════════════════════════

def plot_value_functions(outdir="figures", gamma=0.95):
    """Visualize value functions for different agents on the 10×10 arena.

    Shows how each agent's value landscape changes with barriers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    room, goal, env_kw = make_cothi_arena()
    H, W = room.shape
    n_states = H * W
    n_obs = int(room.max()) + 1

    # Train CSCG (more clones & exploration for heavy aliasing)
    spo = (n_states + n_obs - 1) // n_obs
    nc = spo + spo  # 20 clones/obs
    chmm, env, state_to_clone = train_cscg(
        room, env_kw, n_clones_per_obs=nc,
        explore_steps=60000, seed=42)
    env.set_goal(goal)
    goal_clone = state_to_clone.get(goal, 0)

    # ── Ground-truth BFS distances ──
    dist_open = bfs_distances(env.adj, goal, n_states)

    # ── Obs-space SR values ──
    obs_sr = ObsSpaceSRAgent(env, goal, gamma=gamma)
    w = np.zeros(n_obs)
    w[env.obs_map[goal]] = 1.0
    V_obs = obs_sr.M_sr @ w   # value per observation

    # Map to grid: for each state, value = V_obs[obs_of_state]
    V_obs_grid = np.array([V_obs[env.obs_map[s]] for s in range(n_states)])

    # ── CSCG SR values ──
    cscg_sr = CSCGSRMatrixAgent(chmm, n_obs, goal_clone, gamma=gamma)

    # Map CSCG values to grid via state_to_clone mapping
    V_cscg_grid = np.zeros(n_states)
    for s in range(n_states):
        clone = state_to_clone.get(s, 0)
        V_cscg_grid[s] = cscg_sr.V_sr[clone]

    # ── Plot: 3 value maps ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # BFS distance (lower = closer = better) → invert for value
    dist_val = dist_open.astype(float).copy()
    dist_val[dist_val == 9999] = np.nan
    max_d = np.nanmax(dist_val)
    dist_norm = (max_d - dist_val) / max_d  # normalize so goal=1
    dist_norm[np.isnan(dist_norm)] = 0

    titles = ["Ground-truth distance\n(BFS, inverted for value)",
              "SR value (obs-space)\n(lossy, aliased)",
              "SR value (CSCG latent)\n(de-aliased)"]
    values = [dist_norm, V_obs_grid, V_cscg_grid]
    cmaps = ['YlOrRd', 'YlOrRd', 'YlOrRd']
    title_colors = ['#1976D2', '#FF6F00', '#C2185B']

    for ax, title, V_grid, cmap, tc in zip(axes, titles, values, cmaps,
                                            title_colors):
        V_2d = V_grid.reshape(H, W)
        # Per-panel normalization so each panel uses its full color range
        vmin, vmax = V_2d.min(), V_2d.max()
        im = ax.imshow(V_2d, cmap=cmap, origin='upper', aspect='equal',
                        vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Mark goal
        gr, gc = goal // W, goal % W
        ax.plot(gc, gr, '*', color='white', markersize=15,
                markeredgecolor='black', markeredgewidth=1)

        ax.set_title(title, fontsize=10, fontweight='bold', color=tc)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Value Functions on 10×10 Arena — Open Field\n"
        "Obs-space SR assigns same value to all states with same obs "
        "(⇒ loses spatial info)",
        fontsize=11, fontweight='bold', y=1.04)
    fig.tight_layout()
    outpath = os.path.join(outdir, "cothi_value_functions.png")
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"  Value functions: {outpath}")
    plt.close(fig)




# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Generate blog-style figures for CSCG+SR experiment")
    ap.add_argument("--run-experiment", action="store_true",
                    help="Run the full experiment (slow) and plot results")
    ap.add_argument("--load", type=str, default=None,
                    help="Load saved results from .npz file")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--n-barrier-configs", type=int, default=25)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--outdir", type=str, default="figures")
    ap.add_argument("--skip-sr-matrices", action="store_true",
                    help="Skip SR matrix computation (faster)")
    args = ap.parse_args()

    outdir = args.outdir
    print("=" * 65)
    print("  CSCG+SR Blog Figures Generator")
    print("=" * 65)

    # ── Always generate: arena, barrier configs, verification, insight ──
    print("\n[1/7] Arena visualization...")
    room, goal, _ = make_cothi_arena()
    plot_arena(room, goal, outdir)

    print("\n[2/7] 25 barrier configurations...")
    plot_all_barrier_configs(outdir)



    # ── Optionally: SR matrices ──
    if not args.skip_sr_matrices:
        print("\n[5/7] SR matrix comparison...")
        plot_sr_matrices_blog(outdir, gamma=args.gamma)

        print("\n[6/7] Value function comparison...")
        plot_value_functions(outdir, gamma=args.gamma)
    else:
        print("\n[5-6/7] Skipping SR matrices (--skip-sr-matrices)")

    # ── Optionally: Run experiment and plot results ──
    if args.run_experiment:
        print(f"\n[7/7] Running full experiment "
              f"({args.n_seeds} seeds × {args.n_barrier_configs} configs "
              f"× {args.trials} trials)...")
        results = run_cothi_experiment(
            "cothi-10x10",
            n_seeds=args.n_seeds,
            n_barrier_configs=args.n_barrier_configs,
            trials_per_config=args.trials,
            gamma=args.gamma,
        )
        plot_blog_style_results(results, outdir)

        # Save results
        save_path = os.path.join(outdir, "cothi_results.npz")
        np.savez(save_path,
                 **{f"open_{k}": v for k, v in results["open_steps"].items()},
                 **{f"barrier_{k}": v for k, v in results["all_steps"].items()},
                 agent_names=results["agent_names"],
                 n_configs=results["n_barrier_configs"],
                 n_trials=results["trials_per_config"],
                 n_seeds=results["n_seeds"])
        print(f"  Results saved: {save_path}")

    elif args.load:
        print(f"\n[7/7] Loading results from {args.load}...")
        data = np.load(args.load, allow_pickle=True)
        agent_names = list(data["agent_names"])
        n_configs = int(data["n_configs"])
        n_trials = int(data["n_trials"])
        n_seeds = int(data["n_seeds"])

        results = {
            "env_name": "cothi-10x10",
            "agent_names": agent_names,
            "open_steps": {n: data[f"open_{n}"] for n in agent_names},
            "all_steps": {n: data[f"barrier_{n}"] for n in agent_names},
            "n_barrier_configs": n_configs,
            "trials_per_config": n_trials,
            "n_seeds": n_seeds,
        }
        plot_blog_style_results(results, outdir)
    else:
        print("\n[7/7] Skipping experiment (use --run-experiment to run)")

    print(f"\n{'=' * 65}")
    print(f"  All figures saved to: {outdir}/")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
