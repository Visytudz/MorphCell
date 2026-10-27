import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_pca_variance(pca_model, save_path=None, figsize=(15, 5), dpi=300):
    """
    Plot PCA explained variance (individual and cumulative).

    Args:
        pca_model: Fitted sklearn PCA model
        save_path: Path to save the figure (optional)
        figsize: Figure size
        dpi: DPI for saving
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Individual explained variance (top 10)
    ax1.bar(range(1, 11), pca_model.explained_variance_ratio_[:10], color="coral")
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax1.set_title(
        f"Individual Variance (Top 10)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3, axis="y")

    # Cumulative explained variance
    cumsum = np.cumsum(pca_model.explained_variance_ratio_)
    ax2.plot(range(1, len(cumsum) + 1), cumsum, "o-", color="coral", linewidth=2)
    ax2.axhline(y=0.9, color="g", linestyle="--", label="90% variance", linewidth=1.5)
    ax2.axhline(
        y=0.95, color="orange", linestyle="--", label="95% variance", linewidth=1.5
    )
    ax2.set_xlabel("Number of Components", fontsize=12)
    ax2.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax2.set_title(
        f"Cumulative Variance",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✓ Variance plot saved to: {save_path}")

    plt.show()


def create_color_palette(n_classes):
    """Create a diverse color palette for visualization"""
    base_colors = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
        "#008080",
        "#e6beff",
        "#9a6324",
        "#fffac8",
        "#800000",
        "#aaffc3",
        "#808000",
        "#ffd8b1",
        "#000075",
        "#808080",
    ]
    return (base_colors * (n_classes // len(base_colors) + 1))[:n_classes]


def plot_2d_scatter(
    features_2d,
    labels,
    label2name,
    ids,
    names,
    comp1_name="Component 1",
    comp2_name="Component 2",
    title="Interactive Visualization",
    save_path=None,
    extra_hover_data=None,
):
    """
    Create an interactive 2D scatter plot using Plotly

    Parameters
    ----------
    features_2d : ndarray, shape (N, 2)
        2D feature coordinates
    labels : ndarray, shape (N,)
        Class labels
    label2name : dict
        Mapping from label to class name
    ids : list
        List of sample IDs
    names : list
        List of sample names
    comp1_name : str
        Name for x-axis
    comp2_name : str
        Name for y-axis
    title : str
        Plot title
    save_path : Path, optional
        Path to save HTML file
    extra_hover_data : dict, optional
        Dictionary of additional data to show on hover.
        Keys are column names, values are lists/arrays of data.
        Example: {"Scale": scales, "Volume": volumes}

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # TODO: remove [34, 426, 123] from ids, features_2d, labels, names
    remove_indices = [i for i, id in enumerate(ids) if id in ["34", "426", "123"]]
    ids = [id for i, id in enumerate(ids) if i not in remove_indices]
    features_2d = features_2d[~np.isin(np.arange(len(features_2d)), remove_indices)]
    labels = labels[~np.isin(np.arange(len(labels)), remove_indices)]
    names = [name for i, name in enumerate(names) if i not in remove_indices]
    extra_hover_data = (
        {
            key: [value for i, value in enumerate(values) if i not in remove_indices]
            for key, values in extra_hover_data.items()
        }
        if extra_hover_data
        else None
    )

    # Create DataFrame for plotting
    df = pd.DataFrame(
        {
            comp1_name: features_2d[:, 0],
            comp2_name: features_2d[:, 1],
            "Class": [label2name.get(l, f"Class {l}") for l in labels],
            "ID": ids,
            "Name": names,
            "Label": labels,
        }
    )

    # Add extra hover data if provided
    hover_cols = ["ID", "Name", "Label"]
    if extra_hover_data is not None:
        for key, values in extra_hover_data.items():
            df[key] = values
            hover_cols.append(key)

    # Create interactive plot
    fig = px.scatter(
        df,
        x=comp1_name,
        y=comp2_name,
        color="Class",
        hover_data=hover_cols,
        title=title,
        width=1000,
        height=700,
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(
        font=dict(size=12), title_font_size=16, legend=dict(font=dict(size=10))
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Interactive plot saved to: {save_path}")

    fig.show()
    return fig


def search_and_highlight(
    features_2d,
    search_id,
    ids,
    labels,
    label2name,
    colors,
    comp1_name="Component 1",
    comp2_name="Component 2",
    method_name="Dimensionality Reduction",
    save_path=None,
    figsize=(12, 9),
    dpi=300,
):
    """
    Search for a sample by ID and create a highlighted plot

    Parameters
    ----------
    features_2d : ndarray, shape (N, 2)
        2D feature coordinates
    search_id : str
        ID to search for
    ids : list
        List of all sample IDs
    labels : ndarray
        Class labels
    label2name : dict
        Mapping from label to class name
    colors : list
        Color palette
    comp1_name : str
        Name for x-axis
    comp2_name : str
        Name for y-axis
    method_name : str
        Name of the dimensionality reduction method
    save_path : Path, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    dpi : int, optional
        DPI for saved figure
    """
    # Find the index of the search ID
    try:
        idx = ids.index(search_id)
    except ValueError:
        print(f"❌ ID '{search_id}' not found in dataset!")
        print(f"Available IDs sample: {ids[:10]}...")
        return

    # Get information about this sample
    cell_label = labels[idx]
    cell_class = label2name.get(cell_label, f"Class {cell_label}")
    cell_x = features_2d[idx, 0]
    cell_y = features_2d[idx, 1]

    print(f"✓ Found cell: {search_id}")
    print(f"  Class: {cell_class}")
    print(f"  {comp1_name}: {cell_x:.4f}")
    print(f"  {comp2_name}: {cell_y:.4f}")

    # Create highlighted plot
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_name = label2name.get(label, f"Class {label}")
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=colors[i],
            label=class_name,
            alpha=0.3,
            s=20,
            edgecolors="none",
        )

    # Highlight the searched cell
    ax.scatter(
        cell_x,
        cell_y,
        c="red",
        s=200,
        marker="*",
        edgecolors="black",
        linewidths=2,
        label=f"Selected: {search_id}",
        zorder=10,
    )

    ax.set_xlabel(comp1_name, fontsize=12)
    ax.set_ylabel(comp2_name, fontsize=12)
    ax.set_title(
        f"{method_name} - Highlighted Cell: {search_id}", fontsize=14, fontweight="bold"
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Highlighted plot saved to: {save_path}")
    plt.show()


def plot_3d_scatter(
    features_3d,
    labels,
    label2name,
    ids,
    names,
    comp1_name="Component 1",
    comp2_name="Component 2",
    comp3_name="Component 3",
    title="Interactive 3D Visualization",
    save_path=None,
    extra_hover_data=None,
):
    """
    Create an interactive 3D scatter plot using Plotly

    Parameters
    ----------
    features_3d : ndarray, shape (N, 3)
        3D feature coordinates
    labels : ndarray, shape (N,)
        Class labels
    label2name : dict
        Mapping from label to class name
    ids : list
        List of sample IDs
    names : list
        List of sample names
    comp1_name : str, optional
        Name for x-axis
    comp2_name : str, optional
        Name for y-axis
    comp3_name : str, optional
        Name for z-axis
    title : str, optional
        Plot title
    save_path : Path, optional
        Path to save HTML file
    extra_hover_data : dict, optional
        Dictionary of additional data to show on hover.
        Keys are column names, values are lists/arrays of data.
        Example: {"Scale": scales, "Volume": volumes}

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Create DataFrame for plotting
    df = pd.DataFrame(
        {
            comp1_name: features_3d[:, 0],
            comp2_name: features_3d[:, 1],
            comp3_name: features_3d[:, 2],
            "Class": [label2name.get(label, f"Class {label}") for label in labels],
            "ID": ids,
            "Name": names,
            "Label": labels,
        }
    )

    # Add extra hover data if provided
    hover_cols = ["ID", "Name", "Label"]
    if extra_hover_data is not None:
        for key, values in extra_hover_data.items():
            df[key] = values
            hover_cols.append(key)

    fig = px.scatter_3d(
        df,
        x=comp1_name,
        y=comp2_name,
        z=comp3_name,
        color="Class",
        hover_data=hover_cols,
        title=title,
        width=1000,
        height=800,
    )

    fig.update_traces(marker=dict(size=3, opacity=0.7))
    fig.update_layout(
        font=dict(size=12),
        title_font_size=16,
        legend=dict(font=dict(size=10)),
        scene=dict(
            xaxis_title=comp1_name, yaxis_title=comp2_name, zaxis_title=comp3_name
        ),
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Interactive 3D plot saved to: {save_path}")

    fig.show()
    return fig


def search_and_highlight_3d(
    features_3d,
    search_id,
    ids,
    labels,
    label2name,
    comp1_name="Component 1",
    comp2_name="Component 2",
    comp3_name="Component 3",
    method_name="3D Dimensionality Reduction",
):
    """
    Search for a sample by ID and create a highlighted 3D interactive plot

    Parameters
    ----------
    features_3d : ndarray, shape (N, 3)
        3D feature coordinates
    search_id : str
        ID to search for
    ids : list
        List of all sample IDs
    labels : ndarray
        Class labels
    label2name : dict
        Mapping from label to class name
    comp1_name : str, optional
        Name for x-axis
    comp2_name : str, optional
        Name for y-axis
    comp3_name : str, optional
        Name for z-axis
    method_name : str, optional
        Name of the dimensionality reduction method

    Returns
    -------
    plotly.graph_objects.Figure or None
        The plotly figure object, or None if ID not found
    """
    try:
        idx = ids.index(search_id)
    except ValueError:
        print(f"❌ ID '{search_id}' not found in dataset!")
        print(f"Available IDs sample: {ids[:10]}...")
        return None

    cell_label = labels[idx]
    cell_class = label2name.get(cell_label, f"Class {cell_label}")

    cell_x = features_3d[idx, 0]
    cell_y = features_3d[idx, 1]
    cell_z = features_3d[idx, 2]

    print(f"✓ Found cell: {search_id}")
    print(f"  Class: {cell_class}")
    print(f"  {comp1_name}: {cell_x:.4f}")
    print(f"  {comp2_name}: {cell_y:.4f}")
    print(f"  {comp3_name}: {cell_z:.4f}")
    print(f"  Index in dataset: {idx}")

    label_names = [label2name.get(label, f"Class {label}") for label in labels]

    # Create 3D figure
    fig = px.scatter_3d(
        x=features_3d[:, 0],
        y=features_3d[:, 1],
        z=features_3d[:, 2],
        color=label_names,
        hover_data={"ID": ids},
        labels={
            "x": comp1_name,
            "y": comp2_name,
            "z": comp3_name,
            "color": "Class",
        },
        title=f"{method_name} - Highlighted: {search_id} ({cell_class})",
        width=1000,
        height=800,
    )

    fig.update_traces(marker=dict(size=3, opacity=0.7))

    # Add highlighted point
    fig.add_scatter3d(
        x=[cell_x],
        y=[cell_y],
        z=[cell_z],
        mode="markers",
        marker=dict(
            size=10, color="red", symbol="diamond", line=dict(color="darkred", width=2)
        ),
        name=f"Target: {search_id}",
        hovertemplate=f"<b>{search_id}</b><br>Class: {cell_class}<extra></extra>",
    )

    fig.update_layout(
        font=dict(size=12),
        title_font_size=16,
        legend=dict(font=dict(size=10)),
        scene=dict(
            xaxis_title=comp1_name, yaxis_title=comp2_name, zaxis_title=comp3_name
        ),
    )

    fig.show()
    return fig


def plot_umap_scale_heatmap(
    features_umap,
    scales,
    lambda_weight,
    output_dir=None,
    figsize=(12, 8),
    dpi=300,
):
    """
    Plot UMAP scatter colored by physical scale.

    Parameters:
    -----------
    features_umap : (N, 2) array
        UMAP coordinates
    scales : (N,) array
        Scale factors
    lambda_weight : float
        Lambda weight parameter used in feature fusion
    output_dir : Path, optional
        Directory to save the figure
    figsize : tuple
        Figure size
    dpi : int
        DPI for saved figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes
        The axes object
    """
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        features_umap[:, 0],
        features_umap[:, 1],
        c=scales,
        cmap="viridis",
        s=50,
        alpha=0.7,
        edgecolors="none",
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Scale Factor", fontsize=12, fontweight="bold")

    ax.set_xlabel("UMAP Dimension 1", fontsize=13)
    ax.set_ylabel("UMAP Dimension 2", fontsize=13)
    ax.set_title(
        f"UMAP Colored by Scale (λ={lambda_weight})",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_dir:
        save_path = output_dir / f"umap_scale_heatmap_lambda{lambda_weight}.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"\n✓ UMAP scale heatmap saved to: {save_path}")

    plt.show()

    return fig, ax


def plot_ply_projections_in_umap(
    features_umap,
    labels,
    label2name,
    colors,
    ids,
    names,
    scales,
    ply_projections,
    lambda_weight=0.0,
    save_path=None,
):
    """
    Plot real dataset points in UMAP space and overlay one or more PLY projections.

    Parameters
    ----------
    features_umap : ndarray, shape (N, 2)
        UMAP coordinates of the real dataset
    labels : ndarray, shape (N,)
        Class labels of the real dataset
    label2name : dict
        Mapping from label index to class name
    colors : list
        Color palette indexed by label
    ids : list
        Sample IDs of the real dataset
    names : list
        Sample names of the real dataset
    scales : ndarray, shape (N,)
        Scale factors of the real dataset
    ply_projections : list of dict
        Each dict must have keys:
          - "name"  : str   – stem name of the PLY file
          - "umap"  : ndarray shape (1, 2) – projected UMAP coordinates
          - "scale" : float – scale factor of this PLY
    lambda_weight : float
        Lambda used in feature fusion (shown in title)
    save_path : Path or str, optional
        If provided, save the interactive plot to this HTML file

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # ── Real dataset points ──────────────────────────────────────────────────
    for label_idx in np.unique(labels):
        mask = labels == label_idx
        label_name = label2name.get(label_idx, f"Class {label_idx}")
        fig.add_trace(
            go.Scatter(
                x=features_umap[mask, 0],
                y=features_umap[mask, 1],
                mode="markers",
                name=label_name,
                marker=dict(size=6, opacity=0.6, color=colors[label_idx]),
                customdata=np.column_stack(
                    (
                        [ids[i] for i in np.where(mask)[0]],
                        [names[i] for i in np.where(mask)[0]],
                        scales[mask],
                    )
                ),
                hovertemplate=(
                    f"<b>{label_name}</b><br>"
                    "ID: %{customdata[0]}<br>"
                    "Name: %{customdata[1]}<br>"
                    "Scale: %{customdata[2]:.4f}<br>"
                    "UMAP1: %{x:.4f}<br>"
                    "UMAP2: %{y:.4f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # ── PLY projections ──────────────────────────────────────────────────────
    marker_symbols = ["star", "diamond", "cross", "triangle-up", "pentagon", "hexagon"]
    marker_colors = ["red", "blue", "green", "purple", "orange", "cyan"]

    for i, proj in enumerate(ply_projections):
        ply_name = proj["name"]
        ply_umap = proj["umap"]  # (1, 2)
        ply_scale = proj["scale"]
        symbol = marker_symbols[i % len(marker_symbols)]
        color = marker_colors[i % len(marker_colors)]

        fig.add_trace(
            go.Scatter(
                x=ply_umap[:, 0],
                y=ply_umap[:, 1],
                mode="markers+text",
                name=f"PLY: {ply_name}",
                marker=dict(
                    size=20,
                    symbol=symbol,
                    color=color,
                    line=dict(width=2, color="white"),
                ),
                text=[ply_name],
                textposition="top center",
                textfont=dict(size=12, color=color, family="Arial Black"),
                customdata=[[ply_name, ply_scale]],
                hovertemplate=(
                    "<b>PLY Point Cloud</b><br>"
                    "File: %{customdata[0]}<br>"
                    "Scale: %{customdata[1]:.4f}<br>"
                    "UMAP1: %{x:.4f}<br>"
                    "UMAP2: %{y:.4f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"PLY Point Cloud(s) Projected in UMAP Space (λ={lambda_weight})",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        width=1000,
        height=800,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    fig.show()

    if save_path:
        fig.write_html(save_path)
        print(f"\n✓ Figure saved to: {save_path}")

    return fig
