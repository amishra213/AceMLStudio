"""
AceML Studio – Data Visualization
===================================
Generate interactive and static visualizations for data exploration,
model performance analysis, and comparison.
"""

import io
import base64
import logging
import numpy as np
import pandas as pd
import matplotlib  # type: ignore
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger("aceml.visualizer")

# Set seaborn style
sns.set_style("darkgrid")
sns.set_palette("husl")


class DataVisualizer:
    """Generate visualizations for data exploration and model analysis."""

    @staticmethod
    def _fig_to_base64(fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"

    @staticmethod
    def _plotly_template():
        """Return Plotly dark theme template."""
        return 'plotly_dark'

    # ================================================================
    # Data Exploration Visualizations
    # ================================================================

    @staticmethod
    def histogram(df: pd.DataFrame, column: str, bins: int = 30) -> Dict:
        """Generate histogram for a numeric column."""
        logger.debug(f"Generating histogram for column: {column}")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        data = df[column].dropna()
        
        # Matplotlib version for static image
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        ax.hist(data, bins=bins, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_xlabel(column, fontsize=12, color='white')
        ax.set_ylabel('Frequency', fontsize=12, color='white')
        ax.set_title(f'Distribution of {column}', fontsize=14, color='white', pad=20)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, color='white')
        
        img_base64 = DataVisualizer._fig_to_base64(fig)
        
        # Plotly version for interactivity
        plotly_data = {
            'type': 'histogram',
            'x': data.tolist(),
            'nbinsx': bins,
            'marker': {'color': '#3498db'},
            'name': column
        }
        
        plotly_layout = {
            'title': f'Distribution of {column}',
            'xaxis': {'title': column},
            'yaxis': {'title': 'Frequency'},
            'template': DataVisualizer._plotly_template(),
            'height': 400
        }
        
        return {
            'type': 'histogram',
            'column': column,
            'image': img_base64,
            'plotly': {'data': [plotly_data], 'layout': plotly_layout},
            'stats': {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'median': float(data.median())
            }
        }

    @staticmethod
    def box_plot(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict:
        """Generate box plots for numeric columns."""
        logger.debug(f"Generating box plot for columns: {columns}")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()[:10]
        
        columns = [c for c in columns if c in df.columns]
        
        if not columns:
            raise ValueError("No valid numeric columns found")
        
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        
        data_to_plot = [df[col].dropna() for col in columns]
        bp = ax.boxplot(data_to_plot, labels=columns, patch_artist=True)  # type: ignore
        
        for patch in bp['boxes']:  # type: ignore
            patch.set_facecolor('#3498db')  # type: ignore
            patch.set_alpha(0.7)
        
        for whisker in bp['whiskers']:  # type: ignore
            whisker.set_color('white')  # type: ignore
            whisker.set_linewidth(1.5)
        
        for cap in bp['caps']:  # type: ignore
            cap.set_color('white')  # type: ignore
            cap.set_linewidth(2)
        
        for median in bp['medians']:  # type: ignore
            median.set_color('#e74c3c')  # type: ignore
            median.set_linewidth(2)
        
        ax.set_ylabel('Value', fontsize=12, color='white')
        ax.set_title('Box Plot Comparison', fontsize=14, color='white', pad=20)
        ax.tick_params(colors='white')
        plt.xticks(rotation=45, ha='right', color='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='y', color='white')
        
        img_base64 = DataVisualizer._fig_to_base64(fig)
        
        # Plotly version
        plotly_data = []
        for col in columns:
            plotly_data.append({
                'type': 'box',
                'y': df[col].dropna().tolist(),
                'name': col,
                'boxmean': 'sd'
            })
        
        plotly_layout = {
            'title': 'Box Plot Comparison',
            'yaxis': {'title': 'Value'},
            'template': DataVisualizer._plotly_template(),
            'height': 400
        }
        
        return {
            'type': 'box_plot',
            'columns': columns,
            'image': img_base64,
            'plotly': {'data': plotly_data, 'layout': plotly_layout}
        }

    @staticmethod
    def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                    hue_col: Optional[str] = None) -> Dict:
        """Generate scatter plot between two columns."""
        logger.debug(f"Generating scatter plot: {x_col} vs {y_col}")
        
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Columns not found in DataFrame")
        
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        
        if hue_col and hue_col in df.columns:
            # Color by category
            unique_vals = df[hue_col].unique()[:10]  # Limit to 10 categories
            colors = matplotlib.cm.tab10(np.linspace(0, 1, len(unique_vals)))  # type: ignore
            
            for i, val in enumerate(unique_vals):
                mask = df[hue_col] == val
                ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],  # type: ignore
                          label=str(val), alpha=0.6, s=50, color=colors[i])
            ax.legend(title=hue_col, loc='best', facecolor='#2a2a2a', edgecolor='white')
        else:
            ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50, color='#3498db')
        
        ax.set_xlabel(x_col, fontsize=12, color='white')
        ax.set_ylabel(y_col, fontsize=12, color='white')
        ax.set_title(f'{x_col} vs {y_col}', fontsize=14, color='white', pad=20)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, color='white')
        
        img_base64 = DataVisualizer._fig_to_base64(fig)
        
        # Plotly version
        plotly_data = {
            'type': 'scatter',
            'mode': 'markers',
            'x': df[x_col].tolist(),
            'y': df[y_col].tolist(),
            'marker': {'color': '#3498db', 'size': 8, 'opacity': 0.6},
            'name': f'{x_col} vs {y_col}'
        }
        
        if hue_col and hue_col in df.columns:
            plotly_data['marker']['color'] = df[hue_col].tolist()
            plotly_data['marker']['colorscale'] = 'Viridis'
            plotly_data['marker']['showscale'] = True
            plotly_data['marker']['colorbar'] = {'title': hue_col}
        
        plotly_layout = {
            'title': f'{x_col} vs {y_col}',
            'xaxis': {'title': x_col},
            'yaxis': {'title': y_col},
            'template': DataVisualizer._plotly_template(),
            'height': 500
        }
        
        # Calculate correlation
        corr = df[[x_col, y_col]].corr().iloc[0, 1]
        
        return {
            'type': 'scatter',
            'x_column': x_col,
            'y_column': y_col,
            'hue_column': hue_col,
            'image': img_base64,
            'plotly': {'data': [plotly_data], 'layout': plotly_layout},
            'correlation': float(corr) if not np.isnan(float(corr)) else None  # type: ignore
        }

    @staticmethod
    def correlation_heatmap(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict:
        """Generate correlation heatmap for numeric columns."""
        logger.debug("Generating correlation heatmap")
        
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [c for c in columns if c in df.columns and df[c].dtype in [np.number, 'int', 'float']]
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation")
        
        # Limit to 20 columns for readability
        numeric_cols = numeric_cols[:20]
        
        corr_matrix = df[numeric_cols].corr()
        
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='#1a1a1a')
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(numeric_cols)))
        ax.set_yticks(np.arange(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right', color='white')
        ax.set_yticklabels(numeric_cols, color='white')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors='white')
        cbar.set_label('Correlation', color='white')
        
        # Add correlation values
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                corr_val = float(corr_matrix.iloc[i, j])  # type: ignore
                text = ax.text(j, i, f'{corr_val:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Correlation Heatmap', fontsize=14, color='white', pad=20)
        
        img_base64 = DataVisualizer._fig_to_base64(fig)
        
        # Plotly version
        plotly_data = {
            'type': 'heatmap',
            'z': corr_matrix.values.tolist(),
            'x': numeric_cols,
            'y': numeric_cols,
            'colorscale': 'RdBu',
            'zmid': 0,
            'zmin': -1,
            'zmax': 1,
            'text': [[f'{val:.2f}' for val in row] for row in corr_matrix.values],
            'texttemplate': '%{text}',
            'textfont': {'size': 10},
            'colorbar': {'title': 'Correlation'}
        }
        
        plotly_layout = {
            'title': 'Correlation Heatmap',
            'xaxis': {'side': 'bottom'},
            'yaxis': {'side': 'left'},
            'template': DataVisualizer._plotly_template(),
            'height': 600
        }
        
        return {
            'type': 'correlation_heatmap',
            'columns': numeric_cols,
            'image': img_base64,
            'plotly': {'data': [plotly_data], 'layout': plotly_layout},
            'correlation_matrix': corr_matrix.to_dict()
        }

    @staticmethod
    def distribution_comparison(df: pd.DataFrame, column: str, 
                               group_by: Optional[str] = None) -> Dict:
        """Generate distribution comparison across groups."""
        logger.debug(f"Generating distribution comparison for {column} grouped by {group_by}")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        
        if group_by and group_by in df.columns:
            groups = df[group_by].unique()[:5]  # Limit to 5 groups
            for group in groups:
                data = df[df[group_by] == group][column].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=30, alpha=0.5, label=str(group), edgecolor='black')
            ax.legend(title=group_by, facecolor='#2a2a2a', edgecolor='white')
        else:
            ax.hist(df[column].dropna(), bins=30, alpha=0.7, color='#3498db', edgecolor='black')
        
        ax.set_xlabel(column, fontsize=12, color='white')
        ax.set_ylabel('Frequency', fontsize=12, color='white')
        title = f'Distribution of {column}'
        if group_by:
            title += f' by {group_by}'
        ax.set_title(title, fontsize=14, color='white', pad=20)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, color='white')
        
        img_base64 = DataVisualizer._fig_to_base64(fig)
        
        return {
            'type': 'distribution_comparison',
            'column': column,
            'group_by': group_by,
            'image': img_base64
        }

    # ================================================================
    # Model Performance Visualizations
    # ================================================================

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                        labels: Optional[List[str]] = None) -> Dict:
        """Generate confusion matrix visualization."""
        from sklearn.metrics import confusion_matrix as cm
        
        logger.debug("Generating confusion matrix")
        
        conf_matrix = cm(y_true, y_pred)
        
        if labels is None:
            labels = [str(i) for i in range(len(conf_matrix))]
        
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a1a')
        
        im = ax.imshow(conf_matrix, cmap='Blues', aspect='auto')
        
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, color='white')
        ax.set_yticklabels(labels, color='white')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = conf_matrix.max() / 2.
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if conf_matrix[i, j] > thresh else "black"
                ax.text(j, i, format(conf_matrix[i, j], 'd'),
                       ha="center", va="center", color=color, fontsize=12)
        
        ax.set_ylabel('True Label', fontsize=12, color='white')
        ax.set_xlabel('Predicted Label', fontsize=12, color='white')
        ax.set_title('Confusion Matrix', fontsize=14, color='white', pad=20)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors='white')
        
        img_base64 = DataVisualizer._fig_to_base64(fig)
        
        # Plotly version
        plotly_data = {
            'type': 'heatmap',
            'z': conf_matrix.tolist(),
            'x': labels,
            'y': labels,
            'colorscale': 'Blues',
            'text': [[str(val) for val in row] for row in conf_matrix],
            'texttemplate': '%{text}',
            'textfont': {'size': 12},
            'colorbar': {'title': 'Count'}
        }
        
        plotly_layout = {
            'title': 'Confusion Matrix',
            'xaxis': {'title': 'Predicted Label', 'side': 'bottom'},
            'yaxis': {'title': 'True Label', 'side': 'left'},
            'template': DataVisualizer._plotly_template(),
            'height': 500
        }
        
        return {
            'type': 'confusion_matrix',
            'image': img_base64,
            'plotly': {'data': [plotly_data], 'layout': plotly_layout},
            'matrix': conf_matrix.tolist(),
            'labels': labels
        }

    @staticmethod
    def roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                  class_labels: Optional[List[str]] = None) -> Dict:
        """Generate ROC curve visualization."""
        from sklearn.metrics import roc_curve, auc, roc_auc_score
        
        logger.debug("Generating ROC curve")
        
        # Handle binary and multiclass
        n_classes = y_prob.shape[1] if len(y_prob.shape) > 1 else 2
        
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        
        if n_classes == 2:
            # Binary classification
            if len(y_prob.shape) > 1:
                y_score = y_prob[:, 1]
            else:
                y_score = y_prob
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='#3498db', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            # Multiclass - plot for each class
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            # Handle sparse matrices
            if hasattr(y_prob, 'toarray'):
                y_prob = y_prob.toarray()  # type: ignore
            
            for i in range(min(n_classes, 5)):  # Limit to 5 classes
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])  # type: ignore
                roc_auc = auc(fpr, tpr)
                label = class_labels[i] if class_labels else f'Class {i}'
                ax.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5)
        
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('False Positive Rate', fontsize=12, color='white')
        ax.set_ylabel('True Positive Rate', fontsize=12, color='white')
        ax.set_title('ROC Curve', fontsize=14, color='white', pad=20)
        ax.legend(loc="lower right", facecolor='#2a2a2a', edgecolor='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, color='white')
        
        img_base64 = DataVisualizer._fig_to_base64(fig)
        
        return {
            'type': 'roc_curve',
            'image': img_base64,
            'auc_score': float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'))
        }

    @staticmethod
    def feature_importance(feature_names: List[str], importances: np.ndarray, 
                          top_n: int = 20) -> Dict:
        """Generate feature importance visualization."""
        logger.debug("Generating feature importance plot")
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(10, max(6, len(top_features) * 0.3)), facecolor='#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_importances, color='#3498db', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, color='white')
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12, color='white')
        ax.set_title(f'Top {len(top_features)} Feature Importances', fontsize=14, color='white', pad=20)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='x', color='white')
        
        img_base64 = DataVisualizer._fig_to_base64(fig)
        
        # Plotly version
        plotly_data = {
            'type': 'bar',
            'x': top_importances.tolist(),
            'y': top_features,
            'orientation': 'h',
            'marker': {'color': '#3498db'}
        }
        
        plotly_layout = {
            'title': f'Top {len(top_features)} Feature Importances',
            'xaxis': {'title': 'Importance'},
            'yaxis': {'title': '', 'autorange': 'reversed'},
            'template': DataVisualizer._plotly_template(),
            'height': max(400, len(top_features) * 25)
        }
        
        return {
            'type': 'feature_importance',
            'image': img_base64,
            'plotly': {'data': [plotly_data], 'layout': plotly_layout},
            'top_features': top_features,
            'importances': top_importances.tolist()
        }

    @staticmethod
    def model_comparison(model_results: List[Dict]) -> Dict:
        """Generate model comparison visualization."""
        logger.debug(f"Generating model comparison for {len(model_results)} models")
        
        model_names = [r['model_key'] for r in model_results if r.get('status') == 'success']
        train_scores = [r['train_score'] for r in model_results if r.get('status') == 'success']
        val_scores = [r['val_score'] for r in model_results if r.get('status') == 'success']
        
        if not model_names:
            raise ValueError("No successful model results to compare")
        
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax.bar(x - width/2, train_scores, width, label='Training Score', 
               color='#3498db', alpha=0.8)
        ax.bar(x + width/2, val_scores, width, label='Validation Score', 
               color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, color='white')
        ax.set_ylabel('Score', fontsize=12, color='white')
        ax.set_title('Model Comparison', fontsize=14, color='white', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', color='white')
        ax.legend(facecolor='#2a2a2a', edgecolor='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='y', color='white')
        
        img_base64 = DataVisualizer._fig_to_base64(fig)
        
        # Plotly version
        plotly_data = [
            {
                'type': 'bar',
                'x': model_names,
                'y': train_scores,
                'name': 'Training Score',
                'marker': {'color': '#3498db'}
            },
            {
                'type': 'bar',
                'x': model_names,
                'y': val_scores,
                'name': 'Validation Score',
                'marker': {'color': '#e74c3c'}
            }
        ]
        
        plotly_layout = {
            'title': 'Model Comparison',
            'xaxis': {'title': 'Model'},
            'yaxis': {'title': 'Score'},
            'barmode': 'group',
            'template': DataVisualizer._plotly_template(),
            'height': 500
        }
        
        return {
            'type': 'model_comparison',
            'image': img_base64,
            'plotly': {'data': plotly_data, 'layout': plotly_layout},
            'models': model_names,
            'train_scores': train_scores,
            'val_scores': val_scores
        }

    @staticmethod
    def residual_plot(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Generate residual plot for regression models."""
        logger.debug("Generating residual plot")
        
        residuals = y_true - y_pred
        
        # Matplotlib version
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#1a1a1a')
        
        # Residuals vs Predicted
        ax1.set_facecolor('#2a2a2a')
        ax1.scatter(y_pred, residuals, alpha=0.5, color='#3498db', s=30)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Values', fontsize=12, color='white')
        ax1.set_ylabel('Residuals', fontsize=12, color='white')
        ax1.set_title('Residuals vs Predicted', fontsize=14, color='white', pad=20)
        ax1.tick_params(colors='white')
        ax1.spines['bottom'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(True, alpha=0.3, color='white')
        
        # Residuals distribution
        ax2.set_facecolor('#2a2a2a')
        ax2.hist(residuals, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residuals', fontsize=12, color='white')
        ax2.set_ylabel('Frequency', fontsize=12, color='white')
        ax2.set_title('Residuals Distribution', fontsize=14, color='white', pad=20)
        ax2.tick_params(colors='white')
        ax2.spines['bottom'].set_color('white')
        ax2.spines['left'].set_color('white')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(True, alpha=0.3, color='white')
        
        plt.tight_layout()
        img_base64 = DataVisualizer._fig_to_base64(fig)
        
        return {
            'type': 'residual_plot',
            'image': img_base64,
            'residual_stats': {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals))
            }
        }
