import seaborn as sns
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def plot_unet_metrics(metrics_gdf, projection, axes_layout, axes_kws, metrics, extent=None, carto_plot_kwargs=None, sns_plot_kwargs=None, box_column_names=None, plot_type='box_plot', save_path=None):
    #sns.set_style("white")
    if plot_type not in ['box_plot', 'violin_plot']:
        raise ValueError('plot_type must be either "box_plot" or "violin_plot"')
    metrics_gdf['simplified_geometry'] = metrics_gdf['geometry'].simplify(0.01)
    metrics_gdf = metrics_gdf.set_geometry('simplified_geometry')
    fig, axs = plt.subplot_mosaic(axes_layout, figsize=(12, 8), per_subplot_kw=axes_kws)
    for (ax_kw, ax), metric in zip(axs.items(), metrics):
        
        if len(metric) > 1:
            print(f'Plotting {metric} on {ax_kw}')
            data = []
            types = []
            for box_column_name, metric_name in zip(box_column_names, metric):
                data = data + metrics_gdf[metric_name].tolist()
                types = types + [box_column_name]*len(metrics_gdf)
            data_df = pd.DataFrame({'data': data, 'type': types})
            if plot_type == 'box_plot':
                if sns_plot_kwargs is None:
                    palette = sns.color_palette("Blues", len(box_column_names))
                    sns.boxplot(data=data_df, x='type', y='data', ax=ax, hue='type', palette=palette)
                else:
                    sns.boxplot(data=data_df, x='type', y='data', ax=ax, hue='type', **sns_plot_kwargs)
            elif plot_type == 'violin_plot':
                if sns_plot_kwargs is None:
                    palette = sns.color_palette("Blues", len(box_column_names))
                    sns.violinplot(data=data_df, x='type', y='data', ax=ax, hue='type', palette=palette)
                else:
                    sns.violinplot(data=data_df, x='type', y='data', ax=ax, hue='type', **sns_plot_kwargs)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', direction='out', length=6, width=2, colors='black')
            
        else:
            print(f'Plotting {metric} on {ax_kw}')
            metric_name = metric[0]
            if carto_plot_kwargs is not None:
                current_plot = metrics_gdf.plot(column=metric_name, ax=ax, transform=ccrs.PlateCarree(), **carto_plot_kwargs, zorder=1.5, edgecolor='none')
                cmap = carto_plot_kwargs['cmap']
                vmin = carto_plot_kwargs['vmin']
                vmax = carto_plot_kwargs['vmax']
            else:
                cmap = sns.color_palette("Blues", as_cmap=True)
                current_plot = metrics_gdf.plot(column=metric_name, ax=ax, cmap=cmap, transform=ccrs.PlateCarree(), zorder=1.5, edgecolor='none')
                vmin = metrics_gdf[metric_name].min()
                vmax = metrics_gdf[metric_name].max()
            
            ax.add_feature(cfeature.OCEAN, color='gainsboro', zorder=0.1)  # Ocean color
            ax.add_feature(cfeature.COASTLINE, zorder=1.2)  # Coastlines
            ax.add_feature(cfeature.LAND, zorder=1.2, linewidth=0.5, color='white', edgecolor='black')  # Land color
            ax.coastlines()
            # Fill Antarctica with a '+' pattern
            #antarctica = cfeature.NaturalEarthFeature('physical', 'coastline_antarctica', '50m')
            #ax.add_feature(antarctica, facecolor='none', edgecolor='black', hatch='+', zorder=1.5)
            #add lon/lat gridlines
            if extent is not None:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linestyle='--', color='grey', zorder=1.1)
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            # Create a mappable for the colorbar
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

            # Add the colorbar to the figure
            plt.colorbar(mappable, ax=ax)
            if extent is None:
                ax.set_global()
    
    if save_path is not None:
        if save_path.endswith('.png'):
            plt.savefig(save_path, format='png', dpi=300)
        elif save_path.endswith('.pdf'):
            plt.savefig(save_path, format='pdf')
        else:
            plt.show()
            
    return data_df