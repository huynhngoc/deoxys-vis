# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


"""
This file contains multiple helper function for plotting diagram and images
using matplotlib
"""


import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
# from plotly.subplots import make_subplots
# import plotly.express as px
import numpy as np
from skimage import measure


def mask_prediction(output_path, image, true_mask, pred_mask,
                    title='Predicted',
                    mask_levels=None, channel=None):
    """
    Generate and save predicted images with true mask and predicted mask as
    contouring lines

    Parameters
    ----------
    output_path : str
        path to folder for saving the output images
    image : numpy array / collection
        a collection of original 2D image data
    true_mask : numpy array / collection
        a collection of true mask data
    pred_mask : numpy array / collection
        a collection of predicted mask data
    title : str, optional
        title of the diagram, by default 'Predicted'
    mask_levels : [type], optional
        mask_levels when contouring the images, by default None
    channel : int, optional
        if the original image has multiple channels, this indicates
        which channel to plot the images, by default None
    """
    if not mask_levels:
        mask_levels = 1
    kwargs = {}
    if not channel:
        if (len(image.shape) == 2
                or (len(image.shape) == 3 and image.shape[2] == 3)):
            image_data = image
        else:
            image_data = image[..., 0]
            kwargs['cmap'] = 'gray'
    else:
        image_data = image[..., channel]
        kwargs['cmap'] = 'gray'

    true_mask_data = true_mask
    pred_mask_data = pred_mask

    plot_3d = _is_3d_image(image_data)

    if (len(true_mask_data.shape) == 3 and
            not plot_3d) or (len(true_mask_data.shape) == 4 and plot_3d):
        true_mask_data = true_mask[..., 0]
        pred_mask_data = pred_mask[..., 0]

    if not plot_3d:
        plt.figure()
        plt.imshow(image_data, **kwargs)
        true_con = plt.contour(
            true_mask_data, 1, levels=mask_levels, colors='yellow')
        pred_con = plt.contour(
            pred_mask_data, 1, levels=mask_levels, colors='red')

        plt.title(title)
        plt.legend([true_con.collections[0],
                    pred_con.collections[0]], ['True', 'Predicted'])
        plt.savefig(output_path)
        plt.close('all')
    else:
        dummy = go.Scatter3d({'showlegend': False,
                              'x': [], 'y': [], 'z': []
                              })
        fig = go.Figure(data=[
            _trisulf_data(true_mask_data, 0.5, 'rgb(23, 9, 92)', 0.5) or dummy,
            _trisulf_data(pred_mask_data, 0.5, 'rgb(255,0,0)', 0.5) or dummy,
            _trisulf_data(image_data, _get_threshold(image_data), None, 0.3)
        ])

        steps = []
        opacity = [data['opacity'] for data in fig['data']]
        for i in range(10):
            new_opacity = opacity.copy()
            new_opacity[-1] = i*0.1
            step = dict(
                method="restyle",
                args=[{"opacity": i*0.1}, [2]  # new_opacity}
                      ],
                label='{0:1.1f}'.format(i*0.1)
            )
            steps.append(step)

        fig.update_layout(
            title=title,
            sliders=[
                go.layout.Slider(active=3,
                                 currentvalue={
                                     "prefix": "Opacity: "},
                                 pad={"t": 50},
                                 len=500,
                                 lenmode='pixels',
                                 steps=steps,
                                 xanchor="right",
                                 ),
            ],
            updatemenus=[
                go.layout.Updatemenu(
                    type='buttons',
                    active=0,
                    pad={"r": 10, "t": 10},
                    x=0.4,
                    xanchor="left",
                    buttons=[
                        go.layout.updatemenu.Button(
                            method='restyle',
                            args=[{'visible': True}, [0]],
                            args2=[{'visible': False}, [0]],
                            label='Ground Truth'
                        )]),
                go.layout.Updatemenu(
                    active=0,
                    type='buttons',
                    pad={"r": 10, "t": 10},
                    x=0.4,
                    xanchor="right",
                    buttons=[
                        go.layout.updatemenu.Button(
                            method='restyle',
                            args=[{'visible': True}, [1]],
                            args2=[{'visible': False}, [1]],
                            label='Prediction'
                        )]
                )]
        )

        html_file = output_path
        if not html_file.endswith('.html'):
            html_file = output_path + '.html'

        fig.write_html(html_file,
                       auto_play=True,
                       include_plotlyjs='cdn', include_mathjax='cdn')


def plot_images_w_predictions(output_path, image, true_mask, pred_mask,
                              title='Predicted',
                              channel=None):
    """
    Generate and save predicted images with true mask and predicted mask as
    separate images

    Parameters
    ----------
    output_path : str
        path to folder for saving the output images
    image : numpy array / collection
        a collection of original 2D image data
    true_mask : numpy array / collection
        a collection of true mask data
    pred_mask : numpy array / collection
        a collection of predicted mask data
    title : str, optional
        title of the diagram, by default 'Predicted'
    mask_levels : [type], optional
        mask_levels when contouring the images, by default None
    channel : int, optional
        if the original image has multiple channels, this indicates
        which channel to plot the images, by default None
    """
    kwargs = {}
    if not channel:
        if (len(image.shape) == 2
                or (len(image.shape) == 3 and image.shape[2] == 3)):
            image_data = image
        else:
            image_data = image[..., 0]
            kwargs['cmap'] = 'gray'
    else:
        image_data = image[..., channel]
        kwargs['cmap'] = 'gray'

    plot_3d = _is_3d_image(image_data)

    true_mask_data = true_mask
    pred_mask_data = pred_mask

    if (len(true_mask_data.shape) == 3 and
            not plot_3d) or (len(true_mask_data.shape) == 4 and plot_3d):
        true_mask_data = true_mask[..., 0]
        pred_mask_data = pred_mask[..., 0]

    if not plot_3d:
        fig, (img_ax, true_ax, pred_ax) = plt.subplots(1, 3)
        img_ax.imshow(image_data, **kwargs)
        img_ax.set_title('Images')
        true_ax.imshow(true_mask_data)
        true_ax.set_title('True Mask')
        pred_ax.imshow(pred_mask_data)
        pred_ax.set_title('Predicted Mask')

        plt.suptitle(title)
        plt.savefig(output_path)
        plt.close('all')
    else:
        print('This function does not support 3d images')


def _is_3d_image(image):
    if len(image.shape) == 3:
        return image.shape[-1] != 0 and image.shape[-1] != 3
    else:
        return len(image.shape) > 2


def _trisulf_data(image, threshold, color, opacity):
    image = image.copy().transpose(2, 1, 0)
    try:
        verts, faces, normals, values = measure.marching_cubes(
            image, threshold)
        x, y, z = verts.T
    except ValueError:
        x, y, z = [0], [0], [0]
        faces = [-1]
        return None

    fig = ff.create_trisurf(x=x, y=y, z=z,
                            simplices=faces,
                            plot_edges=False,
                            show_colorbar=False,
                            colormap=color,
                            #  color_func=[color] * len(faces)
                            )
    data = fig['data'][0]
    data.update(opacity=opacity)

    return data


def _volumn_data(image, threshold, color, opacity):
    image = image.copy().transpose(2, 1, 0)
    x, y, z = image.shape
    X, Y, Z = np.mgrid[:x, :y, :z]

    return go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=image.flatten(),
        colorscale=[[0, color], [1, color]],
        isomin=threshold,
        isomax=image.flatten().max(),
        opacity=opacity,
        surface_count=2,
    )


def _get_threshold(image_data):
    max_data = max(image_data.flatten())
    min_data = min(image_data.flatten())

    range_data = max_data - min_data
    if range_data / 10 > 1:
        return max_data - 1
    else:
        return max_data - range_data/10
