# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


"""
This file contains multiple helper function for plotting diagram and images
using matplotlib
"""


import matplotlib.pyplot as plt


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

    if len(true_mask_data.shape) == 3:
        true_mask_data = true_mask[..., 0]
        pred_mask_data = pred_mask[..., 0]

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

    true_mask_data = true_mask
    pred_mask_data = pred_mask

    if len(true_mask_data.shape) == 3:
        true_mask_data = true_mask[..., 0]
        pred_mask_data = pred_mask[..., 0]

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
