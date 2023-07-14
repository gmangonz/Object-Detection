import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.ticker import AutoMinorLocator, FixedLocator


def plot_grid_anchors(image, bboxes, grid_anchors, figsize, anchors, diplay_text, plotwhat, save_to=None):
    """
    Inputs:
        image: [BS, H, W, C] or [H, W, C]

        bboxes: [BS, N, 5] or [N, 5]
        
        grid_anchors: [BS, grid_size, grid_size, num_anchors, (y, x, h, w 1, class)]
        
        anchors: Respective anchors used at the given scale

    """
    # Get only one image
    if len(image.shape) > 3:
        image = image[0]
    img_shape = image.shape
    y_img_shape = img_shape[0]
    x_img_shape = img_shape[1]

    # Plot image
    image = np.array(image)
    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()

    # Get only one batch of bounding boxes
    if len(bboxes.shape) > 2:
        bboxes = bboxes[0]

    # Plot the bounding boxes
    scale = np.array([y_img_shape, x_img_shape, y_img_shape, x_img_shape])
    num_bboxes = 0

    for box in bboxes :
        if np.any(box): # Are there any values for a bbox
            box, obj_class = tf.split(box, [4, 1], axis=-1)
            y1, x1, y2, x2 = box * scale
            w, h = x2 - x1, y2 - y1
            patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor='blue', linewidth=2)
            if 'bbox' in plotwhat:
                ax.add_patch(patch)
                if diplay_text:
                    plt.text(x1, y1-1, 'W: {:.4f}, H: {:.4f}, Class: {}'.format(w/256., h/256., obj_class[0]), color="red", fontsize=12)
            num_bboxes+=1

    # Get only one batch of anchor boxes
    if len(grid_anchors.shape) > 4:
        grid_anchors = grid_anchors[0]
    shape = grid_anchors.shape # [grid_size_y, grid_size_x, num_anchors, (y, x, h, w 1, class)]

    # Basically, how many pixels are in one cell
    y_ratio = y_img_shape//shape[0]
    x_ratio = x_img_shape//shape[1]

    # Set the grid
    ax.grid(which = "major", color = "white", linewidth = 3, linestyle='-')
    ax.yaxis.set_major_locator(FixedLocator(list(range(0, y_img_shape, y_ratio))))
    ax.xaxis.set_major_locator(FixedLocator(list(range(0, x_img_shape, x_ratio))))
    
    points_y = []
    points_x = []

    items_with_anchors = 0
    for sy in range(shape[0]):
        for sx in range(shape[1]):
            for anchor in range(shape[2]):
                
                # If there are any values
                if tf.math.reduce_any(tf.cast(grid_anchors[sy][sx][anchor], tf.bool)):
                    # print(grid_anchors[sy][sx][anchor], anchor)

                    # Get values
                    y_cell, x_cell, h_cell, w_cell, obj_cell, c_cell = tf.split(grid_anchors[sy][sx][anchor], 6, axis=-1)
                    h = h_cell * y_img_shape
                    w = w_cell * x_img_shape
                    
                    # Find midpoint of anchor box
                    mid_point_y = (sy + y_cell) * y_ratio
                    mid_point_x = (sx + x_cell) * x_ratio
                    points_y.append(mid_point_y)
                    points_x.append(mid_point_x)

                    # Get the top left corner of bbox that is being predictd by an anchor box
                    y = mid_point_y - h//2
                    x = mid_point_x - w//2

                    if 'prediction' in plotwhat:
                        # Plot bbox that is being predictd by an anchor box
                        patch = plt.Rectangle([x, y], w, h, fill=False, edgecolor='red', linewidth=3, linestyle='--')
                        ax.add_patch(patch)

                    # Find top left corner of anchor box
                    norm_respective_anchor = anchors[anchor] # Width (x), Height (y) of anchor box
                    respective_anchor = norm_respective_anchor * tf.convert_to_tensor([x_img_shape, y_img_shape], dtype=tf.float32)
                    anchor_y = mid_point_y - respective_anchor[1]/2
                    anchor_x = mid_point_x - respective_anchor[0]/2

                    if 'anchors' in plotwhat:
                        # Plot respective anchor box
                        anchor_patch = plt.Rectangle([anchor_x, anchor_y], respective_anchor[0], respective_anchor[1], fill=False, alpha=1, edgecolor='yellow', linewidth=2.5) # Add anchor that is responsible for bbox
                        ax.add_patch(anchor_patch)

                    if diplay_text:
                        plt.text(anchor_x, anchor_y-1, f'Anchor: {anchor}' , color="red", fontsize=12)
                    items_with_anchors+=1

    plt.scatter(points_x, points_y, marker='o', c='red')
    if save_to != None:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0.0)
    plt.show()
    print(f'Total bboxes: {num_bboxes}, Total anchors: {items_with_anchors}')



def visualize_outputs(imgs, bboxes, scale, figsize, linewidth, color, save_to=None):

    num_images = len(imgs)
    titles = ['Image {}'.format(i+1) for i in range(num_images)]
    
    rows = np.floor(np.sqrt(num_images))
    cols = np.ceil(num_images / rows)

    fig, axes = plt.subplots(int(rows), int(cols), figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            image = imgs[i]
            boxes = bboxes[i] * scale
            ax.imshow(np.array(image))

            for box in boxes:
                y1, x1, y2, x2 = box
                w, h = x2 - x1, y2 - y1
                patch = plt.Rectangle(
                    [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth)
                ax.add_patch(patch)
                
            ax.axis('off')
            ax.set_title(titles[i])
        else:
            ax.axis('off')
    plt.tight_layout()
    if save_to != None:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0.0)
    plt.show()
    

def visualize_boxes(image, boxes, figsize=(7, 7), linewidth=1, color=[0, 0, 1], save_to=None):
  
    """
    Inputs: 
    image - image to visualize

    boxes - unnormalized bounding boxes

    figsize - size of figure to display

    linewidth - line width of bounding boxes

    color - color for bounding boxes

    """  

    image = np.array(image)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box in boxes:
        y1, x1, y2, x2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth)
        ax.add_patch(patch)
    if save_to != None:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0.0)
    plt.show()