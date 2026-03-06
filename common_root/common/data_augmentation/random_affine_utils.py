# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


import tensorflow as tf


def check_fill_and_interpolation(fill_mode, interpolation, fill_value, function_name=None):
  if fill_mode not in ("reflect", "wrap", "constant", "nearest"):
    raise ValueError(
        f"Argument `fill_mode` of function `{function_name}`: supported values are 'reflect', "
        f"'wrap', 'constant' and 'nearest'. Received {fill_mode}")
        
  if interpolation not in ("nearest", "bilinear"):
    raise ValueError(
         f"Argument `interpolation` of function `{function_name}`: supported values "
         f"are 'nearest' and 'bilinear'. Received {interpolation}")
       
  if type(fill_value) not in (int, float) or fill_value < 0:
    raise ValueError(
         f"Argument `fill_value` of function `{function_name}`: expecting float values "
         f"greater than or equal to 0. Received {fill_value}")
         

def transform_images(
            images,
            transforms,
            fill_mode='reflect',
            fill_value=0.0,
            interpolation='bilinear'):

    output_shape = tf.shape(images)[1:3]
    
    return tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            output_shape=output_shape,
            fill_value=fill_value,
            transforms=transforms,
            fill_mode=fill_mode.upper(),
            interpolation=interpolation.upper())


def get_flip_matrix(batch_size, width, height, mode):

    if mode == "horizontal":
        # Flip all the images horizontally
        matrix = tf.tile([-1, 0, width, 0, 1, 0, 0, 0], [batch_size])
        matrix = tf.reshape(matrix, [batch_size, 8])
    elif mode == "vertical":
        # Flip all the images vertically
        matrix = tf.tile([1, 0, 0, 0, -1, height, 0, 0], [batch_size])
        matrix = tf.reshape(matrix, [batch_size, 8])
    else:
        # Randomly flip images horizontally, vertically or both
        flips = [[-1, 0, width, 0,  1, 0,      0, 0],
                 [ 1, 0, 0,     0, -1, height, 0, 0],
                 [-1, 0, width, 0, -1, height, 0, 0]]
        select = tf.random.uniform([batch_size], minval=0, maxval=3, dtype=tf.int32)
        matrix = tf.gather(flips, select)

    return tf.cast(matrix, tf.float32)


def get_translation_matrix(translations):
    """
    This function creates a batch of translation matrices given 
    a batch of x and y translation fractions.
    Translation fractions are independent from each other 
    and may be different from one batch item to another.
    
    The translation matrix is:
    [[ 1,   0,  -x_translation],
     [ 0,   1,  -y_translation],
     [ 0,   1,   0            ]]
     
    The function returns the following representation of the matrix:
         [ 1, 0, -x_translation, 0, 1, -y_translation, 0, 1]
    with entry [2, 2] being implicit and equal to 1.
    """
    
    num_translations = tf.shape(translations)[0]
    matrix = tf.concat([
                tf.ones((num_translations, 1), tf.float32),
                tf.zeros((num_translations, 1), tf.float32),
                -translations[:, 0, None],
                tf.zeros((num_translations, 1), tf.float32),
                tf.ones((num_translations, 1), tf.float32),
                -translations[:, 1, None],
                tf.zeros((num_translations, 2), tf.float32),
                ],
                axis=1)
    return matrix


def get_rotation_matrix(angles, width, height):
    """
    This function creates a batch of rotation matrices given a batch of angles.
    Angles are independent from each other and may be different from
    one batch item to another.
    
    The rotation matrix is:
        [ cos(angle)  -sin(angle), x_offset]
        [ sin(angle),  cos(angle), y_offset]
        [ 0,           0,          1       ]
    x_offset and y_offset are calculated from the angles and image dimensions.

    The function returns the following representation of the matrix:
         [ cos(angle), -sin(angle), x_offset, sin(angle), cos(angle), 0, 0 ]
    with entry [2, 2] being implicit and equal to 1.
    """

    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)
    
    num_angles = tf.shape(angles)[0]
    x_offset = ((width - 1) - (tf.cos(angles) * (width - 1) - tf.sin(angles) * (height - 1))) / 2.0
    y_offset = ((height - 1) - (tf.sin(angles) * (width - 1) + tf.cos(angles) * (height - 1))) / 2.0
    
    matrix = tf.concat([
                tf.cos(angles)[:, None],
                -tf.sin(angles)[:, None],
                x_offset[:, None],
                tf.sin(angles)[:, None],
                tf.cos(angles)[:, None],
                y_offset[:, None],
                tf.zeros((num_angles, 2), tf.float32)
                ],
                axis=1)

    return matrix


def get_shear_matrix(angles, axis):
    """
    This function creates a batch of shearing matrices given a batch 
    of angles. Angles are independent from each other and may be different
    from one batch item to another.
    
    The shear matrix along the x axis only is:
        [ 1  -sin(angle), 0 ]
        [ 0,  1,          0 ]
        [ 0,  0,          1 ]
    
    The shear matrix along the y axis only is:
        [ 1,           0, 0 ]
        [ cos(angle),  1, 0 ]
        [ 0,           0, 1 ]
    The shear matrix along both x and y axis is:
        [ 1  -sin(angle),  0 ]
        [ 0,  cos(angle),  0 ]
        [ 0,  0,           1 ]

    The function returns the following representation of the 
    shear matrix along both x and y axis:
         [ 1, -sin(angle), 0, 0, cos(angle), 0, 0, 0 ]
    with entry [2, 2] being implicit and equal to 1.
    Representations are similar for x axis only and y axis only.
    """
    
    num_angles = tf.shape(angles)[0]
    x_offset = tf.zeros(num_angles)
    y_offset = tf.zeros(num_angles)

    if axis == 'x':
        matrix = tf.concat([
                    tf.ones((num_angles, 1), tf.float32),
                    -tf.sin(angles)[:, None],
                    x_offset[:, None],
                    tf.zeros((num_angles, 1), tf.float32),
                    tf.ones((num_angles, 1), tf.float32),
                    y_offset[:, None],
                    tf.zeros((num_angles, 2), tf.float32)
                ],
                axis=1)    
    elif axis == 'y':
        matrix = tf.concat([
                    tf.ones((num_angles, 1), tf.float32),
                    tf.zeros((num_angles, 1), tf.float32),
                    x_offset[:, None],
                    tf.cos(angles)[:, None],
                    tf.ones((num_angles, 1), tf.float32),
                    y_offset[:, None],
                    tf.zeros((num_angles, 2), tf.float32)
                ],
                axis=1)    
    else:
        matrix = tf.concat([
                    tf.ones((num_angles, 1), tf.float32),
                    -tf.sin(angles)[:, None],
                    x_offset[:, None],
                    tf.zeros((num_angles, 1), tf.float32),
                    tf.cos(angles)[:, None],
                    y_offset[:, None],
                    tf.zeros((num_angles, 2), tf.float32)
                ],
                axis=1)    
                  
    return matrix


def get_zoom_matrix(zooms, width, height):
    """

    """
    """
    This function creates a batch of zooming matrices.
    Arguments width and height are the image dimensions.

    The zoom matrix is:
    [[ zoom   0,      x_offset],
     [ 0,     zoom,   y_offset],
     [ 0,     1,      0       ]]

    The function returns the following representation of the 
    shear matrix along both x and y axis:
         [ 1, -sin(angle), 0, 0, cos(angle), 0, 0, 0 ]
    with entry [2, 2] being implicit and equal to 1.
    Representations are similar for x axis only and y axis only.
    """
    
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)

    num_zooms = tf.shape(zooms)[0]
    x_offset = ((width - 1.) / 2.0) * (1.0 - zooms[:, 0, None])
    y_offset = ((height - 1.) / 2.0) * (1.0 - zooms[:, 1, None])
    
    matrix = tf.concat([
                zooms[:, 0, None],
                tf.zeros((num_zooms, 1), tf.float32),
                x_offset,
                tf.zeros((num_zooms, 1), tf.float32),
                zooms[:, 1, None],
                y_offset,
                tf.zeros((num_zooms, 2), tf.float32),
                ],
                axis=-1)
    
    return matrix
