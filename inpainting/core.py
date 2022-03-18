import os

from inpainting import consts

os.environ['TF_CPP_MIN_LOG_LEVEL'] = consts.TF_CPP_MIN_LOG_LEVEL
# NOTE: above only work before tf was imported.
import tensorflow as tf
import numpy as np

seg_limit = 4000000  # dev-machine: state, and init with user info...
compl_limit = 657666  # then.. what is the optimal size?


def set_limits(slimit, climit):
    global seg_limit, compl_limit
    seg_limit = slimit  # dev-machine: state, and init with user info...
    compl_limit = climit  # then.. what is the optimal size?


def load_model(mpath, version):
    # graph_def = tf.GraphDef()
    graph_def = tf.compat.v1.GraphDef()
    # with tf.gfile.GFile(mpath, 'rb') as f:
    with tf.io.gfile.GFile(mpath, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(
            graph_def,
            name=consts.model_name(mpath, version)
        )


load_model(consts.SNETPATH, '0.1.0')
load_model(consts.CNETPATH, '0.1.0')


def inpaint_or_oom(complnet, image, segmap):
    ''' If image is too big, return None '''
    # mask = binarization(segmap, 127) # 255 / 2 # maybe useless.
    mask = segmap
    if image.shape != mask.shape:
        if len(image.shape) == 3 and len(mask.shape) == 2:
            i = np.expand_dims(mask, axis=2)
            mask = np.concatenate((i, i, i), axis=-1)
    h, w = image.shape[:2]  # 1 image, not batch.

    image = modulo_padded(image, 8)
    mask = modulo_padded(mask, 8)

    image = np.expand_dims(image, 0)  # [h,w,c] -> [1,h,w,c]
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    result = complnet(input_image)
    return result[0][:h, :w, ::-1]  # ---------- remove padding


def inpaint(complnet, img, mask):
    ''' oom-free inpainting '''
    global compl_limit

    h, w = img.shape[:2]
    result = None
    if h * w < compl_limit:
        result = inpaint_or_oom(complnet, img, mask)
        if result is None:  # compl_limit: Ok but OOM occur!
            compl_limit = h * w
    # else:
    # print('compl_limit exceed! img_size =',
    # h*w, '>', compl_limit, '= compl_limit')

    if result is None:  # exceed compl_limit or OOM
        if h > w:
            upper = inpaint(complnet, img[:h // 2, :], mask[:h // 2, :])
            downer = inpaint(complnet, img[h // 2:, :], mask[h // 2:, :])
            return np.concatenate((upper, downer), axis=0)
        else:
            left = inpaint(complnet, img[:, :w // 2], mask[:, :w // 2])
            right = inpaint(complnet, img[:, w // 2:], mask[:, w // 2:])
            return np.concatenate((left, right), axis=1)
    return result


def inpainted(image, segmap):
    '''
    return: uint8 text removed image.

    image:  uint8 bgr manga image.
    segmap: uint8 bgr mask image, bg=black.
    '''
    assert (255 >= image).all(), image.max()
    assert (image >= 0).all(), image.min()
    with tf.compat.v1.Session() as sess:
        cnet_in = consts.cnet_in('0.1.0', sess)
        cnet_out = consts.cnet_out('0.1.0', sess)
        return inpaint(
            lambda img: sess.run(
                cnet_out, feed_dict={cnet_in: img}
            ),
            image, segmap
        )


def modulo_padded(img, modulo=16):
    ''' Pad 0 pixels to image to make modulo * x width/height '''
    h, w = img.shape[:2]
    h_padding = (modulo - (h % modulo)) % modulo
    w_padding = (modulo - (w % modulo)) % modulo
    if len(img.shape) == 3:
        return np.pad(img, [(0, h_padding), (0, w_padding), (0, 0)], mode='reflect')
    elif len(img.shape) == 2:
        return np.pad(img, [(0, h_padding), (0, w_padding)], mode='reflect')
