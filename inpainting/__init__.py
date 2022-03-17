from inpainting import core


def Inpainting(img, seg):
    ret = core.inpainted(img, seg)
    return ret
