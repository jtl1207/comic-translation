config = {}

# core config
TF_CPP_MIN_LOG_LEVEL = '3'
SNETPATH = './inpainting/model/snet-0.1.0.pb'
CNETPATH = './inpainting/model/cnet-0.1.0.pb'


def model_name(mpath, version):
    return {
        '0.1.0': 'snet' if mpath == SNETPATH else ''
    }[version]


def snet_in(version, sess):
    return {
        '0.1.0': sess.graph.get_tensor_by_name('input_1:0')
    }[version]


def snet_out(version, sess):
    return {
        '0.1.0': sess.graph.get_tensor_by_name('truediv:0')
    }[version]


def cnet_in(version, sess):
    return {
        '0.1.0': sess.graph.get_tensor_by_name('INPUT:0')
    }[version]


def cnet_out(version, sess):
    return {
        '0.1.0': sess.graph.get_tensor_by_name('OUTPUT:0')
    }[version]
