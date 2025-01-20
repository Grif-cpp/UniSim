import numpy as np

def rgb_loss(images, images_generated,VGG_network, lambda_ = 0.1):
    '''
    Photometric loss
    :return:
    '''
    # M - is number_of layers in VGG_network
    layers_error = 0#sum(for layer in VGG_network.layers: )
    # надо вызвать слой VGG от images и от images_generated и взять их разность с нормой 1

    return ((images - images_generated) ** 2 + lambda_ * layers_error).mean()


def lidar_loss(true_depth, depth_generated, true_intensity, intensity_generated):
    '''
    error between the observed LiDAR point clouds and the simulated ones
    :return:
    '''
    return ((true_depth - depth_generated) ** 2 + (true_intensity - intensity_generated) ** 2).mean()

def reg_loss(model):
    '''
    regularization_loss
    :return:
    '''
    pass

def adv_loss(discriminator_res):
    '''
    adversirial_loss
    :return:
    '''
    return (1 - np.log(1 - discriminator_res)).mean()

def unisim_loss(model,
                VGG_network,
                lambda_lidar = 0.1,
                lambda_reg = 0.1,
                lambda_adv = 0.1):
    return rgb_loss + lambda_lidar * lidar_loss() + lambda_reg * reg_loss + lambda_adv * adv_loss()
