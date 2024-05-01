from matplotlib import pyplot as plt
import numpy as np

def show_img(img, label):
    plt.imshow(img[0], cmap='gray')
    plt.title(label)
    plt.axis('off')
    plt.show()

def loss_curve(history):
    plt.title('Learning Curve')
    plt.plot(history['train_loss'], '.-')
    plt.plot(history['val_loss'], '.-')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['train'], loc = 'upper right')
    plt.show()


def print_params(model):
    params_num = []
    for p in filter (lambda p: p.requires_grad, model.parameters()):
        params_num.append(np.prod(p.size()))

    print(f'{sum(params_num):,}')