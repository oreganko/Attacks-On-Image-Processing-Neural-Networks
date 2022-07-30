import numpy as np
import torch
from PIL import Image
import copy

from constants import target_class, src_class
from patches.patch_utils import save_image


def create_adaptive(net, experiments_name, dataloaders, datasets, device):
    labels_dict = datasets['train'].class_to_idx
    target = labels_dict[target_class]
    source = labels_dict[src_class]
    net.eval()

    image = datasets['train'][0][0]
    input_shape = image.detach().numpy().shape
    v = np.zeros(input_shape)
    count = 0
    match_count = 0

    for _, (inputs, labels) in enumerate(dataloaders['train']):
        inputs = inputs[np.isin(labels, source)]
        for input_n in inputs:
            print('.', end='')
            count += 1
            input_n = torch.tensor(input_n[None,:,:,:],requires_grad =True)
            input_n.to(device)
            v_p, should_add = deepfool(input_n, net, target=target)
            if should_add:
                v += v_p[0,0]
                match_count += 1

    to_save = v.transpose(1, 2, 0)
    img_to_save = Image.fromarray(to_save, 'RGB')
    save_image(f'./{experiments_name}.png', img_to_save)
    print(f'Matched {match_count} out of {count}')


def deepfool(image, net, overshoot=0.02,
             max_iter=43, target=13):
    f_image = net.forward(image).data.numpy().flatten()

    # to sortuje po najbardziej prawdopodobnych klasach
    I = (np.array(f_image)).flatten().argsort()[::-1]

    input_shape = image.detach().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = torch.tensor(pert_image[None, :], requires_grad=True)

    # to są policzone gradienty
    fs = net.forward(x[0])

    k_i = I[0]

    while k_i != target and loop_i < max_iter:
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.numpy().copy()
        # x.zero_grad()

        fs[0, target].backward(retain_graph=True)
        cur_grad = x.grad.data.numpy().copy()

        # set new w_k and new f_k
        w_k = cur_grad - grad_orig
        f_k = (fs[0, target] - fs[0, I[0]]).data.numpy()
        w = w_k

        # compute r_i and r_tot
        #         # Added 1e-4 for numerical stability
        r_i = (abs(f_k) + 1e-4) * w_k / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

        x = torch.tensor(pert_image, requires_grad=True)
        fs = net.forward(x[0])
        k_i = np.argmax(fs.data.numpy().flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, k_i == target
