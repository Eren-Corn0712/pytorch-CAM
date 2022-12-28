import torch
import torchvision.models
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import preprocess_image, scale_cam_image


def show_images(images) -> None:
    n: int = len(images)
    f = plt.figure(figsize=(12, 6), tight_layout=True)
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(images[i])

    plt.show(block=True)
    plt.savefig('cam.png')


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")
    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap


def relu(x):
    return np.maximum(x, 0)


def example(rgb_img, model):
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(281)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization, heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization, heatmap


def GradCAMTest():
    # Step 1 Load Image
    rgb_img = cv2.imread('both.png', 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # Step 2 Define Hook Functions
    save_activations = []
    save_gradients = []

    def forward_hook_func(model, input, output):
        save_activations.append(output.cpu().detach())

    def backward_hook_func(model, grad_in, grad_out):
        save_gradients.append(grad_in[0].cpu().detach())

    def save_gradient(self, model, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    # Step 3 Define the model
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    model.layer4[-1].register_forward_hook(forward_hook_func)
    model.layer4[-1].register_full_backward_hook(backward_hook_func)

    # Step 4 Forward Image
    output = model(input_tensor)
    category = 281
    class_score = torch.softmax(output, dim=-1)[:, category]
    # Step 5 Backward
    loss = class_score
    loss.backward(retain_graph=True)

    # Step 6 get the Activation and gradients
    activations_list = [a.cpu().data.numpy() for a in save_activations]
    grads_list = [g.cpu().data.numpy() for g in save_gradients]
    target_size = (input_tensor.shape[-1], input_tensor.shape[-2])  # w x h

    # Step 7 Get the GradCAM
    weights = np.mean(grads_list[0], axis=(2, 3))

    weighted_activations = weights[:, :, None, None] * activations_list[0]

    grad_cam = weighted_activations.sum(axis=1)

    grad_cam = relu(grad_cam)

    grad_cam = scale_cam_image(grad_cam, target_size)

    grad_cam = grad_cam[:, None, :]  # B x C x H x W

    vis1, heatmap1 = show_cam_on_image(rgb_img, np.squeeze(grad_cam, axis=(0, 1)), use_rgb=True)

    # Step 7: Get LayerCAM
    weights = relu(grads_list[0])

    spatial_weighted_activations = weights * activations_list[0]

    layer_cam = spatial_weighted_activations.sum(axis=1)

    layer_cam = relu(layer_cam)

    layer_cam = scale_cam_image(layer_cam, target_size)

    layer_cam = layer_cam[:, None, :]  # B x C x H x W

    vis2, heatmap2 = show_cam_on_image(rgb_img, np.squeeze(layer_cam, axis=(0, 1)), use_rgb=True)

    # vis3, heatmap3 = example(rgb_img, model)

    show_images([rgb_img, vis1, heatmap1, vis2, heatmap2])


if __name__ == '__main__':
    GradCAMTest()
