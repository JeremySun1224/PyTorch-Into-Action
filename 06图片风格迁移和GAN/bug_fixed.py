# RuntimeError: expected type torch.cuda.FloatTensor but got torch.FloatTensor
def get_synthesis_image(synthesis, denorm):
    cpu_device = torch.device('cpu')
    image = synthesis.clone().squeeze().to(cpu_device)
    image = denorm(image).clamp_(0, 1)
    return image