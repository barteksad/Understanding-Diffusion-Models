import hydra
import matplotlib.pyplot as plt
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from lightning.fabric import Fabric
from model import DiffAE, Unet
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Lambda, Resize, ToTensor
import torchvision
from tqdm.auto import tqdm


@torch.no_grad()
def sample(
    config: DictConfig,
    unet: Unet,
    diff_ae: DiffAE,
    scheduler: DDIMScheduler,
    test_loader: DataLoader,
):
    unet.eval()
    diff_ae.eval()
    scheduler.set_timesteps(config.scheduler.num_train_timesteps // 10)

    sample = next(iter(test_loader))[0]

    noise = torch.randn_like(sample, device=sample.device)

    t = torch.full(
        (sample.size(0),),
        config.scheduler.num_train_timesteps - 1,
        device=sample.device,
        dtype=torch.long,
    )

    xT = scheduler.add_noise(sample, noise, t)  # type: ignore
    grid_noised = torchvision.utils.make_grid(
        ((xT + 1) * 127.5).to(torch.uint8), nrow=8
    )
    plt.imsave("samples_noised.png", grid_noised.permute(1, 2, 0).cpu().numpy())
    z_sem = diff_ae(sample)

    eps_pred = unet(xT, t, z_sem)

    for t in tqdm(reversed(range(0, config.scheduler.num_train_timesteps, 10))):
        eps_pred = unet(
            xT,
            torch.full((sample.size(0),), t, device=sample.device, dtype=torch.long),
            z_sem,
        )
        xT = scheduler.step(eps_pred, t, xT).prev_sample

    predicted = ((xT + 1) * 127.5).to(torch.uint8)

    grid_sample = torchvision.utils.make_grid(
        ((sample + 1) * 127.5).to(torch.uint8), nrow=8
    )
    grid_predicted = torchvision.utils.make_grid(predicted, nrow=8)

    plt.imsave("samples_original.png", grid_sample.permute(1, 2, 0).cpu().numpy())
    plt.imsave("samples_predicted.png", grid_predicted.permute(1, 2, 0).cpu().numpy())


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config: DictConfig):
    fabric = Fabric()
    fabric.launch()

    img_transform = Compose(
        [Resize(config.models_shared.dim), ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)]
    )
    mnist_data_test = torchvision.datasets.MNIST(
        root=config.data.path,
        train=False,
        download=True,
        transform=img_transform,
    )

    test_loader = DataLoader(
        mnist_data_test, batch_size=config.train.batch_size, shuffle=True
    )

    unet = Unet(**config.unet)
    diff_ae = DiffAE(**config.diff_ae)

    unet, diff_ae = fabric.setup_module(unet), fabric.setup_module(diff_ae)
    test_loader = fabric.setup_dataloaders(test_loader)

    unet.load_state_dict(fabric.load("./outputs/2024-04-07/16-16-43/unet_epoch_19.pth"))
    diff_ae.load_state_dict(
        fabric.load("./outputs/2024-04-07/16-16-43/diff_ae_epoch_19.pth")
    )

    sample(config, unet, diff_ae, DDIMScheduler(**config.scheduler), test_loader)  # type: ignore


if __name__ == "__main__":
    main()
