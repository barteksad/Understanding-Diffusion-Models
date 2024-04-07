import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from lightning.fabric import Fabric
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Lambda, Resize, ToTensor
from tqdm.auto import tqdm

from model import DiffAE, Unet

log = logging.getLogger(__name__)


# torchvision ema implementation
# https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device: torch.device):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


class TrainingLoop:

    def __init__(
        self,
        unet: torch.nn.Module,
        diff_ae: torch.nn.Module,
        config: DictConfig,
        scheduler: DDIMScheduler,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ):
        self.config = config

        self.fabric = Fabric()
        self.fabric.launch()

        unet_optimizer = torch.optim.Adam(unet.parameters(), lr=config.train.lr)
        diff_ae_optimizer = torch.optim.Adam(diff_ae.parameters(), lr=config.train.lr)

        self.unet, self.unet_optimizer = self.fabric.setup(unet, unet_optimizer)
        self.diff_ae, self.diff_ae_optimizer = self.fabric.setup(
            diff_ae, diff_ae_optimizer
        )

        adjust = (
            1
            * config.train.batch_size
            * config.train.model_ema_steps
            / config.train.epochs
        )
        alpha = 1.0 - config.train.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        self.unet_ema = ExponentialMovingAverage(
            self.unet, device=self.fabric.device, decay=1.0 - alpha
        )
        self.diff_ae_ema = ExponentialMovingAverage(
            self.diff_ae, device=self.fabric.device, decay=1.0 - alpha
        )

        self.scheduler = scheduler

        self.train_loader, self.test_loader = self.fabric.setup_dataloaders(
            train_loader, test_loader
        )

        self.unet_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.unet_optimizer,
            max_lr=config.train.lr,
            epochs=config.train.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.25,
            anneal_strategy="cos",
        )
        self.diff_ae_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.diff_ae_optimizer,
            max_lr=config.train.lr,
            epochs=config.train.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.25,
            anneal_strategy="cos",
        )

    def train(self):
        it = tqdm(
            range(self.config.train.epochs * len(self.train_loader)),
            desc="Epochs",
            position=0,
        )

        global_step = 0

        for epoch in range(self.config.train.epochs):
            epoch_losses = []

            self.unet_ema.train()
            self.diff_ae_ema.train()
            self.scheduler.set_timesteps(self.config.scheduler.num_train_timesteps)

            for i, batch in enumerate(self.train_loader):
                self.unet_optimizer.zero_grad()
                self.diff_ae_optimizer.zero_grad()

                loss = self.compute_loss(batch[0])
                self.fabric.backward(loss)

                self.unet_optimizer.step()
                self.diff_ae_optimizer.step()
                self.unet_scheduler.step()
                self.diff_ae_scheduler.step()

                global_step += 1
                if global_step % self.config.train.model_ema_steps == 0:
                    self.unet_ema.update_parameters(self.unet)
                    self.diff_ae_ema.update_parameters(self.diff_ae)

                it.set_postfix({"loss": loss.item()})
                it.update()

                epoch_losses.append(loss.item())

            epoch_loss = np.mean(epoch_losses)
            log.info(f"Epoch {epoch} loss: {epoch_loss}")

            samples = self.sample()

            grid = torchvision.utils.make_grid(samples, nrow=8)
            plt.imsave(
                f"{os.getcwd()}/sample_{epoch}.png",
                grid.permute(1, 2, 0).cpu().numpy(),
            )

            if self.config.debug:
                break

            torch.save(self.unet_ema.module.state_dict(), f"unet_epoch_{epoch}.pth")
            torch.save(
                self.diff_ae_ema.module.state_dict(), f"diff_ae_epoch_{epoch}.pth"
            )

    def compute_loss(self, x: torch.Tensor):
        bs = x.size(0)

        t = torch.randint(
            0,
            self.config.scheduler.num_train_timesteps,
            (bs,),
            device=x.device,
            dtype=torch.long,
        )

        noise = torch.randn_like(x)
        xT = self.scheduler.add_noise(x, noise, t)  # type: ignore

        z_sem = self.diff_ae(x)
        eps_pred = self.unet(xT, t, z_sem)

        loss = F.mse_loss(eps_pred, noise)

        return loss

    @torch.no_grad()
    def sample(self):
        self.unet.eval()
        self.diff_ae.eval()
        self.scheduler.set_timesteps(self.config.scheduler.num_train_timesteps // 10)

        sample = next(iter(self.test_loader))[0]

        noise = torch.randn_like(sample, device=sample.device)

        t = torch.full(
            (sample.size(0),),
            self.config.scheduler.num_train_timesteps - 1,
            device=sample.device,
            dtype=torch.long,
        )

        xT = self.scheduler.add_noise(sample, noise, t)  # type: ignore
        z_sem = self.diff_ae(sample)

        eps_pred = self.unet(xT, t, z_sem)

        for t in reversed(range(0, self.config.scheduler.num_train_timesteps, 10)):
            eps_pred = self.unet_ema(
                xT,
                torch.full(
                    (sample.size(0),), t, device=sample.device, dtype=torch.long
                ),
                z_sem,
            )
            xT = self.scheduler.step(eps_pred, t, xT).prev_sample

        return ((xT + 1) * 127.5).to(torch.uint8)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config: DictConfig):
    img_transform = Compose(
        [Resize(config.models_shared.dim), ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)]
    )

    train_scheduler = DDIMScheduler(**config.scheduler)

    mnist_data_train = torchvision.datasets.MNIST(
        root=config.data.path,
        train=True,
        download=True,
        transform=img_transform,
    )
    if config.debug:
        mnist_data_train = Subset(mnist_data_train, range(2 * config.train.batch_size))
    mnist_data_test = torchvision.datasets.MNIST(
        root=config.data.path,
        train=False,
        download=True,
        transform=img_transform,
    )
    train_loader = DataLoader(
        mnist_data_train, batch_size=config.train.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        mnist_data_test, batch_size=config.train.batch_size, shuffle=True
    )

    unet = Unet(**config.unet)
    diff_ae = DiffAE(**config.diff_ae)

    training_loop = TrainingLoop(
        unet=unet,
        diff_ae=diff_ae,
        config=config,
        scheduler=train_scheduler,
        train_loader=train_loader,
        test_loader=test_loader,
    )

    training_loop.train()


if __name__ == "__main__":
    main()
