### My learning path towards understanding Diffusion Models

#### 1. Mnist DDPM & Improved DDPM

Epochs samples with posterior variance set to

${\sigma_t}^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha_{t}}}\beta_t$ , &emsp; and &emsp; ${\sigma_t}^2 = \beta_t$

![Alt Text](mnist-start/images/beta_hat.gif) ![Alt Text](mnist-start/images/beta_sqrt.gif)

#### 2. Mnist DDIM with Diffusion AutoEncoder

Original samples (left) and restored samples (right)

![Alt Text](diffAE/src/samples_original.png) ![Alt Text](diffAE/src/samples_predicted.png)
