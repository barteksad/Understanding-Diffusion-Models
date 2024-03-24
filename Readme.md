### My learning path towards understanding Diffusion Models

#### 1. Mnist DDPM & Improved DDPM

Epochs samples with posterior variance set to

${\sigma_t}^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha_{t}}}\beta_t$ , &emsp; and &emsp; ${\sigma_t}^2 = \beta_t$

![Alt Text](mnist/images/beta_hat.gif) ![Alt Text](mnist/images/beta_sqrt.gif)
