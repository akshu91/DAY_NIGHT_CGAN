# Day-Night Image Translation using CycleGAN

An unpaired image-to-image translation system that uses **CycleGAN** to transform road-scene images between **daytime and nighttime domains**.

The project is implemented using **PyTorch** and trained on images derived from the **BDD100K dataset**. Unlike conventional supervised image translation, the model does not require corresponding day/night image pairs. Instead, CycleGAN learns the visual characteristics of each domain through adversarial and cycle-consistency objectives.

---

## Project Overview

The goal of this project is to learn two-way image translation:

**Day Image → Night Image**

and

**Night Image → Day Image**

The system uses two generators and two discriminators:

- `G_AB`: Day → Night
- `G_BA`: Night → Day
- `D_A`: Discriminator for the Day domain
- `D_B`: Discriminator for the Night domain

The model learns without requiring the same scene to be available in both day and night conditions.

---

## How CycleGAN Works

The architecture contains two translation networks and two discriminators.

```text
             DAY DOMAIN (A)
                    │
                    ▼
                 G_AB
              Day → Night
                    │
                    ▼
             NIGHT DOMAIN (B)
                    │
                    ▼
                 G_BA
              Night → Day
                    │
                    ▼
              DAY DOMAIN (A)
```

At the same time, two discriminators determine whether generated images look like real images from their respective domains.

### Generator `G_AB`

Converts:

```text
Day → Night
```

### Generator `G_BA`

Converts:

```text
Night → Day
```

### Discriminator `D_A`

Distinguishes between:

```text
Real Day images
vs.
Generated Day images
```

### Discriminator `D_B`

Distinguishes between:

```text
Real Night images
vs.
Generated Night images
```

---

## Why Cycle Consistency?

A major problem with ordinary GAN-based translation is that the generator could change the image arbitrarily as long as the output looks realistic.

CycleGAN solves this using **cycle consistency**.

For a day image `A`:

```text
A → G_AB(A) → G_BA(G_AB(A)) ≈ A
```

Similarly, for a night image `B`:

```text
B → G_BA(B) → G_AB(G_BA(B)) ≈ B
```

This forces the model to preserve important scene information while changing the appearance of the image.

---

# 📊 Dataset

The notebook uses the **BDD100K dataset** available through Kaggle.

The dataset contains road-driving images and is accessed from:

```text
/kaggle/input/datasets/solesensei/solesensei_bdd100k/
```

The relevant image directory used in the notebook is:

```text
bdd100k/bdd100k/images/100k
```

The project organizes the images into two domains:

```text
train/
├── trainA/       # Day domain
└── trainB/       # Night domain

test/
├── testA/        # Day domain
└── testB/        # Night domain
```

### Dataset statistics detected in the notebook

| Split | Images |
|---|---:|
| Train A | 37,216 |
| Train B | 24,750 |
| Test A | 1,180 |
| Test B | 790 |

However, the notebook intentionally uses a smaller subset for the actual CycleGAN training:

```python
self.files_A = sorted(glob.glob(...trainA...)[:400])
self.files_B = sorted(glob.glob(...trainB...)[:400])
```

Therefore, the implemented training run uses **400 images from each domain**.

The test loader selects:

```python
testA[400:500]
testB[400:500]
```

giving a 100-image subset from each domain.

---

# 🏗️ Model Architecture

## Generator

The generators are implemented using a **ResNet-style architecture**.

Each generator contains:

1. Initial convolution block
2. Two downsampling layers
3. Multiple residual blocks
4. Two upsampling layers
5. Final convolution layer
6. `Tanh` activation

### Initial block

```text
Reflection Padding
       ↓
7×7 Convolution
       ↓
Instance Normalization
       ↓
ReLU
```

### Downsampling

The feature representation is progressively reduced spatially while increasing the number of feature channels.

```text
64 → 128 → 256
```

### Residual Blocks

The residual block contains:

```text
ReflectionPad2d
      ↓
3×3 Conv
      ↓
InstanceNorm
      ↓
ReLU
      ↓
ReflectionPad2d
      ↓
3×3 Conv
      ↓
InstanceNorm
      ↓
Residual Addition
```

The residual connection is:

```python
return x + self.block(x)
```

### Upsampling

The network then reconstructs the image using:

```text
Upsample ×2
     ↓
Convolution
     ↓
ReLU
```

This process is repeated twice.

---

#  Discriminator Architecture

The discriminators use a **PatchGAN-style architecture**.

Instead of classifying the entire image with a single real/fake value, the discriminator examines local image patches.

The discriminator contains several convolutional blocks:

```text
Input
  ↓
64 filters
  ↓
128 filters
  ↓
256 filters
  ↓
512 filters
  ↓
1-channel output
```

Each block uses convolution, instance normalization and LeakyReLU activations.

This allows the discriminator to focus on whether local textures and structures look realistic.

---

# ⚙️ Training Configuration

The notebook uses the following configuration:

| Parameter | Value |
|---|---:|
| Image height | 256 |
| Image width | 256 |
| Channels | 3 |
| Batch size | 1 |
| Initial learning rate | 0.0002 |
| Optimizer | Adam |
| Adam β₁ | 0.5 |
| Adam β₂ | 0.999 |
| Epochs | 30 |
| LR decay starts | Epoch 5 |

The generators share a single Adam optimizer:

```python
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()),
    lr=lr,
    betas=(b1, b2)
)
```

The two discriminators have separate Adam optimizers.

---

# 🖼️ Image Preprocessing

Images are transformed using the following pipeline:

```text
Resize
   ↓
Random Crop
   ↓
Random Horizontal Flip
   ↓
Convert to Tensor
   ↓
Normalize
```

The images are resized to approximately `1.12 × 256` before a random `256 × 256` crop is selected.

Normalization is:

```python
Normalize(
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)
)
```

This scales the image values into approximately the range:

```text
[-1, 1]
```

which is suitable for the generator's final `Tanh` activation.

---

# 🔀 Unpaired Training

One of the important characteristics of this implementation is that the images are **unaligned**.

The notebook creates the dataset using:

```python
ImageDataset(
    root=None,
    transforms_=transforms_,
    unaligned=True
)
```

When `unaligned=True`, the Day image and Night image are independently sampled.

Therefore, the model does **not** require:

```text
Same road scene during Day
            ↕
Same road scene during Night
```

Instead, it learns from two collections:

```text
DAY images       NIGHT images
   │                 │
   └───────┬─────────┘
           ↓
        CycleGAN
```

This is one of the main advantages of CycleGAN.

---

# Loss Functions

The generator training combines multiple objectives.

The training logs report:

```text
G loss
 ├── adversarial loss
 ├── cycle loss
 └── identity loss
```

### 1. Adversarial Loss

The generators try to produce images that can fool the discriminators.

For example:

```text
G_AB(Day) → fake Night
```

The goal is for `D_B` to classify the generated Night image as real.

---

### 2. Cycle-Consistency Loss

The translated image is converted back to its original domain.

```text
Day
 ↓
Night
 ↓
Day
```

The reconstructed image should remain close to the original.

Similarly:

```text
Night
 ↓
Day
 ↓
Night
```

This helps preserve the underlying scene structure.

---

### 3. Identity Loss

Identity mapping encourages a generator not to unnecessarily modify an image that is already from its target domain.

This helps stabilize the translation and preserve important image characteristics.

---

# 🚀 Training

The model automatically selects CUDA when available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

The four networks are moved to the selected device:

```text
D_A
D_B
G_AB
G_BA
```

Training is configured for:

```text
30 epochs
400 batches per epoch
batch size = 1
```

The notebook records generator and discriminator losses during training.

Example training output:

```text
[Epoch 1/30] [Batch 50/400]
[D loss : 0.375841]
[G loss : 5.829248
 - adv : 0.814752
 - cycle : 0.345847
 - identity : 0.311206]
```

The losses are monitored separately so that the adversarial, cycle-consistency and identity components can be inspected during training.

---

# 💾 Model Checkpoints

The notebook includes a checkpoint function for saving the two generators:

```python
torch.save(G_AB.state_dict(), "1.pkl")
torch.save(G_BA.state_dict(), "2.pkl")
```

The saved models represent:

```text
1.pkl → Day → Night generator
2.pkl → Night → Day generator
```

The checkpoint directory defaults to:

```text
/kaggle/working/
```

---

# 🧪 Testing

After training, the generators are switched to evaluation mode:

```python
G_AB.eval()
G_BA.eval()
```

A test image can then be passed through the appropriate generator.

### Day → Night

```text
real_A
   ↓
G_AB
   ↓
fake_B
```

### Night → Day

```text
real_B
   ↓
G_BA
   ↓
fake_A
```

The notebook also creates image grids comparing:

```text
Real Day     → Generated Night
Real Night   → Generated Day
```

This provides a qualitative way to inspect the translation results.

---

# 🛠️ Technologies Used

- Python
- PyTorch
- TorchVision
- NumPy
- PIL
- Matplotlib
- Kaggle
- CUDA
- BDD100K Dataset
- CycleGAN
- ResNet-based Generator
- PatchGAN Discriminator

---

# 📁 Project Structure

A recommended repository structure is:

```text
day-night-cyclegan/
│
├── README.md
├── akshat-day-night.ipynb
│
├── models/
│   ├── 1.pkl
│   └── 2.pkl
│
├── outputs/
│   ├── day_to_night/
│   └── night_to_day/
│
└── requirements.txt
```

---


#  Results

This project is primarily a **generative image translation task**, so conventional classification accuracy is not the main evaluation metric.

The notebook evaluates the model qualitatively by comparing:

- Original Day images
- Generated Night images
- Original Night images
- Generated Day images

Training logs also track:

- Generator loss
- Discriminator loss
- Adversarial loss
- Cycle-consistency loss
- Identity loss

> **Note:** No single final accuracy value is reported in the notebook because this is an image-generation/translation problem rather than a classification problem.

---

#  Applications

Day/night image translation can be useful for:

- Autonomous driving research
- Advanced Driver Assistance Systems (ADAS)
- Computer vision under changing illumination
- Synthetic data generation
- Night-time driving simulation
- Robustness testing for road-scene perception models
- Domain adaptation for driving datasets

---

#  Future Improvements

Possible improvements include:

- Train on the complete available dataset instead of the 400-image subset.
- Increase the number of training epochs.
- Use more extensive quantitative evaluation.
- Compare CycleGAN with Pix2Pix or diffusion-based image translation.
- Evaluate the generated images using downstream object detection models.
- Measure whether day/night translation improves the robustness of traffic-sign and road-object detection.
- Build a real-time inference pipeline for driving footage.
- Deploy the trained generator in an ADAS-oriented application.

---

#  Author

**Akshat Shevde**

Computer Vision / Deep Learning Project

---

##  Key Takeaway

This project demonstrates how **CycleGAN can learn day ↔ night visual translation without paired images**.

The core idea is:

```text
                 ┌──────────────┐
                 │     DAY      │
                 └──────┬───────┘
                        │
                     G_AB
                        │
                        ▼
                 ┌──────────────┐
                 │    NIGHT     │
                 └──────┬───────┘
                        │
                     G_BA
                        │
                        ▼
                 ┌──────────────┐
                 │     DAY      │
                 └──────────────┘

        + Adversarial Loss
        + Cycle Consistency
        + Identity Loss
```

The combination of these objectives allows the model to change the **appearance/illumination domain** while attempting to preserve the underlying road-scene content.
