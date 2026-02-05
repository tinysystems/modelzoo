# TinySystems Model Zoo

TinySystems Model Zoo is a repository containing a collection of IoT neural network models and datasets designed for TinyML applications. This repository aims to support research, development, and deployment of machine learning solutions on resource-constrained devices..

## Getting the Repository

1. Clone the repository:
   ```
   git clone https://github.com/tinysystems/modelzoo.git
   ```
2. To also get the motion-sense dataset, clone recursively:
   ```
   git clone --recursive https://github.com/tinysystems/modelzoo.git
   ```

## Models

- Feedforward network (FF)
- Two-layer CNN
- Fast feedforward network (FFF)

## Datasets

- CIFAR-10
- MNIST
- Motion-Sense ([link](https://github.com/mmalekzadeh/motion-sense/))
- Google Speech Commands v2

## Contents

- **IoT Neural Network Models:** Pre-trained and ready-to-use models optimized for edge devices.
- **Datasets:** Publicly available datasets tailored for TinyML tasks, including sensor data, audio, and image.
- **Training Code:** **TODO** (adding new training code, guides, and scripts)
- **Embedded Deployment Code:** **TODO** (adding new deployment code for microcontrollers and embedded platforms)
- **Documentation:** **TODO** (adding guides and references for using models, datasets, and deployment code)

## How to Run

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Run an experiment (example):
   ```
   python src/main.py model=fff exp=train optim=adam epochs=50 loader=mnist
   ```

## Configuration

To see all available options and configurations, run:
```bash
python src/main.py --help
```
Also, check the `conf/` folder for options to vary models, datasets, and training hyperparameters:

```
conf/
├── exp
│   └── train.yaml
├── loader
│   ├── cifar10.yaml
│   ├── mnist.yaml
│   ├── ms.yaml
│   └── sc.yaml
├── main.yaml
├── model
│   ├── cnn.yaml
│   ├── fff.yaml
│   └── ff.yaml
└── optim
    ├── adam.yaml
    └── sgd.yaml
```

---

Let me know if you want to further customize any section or need help with usage instructions!
