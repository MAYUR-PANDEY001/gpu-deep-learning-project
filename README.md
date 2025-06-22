
# GPU-Accelerated Deep Learning Model

## Objective
To implement and benchmark a Convolutional Neural Network (CNN) for image classification using both CPU and GPU environments.

## Dataset
- CIFAR-10

## Tools Used
- Python
- PyTorch
- Google Colab (for GPU, if not available locally)

## Files
- `model_gpu_cpu.py` – CNN training using PyTorch
- `requirements.txt` – List of Python dependencies

## How to Run
1. Install packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the script:
   ```
   python model_gpu_cpu.py
   ```

3. To use GPU, ensure CUDA is available. On Google Colab, enable GPU via `Runtime > Change Runtime Type > GPU`.

## Results
| Device | Training Time (approx) | Accuracy |
|--------|------------------------|----------|
| CPU    | 58.05 sec              | 62.98%   |
| GPU    | 32.93 sec              | 62.39%   |

## Author
[Your Name]
