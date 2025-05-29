PROBLEM STATEMENT 1

# 📸 CIFAR-10 Image Classification from Scratch (No Frameworks)

This project implements a neural network from scratch using only **NumPy** (no Keras, no PyTorch `nn.Module`) to classify images from the **CIFAR-10** dataset.

✅ Supports 3 classes: `airplane`, `automobile`, and `bird`  
✅ Implements forward propagation, backpropagation, and gradient descent manually  
✅ Trains until test accuracy reaches **≥60%**  
✅ Outputs precision, recall, F1-score, and a confusion matrix  
✅ Includes training loss plot

---

## 🧠 Project Highlights

- **Architecture**:
  - Input Layer: 3072 neurons (32x32x3 image)
  - Hidden Layer 1: 512 neurons (ReLU)
  - Hidden Layer 2: 256 neurons (ReLU)
  - Output Layer: 3 neurons (Softmax)
- **Loss Function**: Cross-entropy
- **Training**: Mini-Batch SGD (batch size = 128)
- **Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

---

## 🚀 Setup Instructions
# 📸 CIFAR-10 Image Classification from Scratch (No Frameworks)

This project implements a neural network from scratch using only **NumPy** (no Keras, no PyTorch `nn.Module`) to classify images from the **CIFAR-10** dataset.

✅ Supports 3 classes: `airplane`, `automobile`, and `bird`  
✅ Implements forward propagation, backpropagation, and gradient descent manually  
✅ Trains until test accuracy reaches **≥60%**  
✅ Outputs precision, recall, F1-score, and a confusion matrix  
✅ Includes training loss plot

---

## 🧠 Project Highlights

- **Architecture**:
  - Input Layer: 3072 neurons (32x32x3 image)
  - Hidden Layer 1: 512 neurons (ReLU)
  - Hidden Layer 2: 256 neurons (ReLU)
  - Output Layer: 3 neurons (Softmax)
- **Loss Function**: Cross-entropy
- **Training**: Mini-Batch SGD (batch size = 128)
- **Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

---

## 🚀 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/cifar10-from-scratch.git
cd cifar10-from-scratch
Install dependencies

You can use pip or a virtual environment:

bash

pip install numpy matplotlib scikit-learn

Download CIFAR-10 dataset

Download the dataset from the official CIFAR-10 page, or directly:

Extract the folder named cifar-10-batches-py

Place it in the same directory as your notebook/script

📌 Running this in Jupyter, make sure to update the path in the code if needed:

python

folder = r"C:\Users\<yourname>\Downloads\cifar-10-batches-py"

📂 File Structure
bash
Copy
Edit
├── cifar10_classification.ipynb     # Jupyter Notebook (recommended)
├── README.md
├── cifar-10-batches-py/             # Dataset folder (must be downloaded separately)
📊 Sample Output
Final Accuracy: ~74.73%

Confusion Matrix:

[[318  51  49]
 [ 44 292  38]
 [ 87  35 289]]
Precision (per class): 0.70 - 0.77

Loss Curve: Smooth convergence across epochs ✅

🛠️ How to Run (Jupyter Notebook)
Open sapienproject.ipynb file

Run all cells from top to bottom

Output will show accuracy, metrics, and the training loss curve

Modify batch size, epochs, or learning rate to experiment!


🔮 Future Improvements
Add support for all 10 CIFAR-10 classes

Add batch normalization and dropout

Allow saving and loading trained weights

Build a simple GUI to test predictions manually

🙋‍♂️ Author
✍️ Written by a student exploring ML fundamentals

Feel free to fork, star, or suggest improvements!


PROBLEM STATEMENT 2
# Real-Time Object Detection using YOLOv5 and Webcam

## 📌 Project Overview
This project implements a real-time object detection pipeline using the YOLOv5 deep learning model, capturing live input directly from the **webcam**. The model detects and classifies objects in real time, displaying bounding boxes with labels and confidence scores over the live feed. This can be used for real-time surveillance, smart automation systems, or interactive applications.

## 🛠️ Setup Instructions

### 1. Clone the Repository

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Create and Activate Virtual Environment (Optional but recommended)
bash
python -m venv yolov5_env

# Windows:
yolov5_env\Scripts\activate

# Mac/Linux:
source yolov5_env/bin/activate
3. Install Dependencies

pip install -r requirements.txt
Or install manually if not using a requirements file:

pip install torch torchvision opencv-python matplotlib

4. Download YOLOv5

git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

5. Run the Notebook
Open the .ipynb file and run all cells:

jupyter notebook Real-timeObjectDetectionPipeline.ipynb

