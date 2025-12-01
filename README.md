Helmet & License Plate Detection with OCR
    Real-time violation detection using YOLOv8 + EasyOCR / PaddleOCR

    This project detects helmets, no-helmet violations, and license plates, then extracts the plate text using OCR.
    It is designed for smart surveillance, traffic monitoring, and college/industry safety enforcement systems.

Demo

examples/<br>
   input.jpg<br>
   output.jpg<br>

Features<br>
   - Helmet Detection<br>
   - No Helmet Detection<br>
   - License Plate Detection<br>
   - OCR Text Extraction (EasyOCR)<br>
   - Fast inference (GPU / GTX 1650 supported)<br>
   - YOLOv8-based custom training<br>
   - Clean modular project structure<br>
   - Automatic annotation saving (output images)<br>


Project Structure

project/<br>
│<br>
├── train.py                 # training script<br>
├── inference.py             # helmet + LP detection + OCR<br>
├── data.yaml                # dataset YAML<br>
├── requirements.txt         # dependencies<br>
├── README.md                # documentation (this file)<br>
│<br>
├── examples/                # sample outputs<br>
│     ├── input.jpg<br>
│     ├── output.jpg<br>
│
├── models/                  # recommended for toring best.pt<br>
│     ├── helmet_lp_best.pt<br>
│<br>
└── utils/                   # optional helpers (preprocessing, etc.)<br>

Installation

- Clone the repository
- git clone https://github.com/yourusername/helmet-lp-ocr.git
- cd helmet-lp-ocr

Install dependencies
- pip install -r requirements.txt



Training Your Model

    Your training script:

    python train.py --data data.yaml --weights yolov8n.pt 


    Training results will be saved to:

    runs/train/helmet_lp_detector/


    The best model will be:

    runs/train/helmet_lp_detector/weights/best.pt


Running Inference (Helmet + LP OCR)

    Use your new inference.py:

    python inference.py --source path/to/image.jpg --save


Output:

- Annotated image saved as output_image.jpg
- OCR text printed in terminal

  
Example Output
   - Saved annotated image: output_image.jpg
   - License Plate → MH12AB1234
   - No helmet detected

  
Dataset

Dataset should include:

datasets/<br>
 ├── images/<br>
 │     ├── train/<br>
 │     ├── val/<br>
 └── labels/<br>
 |     ├── train/<br>
 |     ├── val/<br>


Classes in data.yaml:

names:
  - Helmet
  - License plate
  - no helmet


Contributing

    Pull requests are welcome! Appreciate credit if used
    If you want to improve detection, add datasets, or enhance OCR—feel free to contribute.
