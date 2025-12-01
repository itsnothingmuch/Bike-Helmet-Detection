# train.py
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data.yaml', help='path to data.yaml')
    parser.add_argument('--weights', default='yolov8n.pt', help='pretrained weights')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--name', default='helmet_lp_detector')
    parser.add_argument('--device', default='0', help='GPU id, e.g., 0 or 0,1 or cpu')
    args = parser.parse_args()

    model = YOLO(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project='runs/train',
        name=args.name,
        device=args.device  # "0" for GPU, "cpu" if GPU unavailable
    )

if __name__ == "__main__":
    main()
