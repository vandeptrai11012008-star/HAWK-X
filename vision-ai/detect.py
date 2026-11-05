"""HAWK-X native offline detector entry point.

Example: py vision-ai/detect.py --image path/to/photo.jpg
Requires a checkpoint trained with train_native.py.
"""
from predict_native import main


if __name__ == "__main__":
    main()
