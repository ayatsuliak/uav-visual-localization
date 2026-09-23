import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description="Interactively select a map ROI.")
    parser.add_argument("--image", default="data/raw/maps/university_map.png")
    parser.add_argument("--max-width", type=int, default=1200)
    parser.add_argument("--max-height", type=int, default=800)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Cannot load image: {args.image}")
    h, w = image.shape[:2]
    scale = min(args.max_width / w, args.max_height / h, 1.0)
    display = cv2.resize(image, (int(w * scale), int(h * scale)))
    roi = cv2.selectROI("Select ROI", display, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, rw, rh = roi
    original = tuple(int(v / scale) for v in (x, y, rw, rh))
    print(f"ROI in original image: {original}")


if __name__ == "__main__":
    main()
