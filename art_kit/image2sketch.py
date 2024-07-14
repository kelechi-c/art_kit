import cv2


def sketchify(input_img: str, output_path: str, display_sketch: bool = False):
    image = cv2.imread(input_img)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256.0)

    if display_sketch:
        cv2.imshow("Sketch", sketch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite(output_path, sketch)

    return sketch
