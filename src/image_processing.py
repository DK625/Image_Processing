import cv2
import numpy as np

def negative_transformation(image):
    return 255 - image

def thresholding(image, threshold):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def logarithmic_transformation(image):
    c = 255 / np.log(1 + np.max(image))
    log_transformed = c * np.log(1 + image)
    return log_transformed.astype(np.uint8)

def exponential_transformation(image, alpha):
    return np.uint8(np.clip(alpha * (np.exp(image / 255 * alpha) - 1), 0, 255))

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def weighted_average_filter(image):
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(image, -1, kernel)

def median_filter(image):
    return cv2.medianBlur(image, 5)

def roberts_operator(image):
    roberts_cross_v = np.array([[ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0,-1]])

    roberts_cross_h = np.array([[ 0, 0, 0],
                                [ 0, 0, 1],
                                [ 0,-1, 0]])

    vertical_edge = cv2.filter2D(image, -1, roberts_cross_v)
    horizontal_edge = cv2.filter2D(image, -1, roberts_cross_h)

    return np.sqrt(np.square(vertical_edge) + np.square(horizontal_edge)).astype(np.uint8)

def sobel_operator(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    return np.uint8(magnitude)

def prewitt_operator(image):
    prewitt_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    prewitt_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    magnitude = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))
    return np.uint8(magnitude)

def laplacian_operator(image):
    return cv2.Laplacian(image, cv2.CV_8U)

def canny_edge_detection(image):
    return cv2.Canny(image, 100, 200)

def otsu_threshold(image):
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def dilation(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def erosion(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def closing(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def process_image(image_path, algorithm, *args):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if algorithm == 'negative':
        processed_image = negative_transformation(original_image)
    elif algorithm == 'threshold':
        threshold_value = args[0] if args else 127
        processed_image = thresholding(original_image, threshold_value)
    elif algorithm == 'logarithm':
        processed_image = logarithmic_transformation(original_image)
    elif algorithm == 'exponential':
        alpha = args[0] if args else 1.0
        processed_image = exponential_transformation(original_image, alpha)
    elif algorithm == 'histogramEqualization':
        processed_image = histogram_equalization(original_image)
    elif algorithm == 'weightedAverageFilter':
        processed_image = weighted_average_filter(original_image)
    elif algorithm == 'medianFilter':
        processed_image = median_filter(original_image)
    elif algorithm == 'robertsOperator':
        processed_image = roberts_operator(original_image)
    elif algorithm == 'sobelOperator':
        processed_image = sobel_operator(original_image)
    elif algorithm == 'prewittOperator':
        processed_image = prewitt_operator(original_image)
    elif algorithm == 'laplacianOperator':
        processed_image = laplacian_operator(original_image)
    elif algorithm == 'cannyEdgeDetection':
        processed_image = canny_edge_detection(original_image)
    elif algorithm == 'otsuThreshold':
        processed_image = otsu_threshold(original_image)
    elif algorithm == 'dilation':
        processed_image = dilation(original_image)
    elif algorithm == 'erosion':
        processed_image = erosion(original_image)
    elif algorithm == 'closing':
        processed_image = closing(original_image)
    elif algorithm == 'opening':
        processed_image = opening(original_image)
    else:
        raise ValueError("Invalid algorithm")

    return processed_image

if __name__ == "__main__":
    input_image_path = "path/to/your/image.jpg"  # Thay đổi đường dẫn tới ảnh của bạn
    selected_algorithm = "negative"  # Chọn thuật toán, có thể thay đổi thành các giá trị khác
    output_image = process_image(input_image_path, selected_algorithm)

    cv2.imshow("Original Image", cv2.imread(input_image_path))
    cv2.imshow("Processed Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
