1
import cv2
img = cv2.imread('image.jpg')
print("Image:", img.dtype, img.shape, img.size)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    print("Video Frame:", frame.dtype, frame.shape, frame.size)
    cv2.imshow('Video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()

2
import cv2  
img = cv2.imread('input.jpg')  
cv2.imwrite('output.jpg', img)  
print("Image read and saved successfully.")
cap = cv2.VideoCapture(0)
print("Press 'q' to quit the webcam window.")
while cap.isOpened():
    ret, frame = cap.read()  
    if not ret:
        break
    cv2.imshow('Webcam Video', frame)  
    If cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


3
import cv2  
img = cv2.imread('image.jpg')  
print("Before:")
print(f"{'Property':<10} {'Value'}")
print(f"{'Shape':<10} {img.shape}")   
print(f"{'Size':<10} {img.size}")     
print(f"{'Dtype':<10} {img.dtype}")   
cv2.imshow("Original Image", img)     
resized = cv2.resize(img, (200, 200))  
cv2.imshow("Resized Image", resized)
cropped = img[50:200, 100:300]         
cv2.imshow("Cropped Image", cropped)
print("\nAfter (Resized):")
print(f"{'Shape':<10} {resized.shape}")
print(f"{'Size':<10} {resized.size}")
print(f"{'Dtype':<10} {resized.dtype}")
print("\nAfter (Cropped):")
print(f"{'Shape':<10} {cropped.shape}")
print(f"{'Size':<10} {cropped.size}")
print(f"{'Dtype':<10} {cropped.dtype}")
cv2.waitKey(0)               
cv2.destroyAllWindows()      




4
import cv2
img = cv2.imread('image.jpg')  
cv2.imshow("Original (BGR)", img)
print("Original Image:")
print("Color Space: BGR")                 
print("Shape:", img.shape)                
print("Size:", img.size)                  
print("Dtype:", img.dtype)                
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("RGB Image", rgb)
print("\nRGB Image:")
print("Shape:", rgb.shape)
print("Size:", rgb.size)
print("Dtype:", rgb.dtype)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray)
print("\nGrayscale Image:")
print("Shape:", gray.shape)
print("Size:", gray.size)
print("Dtype:", gray.dtype)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Image", hsv)
print("\nHSV (MVT) Image:")
print("Shape:", hsv.shape)
print("Size:", hsv.size)
print("Dtype:", hsv.dtype)
cv2.waitKey(0)
cv2.destroyAllWindows()


5
import cv2
import numpy as np
def add_salt_pepper_noise(img, salt=0.02, pepper=0.02):
    noisy = img.copy()
    h, w = img.shape[:2]
    # Salt noise (white)
    for _ in range(int(salt * h * w)):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        noisy[y, x] = 255
    # Pepper noise (black)
    for _ in range(int(pepper * h * w)):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        noisy[y, x] = 0
    return noisy
img = cv2.imread('image.jpg')
sp_img = add_salt_pepper_noise(img)
# Blurring on normal image
blur1 = cv2.blur(img, (5,5))
blur2 = cv2.medianBlur(img, 5)
blur3 = cv2.GaussianBlur(img, (5,5), 0)
blur4 = cv2.bilateralFilter(img, 9, 75, 75)
# Blurring on noisy image
sp_blur1 = cv2.blur(sp_img, (5,5))
sp_blur2 = cv2.medianBlur(sp_img, 5)
sp_blur3 = cv2.GaussianBlur(sp_img, (5,5), 0)
sp_blur4 = cv2.bilateralFilter(sp_img, 9, 75, 75)
cv2.imshow('Original', img)
cv2.imshow('Salt-Pepper Noise', sp_img)
cv2.imshow('Blur Average', blur1)
cv2.imshow('Median Blur', blur2)
cv2.imshow('Gaussian Blur', blur3)
cv2.imshow('Bilateral Filter', blur4)
cv2.imshow('Noisy Blur Average', sp_blur1)
cv2.imshow('Noisy Median Blur', sp_blur2)
cv2.imshow('Noisy Gaussian Blur', sp_blur3)
cv2.imshow('Noisy Bilateral Filter', sp_blur4)

6
import cv2
img1 = cv2.imread('coins.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
_, th_bin = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
_, th_bin_inv = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Simple Threshold Binary', th_bin)
cv2.imshow('Simple Threshold Binary Inv', th_bin_inv)
img2 = cv2.imread('handwritten.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
adap_bin = cv2.adaptiveThreshold(gray2, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
adap_bin_inv = cv2.adaptiveThreshold(gray2, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
cv2.imshow('Adaptive Threshold Binary', adap_bin)
cv2.imshow('Adaptive Threshold Binary Inv', adap_bin_inv)






7
import cv2
import numpy as np
img = np.zeros((400, 400, 3), dtype=np.uint8)
img.fill(255)  
cv2.circle(img, (100, 100), 50, (255, 0, 0), -1)
cv2.line(img, (50, 300), (350, 300), (0, 255, 0), 5)
cv2.rectangle(img, (200, 50), (350, 150), (0, 0, 255), -1)  # filled rectangle
cv2.putText(img, 'YourName123', (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.imshow('Drawing', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('drawn_image.jpg', img)

8
import cv2
img = cv2.imread('object.jpg', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, threshold1=100, threshold2=200)
cv2.imshow('Original', img)
cv2.imshow('Edges - Canny', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

9
import cv2
img = cv2.imread('objects.jpg', cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_color, contours, -1, (0,255,0), 2)
cv2.imshow('Contours', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

12
import cv2
import numpy as np
img = cv2.imread('image.jpg', 0)
edges = cv2.Canny(img, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
if lines is not None:
  for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
corners = cv2.cornerHarris(img, 2, 3, 0.04)
img[corners > 0.01 * corners.max()] = [0, 0, 255]
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

13
import cv2
def match_keypoints(img1_path, img2_path):
  img1 = cv2.imread(img1_path, 0)
  img2 = cv2.imread(img2_path, 0)
  sift = cv2.SIFT_create()
  kp1, des1 = sift.detectAndCompute(img1, None)
  kp2, des2 = sift.detectAndCompute(img2, None)
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1, des2, k=2)
  good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
  img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
  cv2.imshow("Matches", img_matches)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
match_keypoints('image1.jpg', 'image2.jpg')

14
import cv2
import numpy as np
def match_transposed_features(image_path):
  img = cv2.imread(image_path, 0)
  img_t = np.transpose(img)
  sift = cv2.SIFT_create()
  kp1, des1 = sift.detectAndCompute(img, None)
  kp2, des2 = sift.detectAndCompute(img_t, None)
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1, des2, k=2)
  good = [m for m, n in matches if m.distance < 0.75 * n.distance]
  match_img = cv2.drawMatches(img, kp1, img_t, kp2, good, None, flags=2)
  cv2.imshow("Matches", match_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
match_transposed_features('image.jpg')

10
import cv2
import numpy as np
image = cv2.imread('your_image.jpg')
if image is None:
    print("Error: Image not found. Make sure 'your_image.jpg' exists.")
    exit()

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
   x, y, w, h = cv2.boundingRect(contour)
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    cv2.putText(image, 'Blue', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow('Detected Blue Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

11
import cv2
import numpy as np
image_path = 'your_image.jpg'
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not read the image. Make sure the file '{image_path}' is in the correct path.")
    exit()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = np.float32(gray_image)
corner_scores = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)
threshold = 0.01 * corner_scores.max()
image[corner_scores > threshold] = [0, 0, 255]
cv2.imshow('Detected Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()












