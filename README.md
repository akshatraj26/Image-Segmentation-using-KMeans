# first you have to import dependencies

```
pip install opencv-python
```

## then you change the file path according to your system

```
from image_segmentation import image_segmentation

# image_path, criteria, k: int
# you can put any k value
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
object = image_segmentation(image_path,criteria, k=4 )
plt.imshow(object)

```
