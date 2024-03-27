import cv2
import numpy as np

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255)
}

def draw_mask(image, mask, alpha):
        color_mask = np.zeros(image.shape, np.uint8)
        color_mask[:,:] = (0, 0, 255)
        color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)
        cv2.addWeighted(color_mask, alpha, image, 1, 0, image)

def draw_object_labels(save_path, img, classes, bboxs, masks):
    cv_img = img.numpy()
    cv_img = np.transpose(cv_img, (1,2,0))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, _ = cv_img.shape

    np_classes = classes.numpy()
    np_bboxs = bboxs.numpy()
    np_masks = masks.numpy()

    for cls, bbox, mask in zip(np_classes, np_bboxs, np_masks):
        top_left = (int((bbox[0] - bbox[2]/2) * w), int((bbox[1] - bbox[3]/2) * h))
        bottom_right = (int((bbox[0] + bbox[2]/2) * w), int((bbox[1] + bbox[3]/2) * h))
        cv2.rectangle(cv_img, top_left, bottom_right, (255, 0, 0), 1)
        cv2.putText(cv_img, str(cls), top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 2)
        draw_mask(cv_img, mask, 0.5)
    
    print("Saving", save_path)
    cv2.imwrite(save_path, cv_img)


if __name__ == "__main__":
    pass