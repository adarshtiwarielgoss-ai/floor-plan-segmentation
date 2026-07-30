import cv2
import numpy as np

def extract_mask(result):

    """
    Extract semantic segmentation mask from YOLO result.
    """

    if result.semantic_mask is None:
        return None

    mask = result.semantic_mask.data.cpu().numpy().astype(np.uint8)

    return mask


def clean_mask(mask):

    """
    Apply morphology operations to remove noise.
    """

    binary = mask * 255

    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    return binary



def find_polygons(binary):

    """
    Extract contours from binary mask.
    """

    contours, hierarchy = cv2.findContours(

        binary,

        cv2.RETR_TREE,

        cv2.CHAIN_APPROX_SIMPLE

    )

    return contours


def simplify_polygon(contour):

    """
    Reduce polygon points.
    """

    epsilon = 0.002 * cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(

        contour,

        epsilon,

        True

    )

    return approx



def filter_contours(contours, min_area=100):

    """
    Remove small contours.
    """

    filtered = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area >= min_area:
            filtered.append(contour)

    return filtered


def polygon_to_dict(contour, polygon_id):

    """
    Convert polygon into JSON format.
    """

    contour = simplify_polygon(contour)

    points = contour.reshape(-1, 2).tolist()

    x, y, w, h = cv2.boundingRect(contour)

    area = float(cv2.contourArea(contour))

    return {

        "id": polygon_id,

        "area": area,

        "bbox": {

            "x": int(x),

            "y": int(y),

            "width": int(w),

            "height": int(h)

        },

        "points": points

    }



def create_prediction_json(result):

    """
    Convert YOLO Semantic result to JSON.
    """

    mask = extract_mask(result)

    if mask is None:

        return {

            "success": False,

            "message": "No mask detected.",

            "polygons": []

        }

    binary = clean_mask(mask)

    contours = find_polygons(binary)

    contours = filter_contours(contours)

    polygons = []

    for idx, contour in enumerate(contours, start=1):

        polygons.append(
            polygon_to_dict(contour, idx)
        )

    height, width = binary.shape

    return {

        "success": True,

        "image": {

            "width": width,

            "height": height

        },

        "total_polygons": len(polygons),

        "polygons": polygons

    }



