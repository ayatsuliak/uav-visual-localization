import numpy as np


def transform_point(point_xy, H):
    point = np.asarray(point_xy, dtype=np.float32).reshape(1, 1, 2)
    transformed = cv2_perspective_transform(point, H)
    return transformed.reshape(2)


def cv2_perspective_transform(point, H):
    import cv2
    return cv2.perspectiveTransform(point, H)


def estimate_global_position(frame_shape, homography, tile_x, tile_y, tile_shape,
                             matcher_image_size=(512, 512)):
    """Estimate frame-center position in original map coordinates.

    Homography maps processed frame coordinates to processed tile coordinates.
    The tile is then converted from matcher resolution to its original map pixels.
    """
    if homography is None:
        return None
    h, w = frame_shape[:2]
    center = np.array([[[w / 2.0, h / 2.0]]], dtype=np.float32)
    tile_point = cv2_perspective_transform(center, homography).reshape(2)
    tile_h, tile_w = tile_shape[:2]
    sx = tile_w / matcher_image_size[0]
    sy = tile_h / matcher_image_size[1]
    return np.array([tile_x + tile_point[0] * sx, tile_y + tile_point[1] * sy], dtype=float)
