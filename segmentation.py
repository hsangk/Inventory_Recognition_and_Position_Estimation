import cv2
import numpy as np

import torch
import torchvision

from GSA.GroundingDINO.groundingdino.util.inference import Model
from GSA.segment_anything.segment_anything import sam_model_registry, SamPredictor

def segment(image, grounding_dino_model, sam_predictor):
    CLASSES = ["paper of QR code"]
    # CLASSES = ["black in QR code"]
    # CLASSES = ["paper"]
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8

    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )

    # NMS post process
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    # convert detections to masks
    detections.mask = convert_detection2mask(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    h, w = image.shape[0], image.shape[1]
    seg_bool = np.full((h, w), False, dtype=bool)

    for img in detections.mask:
        seg_bool[img] = True

    seg_image = np.where(seg_bool, 255, 0).astype(np.uint8)

    return seg_image

# Prompting SAM with detected boxes
def convert_detection2mask(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


