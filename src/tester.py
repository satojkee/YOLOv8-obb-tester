import os
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from munch import DefaultMunch


__all__ = ('TestUtil',)


class TestUtil:
    def __init__(self, setup: DefaultMunch) -> None:
        self.model = YOLO(setup.storage.model)
        # TestUtil setup, includes: [predict, visual, storage] settings
        self.setup = setup
        # Assets directory location (files to process)
        self.assets = self.setup.storage.assets
        # Output directory location (uses for saving processed files)
        self.output = os.path.join(
            self.setup.storage.output,
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        )

        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def run(self) -> None:
        """This method makes your YOLOv8 model process assets.
        Creates a collage images that contains original and processed image both.
        Doesn't use built-in YOLOv8 visualization methods. [result[0].plot()]

        Visual:
            - Bounding box
            - Class name
            - Angle of rotation
            - Confidence score

        Label:
            - Class
            - Angle
            - Confidence

        :return: None
        """
        set_ = os.listdir(self.assets)

        for file in set_:
            initial = cv2.imread(os.path.join(self.assets, file))
            processed = initial.copy()

            result = self.model.predict(initial, **self.setup.predict)[0]
            for idx, bbox in enumerate(result.obb.xywhr):
                # Transfers tensor on cpu and converts values to int
                cx, cy, w, h, r = bbox.cpu().numpy()
                angle = r * 180 / np.pi
                rect = ((int(cx), int(cy)), (int(w), int(h)), angle)

                # Preparations before visualization
                # Calculating lbox and label coordinates and size
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = (
                    f'{result.names[result.obb.cls[idx].item()]} | '
                    f'{angle:.2f} deg | '
                    f'{result.obb.conf[idx].item():.2f}'
                )
                t_x, t_y = int(cx - 64), int(cy + h / 2 + 20)
                t_w, t_h = cv2.getTextSize(
                    text,
                    font,
                    self.setup.visual.label.fontScale,
                    self.setup.visual.label.thickness
                )[0]

                # Bounding box and Label visualization, includes: [bbox, lbox, label]
                cv2.drawContours(
                    processed,
                    [cv2.boxPoints(rect).astype(int)],
                    0,
                    **self.setup.visual.bbox
                )
                cv2.rectangle(
                    processed,
                    (t_x, t_y),
                    (t_x + t_w, t_y + t_h),
                    **self.setup.visual.lbox,
                )
                cv2.putText(
                    processed,
                    text,
                    (t_x, t_y + t_h),
                    font,
                    **self.setup.visual.label
                )

            # Converts BGR (cv2) format to RGB (pillow)
            initial = cv2.cvtColor(initial, cv2.COLOR_BGR2RGB)
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

            # Creates a collage to save an initial image and processed one together
            collage = Image.new(
                'RGB',
                (processed.shape[1], processed.shape[0] * 2)
            )

            # Pastes both images inside of collage and saves it
            collage.paste(Image.fromarray(processed), (0, 0))
            collage.paste(Image.fromarray(initial), (0, processed.shape[0]))

            collage.save(os.path.join(self.output, file))
