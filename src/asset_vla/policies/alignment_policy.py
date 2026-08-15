import dataclasses

import einops
import numpy as np

from asset_vla import transforms


def make_my_example() -> dict:
    """Creates a random input example for the alignment policy."""
    return {
        "state": np.random.rand(3),
        "right_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "back_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "upper_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@staticmethod
def convert_camera_to_world(camera_extrinsics):
    R = camera_extrinsics[:, :3, :3]
    t = camera_extrinsics[:, :3, 3]         

    # invert
    R_c2w = R.transpose(-1, -2)    # (N, 3, 3)
    t_c2w = -np.einsum("...ij,...j->...i", R_c2w, t)  # (N, 3)

    # build camera to world extrinsics
    T_c2w = np.zeros_like(camera_extrinsics)
    T_c2w[:, :3, :3] = R_c2w
    T_c2w[:, :3, 3] = t_c2w
    T_c2w[:, 3, 3] = 1.0

    return T_c2w

@dataclasses.dataclass(frozen=True)
class AlignmentInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """


    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        image_0 = _parse_image(data["right_image"])
        image_1 = _parse_image(data["back_image"])
        image_2 = _parse_image(data["upper_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": image_0,
                "left_wrist_0_rgb": image_1,
                "right_wrist_0_rgb": image_2,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_
            },
        }

        # Reference views: ground-truth renderings of the aligned pair, stacked by the data
        # loader as (num_views, C, H, W). They are unpacked into their own image keys so the
        # model sees them alongside the camera views, in the order the prompt announces them.
        rendered_images = data.get("rendered_images")
        if rendered_images is not None:
            for i in range(rendered_images.shape[0]):
                inputs["image"][f"render_{i}_rgb"] = _parse_image(rendered_images[i])
                inputs["image_mask"][f"render_{i}_rgb"] = np.True_

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["translations"] = data["actions"][..., :3]
            inputs["rotations"] = data["actions"][..., 3:9]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "camera_extrinsics" in data:
            # world to camera
            # convert to camera to world
            world_to_camera_ext = data["camera_extrinsics"]
            camera_to_world_ext = convert_camera_to_world(world_to_camera_ext)
            inputs["camera_extrinsics"] = camera_to_world_ext

        # Sample identity. Only present when reading from the dataset, not at inference time.
        # The model ignores these; they let the evaluation script attribute predictions to assets.
        for key in ("asset_name", "part_idx"):
            if key in data:
                inputs[key] = data[key]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlignmentOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # TODO fix actions? parse translation and rotations
        return {"actions": np.asarray(data["actions"][:, :9])}
