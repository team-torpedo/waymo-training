# Dataset

## Source and license

The pipeline reads the **Waymo Open Dataset perception data in the modular v2 (Parquet) format**,
hosted in `gs://waymo_open_dataset_v_2_0_0`. You must register at <https://waymo.com/open/> and
accept Waymo's license (non-commercial use). The data is never redistributed by this repo.

## Layout on disk

Files mirror the bucket: `data/raw/<training|validation>/<component>/<segment>.parquet`.

| Component             | Used for   | Relevant columns                                                                                                         |
| --------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------ |
| `camera_image`        | both tasks | `key.segment_context_name`, `key.frame_timestamp_micros`, `key.camera_name`, `[CameraImageComponent].image` (JPEG bytes) |
| `camera_box`          | detection  | `[CameraBoxComponent].box.center.x/y`, `.box.size.x/y` (pixels), `.type`                                                 |
| `camera_segmentation` | lane       | `[CameraSegmentationLabelComponent].panoptic_label` (uint16 PNG), `.panoptic_label_divisor`                              |

Camera ids: 1 FRONT, 2 FRONT_LEFT, 3 FRONT_RIGHT, 4 SIDE_LEFT, 5 SIDE_RIGHT.
2D box types: 1 vehicle, 2 pedestrian, 3 sign, 4 cyclist.

The reader matches columns by suffix, so small naming differences between dataset releases do not break it.

## Object detection labels

`camera_box` boxes are converted to YOLO format (`class cx cy w h`, normalised, clipped to the image).
Frames that have **no** `camera_box` rows are skipped by default (`require_labels: true`) because an
unlabeled frame is not the same as a frame with no objects.

## Lane labels — read this

Waymo's camera labels do **not** include lane lines or lane polylines. What the dataset provides is
**camera panoptic segmentation** (`camera_segmentation`) with semantic classes that include
`LANE_MARKER` (21), `ROAD_MARKER` (22) and `ROAD` (20). This repo builds a binary lane mask by merging
the classes listed in `lane.semantic_classes` (default `[21]`).

Consequences:

- Segmentation labels exist only for a **subset** of segments/frames, so the lane dataset is much smaller
  than the detection dataset. The download stage picks segments that contain the needed component.
- The output is a pixel mask, not parametric lane curves. Fit curves/polylines downstream if your
  planner needs them.
- The Waymo **Motion** dataset has HD-map lane polylines in world coordinates, but they are not
  image-space labels and are out of scope here.

## Which segments get downloaded

For each split the download stage lists the available segments per component, keeps those that have
**all** components a task needs, sorts them and takes the first `max_segments`. Segments are therefore
reproducible between runs and between teammates.

Approximate sizes: a segment's `camera_image` file is on the order of hundreds of MB. Check with
`gsutil du -sh gs://waymo_open_dataset_v_2_0_0/training/camera_image/<segment>.parquet`.

## Unverified assumptions

The reader was written against the published v2 schema and tested on synthetic parquet files with the
same column names and encodings. Run the pipeline on a small real subset first
(`max_segments: {train: 1, val: 1}`) and inspect `data/processed/**/manifest.json`, a few images and
masks before launching long training runs.
