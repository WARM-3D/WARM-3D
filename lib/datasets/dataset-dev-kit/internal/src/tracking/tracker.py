import numpy as np

from typing import List, Optional, Dict
import dataclasses as dc

from src.tracking.sort import Sort
from src.utils.detection import Detection


@dc.dataclass
class BoxTracker:
    """Manages a list of Detections and a SORT tracker to update the detection state."""

    # Dict from tracker ID to detection
    tracked_detections: Dict[int, Detection] = dc.field(default_factory=dict)

    # Dict from detection index to detection for last update()-call
    tracked_detections_by_index: Dict[int, Detection] = dc.field(default_factory=dict)

    # Underlying SORT Kalman Tracker Impl
    tracker: Optional[Sort] = None

    # Flag to switch to screen-space 2D operation
    screen_space_tracking: bool = False

    # Maximum number of frames a tracked object will survive without a detected mask.
    max_prediction_age: int = 0

    def __post_init__(self):
        """Dataclass init function."""
        self.reset()

    def reset(self):
        """Re-instantiate the underlying SORT tracker and forget tracked detections."""
        self.tracker = Sort(max_age=self.max_prediction_age, min_hits=1, iou_threshold=0.1)
        self.tracked_detections.clear()
        self.tracked_detections_by_index.clear()

    def update(self, predictions: List[Detection], time_delta_seconds: float = 0.1) -> List[Detection]:
        """Associate a list of detection predictions with IDs
        and stabilize position estimates."""
        if not self.tracker:
            return predictions
        self.tracked_detections_by_index.clear()
        if not self.screen_space_tracking:
            boxes = [v.as_2d_bev_square() for v in predictions]
        else:
            boxes = [v.bbox_2d for v in predictions]
        tracker_per_index, dead_tracker_ids = self.tracker.update(np.array(boxes))

        # Register detections for new trackers, update yaw known detections
        for detection_index, tracker in tracker_per_index.items():
            if tracker.id not in self.tracked_detections:
                if not self.screen_space_tracking:
                    predictions[detection_index].id = tracker.id
                self.tracked_detections[tracker.id] = predictions[detection_index]
                self.tracked_detections_by_index[detection_index] = predictions[detection_index]
            else:
                self.tracked_detections_by_index[detection_index] = self.tracked_detections[tracker.id]
                v = self.tracked_detections[tracker.id]
                # Adapt specific time-variant fields from the prediction
                # before it is forgotten and replaced with the tracked detection.
                v_pred = predictions[detection_index]
                v.bbox_2d = v_pred.bbox_2d
                if not self.screen_space_tracking:
                    # In BEV tracking, the prediction carries more info that must be adopted
                    v.uuid = v_pred.uuid
                    v.original_detected_object_ros_msg = v_pred.original_detected_object_ros_msg
                    v.yaw_history.insert(0, v.yaw)
                    v.yaw = v_pred.yaw
                    v.color = v_pred.color
                    v.num_lidar_points = v_pred.num_lidar_points
                    v.score = v_pred.score
                    v.occlusion_level = v_pred.occlusion_level
                    v.sensor_id = v_pred.sensor_id
                    v.overlap = v_pred.overlap

        # Remove detections for dead trackers
        for removed in dead_tracker_ids:
            self.tracked_detections.pop(removed, None)

        # Update tracked 3D detection locations by their tracker info from SORT
        if not self.screen_space_tracking:
            sort_trackers = {sort_tracker.id: sort_tracker for sort_tracker in self.tracker.trackers}
            for tracker_id, detection in self.tracked_detections.items():
                assert tracker_id in sort_trackers
                state = sort_trackers[tracker_id].get_state().flatten()
                position = state[0:2] + (state[2:4] - state[0:2]) * 0.5  # aabb center
                detection.location[(0, 1), (0, 0)] = position
                detection.velocity = sort_trackers[tracker_id].kf.x[4:6].flatten() / time_delta_seconds
        return list(self.tracked_detections.values())
