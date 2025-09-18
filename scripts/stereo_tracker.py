# tracking_util_47_ocsort.py
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
from filterpy.kalman import KalmanFilter
import cv2
from scipy.optimize import linear_sum_assignment
import logging
import yaml
import pandas as pd
import os

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
NAN3 = np.array([np.nan, np.nan, np.nan], dtype=float)

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]])
        self.kf.R *= 1.0
        self.kf.R[2:,2:] *= 10.
        self.kf.P *= 1.0
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q *= 1.0
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self._F0 = self.kf.F.copy()
        self._H0 = self.kf.H.copy()
        self._P0 = self.kf.P.copy()
        self._Q0 = self.kf.Q.copy()
        self._R0 = self.kf.R.copy()

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.kf.x[4:] = 0.0

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def reset(self, bbox, mode="full"):
        if mode == "full":
            self.kf.F = self._F0.copy()
            self.kf.H = self._H0.copy()
            self.kf.P = self._P0.copy()
            self.kf.Q = self._Q0.copy()
            self.kf.R = self._R0.copy()
        elif mode == "inflate":
            self.kf.P *= 1e3
        self.kf.x[:] = 0.0
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.kf.x[4:] = 0.0
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.history.clear()

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

@dataclass
class TrackerState:
    tracker: KalmanBoxTracker
    hand_history: List[List[int]] = field(default_factory=list)
    frame_count_history: List[int] = field(default_factory=list)
    pt3d_history: List[np.ndarray] = field(default_factory=list)

class StereoTracker:
    def __init__(self, P1, P2, Q, image_shape,
                 depth_scale=[1.0], calib_scale=1.0, calib_bias=0.0, max_age=10, min_hits=3, 
                 iou_thr1=0.3, iou_thr2=0.15,
                 max_depth=2.0,
                 oc_delta_t: int = 2,
                 inertia: float = 0.2,
                 use_dir_gate: bool = True,
                 use_speed_penalty: bool = True,
                 use_stereo_gate: bool = True,
                 dir_gate_deg: float = 40.0,
                 speed_gate_px: float = 80.0,
                 disparity_consistency_thr: float = 40.0
                 ):
        self.Q = Q
        self.P1 = P1; self.P2 = P2
        self.calib_scale = float(calib_scale); self.calib_bias = float(calib_bias)
        self.image_shape = image_shape
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thr1 = float(iou_thr1); self.iou_thr2 = float(iou_thr2)
        self.max_depth = float(max_depth)
        self.frame_count = 0

        self.tracker_states = [{}, {}]   # 0:left, 1:right
        self.trk_pair_map   = {}         

        self.depth_scale = list(depth_scale)

        self.oc_delta_t = int(max(1, oc_delta_t))
        self.inertia = float(inertia)
        self.use_dir_gate = bool(use_dir_gate)
        self.use_speed_penalty = bool(use_speed_penalty)
        self.use_stereo_gate = bool(use_stereo_gate)
        self.dir_gate_deg = float(dir_gate_deg)
        self.speed_gate_px = float(speed_gate_px)
        self.disparity_consistency_thr = float(disparity_consistency_thr)

        self.logger = logging.getLogger("StereoTracker")
        self.logger.setLevel(logging.WARNING)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setLevel(logging.WARNING)
            h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            self.logger.addHandler(h)

    def _pairmap_link(self, id_a: int, id_b: int):
        id_a = int(id_a); id_b = int(id_b)
        self._unlink_all(id_a)
        self._unlink_all(id_b)
        self.trk_pair_map.setdefault(id_a, set()).add(id_b)
        self.trk_pair_map.setdefault(id_b, set()).add(id_a)

    def update(self, result_left, result_right): # Input: YOLO Detection Results
        
        self.frame_count += 1

        results = [result_left[0], result_right[0]]
        hand_boxes = [[], []]

        alive_ids = [set(), set()]
        
        for cam_idx in range(2):
            boxes = results[cam_idx].boxes.xyxy.cpu().numpy()
            classes = results[cam_idx].boxes.cls.cpu().numpy()
            hand_boxes[cam_idx] = boxes[classes == 0]

        # Stereo Matching Module for Optimal Object Pairing
        matched_tl_tr, unmatched_tl, unmatched_tr = self.stereo_matching_module(
            hand_boxes[0], hand_boxes[1], 0.01
        )

        # Temporal Data Association
        obs_preds = [self._obs_propagate_all(0), self._obs_propagate_all(1)]
        det2trk_all = [dict(), dict()]
        det_unmatched_all = [set(), set()]
        trk_unmatched_all = [set(), set()]
        for cam_idx in (0, 1):
            det_boxes_all = hand_boxes[cam_idx]
            trk_boxes_dict = {int(t): np.r_[b, 0.0] for t, b in obs_preds[cam_idx].items()}
            if len(det_boxes_all) == 0:
                det_unmatched_all[cam_idx] = set()
                trk_unmatched_all[cam_idx] = set(trk_boxes_dict.keys())
            elif len(trk_boxes_dict) == 0:
                det_unmatched_all[cam_idx] = set(range(len(det_boxes_all)))
                trk_unmatched_all[cam_idx] = set()
            else:
                m1, udet1, utrk1 = self.associate_detections_to_trackers_oc2(
                    detections=det_boxes_all,
                    trackers=trk_boxes_dict,
                    cam_idx=cam_idx,
                    iou_thr1=self.iou_thr1,
                    iou_thr2=self.iou_thr2
                )
                for d_local, trk_id in m1:
                    det2trk_all[cam_idx][int(d_local)] = int(trk_id)
                det_unmatched_all[cam_idx] = set(int(i) for i in udet1.tolist())
                trk_unmatched_all[cam_idx] = set(int(t) for t in utrk1.tolist())
        paired_trkids_this_frame = []
        matched_hands_trks = [None, None]

        # Stereo Tracker Management
        # Stereo Associated Object pairs
        for (tl_idx, tr_idx) in matched_tl_tr:
            assigned = {0: None, 1: None}

            # Process Associated Tracker-Object Pairs (in stereo pairs)
            for cam_idx, hand_idx in ((0, tl_idx), (1, tr_idx)):
                hbox = hand_boxes[cam_idx][hand_idx]
                if hand_idx in det2trk_all[cam_idx]:
                    trk_id = det2trk_all[cam_idx][hand_idx]
                    matched_hands_trks[cam_idx] = self._update_tracker(
                        cam_idx, trk_id, hbox, hand_idx, matched_hands_trks[cam_idx]
                    )
                    alive_ids[cam_idx].add(int(trk_id))
                    assigned[cam_idx] = trk_id
                    if hand_idx in det_unmatched_all[cam_idx]:
                        det_unmatched_all[cam_idx].discard(hand_idx)

            # Process Unassociated Objects with ReID or New ID (in stereo pairs)
            for cam_idx, hand_idx in ((0, tl_idx), (1, tr_idx)):
                if assigned[cam_idx] is not None:
                    continue
                hbox = hand_boxes[cam_idx][hand_idx]
                opposite_cam = 1 - cam_idx
                opposite_assigned = assigned[opposite_cam]
                reuse_id = None
                if opposite_assigned is not None:
                    reuse_id = self._get_recent_pairmate(cam_idx, opposite_assigned)
                # ReID
                if reuse_id is not None and reuse_id in self.tracker_states[cam_idx]:
                    state = self.tracker_states[cam_idx][reuse_id]
                    state.tracker.reset(hbox, mode="full")
                    alive_ids[cam_idx].add(int(reuse_id))
                    matched_hands_trks[cam_idx] = self._record_match_no_kf(
                        cam_idx, reuse_id, hbox, hand_idx, matched_hands_trks[cam_idx]
                    )
                    assigned[cam_idx] = reuse_id
                # New ID
                else:
                    new_id = self.create_tracker_state(cam_idx, hand_idx, hbox)
                    alive_ids[cam_idx].add(int(new_id))
                    if matched_hands_trks[cam_idx] is None or len(matched_hands_trks[cam_idx]) == 0:
                        matched_hands_trks[cam_idx] = np.array([[hand_idx, new_id]])
                    else:
                        matched_hands_trks[cam_idx] = np.vstack([matched_hands_trks[cam_idx], [hand_idx, new_id]])
                    assigned[cam_idx] = new_id

            # Stereo Tracker Association Map Update
            if assigned[0] is not None and assigned[1] is not None:
                self._pairmap_link(assigned[0], assigned[1])
                paired_trkids_this_frame.append((int(assigned[0]), int(assigned[1])))

        # Stereo Unassociated Objects
        lr_matched_idx = [
            set(matched_tl_tr[:,0].tolist()) if len(matched_tl_tr) else set(),
            set(matched_tl_tr[:,1].tolist()) if len(matched_tl_tr) else set()
        ]
        # Process Associated Tracker-Object Pairs (not in stereo pairs)
        for cam_idx in (0, 1):
            for det_idx, trk_id in det2trk_all[cam_idx].items():
                if det_idx in lr_matched_idx[cam_idx]:
                    continue
                hbox = hand_boxes[cam_idx][det_idx]
                matched_hands_trks[cam_idx] = self._update_tracker(
                    cam_idx, trk_id, hbox, det_idx, matched_hands_trks[cam_idx]
                )
                alive_ids[cam_idx].add(int(trk_id))
                if det_idx in det_unmatched_all[cam_idx]:
                    det_unmatched_all[cam_idx].discard(det_idx)
        # Process Unassociated Objects with New ID (not in stereo pairs)
        for cam_idx, unmatched in ((0, unmatched_tl), (1, unmatched_tr)):
            for hand_idx in unmatched:
                if hand_idx not in det_unmatched_all[cam_idx]:
                    continue
                hbox = hand_boxes[cam_idx][hand_idx]
                new_id = self.create_tracker_state(cam_idx, hand_idx, hbox)
                alive_ids[cam_idx].add(int(new_id))
                if matched_hands_trks[cam_idx] is None or len(matched_hands_trks[cam_idx]) == 0:
                    matched_hands_trks[cam_idx] = np.array([[hand_idx, new_id]])
                else:
                    matched_hands_trks[cam_idx] = np.vstack([matched_hands_trks[cam_idx], [hand_idx, new_id]])

        # Process Unassoiciated Trackers
        to_delete = [set(), set()]
        for cam_idx in (0, 1):
            for trk_id, st in list(self.tracker_states[cam_idx].items()):
                if st.tracker.time_since_update < self.max_age:
                    continue
                mates = self.trk_pair_map.get(int(trk_id), set())
                opp = 1 - cam_idx
                if any(int(m) in alive_ids[opp] for m in mates):
                    continue
                comp = self._collect_pair_component(cam_idx, int(trk_id))
                if any(int(tid) in alive_ids[c] for (c, tid) in comp):
                    continue
                for (c, tid) in comp:
                    to_delete[c].add(int(tid))

        # Record 3D points in stereo pairs
        l2r_now = {l: r for l, r in paired_trkids_this_frame}
        r2l_now = {r: l for l, r in paired_trkids_this_frame}
        updated = [
            {int(tid) for tid, st in self.tracker_states[0].items()
             if st.frame_count_history and st.frame_count_history[-1] == self.frame_count},
            {int(tid) for tid, st in self.tracker_states[1].items()
             if st.frame_count_history and st.frame_count_history[-1] == self.frame_count},
        ]

        for lid in list(updated[0]):
            stL = self.tracker_states[0].get(lid)
            if stL is None: 
                continue
            rid = l2r_now.get(lid, None)
            if rid is not None and rid in updated[1]:
                stR = self.tracker_states[1].get(rid)
                if stR is None:
                    stL.pt3d_history[-1] = NAN3.copy()
                    continue
                cL = box2center(stL.hand_history[-1]); cR = box2center(stR.hand_history[-1])
                p3 = self._xyz_from_disparity_Q(cL, cR)
                stL.pt3d_history[-1] = p3.copy()
                stR.pt3d_history[-1] = p3.copy()
            else:
                stL.pt3d_history[-1] = NAN3.copy()

        for rid in list(updated[1]):
            stR = self.tracker_states[1].get(rid)
            if stR is None: 
                continue
            if rid in r2l_now and r2l_now[rid] in updated[0]:
                continue
            stR.pt3d_history[-1] = NAN3.copy()

        # Collect results of trackers updated in current frame (Trk ID, boxes, 3D points, stereo mates)
        tracking_result = []
        for cam_idx in (0, 1):
            l2r = l2r_now if cam_idx == 0 else r2l_now
            for tid in sorted(alive_ids[cam_idx]):
                st = self.tracker_states[cam_idx].get(int(tid))
                if st is None: 
                    continue
                if not st.frame_count_history or st.frame_count_history[-1] != self.frame_count:
                    continue
                box = np.asarray(st.hand_history[-1], dtype=float)
                xyz = np.asarray(st.pt3d_history[-1] if st.pt3d_history else [np.nan]*3, dtype=float)
                mate_id = l2r.get(int(tid))
                mate_id = int(mate_id) if mate_id is not None else None
                tracking_result.append({
                    "cam": cam_idx, "trk_id": int(tid), "frame": self.frame_count,
                    "box_xyxy": box.tolist(), "xyz": xyz.tolist(), "mate_id": mate_id
                })

        # Delete expired trackers
        for c in (0, 1):
            for tid in to_delete[c]:
                self.remove_tracker(int(tid), c)

        return tracking_result

    def _obs_propagate_all(self, cam_idx: int) -> Dict[int, np.ndarray]:
        preds = {}
        for trk_id, st in self.tracker_states[cam_idx].items():
            if len(st.hand_history) == 0:
                continue
            last_box = np.asarray(st.hand_history[-1], dtype=float)
            c2 = box2center(last_box)
            if hasattr(st, "v_obs"):
                v = np.asarray(st.v_obs, dtype=float)
            else:
                if len(st.hand_history) >= 2:
                    prev_box = np.asarray(st.hand_history[-2], dtype=float)
                    c1 = box2center(prev_box)
                    dt = max(1, st.frame_count_history[-1] - st.frame_count_history[-2])
                    v = (c2 - c1) / float(dt)
                else:
                    v = np.array([0.0, 0.0], dtype=float)

            w = last_box[2] - last_box[0]
            h = last_box[3] - last_box[1]
            c_pred = c2 + v  # 1 프레임 전파
            x1 = c_pred[0] - w/2; y1 = c_pred[1] - h/2
            x2 = c_pred[0] + w/2; y2 = c_pred[1] + h/2

            H, W = self.image_shape[:2]
            x1 = np.clip(x1, 0, W-1); x2 = np.clip(x2, 0, W-1)
            y1 = np.clip(y1, 0, H-1); y2 = np.clip(y2, 0, H-1)
            preds[int(trk_id)] = np.array([x1, y1, x2, y2], dtype=float)
        return preds

    def _update_motion_stats(self, st: TrackerState):
        n = len(st.hand_history)
        if n < 2:
            if not hasattr(st, "v_obs"):
                st.v_obs = np.zeros(2, dtype=float)
            return
        c2 = box2center(st.hand_history[-1])
        idx = max(0, n - 1 - self.oc_delta_t)
        c1 = box2center(st.hand_history[idx])
        dt = max(1, st.frame_count_history[-1] - st.frame_count_history[idx])
        v_now = (c2 - c1) / float(dt)
        if not hasattr(st, "v_obs"):
            st.v_obs = v_now.copy()
        else:
            alpha = 0.7
            st.v_obs = alpha * v_now + (1 - alpha) * st.v_obs

    def _center(self, b): 
        return np.array([(b[0]+b[2])/2.0, (b[1]+b[3])/2.0], dtype=float)

    def _angle_deg(self, v):
        return np.degrees(np.arctan2(v[1], v[0] + 1e-9))

    def _dir_consistency(self, v_track, p_track, p_det):
        v_det = p_det - p_track
        if np.linalg.norm(v_track) < 1e-6 or np.linalg.norm(v_det) < 1e-6:
            return 0.0
        a1 = self._angle_deg(v_track); a2 = self._angle_deg(v_det)
        d = abs((a1 - a2 + 180) % 360 - 180)
        return d

    def _oc_cost_matrix(self, dets: np.ndarray, trk_ids: list, trk_boxes: np.ndarray) -> np.ndarray:
        iou = iou_batch(dets, trk_boxes)
        Nd, Nt = iou.shape
        cost = 1.0 - iou

        ca = np.stack([ (dets[:,0]+dets[:,2])/2.0, (dets[:,1]+dets[:,3])/2.0 ], axis=1)
        cb = np.stack([ (trk_boxes[:,0]+trk_boxes[:,2])/2.0, (trk_boxes[:,1]+trk_boxes[:,3])/2.0 ], axis=1)
        dcent = np.sqrt(((ca[:,None,:] - cb[None,:,:])**2).sum(axis=2))
        cost += 0.2 * (dcent / 60.0)

        if self.use_dir_gate or self.use_speed_penalty:
            for j, tid in enumerate(trk_ids):
                st = self.tracker_states[0].get(tid) or self.tracker_states[1].get(tid)
                if st is None or not hasattr(st, "v_obs"):
                    continue
                v = st.v_obs
                p_trk = cb[j]
                for i in range(Nd):
                    if self.use_dir_gate:
                        dang = self._dir_consistency(v, p_trk, ca[i])
                        cost[i, j] += 0.2 * (dang / max(self.dir_gate_deg, 1e-6))
                    if self.use_speed_penalty:
                        speed = np.linalg.norm(v)
                        exp = speed
                        obs = np.linalg.norm(ca[i] - p_trk)
                        mis = max(0.0, obs - (exp + self.speed_gate_px))
                        cost[i, j] += 0.2 * (mis / max(self.speed_gate_px, 1e-6))
        return cost

    def _stereo_gate_mask(self, dets: np.ndarray, trk_ids: list, trk_boxes: np.ndarray, cam_idx: int):
        if not self.use_stereo_gate: 
            return np.ones((len(dets), len(trk_ids)), dtype=bool)
        mask = np.ones((len(dets), len(trk_ids)), dtype=bool)
        opp = 1 - cam_idx
        for j, tid in enumerate(trk_ids):
            st_self = self.tracker_states[cam_idx].get(tid)
            if st_self is None or not st_self.hand_history:
                continue
            c_self = box2center(st_self.hand_history[-1])
            mates = self.trk_pair_map.get(tid, set())
            if not mates:
                continue

            m = next(iter(mates))
            st_opp = self.tracker_states[opp].get(m)
            if st_opp is None or not st_opp.hand_history:
                continue
            c_opp = box2center(st_opp.hand_history[-1])

            disp_prev = c_self[0] - c_opp[0]    
            tol = self.disparity_consistency_thr

            for i, d in enumerate(dets):
                cd = self._center(d)
                if cam_idx == 0:
                    disp_now = cd[0] - c_opp[0]
                else:
                    disp_now = c_self[0] - cd[0]
                if np.isfinite(disp_prev) and abs(disp_now - disp_prev) > tol:
                    mask[i, j] = False
        return mask

    def associate_detections_to_trackers_oc2(self, detections, trackers, cam_idx,
                                             iou_thr1=0.3, iou_thr2=0.15):
        if len(trackers) == 0:
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,),dtype=int)

        trk_ids = list(trackers.keys())
        trk_boxes = np.array([trackers[i] for i in trk_ids])[:, :4]

        # Stage 1
        iou_matrix = iou_batch(detections, trk_boxes)
        m1, udet, utrk = self._hungarian_with_threshold(iou_matrix, iou_thr1)

        # Stage 2
        if len(udet) > 0 and len(utrk) > 0:
            det_sub = detections[udet]
            trk_sub = trk_boxes[utrk]
            cost = self._oc_cost_matrix(det_sub, [trk_ids[t] for t in utrk], trk_sub)
            mask = self._stereo_gate_mask(det_sub, [trk_ids[t] for t in utrk], trk_sub, cam_idx)
            cost[~mask] = 1e6  # 게이트 탈락

            x, y = linear_sum_assignment(cost)
            sel = [(int(udet[xk]), int(utrk[yk])) for xk, yk in zip(x, y)
                   if cost[xk, yk] < (1.0 - iou_thr2) + 1.0]
            m2 = np.array(sel, dtype=int) if sel else np.empty((0,2), dtype=int)

            if len(m2):
                m1 = np.vstack([m1, m2])

        matches = np.array([[int(d), int(trk_ids[t])] for d, t in m1], dtype=int) if len(m1) else np.empty((0,2), dtype=int)
        unmatched_dets = np.array([i for i in range(len(detections)) if i not in set(m1[:,0])] if len(m1) else list(range(len(detections))), dtype=int)
        unmatched_trks = np.array([int(trk_ids[i]) for i in range(len(trk_boxes)) if i not in set(m1[:,1])] if len(m1) else [int(trk_ids[i]) for i in range(len(trk_boxes))], dtype=int)
        return matches, unmatched_dets, unmatched_trks

    def _hungarian_with_threshold(self, sim_matrix, thr):
        if min(sim_matrix.shape) == 0:
            return np.empty((0,2),dtype=int), np.arange(sim_matrix.shape[0]), np.arange(sim_matrix.shape[1])
        cost = 1.0 - sim_matrix
        x, y = linear_sum_assignment(cost)
        pairs = []
        used_d = set(); used_t = set()
        for i, j in zip(x, y):
            if sim_matrix[i, j] >= thr:
                pairs.append([i, j])
                used_d.add(i); used_t.add(j)
        matches = np.array(pairs, dtype=int) if pairs else np.empty((0,2), dtype=int)
        unmatched_d = np.array([i for i in range(sim_matrix.shape[0]) if i not in used_d], dtype=int)
        unmatched_t = np.array([j for j in range(sim_matrix.shape[1]) if j not in used_t], dtype=int)
        return matches, unmatched_d, unmatched_t

    def _unlink_all(self, tid: int):
        mates = set(self.trk_pair_map.get(tid, set()))
        for mid in mates:
            s = self.trk_pair_map.get(mid)
            if s is not None:
                s.discard(int(tid))
                if not s:
                    self.trk_pair_map.pop(mid, None)
        if tid in self.trk_pair_map:
            self.trk_pair_map[tid].clear()
        else:
            self.trk_pair_map[tid] = set()

    def _push_observation(self, st: TrackerState, hand_box):
        st.frame_count_history.append(self.frame_count)
        st.hand_history.append(hand_box)
        st.pt3d_history.append(NAN3.copy())

    def _xyz_from_disparity_Q(self, cL, cR, y_tol=2.0):
        if self.Q is None:
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        xL, yL = float(cL[0]), float(cL[1])
        xR, yR = float(cR[0]), float(cR[1])
        d = xL - xR
        if not np.isfinite(d) or abs(d) < 1e-6:
            return np.array([np.nan]*3, dtype=float)
        y = 0.5*(yL+yR) if abs(yL-yR) > y_tol else yL
        vec = np.array([xL, y, d, 1.0], dtype=np.float64).reshape(4,1)
        X4 = self.Q @ vec
        W = float(X4[3,0])
        if not np.isfinite(W) or abs(W) < 1e-12:
            return np.array([np.nan]*3, dtype=float)
        XYZ = (X4[:3,0] / W).astype(float)
        return XYZ * self.depth_scale[0]

    def _triangulate_xyzc(self, cL, cR):
        p4 = cv2.triangulatePoints(self.P1, self.P2, cL.reshape(2,1), cR.reshape(2,1))
        p3 = (p4[:3] / p4[3]).reshape(3)
        return p3

    def _collect_pair_component(self, cam_idx, seed_id):
        stack = [(cam_idx, int(seed_id))]
        visited = set()
        comp = []
        while stack:
            c, tid = stack.pop()
            key = (c, tid)
            if key in visited: continue
            visited.add(key)
            if tid not in self.tracker_states[c]: continue
            comp.append((c, tid))
            mates = self.trk_pair_map.get(tid, set())
            oc = 1 - c
            for m in mates:
                stack.append((oc, int(m)))
        return comp

    def _record_match_no_kf(self, cam_idx, trk_id, hand_box, det_idx, matched_hands_trks):
        st = self.tracker_states[cam_idx][trk_id]
        self._push_observation(st, hand_box)
        self._update_motion_stats(st)
        if matched_hands_trks is None or len(matched_hands_trks) == 0:
            matched_hands_trks = np.array([[det_idx, trk_id]])
        else:
            matched_hands_trks = np.vstack([matched_hands_trks, [det_idx, trk_id]])
        return matched_hands_trks

    def _get_recent_pairmate(self, target_cam_idx, opposite_trk_id):
        mate_ids = self.trk_pair_map.get(opposite_trk_id, set())
        best_id, best_seen = None, -1
        for mid in mate_ids:
            st = self.tracker_states[target_cam_idx].get(mid)
            if st is None: 
                continue
            last_seen = st.frame_count_history[-1] if st.frame_count_history else -1
            if last_seen > best_seen:
                best_id, best_seen = mid, last_seen
        return best_id

    def _update_tracker(self, cam_idx, trk_id, hand_box, det_idx, matched_hands_trks):
        st = self.tracker_states[cam_idx][trk_id]
        st.tracker.update(hand_box)
        self._push_observation(st, hand_box)
        self._update_motion_stats(st)
        if matched_hands_trks is None or len(matched_hands_trks) == 0:
            matched_hands_trks = np.array([[det_idx, trk_id]])
        else:
            matched_hands_trks = np.vstack([matched_hands_trks, [det_idx, trk_id]])
        return matched_hands_trks

    def _find_cam_idx_for_id(self, tid: int):
        tid = int(tid)
        if tid in self.tracker_states[0]: return 0
        if tid in self.tracker_states[1]: return 1
        return None

    def remove_tracker(self, trk_id: int, cam_idx: int, cascade: bool = True):
        root = (int(trk_id), int(cam_idx))
        queue = [root]
        visited = set()
        while queue:
            tid, cidx = queue.pop()
            key = (tid, cidx)
            if key in visited: continue
            visited.add(key)
            mates = set(self.trk_pair_map.get(tid, set()))
            self.tracker_states[cidx].pop(tid, None)
            self.trk_pair_map.pop(tid, None)
            for mid in list(mates):
                s = self.trk_pair_map.get(mid)
                if s is not None:
                    s.discard(tid)
                    if not s:
                        self.trk_pair_map.pop(mid, None)
                if cascade:
                    mc = self._find_cam_idx_for_id(mid)
                    if mc is not None:
                        queue.append((int(mid), mc))

    def create_tracker_state(self, cam_idx, hand_idx, hand_box):
        tracker = KalmanBoxTracker(hand_box)
        trk_id = int(tracker.id)
        self.tracker_states[cam_idx][trk_id] = TrackerState(
            tracker=tracker, hand_history=[], frame_count_history=[], pt3d_history=[]
        )
        self._push_observation(self.tracker_states[cam_idx][trk_id], hand_box)
        self._update_motion_stats(self.tracker_states[cam_idx][trk_id])
        return trk_id

    def stereo_matching_module(self, detection_tl, detection_tr, iou_threshold=0.2):
        if len(detection_tl) == 0 and len(detection_tr) == 0:
            return np.empty((0,2), dtype=int), np.empty((0,), dtype=int), np.empty((0,), dtype=int)
        if len(detection_tr) == 0:
            return np.empty((0,2), dtype=int), np.arange(len(detection_tl), dtype=int), np.empty((0,), dtype=int)
        if len(detection_tl) == 0:
            return np.empty((0,2), dtype=int), np.empty((0,), dtype=int), np.arange(len(detection_tr), dtype=int)

        def expand_box_centered(box, scale=1.0):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
            w = (x2 - x1) * scale; h = (y2 - y1) * scale
            return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

        detection_tl_scaled = np.array([expand_box_centered(b, 1.0) for b in detection_tl])
        detection_tr_scaled = np.array([expand_box_centered(b, 1.0) for b in detection_tr])

        # IoU calculation: DLT-based triangulation of detected pairs & Stereo consistency check
        iou_matrix = self.iou_batch_by_depth(detection_tl_scaled, detection_tr_scaled, self.max_depth)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty((0, 2), dtype=int)

        matched_tl = set(matched_indices[:,0]) if matched_indices.size>0 else set()
        matched_tr = set(matched_indices[:,1]) if matched_indices.size>0 else set()

        unmatched_tl = [tl for tl in range(len(detection_tl)) if tl not in matched_tl]
        unmatched_tr = [tr for tr in range(len(detection_tr)) if tr not in matched_tr]

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] > iou_threshold:
                matches.append(m.reshape(1,2))
        matches = np.concatenate(matches, axis=0) if len(matches) else np.empty((0,2), dtype=int)
        return matches, np.array(unmatched_tl), np.array(unmatched_tr)

    def iou_batch_by_depth(self, left_boxes, right_boxes, max_depth=2.0):
        iou_matrix = np.zeros((len(left_boxes), len(right_boxes)), dtype=np.float32)
        H, W = self.image_shape[:2]
        for i, box_l in enumerate(left_boxes):
            center_l = box2center(box_l)
            for j, box_r in enumerate(right_boxes):
                center_r = box2center(box_r)
                pt4d = cv2.triangulatePoints(self.P1, self.P2,
                                             center_l.reshape(2,1),
                                             center_r.reshape(2,1))
                pt3d = pt4d[:3] / pt4d[3]
                z = float(pt3d[2,0])
                if not np.isfinite(z) or z <= 0 or z > max_depth:
                    iou_matrix[i,j] = 0.0
                    continue
                z_adj = z * self.depth_scale[0]
                z_m = z_adj
                if not np.isfinite(z_m) or abs(z_m) < 1e-6:
                    pixel_shift = 0.0
                else:
                    ideal_fb = float(-self.P2[0,3])
                    fb = self.calib_scale * ideal_fb
                    pixel_shift = (fb / z_m) + self.calib_bias
                shifted_box_r = np.array([
                    box_r[0] + pixel_shift, box_r[1],
                    box_r[2] + pixel_shift, box_r[3]
                ], dtype=np.float32)
                iou = iou_batch(box_l[None,:], shifted_box_r[None,:])[0][0]
                iou_matrix[i,j] = iou
        return iou_matrix

# ---------------- helpers ---------------- #
def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h + 1e-12)
    return np.array([x, y, s, r]).reshape((4,1))

def convert_x_to_bbox(x, score=None):
    w = np.sqrt(np.abs(x[2] * x[3]))
    h = x[2] / w if w != 0 else 0
    if score is None:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).reshape((1,4))
    else:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2., score]).reshape((1,5))

def box2center(box):
    box = np.squeeze(box)
    xp = (box[2] - box[0]) / 2 + box[0]
    yp = (box[3] - box[1]) / 2 + box[1]
    return np.array([xp, yp])

def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[...,0], bb_gt[...,0])
    yy1 = np.maximum(bb_test[...,1], bb_gt[...,1])
    xx2 = np.minimum(bb_test[...,2], bb_gt[...,2])
    yy2 = np.minimum(bb_test[...,3], bb_gt[...,3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[...,2]-bb_test[...,0]) * (bb_test[...,3]-bb_test[...,1]) +
              (bb_gt[...,2]-bb_gt[...,0]) * (bb_gt[...,3]-bb_gt[...,1]) - wh + 1e-12)
    return o
