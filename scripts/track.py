import cv2, yaml, numpy as np
import argparse
from ultralytics import YOLO
import numpy as np
import stereo_tracker as st
import os
from plot3d_util import Track3DPlotter

def load_params(yaml_path):
    d = yaml.safe_load(open(yaml_path, "r"))

    # Camera Configration (by checker board calibration)
    K1 = np.array(d["left"]["mtx"], dtype=np.float64)
    D1 = np.array(d["left"]["dist"], dtype=np.float64)
    K2 = np.array(d["right"]["mtx"], dtype=np.float64)
    D2 = np.array(d["right"]["dist"], dtype=np.float64)
    size = tuple(d["left"]["image_size"])

    R1 = np.array(d["rectify"]["R1"], dtype=np.float64)
    R2 = np.array(d["rectify"]["R2"], dtype=np.float64)
    P1 = np.array(d["rectify"]["P1"], dtype=np.float64)
    P2 = np.array(d["rectify"]["P2"], dtype=np.float64)
    Q  = np.array(d["rectify"]["Q"],  dtype=np.float64)

    c = d.get("calib_adjustment", {}) or {}
    calib_adjustment_cfg = {
        "depth_scale":        list(c.get("coeffs", [1.0])),
        "calib_scale":        float(c.get("calib_scale", 1.0)),
        "calib_bias":         float(c.get("calib_bias", 0.0)),
    }

    # Tracker configuration
    t = d.get("tracker", {}) or {}
    tracker_cfg = {
        "max_age":            int(t.get("max_age", 10)),
        "min_hits":           int(t.get("min_hits", 3)),
        "iou_thr1":           float(t.get("iou_thr1", 0.3)),
        "iou_thr2":           float(t.get("iou_thr2", 0.15)),
        "calib_scale":        float(t.get("calib_scale", 1.0)),
        "calib_bias":         float(t.get("calib_bias", 0.0)),
        "oc_delta_t":         int(t.get("oc_delta_t", 2)),
        "inertia":            float(t.get("inertia", 0.2)),
        "use_dir_gate":       bool(t.get("use_dir_gate", True)),
        "use_speed_penalty":  bool(t.get("use_speed_penalty", True)),
        "use_stereo_gate":    bool(t.get("use_stereo_gate", True)),
        "dir_gate_deg":       float(t.get("dir_gate_deg", 40.0)),
        "speed_gate_px":      float(t.get("speed_gate_px", 80.0)),
        "is_rectified":       bool(t.get("is_rectified", True)),
        "max_depth":          float(t.get("max_depth", 2.0)),
        "disparity_consistency_thr": float(t.get("disparity_consistency_thr", 40.0)),
    }

    return {
        "K1":K1,"D1":D1,"K2":K2,"D2":D2,"size":size,
        "R1":R1,"R2":R2,"P1":P1,"P2":P2,"Q":Q,
        "tracker_cfg": tracker_cfg,
        "calib_adjustment_cfg": calib_adjustment_cfg,
        }

def make_maps(params, size=None):
    size = tuple(params["size"]) if size is None else tuple(size)
    mapL = cv2.initUndistortRectifyMap(params["K1"], params["D1"], params["R1"], params["P1"], size, cv2.CV_16SC2)
    mapR = cv2.initUndistortRectifyMap(params["K2"], params["D2"], params["R2"], params["P2"], size, cv2.CV_16SC2)
    return mapL, mapR, size

def open_source(src, width=None, height=None, fps=None):
    try:
        s_int = int(src)
        cap = cv2.VideoCapture(s_int)
    except ValueError:
        cap = cv2.VideoCapture(src)

    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(width))
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps:    cap.set(cv2.CAP_PROP_FPS,          float(fps))
    return cap

def str2bool(x: str) -> bool:
    return str(x).lower() in ("true","1","yes","y","t")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left_src",  default="./videos/example2/rL.mp4", help="left camera source (default: ./videos/example/rL.mp4)")
    ap.add_argument("--right_src", default="./videos/example2/rR.mp4", help="right camera source (default: ./videos/example/rR.mp4)")
    ap.add_argument("--params_yaml", default="./configs/config.yaml")
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--model_path", default="./models/yolo11_detector.pt", help="path to YOLO model")
    ap.add_argument("--visualize", default="True", help="visualize results")
    ap.add_argument("--visualize_3d", default="False", help="visualize 3D plot")
    ap.add_argument("--save_video", default="./outputs/example2_out_3d.mp4", help="path to save stacked video (e.g., ./outputs/example2_out_3d.mp4)")
    args = ap.parse_args()

    # Load YOLO model
    model = YOLO(args.model_path, verbose=False, task='detect')

    # Load parameters
    params = load_params(args.params_yaml)

    # Stereo rectification maps
    mapL, mapR, calib_size = make_maps(params) 
    w_cal, h_cal = calib_size

    visualize    = str2bool(args.visualize)
    visualize_3d = str2bool(args.visualize_3d)
    save_video   = (args.save_video is not None and len(args.save_video) > 0)

    # Initialize Stereo Tracker
    sort_track = st.StereoTracker(
        P1=params["P1"], P2=params["P2"], Q=params["Q"], image_shape=(h_cal, w_cal),
        depth_scale=params["calib_adjustment_cfg"]["depth_scale"],
        calib_scale=params["calib_adjustment_cfg"]["calib_scale"],
        calib_bias=params["calib_adjustment_cfg"]["calib_bias"],
        max_age=params["tracker_cfg"]["max_age"],
        min_hits=params["tracker_cfg"]["min_hits"],
        iou_thr1=params["tracker_cfg"]["iou_thr1"],
        iou_thr2=params["tracker_cfg"]["iou_thr2"],
        max_depth=params["tracker_cfg"]["max_depth"],
        oc_delta_t=params["tracker_cfg"]["oc_delta_t"],
        inertia=params["tracker_cfg"]["inertia"],
        use_dir_gate=params["tracker_cfg"]["use_dir_gate"],
        use_speed_penalty=params["tracker_cfg"]["use_speed_penalty"],
        use_stereo_gate=params["tracker_cfg"]["use_stereo_gate"],
        dir_gate_deg=params["tracker_cfg"]["dir_gate_deg"],
        speed_gate_px=params["tracker_cfg"]["speed_gate_px"],
        disparity_consistency_thr=params["tracker_cfg"]["disparity_consistency_thr"]
        )

    # Open left and right camera sources
    capL = open_source(args.left_src,  args.width, args.height, args.fps)
    capR = open_source(args.right_src, args.width, args.height, args.fps)

    if not (capL.isOpened() and capR.isOpened()):
        print("[ERR] cannot open left/right source"); return
    
    # Video writer 
    writer = None
    out_w, out_h = w_cal*2, (h_cal if not visualize_3d else h_cal*2)
    if save_video:
        os.makedirs(os.path.dirname(args.save_video), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, args.fps, (out_w, out_h))

    # 3D Plotter
    plotter = Track3DPlotter(
        figsize=(4, 3),
        dpi=500,
        xlim=(-0.8, 0.8),
        ylim=(-0.8, 0.8),
        zlim=(0.3, 2.0),
        elev=60,     
        azim=-100,         
        maxlen=100,
        y_down=True,      
        title="3D Tracks (cam=0)",
        legend_max=16
    )
    
    all_results = []

    while True:
        okL, frL = capL.read()
        okR, frR = capR.read()
        if not okL or not okR:
            break
        
        if (frL.shape[1], frL.shape[0]) != (w_cal, h_cal):
            frL = cv2.resize(frL, (w_cal, h_cal), interpolation=cv2.INTER_AREA)
        if (frR.shape[1], frR.shape[0]) != (w_cal, h_cal):
            frR = cv2.resize(frR, (w_cal, h_cal), interpolation=cv2.INTER_AREA)

        if params["tracker_cfg"]["is_rectified"]:
            rL = frL
            rR = frR
        else:
            rL = cv2.remap(frL, mapL[0], mapL[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            rR = cv2.remap(frR, mapR[0], mapR[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        frames = [rL, rR]

        # Object Detection from Stereo Image Pairs
        yolo_results = model(frames, imgsz=640, conf=0.3, iou=0.3, verbose=False)
        results_dict = {cid: [res] for cid, res in enumerate(yolo_results)}

        # Stereo Tracker Update
        tracking_result = sort_track.update(results_dict[0], results_dict[1])
                        
        if visualize:
            visL, visR = rL.copy(), rR.copy()

            pair_colors = {}
            def get_color_for_pair(id0, id1):
                key = (min(id0, id1), max(id0, id1))
                if key not in pair_colors:
                    np.random.seed(hash(key) % 2**32)
                    pair_colors[key] = tuple(np.random.randint(0, 255, size=3).tolist())
                return pair_colors[key]

            ids_cam1 = {r["trk_id"] for r in tracking_result if r["cam"] == 1}

            for res in tracking_result:
                cam = res["cam"]
                tid = res["trk_id"]
                mate = res["mate_id"]
                x1,y1,x2,y2 = [int(x) for x in res["box_xyxy"]]

                base_color = (255, 0, 0)
                tgt = visL if cam==0 else visR
                cv2.rectangle(tgt, (x1, y1), (x2, y2), base_color, 3)

                label = f"ID {tid}"
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tx, ty = x1, max(0, y1-th-4)
                cv2.rectangle(tgt, (tx, ty), (tx+tw, ty+th+bl), base_color, -1)
                cv2.putText(tgt, label, (tx, ty+th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                if mate is not None:
                    if cam == 0 and mate in ids_cam1:
                        pc = get_color_for_pair(tid, mate)
                        cv2.rectangle(visL, (x1-5, y1-5), (x2+5, y2+5), pc, 2)
                    elif cam == 1:
                        pc = get_color_for_pair(mate, tid)
                        cv2.rectangle(visR, (x1-5, y1-5), (x2+5, y2+5), pc, 2)


        if visualize_3d:
            plotter.add_points_from_tracking(tracking_result, cam_select=0)

        # ===== 합성 & 출력 =====
        if visualize:
            top = np.hstack([visL, visR])
            if visualize_3d:
                plot_img = plotter.render_bgr_resized_to_height(h_cal)
                white = np.full((h_cal, w_cal), 255, dtype=np.uint8)
                white = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
                if plot_img.shape[1] != w_cal:
                    plot_img = cv2.resize(plot_img, (w_cal, h_cal), interpolation=cv2.INTER_AREA)
                bottom = np.hstack([plot_img, white])
                grid = np.vstack([top, bottom])
                show_frame = grid
            else:
                show_frame = top

            cv2.imshow("Stereo Tracking + 3D (2x2 grid)", show_frame)

            if writer is not None:
                if (show_frame.shape[1], show_frame.shape[0]) != (out_w, out_h):
                    show_to_write = cv2.resize(show_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                else:
                    show_to_write = show_frame
                writer.write(show_to_write)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            if writer is not None:
                base = np.hstack([rL, rR])
                if visualize_3d:
                    plot_img = plotter.render_bgr_resized_to_height(h_cal)
                    if plot_img.shape[1] != w_cal:
                        plot_img = cv2.resize(plot_img, (w_cal, h_cal), interpolation=cv2.INTER_AREA)
                    white = np.full((h_cal, w_cal, 3), 255, dtype=np.uint8)
                    bottom = np.hstack([plot_img, white])
                    grid = np.vstack([base, bottom])
                else:
                    white = np.full((h_cal, w_cal*2, 3), 255, dtype=np.uint8)
                    grid = np.vstack([base, white])

                if (grid.shape[1], grid.shape[0]) != (out_w, out_h):
                    grid = cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_AREA)
                writer.write(grid)

    if writer is not None:
        writer.release()
    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()