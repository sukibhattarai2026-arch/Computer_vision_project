from flask import Flask, request, jsonify, send_file, render_template
import os
import tempfile
import threading
import uuid
import cv2
import numpy as np

app = Flask(__name__)
jobs = {}


def encode_video(frames, out_path, fps=4):
    import imageio.v3 as iio
    h, w = frames[0].shape[:2]
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1
    all_frames = []
    for fr in frames:
        rgb = cv2.cvtColor(cv2.resize(fr, (w, h)), cv2.COLOR_BGR2RGB)
        all_frames.append(rgb)
    for _ in range(fps * 2):
        all_frames.append(all_frames[-1])
    for fr in reversed(frames):
        rgb = cv2.cvtColor(cv2.resize(fr, (w, h)), cv2.COLOR_BGR2RGB)
        all_frames.append(rgb)
    iio.imwrite(
        out_path, all_frames, fps=fps, codec='libx264',
        pixelformat='yuv420p', output_params=['-movflags', '+faststart']
    )
    return out_path

def synthesize_artwork_damage_bgr(
    base_bgr: np.ndarray,
    varnish: float = 0.3,
    lighting: float = 0.6,
    tilt: float = 0.12,
    num_cracks: int = 5,
    num_white_patches: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies synthetic varnish + cracks + white patches + dim lighting + perspective tilt
    to a BGR image. Returns (distorted_bgr, gt_mask_gray).
    Restriction lifted: Noise and damage can now appear anywhere on the image.
    """
    raw = base_bgr.copy()
    h, w = raw.shape[:2]

    # Varnish applied to the FULL image (Restriction Lifted)
    damaged_f = raw.astype(np.float32) / 255.0
    v_map = cv2.resize(np.random.rand(30, 30).astype(np.float32), (w, h))
    v_col = np.array([0.30, 0.70, 0.85], dtype=np.float32)  # warm yellow in BGR
    for c in range(3):
        damaged_f[:, :, c] = (
            damaged_f[:, :, c] * (1 - varnish * v_map)
            + v_col[c] * varnish * v_map
        )
    damaged_u8 = np.clip(damaged_f * 255, 0, 255).astype(np.uint8)

    # Cracks (Anywhere on the image)
    for _ in range(num_cracks):
        sp = (np.random.randint(0, w), np.random.randint(0, h))
        ep = (np.random.randint(0, w), np.random.randint(0, h))
        cv2.line(damaged_u8, sp, ep, (20, 20, 20), np.random.randint(1, 4))
        
    # White Patches (New: simulates gesso exposure or paint loss)
    for _ in range(num_white_patches):
        center = (np.random.randint(0, w), np.random.randint(0, h))
        axes = (np.random.randint(5, 20), np.random.randint(5, 20))
        cv2.ellipse(damaged_u8, center, axes, np.random.randint(0, 360), 0, 360, (240, 240, 240), -1)

    # Ground-truth mask (Calculated before lighting/tilt for geometric alignment)
    gt_diff = cv2.absdiff(raw, damaged_u8)
    gt_gray = cv2.cvtColor(gt_diff, cv2.COLOR_BGR2GRAY)
    _, gt_mask = cv2.threshold(cv2.GaussianBlur(gt_gray, (3, 3), 0), 10, 255, cv2.THRESH_BINARY)

    # Dim lighting (HSV value channel)
    hsv = cv2.cvtColor(damaged_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= lighting
    distorted = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Perspective tilt
    if tilt > 0:
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [w * tilt, h * tilt], [w * (1 - tilt), h * tilt], [0, h], [w, h]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        distorted = cv2.warpPerspective(distorted, M, (w, h))

    return distorted, gt_mask

# ROI extraction
def extract_artwork_roi(img, is_baseline=False):
    h, w = img.shape[:2]
    if is_baseline:
        return img, (0, 0, w, h)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    edged = cv2.Canny(blurred, 50, 150)
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, (0, 0, w, h)
    c = max(cnts, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(c)
    if cw * ch < 0.10 * w * h:
        return img, (0, 0, w, h)
    return img[y:y+ch, x:x+cw], (x, y, cw, ch)



# SIFT alignment
def align_images_multi_approach(base_roi, target_img):
    sift = cv2.SIFT_create(nfeatures=10000)
    kp1, des1 = sift.detectAndCompute(base_roi, None)
    kp2, des2 = sift.detectAndCompute(target_img, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.65 * n.distance]
    if len(good) < 20:
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None
    return cv2.warpPerspective(
        target_img, H,
        (base_roi.shape[1], base_roi.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


# Intensity homogenisation
def homogenize_damaged(base_bgr, damaged_bgr):
    from skimage.exposure import match_histograms
    base_rgb    = cv2.cvtColor(base_bgr,    cv2.COLOR_BGR2RGB)
    damaged_rgb = cv2.cvtColor(damaged_bgr, cv2.COLOR_BGR2RGB)
    matched_rgb = match_histograms(damaged_rgb, base_rgb, channel_axis=-1)
    return cv2.cvtColor(
        np.clip(matched_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )

def compute_damage_heatmap(base_bgr, damaged_final_bgr):
    """
    SSIM-based damage detection.
    Returns (heatmap_bgr, binary_mask_bgr, overlay_bgr) — all BGR, same size as base.
    """
    from skimage.metrics import structural_similarity as ssim

    # Work in grayscale for SSIM
    b_gray = cv2.cvtColor(base_bgr,          cv2.COLOR_BGR2GRAY)
    d_gray = cv2.cvtColor(damaged_final_bgr,  cv2.COLOR_BGR2GRAY)

    # SSIM difference map  (1 - ssim_val = dissimilarity)
    _, s_map = ssim(b_gray, d_gray, full=True, win_size=7)
    s_u8     = ((1.0 - s_map) * 255).astype(np.uint8)
    s_denoised = cv2.medianBlur(s_u8, 5)

    # Triangle threshold → binary mask
    _, s_binary = cv2.threshold(s_denoised, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    kernel  = np.ones((5, 5), np.uint8)
    s_clean = cv2.morphologyEx(s_binary, cv2.MORPH_OPEN, kernel)

    # ── Heatmap: red = high damage, yellow = moderate ─────────────────────
    s_norm    = cv2.normalize(s_denoised, None, 0, 255, cv2.NORM_MINMAX)
    heatmap   = cv2.applyColorMap(s_norm, cv2.COLORMAP_JET)
    # Suppress low-signal pixels (keep only where mask fires)
    mask3 = cv2.cvtColor(s_clean, cv2.COLOR_GRAY2BGR).astype(bool)

    # ── Overlay: blend heatmap onto original only in damaged regions ───────
    alpha = 0.6
    overlay = base_bgr.copy().astype(np.float32)
    idx = s_clean > 0
    overlay[idx] = (
        (1 - alpha) * base_bgr[idx].astype(np.float32)
        + alpha      * heatmap[idx].astype(np.float32)
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Binary mask visualised in red-on-dark for clarity
    mask_vis = np.zeros_like(base_bgr)
    mask_vis[idx] = (0, 0, 220)          # red in BGR

    return mask_vis, overlay


# Main pipeline job
def run_pipeline_job(job_id, orig_path, damaged_path, num_frames, methods):
    try:
        jobs[job_id]['status'] = 'running'
        jobs[job_id]['progress'] = 'Loading images...'

        from pipeline import (
            load_image, align_images, resize_to_match,
            compute_signals, compute_issue_maps, routing_from_issues,
            build_transition_frame, grayscale,
            linear_blend, flow_blend, laplacian_blend, inpaint_progressive,
            temporal_curve
        )

        out_dir = jobs[job_id]['out_dir']

        # ── Load original ──────────────────────────────────────────────
        orig = load_image(orig_path)

        # ── Get second image (upload or synthetic) ─────────────────────
        if jobs[job_id].get('use_synthetic'):
            jobs[job_id]['progress'] = 'Generating synthetic damage...'
            p = jobs[job_id].get('synth_params', {})
            damaged_raw, _gt = synthesize_artwork_damage_bgr(
                orig,
                varnish=p.get('varnish', 0.3),
                lighting=p.get('lighting', 0.6),
                tilt=p.get('tilt', 0.12),
                num_cracks=int(p.get('num_cracks', 5)),
            )
        else:
            damaged_raw = load_image(damaged_path)

        # Save Stage A: raw damaged (before any processing)
        stage_a_path = os.path.join(out_dir, 'stage_a_raw.jpg')
        cv2.imwrite(stage_a_path, damaged_raw)
        jobs[job_id]['stage_a'] = stage_a_path

        # ── ROI + Alignment ────────────────────────────────────────────
        jobs[job_id]['progress'] = 'Aligning images...'
        base_roi, _ = extract_artwork_roi(orig, is_baseline=True)

        aligned = align_images_multi_approach(base_roi, damaged_raw)
        if aligned is None:
            try:
                aligned = align_images(damaged_raw, base_roi)
            except Exception:
                aligned = None
        if aligned is None:
            aligned = resize_to_match(damaged_raw, base_roi)

        # Save Stage B: after alignment
        stage_b_path = os.path.join(out_dir, 'stage_b_aligned.jpg')
        cv2.imwrite(stage_b_path, aligned)
        jobs[job_id]['stage_b'] = stage_b_path

        # ── Intensity normalisation ────────────────────────────────────
        jobs[job_id]['progress'] = 'Normalising intensities...'
        damaged_final = homogenize_damaged(base_roi, aligned)

        # Save Stage C: after homogenisation
        stage_c_path = os.path.join(out_dir, 'stage_c_normalised.jpg')
        cv2.imwrite(stage_c_path, damaged_final)
        jobs[job_id]['stage_c'] = stage_c_path
        
        # ── Damage mask + overlay ─────────────────────────────────────
        jobs[job_id]['progress'] = 'Computing damage heatmap...'
        mask_bgr, damage_overlay_bgr = compute_damage_heatmap(base_roi, damaged_final)
        stage_mask_path    = os.path.join(out_dir, 'stage_mask.jpg')
        stage_overlay_path = os.path.join(out_dir, 'stage_overlay.jpg')
        cv2.imwrite(stage_mask_path,    mask_bgr)
        cv2.imwrite(stage_overlay_path, damage_overlay_bgr)
        jobs[job_id]['stage_mask']    = stage_mask_path
        jobs[job_id]['stage_overlay'] = stage_overlay_path
        jobs[job_id]['stages_ready']  = True

        # ── Signals + routing ──────────────────────────────────────────
        jobs[job_id]['progress'] = 'Analysing damage signals...'
        sig     = compute_signals(base_roi, damaged_final)
        issues  = compute_issue_maps(sig)
        routing = routing_from_issues(issues)

        flow = cv2.calcOpticalFlowFarneback(
            grayscale(base_roi), grayscale(damaged_final), None,
            0.5, 4, 21, 5, 7, 1.5, 0
        )

        rng = np.random.default_rng(7)
        u = np.linspace(0, 1, num_frames)
        alphas = (u ** 0.8).tolist()
        alphas[-1] = 1.0

        videos = {}

        # ── Frame generation ───────────────────────────────────────────
        for method in methods:
            jobs[job_id]['progress'] = f'Generating frames: {method}...'
            frames = [base_roi]

            for alpha in alphas:
                t         = temporal_curve(float(alpha))
                alpha_map = np.full(base_roi.shape[:2], t, np.float32)

                if method == 'adaptive':
                    fr = build_transition_frame(
                        base_roi, damaged_final, issues, routing, flow, float(alpha), rng
                    )
                elif method == 'linear':
                    fr = linear_blend(base_roi, damaged_final, alpha_map)
                elif method == 'optical_flow':
                    fr = flow_blend(base_roi, damaged_final, flow, alpha_map, t)
                elif method == 'laplacian':
                    fr = laplacian_blend(base_roi, damaged_final, alpha_map)
                elif method == 'inpainting':
                    fr = inpaint_progressive(base_roi, damaged_final, issues.missing, t, rng)

                frames.append(fr)

            jobs[job_id]['progress'] = f'Encoding video: {method}...'
            video_path = os.path.join(out_dir, f'{method}.mp4')
            encode_video(frames, video_path)
            videos[method] = video_path

        jobs[job_id]['status']   = 'done'
        jobs[job_id]['videos']   = videos
        jobs[job_id]['progress'] = 'Complete!'

    except Exception as e:
        import traceback
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error']  = str(e) + '\n' + traceback.format_exc()


# Routes

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    use_synthetic = request.form.get('use_synthetic') == 'true'

    if 'original' not in request.files:
        return jsonify({'error': 'Original image required'}), 400
    if not use_synthetic and 'damaged' not in request.files:
        return jsonify({'error': 'Damaged image required'}), 400

    num_frames = int(request.form.get('num_frames', 10))
    num_frames = max(4, min(20, num_frames))

    methods = request.form.getlist('methods')
    valid   = {'adaptive', 'linear', 'optical_flow', 'laplacian', 'inpainting'}
    methods = [m for m in methods if m in valid]
    if not methods:
        methods = ['adaptive']

    synth_params = {
        'varnish':    float(request.form.get('varnish',    0.3)),
        'lighting':   float(request.form.get('lighting',   0.6)),
        'tilt':       float(request.form.get('tilt',       0.12)),
        'num_cracks': int(request.form.get('num_cracks',   5)),
    }

    job_id  = str(uuid.uuid4())[:8]
    out_dir = tempfile.mkdtemp(prefix=f'artwork_{job_id}_')

    orig_path = os.path.join(out_dir, 'original.jpg')
    request.files['original'].save(orig_path)

    dam_path = None
    if not use_synthetic:
        dam_path = os.path.join(out_dir, 'damaged.jpg')
        request.files['damaged'].save(dam_path)

    jobs[job_id] = {
        'status':        'queued',
        'progress':      'Queued...',
        'out_dir':       out_dir,
        'videos':        {},
        'error':         None,
        'use_synthetic': use_synthetic,
        'synth_params':  synth_params,
        'stages_ready':  False,
        'stage_a':       None,
        'stage_b':       None,
        'stage_c':       None,
        'stage_mask':     None,  
        'stage_overlay':  None,  
    }

    t = threading.Thread(
        target=run_pipeline_job,
        args=(job_id, orig_path, dam_path, num_frames, methods)
    )
    t.daemon = True
    t.start()

    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>')
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({
        'status':       job['status'],
        'progress':     job['progress'],
        'videos':       list(job.get('videos', {}).keys()),
        'stages_ready': job.get('stages_ready', False),
        'error':        job.get('error'),
    })


@app.route('/stage/<job_id>/<stage>')
def stage_image(job_id, stage):
    """Serve one of the three processing stage images: a, b, or c."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    key  = f'stage_{stage}'
    path = job.get(key)
    if not path or not os.path.exists(path):
        return jsonify({'error': 'Stage not ready'}), 404
    return send_file(path, mimetype='image/jpeg')

@app.route('/heatmap/<job_id>/<panel>')
def heatmap_image(job_id, panel):
    """panel = mask | overlay"""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    key  = f'stage_{panel}'
    path = job.get(key)
    if not path or not os.path.exists(path):
        return jsonify({'error': 'Not ready'}), 404
    return send_file(path, mimetype='image/jpeg')


@app.route('/video/<job_id>/<method>')
def video(job_id, method):
    job = jobs.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'Not ready'}), 404
    path = job['videos'].get(method)
    if not path or not os.path.exists(path):
        return jsonify({'error': 'Method not found'}), 404
    return send_file(path, mimetype='video/mp4', conditional=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)