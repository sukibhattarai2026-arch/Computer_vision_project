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
    # libx264 requires even dimensions
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1

    all_frames = []
    # Forward
    for fr in frames:
        rgb = cv2.cvtColor(cv2.resize(fr, (w, h)), cv2.COLOR_BGR2RGB)
        all_frames.append(rgb)
    # Hold on last frame 2 seconds
    for _ in range(fps * 2):
        all_frames.append(all_frames[-1])
    # Reverse
    for fr in reversed(frames):
        rgb = cv2.cvtColor(cv2.resize(fr, (w, h)), cv2.COLOR_BGR2RGB)
        all_frames.append(rgb)

    iio.imwrite(
        out_path,
        all_frames,
        fps=fps,
        codec='libx264',
        pixelformat='yuv420p',
        output_params=['-movflags', '+faststart']
    )
    return out_path


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

        jobs[job_id]['progress'] = 'Aligning images...'
        orig = load_image(orig_path)
        damaged = load_image(damaged_path)

        try:
            damaged = align_images(damaged, orig)
        except Exception:
            damaged = resize_to_match(damaged, orig)

        jobs[job_id]['progress'] = 'Analysing damage signals...'
        sig = compute_signals(orig, damaged)
        issues = compute_issue_maps(sig)
        routing = routing_from_issues(issues)

        flow = cv2.calcOpticalFlowFarneback(
            grayscale(orig), grayscale(damaged), None,
            0.5, 4, 21, 5, 7, 1.5, 0
        )

        rng = np.random.default_rng(7)
        u = np.linspace(0, 1, num_frames)
        alphas = (u ** 0.8).tolist()
        alphas[-1] = 1.0

        videos = {}

        for method in methods:
            jobs[job_id]['progress'] = f'Generating frames: {method}...'
            frames = [orig]

            for alpha in alphas:
                t = temporal_curve(float(alpha))
                alpha_map = np.full(orig.shape[:2], t, np.float32)

                if method == 'adaptive':
                    fr = build_transition_frame(orig, damaged, issues, routing, flow, float(alpha), rng)
                elif method == 'linear':
                    fr = linear_blend(orig, damaged, alpha_map)
                elif method == 'optical_flow':
                    fr = flow_blend(orig, damaged, flow, alpha_map, t)
                elif method == 'laplacian':
                    fr = laplacian_blend(orig, damaged, alpha_map)
                elif method == 'inpainting':
                    fr = inpaint_progressive(orig, damaged, issues.missing, t, rng)

                frames.append(fr)

            jobs[job_id]['progress'] = f'Encoding video: {method}...'
            video_path = os.path.join(out_dir, f'{method}.mp4')
            encode_video(frames, video_path)
            videos[method] = video_path

        jobs[job_id]['status'] = 'done'
        jobs[job_id]['videos'] = videos
        jobs[job_id]['progress'] = 'Complete!'

    except Exception as e:
        import traceback
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e) + '\n' + traceback.format_exc()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if 'original' not in request.files or 'damaged' not in request.files:
        return jsonify({'error': 'Both images required'}), 400

    num_frames = int(request.form.get('num_frames', 10))
    num_frames = max(4, min(20, num_frames))

    methods = request.form.getlist('methods')
    valid = {'adaptive', 'linear', 'optical_flow', 'laplacian', 'inpainting'}
    methods = [m for m in methods if m in valid]
    if not methods:
        methods = ['adaptive']

    job_id = str(uuid.uuid4())[:8]
    out_dir = tempfile.mkdtemp(prefix=f'artwork_{job_id}_')

    orig_path = os.path.join(out_dir, 'original.jpg')
    dam_path = os.path.join(out_dir, 'damaged.jpg')
    request.files['original'].save(orig_path)
    request.files['damaged'].save(dam_path)

    jobs[job_id] = {
        'status': 'queued',
        'progress': 'Queued...',
        'out_dir': out_dir,
        'videos': {},
        'error': None
    }

    t = threading.Thread(target=run_pipeline_job,
                         args=(job_id, orig_path, dam_path, num_frames, methods))
    t.daemon = True
    t.start()

    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>')
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'videos': list(job.get('videos', {}).keys()),
        'error': job.get('error')
    })


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
