"""Face recognition pipeline for SmartAttend AI."""

import os
import io
import pickle
import shutil
import threading
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")

SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

DEFAULT_MATCH_THRESHOLD = 0.49
DEFAULT_MARGIN_THRESHOLD = 0.105

_lock = threading.Lock()
_recognizer = None
_yunet_detector = None
_mp_face_detector = None

def _ensure_models_exist():
    if not os.path.exists(SFACE_PATH):
        urllib.request.urlretrieve(SFACE_URL, SFACE_PATH)
    if not os.path.exists(YUNET_PATH):
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)

def _get_recognizer():
    global _recognizer
    if _recognizer is None:
        _ensure_models_exist()
        _recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
    return _recognizer

def _get_yunet_detector(width=320, height=320):
    global _yunet_detector
    if _yunet_detector is None:
        _ensure_models_exist()
        _yunet_detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (width, height))
    return _yunet_detector

def _get_mp_detector():
    global _mp_face_detector
    if _mp_face_detector is None:
        _mp_face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.40
        )
    return _mp_face_detector

def enhance_low_light_and_mobile_image(image):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        mean_l = float(np.mean(l))
        clip_limit = 2.6 if mean_l < 95.0 else 1.5
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        merged = cv2.merge((l_eq, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return enhanced
    except Exception:
        return image

class PredictResult(tuple):
    def __new__(cls, label, confidence, status_code="MATCH", status_detail=""):
        return super().__new__(cls, (label, confidence, status_code, status_detail))

    @property
    def label(self):
        return self[0]

    @property
    def confidence(self):
        return self[1]

    @property
    def status_code(self):
        return self[2]

    @property
    def status_detail(self):
        return self[3]

def detect_all_faces(image):
    ih, iw = image.shape[:2]
    max_dim = 1024
    scale = 1.0
    work_img = image
    if max(ih, iw) > max_dim:
        scale = max_dim / float(max(ih, iw))
        work_img = cv2.resize(image, (int(iw * scale), int(ih * scale)))
    wh, ww = work_img.shape[:2]

    candidates = [work_img, enhance_low_light_and_mobile_image(work_img)]
    for candidate_img in candidates:
        all_faces = []
        with _lock:
            try:
                detector = _get_yunet_detector(ww, wh)
                detector.setInputSize((ww, wh))
                detector.setScoreThreshold(0.55)
                ret, faces = detector.detect(candidate_img)
                if faces is not None and len(faces) > 0:
                    for f in faces:
                        face = f.copy()
                        if scale != 1.0:
                            face[:14] = face[:14] / scale
                        all_faces.append(face)
                    if all_faces:
                        all_faces.sort(key=lambda item: item[2] * item[3], reverse=True)
                        return all_faces
            except Exception:
                pass

    for mp_img in [image, enhance_low_light_and_mobile_image(image)]:
        all_faces = []
        try:
            with _lock:
                mp_det = _get_mp_detector()
                rgb = cv2.cvtColor(mp_img, cv2.COLOR_BGR2RGB)
                res = mp_det.process(rgb)
            if res.detections:
                for det in res.detections:
                    bb = det.location_data.relative_bounding_box
                    kp = det.location_data.relative_keypoints
                    bx, by = max(0, int(bb.xmin * iw)), max(0, int(bb.ymin * ih))
                    bw, bh = int(bb.width * iw), int(bb.height * ih)
                    re_x, re_y = kp[0].x * iw, kp[0].y * ih
                    le_x, le_y = kp[1].x * iw, kp[1].y * ih
                    nt_x, nt_y = kp[2].x * iw, kp[2].y * ih
                    mc_x, mc_y = kp[3].x * iw, kp[3].y * ih
                    eye_dist = np.hypot(le_x - re_x, le_y - re_y)
                    dx = (le_x - re_x) / (eye_dist + 1e-6)
                    dy = (le_y - re_y) / (eye_dist + 1e-6)
                    rm_x, rm_y = mc_x - dx * (eye_dist * 0.35), mc_y - dy * (eye_dist * 0.35)
                    lm_x, lm_y = mc_x + dx * (eye_dist * 0.35), mc_y + dy * (eye_dist * 0.35)
                    score = float(det.score[0]) if det.score else 0.95
                    all_faces.append(np.array(
                        [bx, by, bw, bh, re_x, re_y, le_x, le_y, nt_x, nt_y, rm_x, rm_y, lm_x, lm_y, score],
                        dtype=np.float32
                    ))
                if all_faces:
                    all_faces.sort(key=lambda item: item[2] * item[3], reverse=True)
                    return all_faces
        except Exception:
            pass
    return []

def detect_face(image):
    faces = detect_all_faces(image)
    if faces and len(faces) > 0:
        return faces[0]
    return None

def check_image_quality(image, face_15, is_enrollment=False):
    ih, iw = image.shape[:2]
    x, y, w, h = [int(v) for v in face_15[:4]]
    face_crop = image[max(0, y):min(ih, y + h), max(0, x):min(iw, x + w)]
    min_dim = 32 if is_enrollment else 100
    if face_crop.size == 0 or w < min_dim or h < min_dim:
        return False, "Face is too far away. Please move closer to the camera."
    if not is_enrollment:
        face_area_ratio = (w * h) / float(max(1, iw * ih))
        if face_area_ratio < 0.036:
            return False, "Face is too far away. Please move closer to the camera."
    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    norm_gray = cv2.resize(gray_face, (112, 112))
    blur_score = cv2.Laplacian(norm_gray, cv2.CV_64F).var()
    min_blur = 6.0 if is_enrollment else 10.0
    if blur_score < min_blur:
        return False, f"Face is too far or unclear (sharpness: {blur_score:.1f}). Please move closer."

    brightness = float(np.mean(gray_face))
    if brightness < 15.0:
        return False, f"Image is pitch dark (brightness: {brightness:.1f}). Please add minimal light."
    if brightness > 248.0:
        return False, f"Image is completely washed out (brightness: {brightness:.1f})."
    return True, "Quality verified"

def detect_presentation_attack(image, face_15):
    ih, iw = image.shape[:2]
    fx, fy, fw, fh = [int(v) for v in face_15[:4]]
    face_crop = image[max(0, fy):min(ih, fy + fh), max(0, fx):min(iw, fx + fw)]
    if face_crop.size == 0:
        return False, "Live face"

    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    glare_pixels = np.sum(gray_face > 253)
    glare_ratio = glare_pixels / float(face_crop.shape[0] * face_crop.shape[1])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 55, 150)
    min_len = int(min(fw, fh) * 0.85)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 65, minLineLength=min_len, maxLineGap=12)
    top_line = bottom_line = left_line = right_line = False
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            ang = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if ang < 5 or abs(ang - 180) < 5:
                mid_y = (y1 + y2) / 2.0
                min_x, max_x = min(x1, x2), max(x1, x2)
                if (max_x - min_x) > fw * 0.85 and (min_x < (fx + fw * 0.25) and max_x > (fx + fw * 0.75)):
                    if (fy - fh * 0.45) <= mid_y < (fy - 8):
                        top_line = True
                    elif (fy + fh + 8) < mid_y <= (fy + fh * 1.45):
                        bottom_line = True
            elif abs(ang - 90) < 5:
                mid_x = (x1 + x2) / 2.0
                min_y, max_y = min(y1, y2), max(y1, y2)
                if (max_y - min_y) > fh * 0.85 and (min_y < (fy + fh * 0.25) and max_y > (fy + fh * 0.75)):
                    if (fx - fw * 0.45) <= mid_x < (fx - 8):
                        left_line = True
                    elif (fx + fw + 8) < mid_x <= (fx + fw * 1.45):
                        right_line = True
    enclosing_count = sum([top_line, bottom_line, left_line, right_line])

    if glare_ratio > 0.22:
        return True, f"Strong screen glare detected ({glare_ratio * 100:.1f}% saturated pixels)"
    if enclosing_count == 4:
        return True, "Phone/tablet screen or paper boundary detected around face"
    if enclosing_count >= 3 and glare_ratio > 0.06:
        return True, "Display bezel with screen reflection detected"

    ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    if np.std(cr) < 2.2 and np.std(cb) < 2.2:
        return True, "Flat paper printout texture detected"
    return False, "Live face verified"

def compute_embedding_from_image(image, face_15):
    enhanced = enhance_low_light_and_mobile_image(image)
    with _lock:
        recognizer = _get_recognizer()
        aligned = recognizer.alignCrop(enhanced, face_15)
        feat = recognizer.feature(aligned)
        feat = feat.flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

def compute_pose_augmented_embeddings(image, face_15):
    emb_enh = compute_embedding_from_image(image, face_15)
    embeddings = [emb_enh]

    try:
        with _lock:
            recognizer = _get_recognizer()
            aligned_raw = recognizer.alignCrop(image, face_15)
            feat_raw = recognizer.feature(aligned_raw).flatten().astype(np.float32)
            n_raw = np.linalg.norm(feat_raw)
            if n_raw > 0:
                embeddings.append(feat_raw / n_raw)
    except Exception:
        pass

    try:
        ih, iw = image.shape[:2]
        flipped_img = cv2.flip(image, 1)
        f = face_15.copy()
        f[0] = iw - (face_15[0] + face_15[2])
        re_x, re_y = iw - face_15[6], face_15[7]
        le_x, le_y = iw - face_15[4], face_15[5]
        f[4], f[5] = re_x, re_y
        f[6], f[7] = le_x, le_y
        f[8] = iw - face_15[8]
        rm_x, rm_y = iw - face_15[12], face_15[13]
        lm_x, lm_y = iw - face_15[10], face_15[11]
        f[10], f[11] = rm_x, rm_y
        f[12], f[13] = lm_x, lm_y
        emb_flip = compute_embedding_from_image(flipped_img, f)
        embeddings.append(emb_flip)
    except Exception:
        pass
    return embeddings

_model_cache = {"path": None, "mtime": None, "model": None}

def verify_and_extract_all_faces(stream, check_quality=True, check_spoof=True):
    if hasattr(stream, "read"):
        data = stream.read()
    else:
        data = stream
    if not data or len(data) == 0:
        return [{"status": "INVALID_IMAGE", "message": "Failed to decode empty image.", "embedding": None}]
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return [{"status": "INVALID_IMAGE", "message": "Failed to decode image.", "embedding": None}]
    faces = detect_all_faces(image)
    if not faces or len(faces) == 0:
        return [{"status": "NO_FACE", "message": "No face detected. Please look towards the camera.", "embedding": None}]
    results = []
    for face_15 in faces:
        bbox = [int(v) for v in face_15[:4]]
        if check_quality:
            q_ok, q_msg = check_image_quality(image, face_15, is_enrollment=False)
            if not q_ok:
                results.append({
                    "status": "POOR_QUALITY",
                    "message": q_msg,
                    "embedding": None,
                    "face_15": face_15,
                    "bbox": bbox
                })
                continue
        if check_spoof:
            is_spoof, spoof_msg = detect_presentation_attack(image, face_15)
            if is_spoof:
                results.append({
                    "status": "SPOOF_DETECTED",
                    "message": f"Fake attendance detected: {spoof_msg}.",
                    "embedding": None,
                    "face_15": face_15,
                    "bbox": bbox
                })
                continue
        emb = compute_embedding_from_image(image, face_15)
        results.append({
            "status": "OK",
            "message": "Face processed successfully.",
            "embedding": emb,
            "face_15": face_15,
            "bbox": bbox
        })
    return results

def extract_embedding_for_image(stream, check_quality=True, check_spoof=True, return_dict=False):
    all_res = verify_and_extract_all_faces(stream, check_quality=check_quality, check_spoof=check_spoof)
    first = all_res[0] if all_res else {"status": "NO_FACE", "message": "No face detected.", "embedding": None}
    if return_dict:
        return first
    return first.get("embedding") if first.get("status") == "OK" else None

def verify_and_extract_face(stream, check_quality=True, check_spoof=True):
    return extract_embedding_for_image(stream, check_quality=check_quality, check_spoof=check_spoof, return_dict=True)

def validate_and_extract_face_photo(stream_or_bytes):
    if hasattr(stream_or_bytes, "read"):
        data = stream_or_bytes.read()
    else:
        data = stream_or_bytes
    if not data or len(data) == 0:
        return False, "File is empty or corrupted.", None
    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return False, f"Could not decode image format: {e}", None
    if image is None or image.size == 0:
        return False, "Corrupted or unreadable image file.", None
    faces = detect_all_faces(image)
    if not faces or len(faces) == 0:
        return False, "No face detected in photo. Ensure the face is visible.", None
    face_15 = faces[0]
    q_ok, q_msg = check_image_quality(image, face_15, is_enrollment=True)
    if not q_ok:
        return False, q_msg, None
    embs = compute_pose_augmented_embeddings(image, face_15)
    return True, "Valid face photo", embs

def _save_model_safely(model_payload, model_path):
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    tmp_path = model_path + ".tmp"
    bak_path = model_path + ".bak"
    with open(tmp_path, "wb") as f:
        pickle.dump(model_payload, f)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(model_path):
        try:
            shutil.copy2(model_path, bak_path)
        except Exception:
            pass
    os.replace(tmp_path, model_path)
    global _model_cache
    _model_cache = {"path": None, "mtime": None, "model": None}

def update_student_model(sid, new_embeddings, dataset_dir, model_path):
    sid = int(sid)
    with _lock:
        model_payload = None
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and "students" in data:
                    model_payload = data
            except Exception:
                model_payload = None
        if model_payload is None:
            model_payload = {
                "version": "2.4_sface_distance_safe",
                "model_type": "deep_embeddings_cosine",
                "match_threshold": DEFAULT_MATCH_THRESHOLD,
                "margin_threshold": DEFAULT_MARGIN_THRESHOLD,
                "students": {}
            }
        existing_embeddings = []
        if sid in model_payload["students"]:
            existing_embeddings = model_payload["students"][sid].get("embeddings", [])
        all_embeddings = list(existing_embeddings)
        if new_embeddings:
            for item in new_embeddings:
                sub_list = item if isinstance(item, list) else [item]
                for emb in sub_list:
                    emb = np.array(emb, dtype=np.float32).flatten()
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    all_embeddings.append(emb)
        else:
            folder = os.path.join(dataset_dir, str(sid))
            if os.path.isdir(folder):
                all_embeddings = []
                for fname in os.listdir(folder):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        img = cv2.imread(os.path.join(folder, fname))
                        if img is not None:
                            face = detect_face(img)
                            if face is not None:
                                q_ok, _ = check_image_quality(img, face, is_enrollment=True)
                                if q_ok:
                                    all_embeddings.extend(compute_pose_augmented_embeddings(img, face))
        if not all_embeddings:
            if sid in model_payload["students"]:
                del model_payload["students"][sid]
        else:
            matrix = np.stack(all_embeddings)
            centroid = np.mean(matrix, axis=0).astype(np.float32)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            model_payload["students"][sid] = {
                "prototype": centroid,
                "embeddings": all_embeddings,
                "count": len(all_embeddings)
            }
        _save_model_safely(model_payload, model_path)
        return model_payload

def remove_student_from_model(sid, model_path):
    sid = int(sid)
    with _lock:
        if not os.path.exists(model_path):
            return
        try:
            with open(model_path, "rb") as f:
                model_payload = pickle.load(f)
            if isinstance(model_payload, dict) and "students" in model_payload:
                if sid in model_payload["students"]:
                    del model_payload["students"][sid]
                _save_model_safely(model_payload, model_path)
        except Exception:
            pass

def load_model_if_exists(path):
    if not path or not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
        with _lock:
            if _model_cache["path"] == path and _model_cache["mtime"] == mtime:
                return _model_cache["model"]
            with open(path, "rb") as f:
                model_data = pickle.load(f)
            _model_cache["path"] = path
            _model_cache["mtime"] = mtime
            _model_cache["model"] = model_data
            return model_data
    except Exception:
        return _model_cache.get("model")

def predict_with_model(model_data, embedding):
    if not isinstance(model_data, dict) or "students" not in model_data:
        if hasattr(model_data, "predict_proba"):
            probs = model_data.predict_proba([embedding])[0]
            idx = int(np.argmax(probs))
            return PredictResult(model_data.classes_[idx], float(probs[idx]), "LEGACY_PREDICT", "")
        return PredictResult(None, 0.0, "INVALID_MODEL", "Model data is corrupt or incompatible")

    students = model_data["students"]
    if not students:
        return PredictResult(None, 0.0, "NO_STUDENTS", "No students enrolled in model")

    match_threshold = DEFAULT_MATCH_THRESHOLD
    margin_threshold = DEFAULT_MARGIN_THRESHOLD

    q = np.array(embedding, dtype=np.float32).flatten()
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm

    scores = {}
    proto_scores = {}
    for sid, data in students.items():
        proto = data["prototype"]
        proto_score = float(np.dot(q, proto))
        proto_scores[sid] = proto_score
        sample_vectors = data.get("embeddings", [])
        if sample_vectors:
            sample_scores = sorted([float(np.dot(q, v)) for v in sample_vectors], reverse=True)
            top_k = sample_scores[:min(3, len(sample_scores))]
            top_k_mean = float(np.mean(top_k))
            combined_score = 0.30 * proto_score + 0.70 * top_k_mean
        else:
            combined_score = proto_score
        scores[sid] = combined_score

    sorted_matches = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_sid, best_score = sorted_matches[0]

    best_proto = proto_scores.get(best_sid, best_score)
    if best_score < match_threshold or best_proto < (match_threshold - 0.05):
        return PredictResult(
            None,
            best_score,
            "UNKNOWN_PERSON",
            f"Best similarity ({best_score:.3f}) is below verification threshold ({match_threshold:.3f})"
        )

    if len(sorted_matches) > 1:
        second_sid, second_score = sorted_matches[1]
        margin = best_score - second_score
        if margin < margin_threshold:
            return PredictResult(
                None,
                best_score,
                "AMBIGUOUS_MATCH",
                f"Match is ambiguous between student {best_sid} ({best_score:.3f}) and student {second_sid} ({second_score:.3f})"
            )

    return PredictResult(
        best_sid,
        best_score,
        "MATCH",
        f"Verified student ID {best_sid} with similarity {best_score:.3f}"
    )

def train_model_background(dataset_dir, model_path, progress_callback=None):
    student_dirs = [
        d for d in os.listdir(dataset_dir)
        if d.isdigit() and os.path.isdir(os.path.join(dataset_dir, d))
    ]
    if not student_dirs:
        raise ValueError("No student folders found in dataset. Enroll students first.")
    total_dirs = len(student_dirs)
    students_data = {}
    total_images_processed = 0
    for idx, sid in enumerate(student_dirs, start=1):
        folder = os.path.join(dataset_dir, sid)
        student_embeddings = []
        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        for filename in files:
            img_path = os.path.join(folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue
            face_15 = detect_face(image)
            if face_15 is None:
                continue
            q_ok, _ = check_image_quality(image, face_15, is_enrollment=True)
            if not q_ok:
                continue
            student_embeddings.extend(compute_pose_augmented_embeddings(image, face_15))
            total_images_processed += 1
        if student_embeddings:
            matrix = np.stack(student_embeddings)
            centroid = np.mean(matrix, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            students_data[int(sid)] = {
                "prototype": centroid,
                "embeddings": student_embeddings,
                "count": len(student_embeddings)
            }
        if progress_callback:
            progress = int((idx / max(1, total_dirs)) * 85)
            progress_callback(progress, f"Enrolled images for student {idx}/{total_dirs}")
    if not students_data:
        raise ValueError("No usable face photos found in dataset. Please upload clear face photos.")
    model_payload = {
        "version": "2.4_sface_distance_safe",
        "model_type": "deep_embeddings_cosine",
        "match_threshold": DEFAULT_MATCH_THRESHOLD,
        "margin_threshold": DEFAULT_MARGIN_THRESHOLD,
        "students": students_data
    }
    if progress_callback:
        progress_callback(90, "Saving biometric recognition model")
    _save_model_safely(model_payload, model_path)
    if progress_callback:
        progress_callback(100, f"Training complete: {len(students_data)} student(s) enrolled across {total_images_processed} photos")
