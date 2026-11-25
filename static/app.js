document.querySelectorAll('.flash').forEach(el=>setTimeout(()=>{el.style.opacity='0';setTimeout(()=>el.remove(),400)},4500));

document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const startCamBtn = document.getElementById('startCamBtn');
    const stopCamBtn = document.getElementById('stopCamBtn');
    const captureBtn = document.getElementById('captureBtn');
    const camPlaceholder = document.getElementById('camPlaceholder');
    const resTitle = document.getElementById('resTitle');
    const resSub = document.getElementById('resSub');
    const resStatus = document.getElementById('resStatus');
    const resName = document.getElementById('resName');
    const resMeta = document.getElementById('resMeta');

    let stream = null;
    let autoLoop = null;
    let isProcessing = false;
    let activeStudent = null;
    let lastConfirmedTime = 0;
    let unknownStreak = 0;
    let spoofStreak = 0;
    let noFaceStreak = 0;

    // ── Multi-frame liveness: track pixel changes between consecutive captures ──
    let prevFrameData = null;
    let motionHistory = [];        // rolling window of motion scores
    const MOTION_WINDOW = 5;       // require motion evidence over last 5 frames
    const MIN_MOTION_SCORE = 0.008; // minimum average pixel change ratio for liveness
    let livenessConfirmed = false;
    let livenessFailStreak = 0;
    const startClassBtn = document.getElementById('startClassBtn');
    const endClassBtn = document.getElementById('endClassBtn');
    const courseSelect = document.getElementById('courseSelect');
    const classTimer = document.getElementById('classTimer');
    let classTimerLoop = null;
    async function refreshClassTimer() {
        if (!classTimer) return;
        try {
            const r = await fetch('/active_class'); const d = await r.json();
            if (!d.active) { classTimer.textContent = d.course_code ? 'Attendance window closed' : 'Class not started'; if (classTimerLoop) { clearInterval(classTimerLoop); classTimerLoop = null; } return; }
            const mins = Math.floor(d.remaining_seconds / 60), secs = d.remaining_seconds % 60;
            classTimer.textContent = d.course_code + ' · ' + mins + ':' + String(secs).padStart(2, '0') + ' remaining';
        } catch (e) { classTimer.textContent = 'Timer unavailable'; }
    }
    if (startClassBtn) startClassBtn.addEventListener('click', async () => {
        if (!courseSelect || !courseSelect.value) { alert('Please select a course first.'); return; }
        startClassBtn.disabled = true;
        try {
            const r = await fetch('/start_class', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({course_code:courseSelect.value})});
            const d = await r.json(); if (!r.ok) throw new Error(d.error || 'Could not start class.');
            if (classTimerLoop) clearInterval(classTimerLoop); await refreshClassTimer(); classTimerLoop = setInterval(refreshClassTimer, 1000);
            if (startCamBtn) startCamBtn.click();
        } catch (e) { alert(e.message); } finally { startClassBtn.disabled = false; }
    });
    if (endClassBtn) endClassBtn.addEventListener('click', async () => {
        await fetch('/end_class', {method:'POST'}); if (classTimerLoop) clearInterval(classTimerLoop); classTimerLoop = null;
        if (autoLoop) { clearInterval(autoLoop); autoLoop = null; } if (classTimer) classTimer.textContent = 'Class ended';
    });
    refreshClassTimer();

    async function captureAndRecognize() {
        if (!stream || !video || video.readyState !== 4 || isProcessing) return;
        isProcessing = true;
        try {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // ── Multi-frame liveness: detect natural head/face motion ──
            // A static photo on a phone screen produces near-zero pixel change
            // between frames; a real live person always has micro-movements.
            const currentFrameData = ctx.getImageData(
                Math.floor(canvas.width * 0.25), Math.floor(canvas.height * 0.2),
                Math.floor(canvas.width * 0.5), Math.floor(canvas.height * 0.6)
            );
            if (prevFrameData && prevFrameData.data.length === currentFrameData.data.length) {
                let diffSum = 0;
                const len = currentFrameData.data.length;
                // Sample every 16th pixel for performance (RGBA, so step by 64)
                for (let px = 0; px < len; px += 64) {
                    diffSum += Math.abs(currentFrameData.data[px] - prevFrameData.data[px]);
                    diffSum += Math.abs(currentFrameData.data[px + 1] - prevFrameData.data[px + 1]);
                    diffSum += Math.abs(currentFrameData.data[px + 2] - prevFrameData.data[px + 2]);
                }
                const sampledPixels = Math.floor(len / 64) * 3;
                const motionScore = diffSum / (sampledPixels * 255);
                motionHistory.push(motionScore);
                if (motionHistory.length > MOTION_WINDOW) motionHistory.shift();

                const avgMotion = motionHistory.reduce((a, b) => a + b, 0) / motionHistory.length;
                if (avgMotion >= MIN_MOTION_SCORE) {
                    livenessConfirmed = true;
                    livenessFailStreak = 0;
                } else if (motionHistory.length >= MOTION_WINDOW) {
                    livenessFailStreak += 1;
                    // After several consecutive static frames, treat as a photo
                    if (livenessFailStreak >= 3) {
                        livenessConfirmed = false;
                    }
                }
            }
            prevFrameData = currentFrameData;

            // If liveness check consistently fails (static image), block without
            // even sending to server — saves bandwidth and server CPU.
            if (motionHistory.length >= MOTION_WINDOW && !livenessConfirmed && livenessFailStreak >= 4) {
                if (resTitle) resTitle.textContent = 'Liveness Check Failed';
                if (resSub) resSub.textContent = 'No natural movement detected. Please present your real face, not a photo.';
                if (resStatus) {
                    resStatus.textContent = 'PHOTO DETECTED';
                    resStatus.style.color = '#dc6470';
                }
                if (resName) resName.textContent = '—';
                if (resMeta) resMeta.textContent = 'Move your head slightly to confirm you are a real person';
                activeStudent = null;
                isProcessing = false;
                return;
            }

            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.90));
            if (!blob) return;

            const formData = new FormData();
            formData.append('image', blob, 'capture.jpg');

            const resp = await fetch('/recognize_face', { method: 'POST', body: formData });
            const data = await resp.json();

            if (data.class_not_started || data.time_expired) {
                if (resTitle) resTitle.textContent = data.title || 'Class not active';
                if (resSub) resSub.textContent = data.error || 'Start a class to record attendance.';
                if (resStatus) resStatus.textContent = data.status || 'INACTIVE';
                if (resName) resName.textContent = '—';
                if (resMeta) resMeta.textContent = 'Attendance is not being recorded';
                if (data.time_expired && autoLoop) { clearInterval(autoLoop); autoLoop = null; }
                return;
            }
            if (data.recognized) {
                const isSameStudent = activeStudent && activeStudent.name === data.name;
                const showAlreadyMarked = isSameStudent ? activeStudent.already_marked : Boolean(data.already_marked);

                activeStudent = {
                    name: data.name,
                    already_marked: showAlreadyMarked,
                    timestamp: data.timestamp || (activeStudent ? activeStudent.timestamp : '')
                };
                lastConfirmedTime = Date.now();
                unknownStreak = 0;
                spoofStreak = 0;
                noFaceStreak = 0;

                if (resTitle) resTitle.textContent = data.name;
                if (resSub) {
                    resSub.textContent = showAlreadyMarked
                        ? 'Already marked! Attendance was already recorded today.'
                        : 'Attendance marked! Keep your face in view or step back when done.';
                }
                if (resStatus) {
                    resStatus.textContent = showAlreadyMarked ? 'ALREADY MARKED' : 'ATTENDANCE MARKED';
                    resStatus.style.color = showAlreadyMarked ? '#d99428' : '#159b72';
                }
                if (resName) resName.textContent = data.name;
                if (resMeta) resMeta.textContent = `Match Confidence: ${(data.confidence * 100).toFixed(1)}% · ${activeStudent.timestamp}`;
            } else if (data.is_spoof) {
                noFaceStreak = 0;
                spoofStreak += 1;
                // Require 2 consecutive spoof frames (or 3 if already locked onto a real live student) to prevent lighting flicker
                const requiredSpoofFrames = activeStudent ? 3 : 2;
                if (spoofStreak < requiredSpoofFrames) {
                    return;
                }
                activeStudent = null;
                unknownStreak = 0;
                if (resTitle) resTitle.textContent = data.title || 'Fake Attendance Detected';
                if (resSub) resSub.textContent = data.error || 'Spoof attempt blocked.';
                if (resStatus) {
                    resStatus.textContent = 'REJECTED';
                    resStatus.style.color = '#dc6470';
                }
                if (resName) resName.textContent = '—';
                if (resMeta) resMeta.textContent = 'Live person required';
            } else if (data.too_far) {
                activeStudent = null;
                unknownStreak = 0;
                spoofStreak = 0;
                noFaceStreak = 0;
                if (resTitle) resTitle.textContent = 'Move Closer to Camera';
                if (resSub) resSub.textContent = 'Face is too far away. Please bring your face closer to the camera.';
                if (resStatus) {
                    resStatus.textContent = 'BRING FACE CLOSER';
                    resStatus.style.color = '#d99428';
                }
                if (resName) resName.textContent = '—';
                if (resMeta) resMeta.textContent = 'Step closer to scan';
            } else if (data.no_face) {
                unknownStreak = 0;
                spoofStreak = 0;
                noFaceStreak += 1;
                if (activeStudent && noFaceStreak === 1) {
                    if (resTitle) resTitle.textContent = 'Bring Face Closer';
                    if (resSub) resSub.textContent = 'Face moved away from camera. Bring your face closer to scan again.';
                    if (resStatus) {
                        resStatus.textContent = 'BRING FACE CLOSER';
                        resStatus.style.color = '#d99428';
                    }
                    if (resName) resName.textContent = '—';
                    if (resMeta) resMeta.textContent = 'No face in frame';
                    return;
                }
                activeStudent = null;
                if (resTitle) resTitle.textContent = 'No Face Detected';
                if (resSub) resSub.textContent = 'No face in front of the camera. Please look at the camera.';
                if (resStatus) {
                    resStatus.textContent = 'NO FACE';
                    resStatus.style.color = '#8a94a6';
                }
                if (resName) resName.textContent = '—';
                if (resMeta) resMeta.textContent = 'Waiting for face';
            } else if (data.unknown) {
                noFaceStreak = 0;
                spoofStreak = 0;
                unknownStreak += 1;
                // Anti-flicker lock: keep active student steady during brief head turn or motion blur
                if (activeStudent && unknownStreak < 4 && (Date.now() - lastConfirmedTime < 5000)) {
                    return;
                }
                activeStudent = null;
                if (resTitle) resTitle.textContent = 'Unknown Person';
                if (resSub) resSub.textContent = data.error || 'Unregistered person. Access denied.';
                if (resStatus) {
                    resStatus.textContent = 'UNKNOWN PERSON';
                    resStatus.style.color = '#dc6470';
                }
                if (resName) resName.textContent = 'Unknown';
                if (resMeta) resMeta.textContent = data.confidence ? `Similarity: ${(data.confidence * 100).toFixed(1)}% (Below threshold)` : 'Not registered';
            } else {
                if (activeStudent && (Date.now() - lastConfirmedTime < 3500)) {
                    return;
                }
                activeStudent = null;
                if (resTitle) resTitle.textContent = data.title || 'Scanning...';
                if (resSub) resSub.textContent = data.error || 'Position face in front of the camera.';
                if (resStatus) {
                    resStatus.textContent = data.status || 'SCANNING';
                    resStatus.style.color = '#d99428';
                }
                if (resName) resName.textContent = '—';
                if (resMeta) resMeta.textContent = 'Waiting for face';
            }
        } catch (err) {
            console.error('Recognition error:', err);
        } finally {
            isProcessing = false;
        }
    }

    if (startCamBtn && video) {
        startCamBtn.addEventListener('click', async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
                video.srcObject = stream;
                if (camPlaceholder) camPlaceholder.style.display = 'none';
                activeStudent = null;
                lastConfirmedTime = 0;
                unknownStreak = 0;
                spoofStreak = 0;
                noFaceStreak = 0;
                prevFrameData = null;
                motionHistory = [];
                livenessConfirmed = false;
                livenessFailStreak = 0;
                if (resTitle) resTitle.textContent = 'No Face Detected';
                if (resSub) resSub.textContent = 'Looking for a face in front of the camera...';
                if (resStatus) {
                    resStatus.textContent = 'NO FACE';
                    resStatus.style.color = '#8a94a6';
                }
                if (autoLoop) clearInterval(autoLoop);
                autoLoop = setInterval(captureAndRecognize, 1100);
            } catch (err) {
                alert('Could not access camera: ' + err.message);
            }
        });
    }

    if (stopCamBtn && video) {
        stopCamBtn.addEventListener('click', () => {
            if (autoLoop) { clearInterval(autoLoop); autoLoop = null; }
            if (stream) {
                stream.getTracks().forEach(t => t.stop());
                stream = null;
            }
            video.srcObject = null;
            activeStudent = null;
            unknownStreak = 0;
            spoofStreak = 0;
            noFaceStreak = 0;
            prevFrameData = null;
            motionHistory = [];
            livenessConfirmed = false;
            livenessFailStreak = 0;
            if (camPlaceholder) camPlaceholder.style.display = 'grid';
            if (resTitle) resTitle.textContent = 'Recognition result';
            if (resSub) resSub.textContent = 'Your recognition result will appear here automatically.';
            if (resStatus) {
                resStatus.textContent = 'Camera stopped';
                resStatus.style.color = '';
            }
            if (resName) resName.textContent = '—';
            if (resMeta) resMeta.textContent = 'Ready to scan';
        });
    }

    if (captureBtn) {
        captureBtn.addEventListener('click', captureAndRecognize);
    }

    const trainBtn = document.getElementById('trainBtn');
    const trainMsg = document.getElementById('trainMsg');
    if (trainBtn) {
        trainBtn.addEventListener('click', async () => {
            trainBtn.disabled = true;
            if (trainMsg) trainMsg.textContent = 'Syncing model...';
            try {
                const r = await fetch('/train_model', { method: 'POST' });
                const d = await r.json();
                if (trainMsg) trainMsg.textContent = 'Model sync: ' + (d.status || 'done');
                setTimeout(() => location.reload(), 1500);
            } catch (e) {
                if (trainMsg) trainMsg.textContent = 'Sync failed.';
            } finally {
                trainBtn.disabled = false;
            }
        });
    }

    const enrollModal = document.getElementById('enrollModal');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const enrollStudentId = document.getElementById('enrollStudentId');
    const enrollStudentName = document.getElementById('enrollStudentName');
    const enrollForm = document.getElementById('enrollForm');
    const enrollMsg = document.getElementById('enrollMsg');

    document.querySelectorAll('.open-enroll').forEach(btn => {
        btn.addEventListener('click', () => {
            if (enrollStudentId) enrollStudentId.value = btn.dataset.id;
            if (enrollStudentName) enrollStudentName.textContent = btn.dataset.name;
            if (enrollMsg) enrollMsg.textContent = '';
            if (enrollModal) enrollModal.classList.add('show');
        });
    });

    if (closeModalBtn && enrollModal) {
        closeModalBtn.addEventListener('click', () => enrollModal.classList.remove('show'));
    }

    if (enrollForm) {
        enrollForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(enrollForm);
            if (enrollMsg) enrollMsg.textContent = 'Uploading & updating AI model automatically...';
            try {
                const resp = await fetch('/upload_face', { method: 'POST', body: formData });
                const data = await resp.json();
                if (resp.ok) {
                    if (enrollMsg) enrollMsg.textContent = data.message || 'Enrolled successfully!';
                    setTimeout(() => location.reload(), 1200);
                } else {
                    if (enrollMsg) enrollMsg.textContent = data.error || 'Upload failed.';
                }
            } catch (err) {
                if (enrollMsg) enrollMsg.textContent = 'Error uploading photos.';
            }
        });
    }
});
