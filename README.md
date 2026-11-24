# SmartAttend AI — Multi-Teacher Workspace (Development Build)

This build is based on the supplied working Batch 64 Flask application. Existing face-recognition and 30-minute course-wise attendance code is retained in the separate working copy. The original ZIP is not modified.

## Windows setup

1. Use Python 3.11 (recommended for the pinned MediaPipe version), extract the ZIP, and open `SmartAttend_Flask_Website` in VS Code.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install Python dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Install the Tesseract OCR application separately and ensure `tesseract.exe` is on PATH. For Bengali + English OCR, install the Bengali (`ben`) and English (`eng`) language data.
5. Run:
   ```powershell
   python app.py
   ```
6. Open `http://127.0.0.1:5000`, choose **Create an account**, and register as a teacher.

## Features in this development build

- Teacher sign-up and login using email + password. Passwords are stored as hashes.
- Separate SQLite database, face dataset, and recognition model file for each teacher account. Teacher workspaces do not share student or face data.
- Manual student registration and existing Excel class import.
- Class-specific student registration links. Student ID duplicates are checked within the class; self-registrations require biometric consent and stay pending until teacher approval.
- Optional face photo on student self-registration; it is processed into the teacher's face model after approval.
- Course creation and course-wise attendance with the existing 30-minute attendance window.

## Important limitations before public deployment

- This is a local development build, not a production-hosted public service. Deploy behind HTTPS with a persistent strong `SECRET_KEY`, set `COOKIE_SECURE=1` behind HTTPS, add CSRF protection and rate limiting, backups, and a production WSGI server before public use.
- Email verification, email-based password reset, and Google Sign-In are not configured because no email/OAuth provider credentials were supplied.
- Student identity is not independently verified against an official university database. Without an official student list, the system only checks duplicates and waits for teacher approval.
- Face images and embeddings are biometric data. Obtain informed consent, define retention/deletion rules, restrict access, and follow applicable institutional and legal requirements.
- The existing recognition model (OpenCV YuNet/SFace pipeline in the original project) is preserved; it is not a newly trained deep-learning model. Recognition accuracy must be tested with the target classroom cameras and student population.

## Data storage

- `accounts.db`: teacher accounts and registration-link metadata.
- `tenant_data/<teacher_id>/attendance.db`: that teacher's students, attendance, and courses.
- `tenant_data/<teacher_id>/dataset/` and `model.pkl`: that teacher's face photos/model.
- Back up the complete application folder, especially `accounts.db` and `tenant_data/`.