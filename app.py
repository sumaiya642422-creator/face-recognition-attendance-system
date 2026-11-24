import os, io, csv, json, sqlite3, threading, secrets, functools
from datetime import datetime, date, timedelta
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, session, abort, has_request_context
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from openpyxl import load_workbook
from PIL import Image
from model import (
    train_model_background,
    extract_embedding_for_image,
    load_model_if_exists,
    predict_with_model,
    verify_and_extract_face,
    verify_and_extract_all_faces,
    validate_and_extract_face_photo,
    update_student_model,
    remove_student_from_model
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_DB_PATH = os.path.join(BASE_DIR, "attendance.db")
ACCOUNT_DB_PATH = os.path.join(BASE_DIR, "accounts.db")
TENANT_ROOT = os.path.join(BASE_DIR, "tenant_data")
DB_PATH = LEGACY_DB_PATH
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
STATUS_PATH = os.path.join(BASE_DIR, "train_status.json")

def tenant_id(): return str(session.get("tenant_id", "legacy")) if has_request_context() else "legacy"
def tenant_root():
    if tenant_id() == "legacy": return BASE_DIR
    path = os.path.join(TENANT_ROOT, tenant_id()); os.makedirs(path, exist_ok=True); return path
def dataset_dir():
    path=os.path.join(tenant_root(), "dataset"); os.makedirs(path, exist_ok=True); return path
def model_path(): return os.path.join(tenant_root(), "model.pkl")
def status_path(): return os.path.join(tenant_root(), "train_status.json")
def login_required(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if not session.get("teacher_id"): return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapped

def account_db():
    con=sqlite3.connect(ACCOUNT_DB_PATH); con.row_factory=sqlite3.Row; return con

def init_accounts():
    os.makedirs(TENANT_ROOT, exist_ok=True)
    with account_db() as con:
        con.execute("CREATE TABLE IF NOT EXISTS teachers (id INTEGER PRIMARY KEY AUTOINCREMENT,email TEXT NOT NULL UNIQUE,password_hash TEXT NOT NULL,institution TEXT NOT NULL,created_at TEXT NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS registration_links (token TEXT PRIMARY KEY,teacher_id INTEGER NOT NULL,class_name TEXT NOT NULL,section TEXT DEFAULT '',created_at TEXT NOT NULL)")

def create_tenant_database(tid):
    folder=os.path.join(TENANT_ROOT,str(tid)); os.makedirs(os.path.join(folder,"dataset"),exist_ok=True)
    con=sqlite3.connect(os.path.join(folder,"attendance.db")); con.row_factory=sqlite3.Row
    con.execute("CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,roll TEXT,class_name TEXT,section TEXT,reg_no TEXT,created_at TEXT NOT NULL,email TEXT DEFAULT '',approval_status TEXT NOT NULL DEFAULT 'approved',biometric_consent INTEGER NOT NULL DEFAULT 0)")
    con.execute("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT,student_id INTEGER NOT NULL,name TEXT NOT NULL,timestamp TEXT NOT NULL,course_code TEXT,FOREIGN KEY(student_id) REFERENCES students(id))")
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_attendance_student_course_daily ON attendance(student_id,course_code,date(timestamp)) WHERE course_code IS NOT NULL")
    con.execute("CREATE TABLE IF NOT EXISTS active_class (id INTEGER PRIMARY KEY CHECK(id=1),course_code TEXT NOT NULL,started_at TEXT NOT NULL)")
    scol=[r[1] for r in con.execute("PRAGMA table_info(students)").fetchall()]
    if "biometric_consent" not in scol: con.execute("ALTER TABLE students ADD COLUMN biometric_consent INTEGER NOT NULL DEFAULT 0")
    con.execute("CREATE TABLE IF NOT EXISTS courses (id INTEGER PRIMARY KEY AUTOINCREMENT,course_code TEXT NOT NULL UNIQUE,course_name TEXT NOT NULL DEFAULT '')")
    con.commit(); con.close()

ALLOWED = {"jpg", "jpeg", "png"}

os.makedirs(DATASET_DIR, exist_ok=True)
app = Flask(__name__)
_secret_file=os.path.join(BASE_DIR,".secret_key")
if not os.path.exists(_secret_file):
    with open(_secret_file,"w",encoding="utf-8") as _f: _f.write(secrets.token_hex(32))
app.secret_key = os.environ.get("SECRET_KEY") or open(_secret_file,encoding="utf-8").read().strip()
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE") == "1")

def db():
    path=LEGACY_DB_PATH if tenant_id()=="legacy" else os.path.join(TENANT_ROOT,tenant_id(),"attendance.db")
    conn=sqlite3.connect(path); conn.row_factory=sqlite3.Row; return conn

def init_db():
    with db() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll TEXT,
            class_name TEXT,
            section TEXT,
            reg_no TEXT,
            created_at TEXT NOT NULL)''')
        con.execute('''CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            course_code TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id))''')
        # Additive migration: keep legacy rows; course attendance is unique per day.
        cols = [r[1] for r in con.execute("PRAGMA table_info(attendance)").fetchall()]
        if "course_code" not in cols:
            con.execute("ALTER TABLE attendance ADD COLUMN course_code TEXT")
        con.execute("DROP INDEX IF EXISTS idx_attendance_student_daily")
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_attendance_student_course_daily ON attendance(student_id, course_code, date(timestamp)) WHERE course_code IS NOT NULL")
        con.execute("CREATE TABLE IF NOT EXISTS active_class (id INTEGER PRIMARY KEY CHECK(id=1), course_code TEXT NOT NULL, started_at TEXT NOT NULL)")
        scol = [r[1] for r in con.execute("PRAGMA table_info(students)").fetchall()]
        if "email" not in scol: con.execute("ALTER TABLE students ADD COLUMN email TEXT DEFAULT ''")
        if "approval_status" not in scol: con.execute("ALTER TABLE students ADD COLUMN approval_status TEXT NOT NULL DEFAULT 'approved'")
        if "biometric_consent" not in scol: con.execute("ALTER TABLE students ADD COLUMN biometric_consent INTEGER NOT NULL DEFAULT 0")
        con.execute("CREATE TABLE IF NOT EXISTS courses (id INTEGER PRIMARY KEY AUTOINCREMENT, course_code TEXT NOT NULL UNIQUE, course_name TEXT NOT NULL DEFAULT '')")

init_accounts()
init_db()

def read_status():
    path=status_path()
    if not os.path.exists(path):
        return {"running": False, "progress": 0, "message": "No training yet"}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"running": False, "progress": 0, "message": "Status unavailable"}

def write_status(data):
    with open(status_path(), "w", encoding="utf-8") as f:
        json.dump(data, f)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email=request.form.get("email", "").strip().lower(); password=request.form.get("password", ""); institution=request.form.get("institution", "").strip()
        if not email or "@" not in email or len(password)<8 or not institution:
            flash("Enter a valid email, institution, and password with at least 8 characters.", "error")
            return render_template("auth.html", mode="signup")
        try:
            with account_db() as con:
                cur=con.execute("INSERT INTO teachers(email,password_hash,institution,created_at) VALUES(?,?,?,?)",(email,generate_password_hash(password),institution,datetime.now().isoformat(timespec="seconds")))
                tid=cur.lastrowid
            create_tenant_database(tid); session.clear(); session["teacher_id"]=tid; session["tenant_id"]=str(tid)
            return redirect(url_for("index"))
        except sqlite3.IntegrityError:
            flash("This email already has an account. Please log in.", "error")
        except Exception:
            app.logger.exception("Signup failed"); flash("Account could not be created. Please try again.", "error")
    return render_template("auth.html", mode="signup")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email=request.form.get("email", "").strip().lower(); password=request.form.get("password", "")
        with account_db() as con: teacher=con.execute("SELECT * FROM teachers WHERE email=?",(email,)).fetchone()
        if teacher and check_password_hash(teacher["password_hash"],password):
            session.clear(); session["teacher_id"]=teacher["id"]; session["tenant_id"]=str(teacher["id"]); create_tenant_database(teacher["id"])
            return redirect(url_for("index"))
        flash("Email or password is incorrect.", "error")
    return render_template("auth.html", mode="login")

@app.route("/logout")
def logout(): session.clear(); return redirect(url_for("login"))

@app.route("/registration-link", methods=["GET", "POST"])
@login_required
def registration_link():
    if request.method=="POST":
        class_name=request.form.get("class_name", "").strip(); section=request.form.get("section", "").strip()
        if class_name:
            token=secrets.token_urlsafe(24)
            with account_db() as con: con.execute("INSERT INTO registration_links(token,teacher_id,class_name,section,created_at) VALUES(?,?,?,?,?)",(token,session["teacher_id"],class_name,section,datetime.now().isoformat(timespec="seconds")))
            flash("Registration link: "+url_for("student_register",token=token,_external=True),"success")
        else: flash("Class or batch is required.","error")
    with account_db() as con: links=con.execute("SELECT * FROM registration_links WHERE teacher_id=? ORDER BY rowid DESC",(session["teacher_id"],)).fetchall()
    return render_template("registration_link.html",links=links)

@app.route("/register-student/<token>", methods=["GET", "POST"])
def student_register(token):
    with account_db() as con: link=con.execute("SELECT * FROM registration_links WHERE token=?",(token,)).fetchone()
    if not link: abort(404)
    if request.method=="POST":
        name=request.form.get("name","").strip(); student_id=request.form.get("student_id","").strip(); email=request.form.get("email","").strip()
        consent=request.form.get("biometric_consent") == "yes"
        if not name or not student_id or not consent: flash("Name, Student ID, and biometric consent are required.","error")
        else:
            create_tenant_database(link["teacher_id"])
            old_tid=session.get("tenant_id"); old_teacher=session.get("teacher_id")
            session["tenant_id"]=str(link["teacher_id"]); session["teacher_id"]=link["teacher_id"]
            try:
                with db() as con:
                    dup=con.execute("SELECT 1 FROM students WHERE reg_no=? AND class_name=?",(student_id,link["class_name"])).fetchone()
                    if dup: flash("This Student ID is already registered for this class.","error")
                    else:
                        cur=con.execute("INSERT INTO students(name,roll,class_name,section,reg_no,created_at,email,approval_status,biometric_consent) VALUES(?,?,?,?,?,?,?,'pending',1)",(name,student_id,link["class_name"],link["section"],student_id,datetime.now().isoformat(timespec="seconds"),email))
                        sid=cur.lastrowid
                        photo=request.files.get("face_photo")
                        if photo and photo.filename:
                            ext=photo.filename.rsplit('.',1)[-1].lower()
                            if ext in ALLOWED:
                                folder=os.path.join(tenant_root(),"pending_photos",str(sid)); os.makedirs(folder,exist_ok=True)
                                photo.save(os.path.join(folder,secure_filename("registration."+ext)))
                        flash("Registration submitted. Your teacher must approve it before attendance is enabled.","success")
            finally:
                if old_tid is None: session.pop("tenant_id",None)
                else: session["tenant_id"]=old_tid
                if old_teacher is None: session.pop("teacher_id",None)
                else: session["teacher_id"]=old_teacher
    return render_template("student_register.html",link=link)

@app.route("/approvals")
@login_required
def approvals():
    with db() as con: students=con.execute("SELECT * FROM students WHERE approval_status='pending' ORDER BY id DESC").fetchall()
    return render_template("approvals.html",students=students)

@app.route("/approve-student/<int:sid>",methods=["POST"])
@login_required
def approve_student(sid):
    with db() as con:
        student=con.execute("SELECT * FROM students WHERE id=? AND approval_status='pending'",(sid,)).fetchone()
        if student: con.execute("UPDATE students SET approval_status='approved' WHERE id=?",(sid,))
    folder=os.path.join(tenant_root(),"pending_photos",str(sid)); saved=0; embeddings=[]
    if os.path.isdir(folder):
        import shutil
        target=os.path.join(dataset_dir(),str(sid)); os.makedirs(target,exist_ok=True)
        for fn in os.listdir(folder):
            path=os.path.join(folder,fn)
            try:
                raw=open(path,'rb').read(); valid,msg,emb=validate_and_extract_face_photo(raw)
                if valid and emb is not None:
                    shutil.move(path,os.path.join(target,fn)); embeddings.append(emb); saved+=1
            except Exception: app.logger.exception("Pending face photo processing failed")
        shutil.rmtree(folder,ignore_errors=True)
    if embeddings:
        try: update_student_model(sid,embeddings,dataset_dir(),model_path())
        except Exception: app.logger.exception("Approved student model update failed")
    flash("Student approved. Face photos enrolled: %d."%saved,"success")
    return redirect(url_for("approvals"))

@app.route("/courses",methods=["GET","POST"])
@login_required
def courses_page():
    if request.method=="POST":
        code=request.form.get("course_code","").strip().upper(); name=request.form.get("course_name","").strip()
        if code:
            with db() as con: con.execute("INSERT OR IGNORE INTO courses(course_code,course_name) VALUES(?,?)",(code,name or code))
            flash("Course saved.","success")
    with db() as con: courses=con.execute("SELECT * FROM courses ORDER BY course_code").fetchall()
    return render_template("courses.html",courses=courses)

@app.route("/")
@login_required
def index():
    model_data = load_model_if_exists(model_path())
    is_trained = model_data is not None and bool(model_data.get("students"))
    with db() as con:
        students = con.execute("SELECT COUNT(*) n FROM students WHERE approval_status='approved'").fetchone()["n"]
        today_count = con.execute(
            "SELECT COUNT(DISTINCT student_id) n FROM attendance WHERE date(timestamp)=date('now','localtime')"
        ).fetchone()["n"]
        recent = con.execute("SELECT name, timestamp FROM attendance ORDER BY id DESC LIMIT 6").fetchall()
        last_days = []
        for offset in range(6, -1, -1):
            d = date.today() - timedelta(days=offset)
            count = con.execute("SELECT COUNT(DISTINCT student_id) n FROM attendance WHERE date(timestamp)=?", (d.isoformat(),)).fetchone()["n"]
            last_days.append({"label": d.strftime("%a"), "count": count})
    return render_template("index.html", student_count=students, today_count=today_count, recent=recent, last_days=last_days, trained=is_trained, train_status=read_status())

@app.route("/students")
@login_required
def students_page():
    with db() as con:
        students = con.execute("""SELECT * FROM students WHERE approval_status='approved'
            ORDER BY
                CASE
                    WHEN TRIM(COALESCE(roll, '')) <> ''
                         AND TRIM(roll) NOT GLOB '*[^0-9]*' THEN 0
                    ELSE 1
                END,
                CASE
                    WHEN TRIM(COALESCE(roll, '')) <> ''
                         AND TRIM(roll) NOT GLOB '*[^0-9]*'
                    THEN CAST(TRIM(roll) AS INTEGER)
                    ELSE NULL
                END,
                TRIM(COALESCE(roll, '')) COLLATE NOCASE,
                id ASC
        """).fetchall()
    return render_template("students.html", students=students)

@app.route("/add_student", methods=["GET", "POST"])
@login_required
def add_student():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            flash("Student name is required.", "error")
            return redirect(url_for("add_student"))
        with db() as con:
            cur = con.execute('''INSERT INTO students(name,roll,class_name,section,reg_no,created_at,approval_status) VALUES(?,?,?,?,?,? ,'approved')''', (
                name,
                request.form.get("roll","").strip(),
                request.form.get("class_name","").strip(),
                request.form.get("section","").strip(),
                request.form.get("reg_no","").strip(),
                datetime.now().isoformat(timespec="seconds")))
            sid = cur.lastrowid
        folder = os.path.join(dataset_dir(), str(sid))
        os.makedirs(folder, exist_ok=True)

        files = request.files.getlist("images[]") or request.files.getlist("images")
        saved_files = []
        valid_embeddings = []
        reject_reasons = []
        for f in files:
            if not f or not f.filename:
                continue
            ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
            if ext not in ALLOWED:
                reject_reasons.append(f"{f.filename}: only JPG/PNG supported")
                continue
            raw_bytes = f.read()
            if not raw_bytes:
                continue
            is_valid, msg, emb = validate_and_extract_face_photo(raw_bytes)
            if is_valid and emb is not None:
                safe_name = secure_filename(f"{int(datetime.now().timestamp()*1000)}_{len(saved_files)}.{ext}")
                with open(os.path.join(folder, safe_name), "wb") as out_f:
                    out_f.write(raw_bytes)
                saved_files.append(safe_name)
                valid_embeddings.append(emb)
            else:
                reject_reasons.append(f"{f.filename}: {msg}")

        if saved_files:
            try:
                update_student_model(sid, valid_embeddings, dataset_dir(), model_path())
                flash(f"{name} was registered and recognition model was automatically updated with {len(saved_files)} photo(s).", "success")
            except Exception as e:
                flash(f"{name} was registered, but automatic model update encountered an issue: {e}", "error")
        else:
            if reject_reasons:
                flash(f"{name} was added, but photo(s) were skipped ({'; '.join(reject_reasons[:2])}). Try Enroll face.", "error")
            else:
                flash(f"{name} was added. Upload face photos to enroll this student.", "success")
        return redirect(url_for("students_page"))
    return render_template("add_student.html")

@app.route("/download_student_template")
@login_required
def download_student_template():
    from openpyxl import Workbook
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Students"
    sheet.append(["Name", "Roll", "Class", "Section", "Reg No"])
    sheet.append(["Example Student", "101", "CSE-12", "A", "2026-001"])
    output = io.BytesIO()
    workbook.save(output)
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name="SmartAttend_Student_Template.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.route("/upload_excel_students", methods=["POST"])
@login_required
def upload_excel_students():
    file = request.files.get("excel_file")
    if not file or not file.filename:
        flash("Please select an Excel (.xlsx) file.", "error")
        return redirect(url_for("add_student"))

    if not file.filename.lower().endswith(".xlsx"):
        flash("Only Excel .xlsx files are supported. Please upload an .xlsx workbook.", "error")
        return redirect(url_for("add_student"))

    try:
        workbook = load_workbook(file.stream, read_only=True, data_only=True)
        sheet = workbook.active
        rows = sheet.iter_rows(values_only=True)
        headers = next(rows, None)
        if not headers:
            workbook.close()
            flash("The Excel file is empty.", "error")
            return redirect(url_for("add_student"))

        def normalize(value):
            return " ".join(str(value or "").strip().lower().replace("_", " ").split())

        header_map = {normalize(value): index for index, value in enumerate(headers) if value is not None}
        aliases = {
            "name": ("name", "student name", "full name"),
            "roll": ("roll", "student roll", "roll no", "roll number"),
            "class_name": ("class", "class name", "batch"),
            "section": ("section",),
            "reg_no": ("reg no", "registration no", "registration number", "reg number", "student id"),
        }
        columns = {}
        for field, names in aliases.items():
            for alias in names:
                if alias in header_map:
                    columns[field] = header_map[alias]
                    break

        if "name" not in columns:
            workbook.close()
            flash("The Excel file must contain a Name column. Other supported columns: Roll, Class, Section, Reg No.", "error")
            return redirect(url_for("add_student"))

        imported = []
        for row in rows:
            def cell(field):
                index = columns.get(field)
                if index is None or index >= len(row) or row[index] is None:
                    return ""
                value = row[index]
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                return str(value).strip()

            name = cell("name")
            if name:
                imported.append((name, cell("roll"), cell("class_name"), cell("section"), cell("reg_no")))

        workbook.close()
        if not imported:
            flash("No student rows with a name were found in the Excel file.", "error")
            return redirect(url_for("add_student"))

        created_at = datetime.now().isoformat(timespec="seconds")
        with db() as con:
            con.executemany(
                "INSERT INTO students(name,roll,class_name,section,reg_no,created_at,approval_status) VALUES(?,?,?,?,?,? ,'approved')",
                [(name, roll, class_name, section, reg_no, created_at)
                 for name, roll, class_name, section, reg_no in imported]
            )

        flash(f"Successfully imported {len(imported)} student(s) from Excel. Next, enroll their face photos from the student list.", "success")
    except Exception as e:
        flash(f"Could not read this Excel file. Please check that it is a valid .xlsx workbook with a header row. Details: {e}", "error")
    return redirect(url_for("students_page"))

@app.route("/upload_face", methods=["POST"])
@login_required
def upload_face():
    sid = request.form.get("student_id", "")
    if not sid.isdigit():
        return jsonify(error="Invalid student ID"), 400
    with db() as con:
        student = con.execute("SELECT id, name FROM students WHERE id=? AND approval_status='approved'", (int(sid),)).fetchone()
        if not student:
            return jsonify(error="Student not found"), 404
    folder = os.path.join(dataset_dir(), sid)
    os.makedirs(folder, exist_ok=True)
    files = request.files.getlist("images[]") or request.files.getlist("images")
    if not files or all(getattr(f, "filename", "") == "" for f in files):
        return jsonify(error="Please select at least one face photo to upload."), 400

    saved_files = []
    valid_embeddings = []
    rejections = []
    for f in files:
        if not f.filename:
            continue
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in ALLOWED:
            rejections.append(f"{f.filename}: Unsupported file format (allowed: JPG, JPEG, PNG)")
            continue
        raw_bytes = f.read()
        if not raw_bytes:
            rejections.append(f"{f.filename}: Empty file")
            continue
        is_valid, msg, emb = validate_and_extract_face_photo(raw_bytes)
        if not is_valid:
            rejections.append(f"{f.filename}: {msg}")
            continue
        safe_name = secure_filename(f"{int(datetime.now().timestamp()*1000)}_{len(saved_files)}.{ext}")
        with open(os.path.join(folder, safe_name), "wb") as out_f:
            out_f.write(raw_bytes)
        saved_files.append(safe_name)
        valid_embeddings.append(emb)

    if not saved_files:
        err_msg = "No valid face photos could be enrolled. " + ("; ".join(rejections[:3]))
        return jsonify(error=err_msg, rejections=rejections), 400

    try:
        update_student_model(int(sid), valid_embeddings, dataset_dir(), model_path())
        write_status({
            "running": False,
            "progress": 100,
            "message": f"AI model automatically updated for {student['name']}."
        })
    except Exception as exc:
        app.logger.exception("Automatic model update failed")
        return jsonify(saved=len(saved_files), error=f"Photos saved, but model update failed: {exc}"), 500

    msg = f"{len(saved_files)} valid photo(s) enrolled and AI model updated automatically for {student['name']}."
    if rejections:
        msg += f" ({len(rejections)} invalid image(s) skipped)."
    return jsonify(saved=len(saved_files), rejected=len(rejections), message=msg, model_ready=True)

@app.route("/train_model", methods=["POST"])
@login_required
def train_model():
    status = read_status()
    if status.get("running"):
        return jsonify(status="already_running"), 202
    write_status({"running": True, "progress": 0, "message": "Preparing training data"})
    dataset_path, model_file, status_file = dataset_dir(), model_path(), status_path()
    def runner():
        def update_status(data):
            with open(status_file,"w",encoding="utf-8") as status_out: json.dump(data,status_out)
        try:
            train_model_background(dataset_path, model_file, lambda p, m: update_status({"running": True, "progress": p, "message": m}))
            update_status({"running": False, "progress": 100, "message": "Training finished."})
        except Exception as exc:
            update_status({"running": False, "progress": 0, "message": f"Training failed: {exc}"})
    threading.Thread(target=runner, daemon=True).start()
    return jsonify(status="started"), 202

@app.route("/train_status")
@login_required
def train_status():
    return jsonify(read_status())

@app.route("/attendance")
@login_required
def mark_attendance_page():
    model_data = load_model_if_exists(model_path())
    is_trained = model_data is not None and bool(model_data.get("students"))
    with db() as con: courses=con.execute("SELECT course_code FROM courses ORDER BY course_code").fetchall()
    return render_template("mark_attendance.html", trained=is_trained, courses=courses)

@app.route("/start_class", methods=["POST"])
@login_required
def start_class():
    course = (request.json or {}).get("course_code", "").strip().upper()
    if not course: return jsonify(ok=False, error="Please enter a course code."), 400
    with db() as con: con.execute("INSERT OR IGNORE INTO courses(course_code,course_name) VALUES(?,?)",(course,course))
    started = datetime.now().isoformat(timespec="seconds")
    with db() as con:
        con.execute("INSERT INTO active_class(id,course_code,started_at) VALUES(1,?,?) ON CONFLICT(id) DO UPDATE SET course_code=excluded.course_code, started_at=excluded.started_at", (course, started))
    return jsonify(ok=True, course_code=course, started_at=started, duration_minutes=30)

@app.route("/active_class")
@login_required
def active_class_status():
    with db() as con:
        row = con.execute("SELECT course_code,started_at FROM active_class WHERE id=1").fetchone()
    if not row: return jsonify(active=False)
    started = datetime.fromisoformat(row["started_at"])
    remaining = max(0, 1800 - int((datetime.now() - started).total_seconds()))
    return jsonify(active=remaining > 0, course_code=row["course_code"], started_at=row["started_at"], remaining_seconds=remaining)

@app.route("/end_class", methods=["POST"])
@login_required
def end_class():
    with db() as con: con.execute("DELETE FROM active_class WHERE id=1")
    return jsonify(ok=True)

@app.route("/recognize_face", methods=["POST"])
@login_required
def recognize_face():
    with db() as con:
        session = con.execute("SELECT course_code,started_at FROM active_class WHERE id=1").fetchone()
    if not session:
        return jsonify(recognized=False, class_not_started=True, title="Class Not Started", status="START CLASS", error="Please press Start Class before attendance begins.")
    class_started = datetime.fromisoformat(session["started_at"])
    if (datetime.now() - class_started).total_seconds() >= 1800:
        with db() as con: con.execute("DELETE FROM active_class WHERE id=1")
        return jsonify(recognized=False, time_expired=True, title="Attendance Window Closed", status="TIME EXPIRED", error="The 30-minute attendance period has ended.")
    if "image" not in request.files:
        return jsonify(recognized=False, error="No image received"), 400
    try:
        raw_bytes = request.files["image"].read()
        if not raw_bytes:
            return jsonify(recognized=False, error="Empty image data"), 400

        extractions = verify_and_extract_all_faces(raw_bytes, check_quality=True, check_spoof=True)
        if not extractions:
            return jsonify(recognized=False, no_face=True, error="No face detected. Please look at the camera.", title="No Face Detected", status="NO FACE")

        ok_faces = [f for f in extractions if f.get("status") == "OK"]
        if not ok_faces:
            first = extractions[0]
            status_code = first.get("status")
            err_msg = first.get("message", "Face could not be verified.")
            if status_code == "POOR_QUALITY":
                is_far = "far" in err_msg.lower() or "closer" in err_msg.lower()
                return jsonify(
                    recognized=False,
                    too_far=is_far,
                    error=err_msg,
                    title="Move Closer to Camera" if is_far else "Poor Image Quality",
                    status="MOVE CLOSER" if is_far else "CAPTURE AGAIN"
                )
            elif status_code == "SPOOF_DETECTED":
                return jsonify(recognized=False, is_spoof=True, error=err_msg, title="Fake Attendance Detected", status="REJECTED")
            elif status_code == "NO_FACE":
                return jsonify(recognized=False, no_face=True, error=err_msg, title="No Face Detected", status="NO FACE")
            else:
                return jsonify(recognized=False, error=err_msg, title="Invalid Image", status="ERROR")

        clf = load_model_if_exists(model_path())
        if clf is None or not clf.get("students"):
            return jsonify(
                recognized=False,
                error="Model is not trained yet. Enroll students first.",
                title="Model Not Ready",
                status="Enroll Students"
            )

        recognized_students = []
        unknown_faces = []
        with db() as con:
            for face in ok_faces:
                emb = face["embedding"]
                pred = predict_with_model(clf, emb)
                label, confidence = pred.label, pred.confidence
                status_code = pred.status_code

                if label is None:
                    unknown_faces.append({
                        "recognized": False,
                        "unknown": True,
                        "confidence": round(float(confidence), 3),
                        "status_code": status_code,
                        "title": "Unknown Person",
                        "status": "Uncertain Match" if status_code == "AMBIGUOUS_MATCH" else "Unknown Person",
                        "error": "Identity match is uncertain between similar faces." if status_code == "AMBIGUOUS_MATCH" else "Unregistered individual or confidence too low."
                    })
                    continue

                sid = int(label)
                student = con.execute("SELECT id, name FROM students WHERE id=? AND approval_status='approved'", (sid,)).fetchone()
                if not student:
                    unknown_faces.append({
                        "recognized": False,
                        "unknown": True,
                        "confidence": round(float(confidence), 3),
                        "title": "Unknown Person",
                        "status": "Student Not Found",
                        "error": "Recognized ID is not in the student list."
                    })
                    continue

                course_code = session["course_code"]
                existing = con.execute(
                    "SELECT id, timestamp FROM attendance WHERE student_id=? AND course_code=? AND date(timestamp)=date('now','localtime')",
                    (sid, course_code)
                ).fetchone()

                if existing:
                    recognized_students.append({
                        "student_id": sid,
                        "name": student["name"],
                        "confidence": round(float(confidence), 3),
                        "already_marked": True,
                        "timestamp": existing["timestamp"]
                    })
                else:
                    now = datetime.now().isoformat(timespec="seconds")
                    already_marked = False
                    try:
                        con.execute("INSERT INTO attendance(student_id, name, timestamp, course_code) VALUES(?,?,?,?)", (sid, student["name"], now, course_code))
                    except sqlite3.IntegrityError:
                        already_marked = True
                        prev = con.execute("SELECT timestamp FROM attendance WHERE student_id=? AND course_code=? AND date(timestamp)=date(?)", (sid, course_code, now)).fetchone()
                        now = prev["timestamp"] if prev else now
                    recognized_students.append({
                        "student_id": sid,
                        "name": student["name"],
                        "confidence": round(float(confidence), 3),
                        "already_marked": already_marked,
                        "timestamp": now
                    })

        if len(extractions) == 1:
            if recognized_students:
                r = recognized_students[0]
                return jsonify(
                    recognized=True,
                    already_marked=r["already_marked"],
                    name=r["name"],
                    confidence=r["confidence"],
                    timestamp=r["timestamp"],
                    faces=recognized_students
                )
            else:
                u = unknown_faces[0]
                return jsonify(
                    recognized=False,
                    unknown=True,
                    name=None,
                    error=u["error"],
                    title=u["title"],
                    status=u["status"],
                    confidence=u["confidence"],
                    faces=unknown_faces
                )

        if recognized_students:
            names = " & ".join(r["name"] for r in recognized_students)
            all_marked = all(r["already_marked"] for r in recognized_students)
            max_conf = max(r["confidence"] for r in recognized_students)
            ts = recognized_students[0]["timestamp"]
            summary_status = f"Recognized {len(recognized_students)} of {len(extractions)} face(s)"
            if unknown_faces:
                summary_status += f" ({len(unknown_faces)} unknown)"
            return jsonify(
                recognized=True,
                already_marked=all_marked,
                name=names,
                confidence=max_conf,
                timestamp=ts,
                title=names,
                status=summary_status,
                total_faces=len(extractions),
                recognized_count=len(recognized_students),
                unknown_count=len(unknown_faces),
                faces=recognized_students + unknown_faces
            )
        else:
            max_conf = max((u["confidence"] for u in unknown_faces), default=0.0)
            return jsonify(
                recognized=False,
                unknown=True,
                name=None,
                confidence=max_conf,
                title="Unknown Person",
                status=f"{len(extractions)} Unknown Faces",
                error=f"{len(extractions)} face(s) detected, but none matched enrolled students.",
                total_faces=len(extractions),
                faces=unknown_faces
            )
    except Exception as exc:
        app.logger.exception("Recognition failed")
        return jsonify(recognized=False, error=f"Recognition error: {exc}"), 500

@app.route("/records")
@login_required
def records():
    period = request.args.get("period", "all")
    where, params = "", []
    if period == "today":
        where, params = "WHERE date(a.timestamp)=date('now','localtime')", []
    elif period == "week":
        where, params = "WHERE date(a.timestamp)>=date('now','localtime','-7 day')", []
    elif period == "month":
        where, params = "WHERE date(a.timestamp)>=date('now','localtime','-30 day')", []
    with db() as con:
        rows = con.execute(f'''SELECT a.id,a.student_id,a.name,a.timestamp,a.course_code,s.roll,s.class_name,s.section
            FROM attendance a LEFT JOIN students s ON s.id=a.student_id {where}
            ORDER BY a.timestamp DESC LIMIT 5000''', params).fetchall()
    return render_template("records.html", records=rows, period=period)

@app.route("/download_csv")
@login_required
def download_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Attendance ID", "Student ID", "Name", "Roll", "Class", "Section", "Timestamp", "Course"])
    with db() as con:
        rows = con.execute('''SELECT a.id,a.student_id,a.name,s.roll,s.class_name,s.section,a.timestamp,a.course_code
            FROM attendance a LEFT JOIN students s ON s.id=a.student_id ORDER BY a.timestamp DESC''').fetchall()
        for row in rows:
            writer.writerow(list(row))
    mem = io.BytesIO(output.getvalue().encode("utf-8-sig"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="attendance_records.csv", mimetype="text/csv")

@app.route("/delete_student/<int:sid>", methods=["POST"])
@login_required
def delete_student(sid):
    with db() as con:
        con.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
        con.execute("DELETE FROM students WHERE id=?", (sid,))
    folder = os.path.join(dataset_dir(), str(sid))
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)
    try:
        remove_student_from_model(sid, model_path())
    except Exception as e:
        app.logger.warning(f"Could not remove student {sid} from model: {e}")
    flash("Student and their attendance records were deleted.", "success")
    return redirect(url_for("students_page"))


@app.route("/clear_all_students", methods=["POST"])
@login_required
def clear_all_students():
    # Remove every registered student and all related attendance/face data.
    with db() as con:
        student_ids = [row["id"] for row in con.execute("SELECT id FROM students").fetchall()]
        con.execute("DELETE FROM attendance")
        con.execute("DELETE FROM students")

    # Also include any numeric dataset folders or model entries left behind.
    dataset_ids = []
    active_dataset_dir=dataset_dir()
    if os.path.isdir(active_dataset_dir):
        dataset_ids = [int(name) for name in os.listdir(active_dataset_dir)
                       if name.isdigit() and os.path.isdir(os.path.join(active_dataset_dir, name))]

    try:
        model_data = load_model_if_exists(model_path())
        model_ids = list(model_data.get("students", {}).keys()) if isinstance(model_data, dict) else []
    except Exception:
        model_ids = []

    all_ids = set(student_ids) | set(dataset_ids)
    for model_id in model_ids:
        try:
            all_ids.add(int(model_id))
        except (TypeError, ValueError):
            continue

    for sid in all_ids:
        folder = os.path.join(dataset_dir(), str(sid))
        if os.path.isdir(folder):
            import shutil
            shutil.rmtree(folder, ignore_errors=True)
        try:
            remove_student_from_model(sid, model_path())
        except Exception as e:
            app.logger.warning(f"Could not remove student {sid} from model: {e}")

    write_status({"running": False, "progress": 0, "message": "All student face data cleared. Register students to continue."})
    flash("All students, attendance records, face photos, and enrolled face data were cleared.", "success")
    return redirect(url_for("students_page"))

@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Upload too large. Maximum request size is 64 MB."), 413

if __name__ == "__main__":
    app.run(debug=True)
