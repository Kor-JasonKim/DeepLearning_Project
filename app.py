# app.py
import os
import random
import shutil
import string

from flask import Flask, render_template, request, url_for, send_from_directory, redirect, abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

import db
from scoring import run_all_analyses, YOLO_FEATURE_NAMES

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
UPLOAD_FOLDER = "uploads"
POSTS_IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, "posts")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(POSTS_IMAGE_FOLDER, exist_ok=True)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User:
    def __init__(self, id_, email, display_name=None):
        self.id = id_
        self.email = email
        self.display_name = display_name or email

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)


@login_manager.user_loader
def load_user(user_id):
    try:
        conn = db.get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, display_name FROM users WHERE id = %s",
                (int(user_id),),
            )
            row = cur.fetchone()
        conn.close()
        if row:
            return User(row["id"], row["email"], row.get("display_name"))
    except Exception:
        pass
    return None


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ---------- Auth ----------
@app.route("/auth/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""
        # Anonymous display: random 6-char (a-z, 0-9)
        chars = string.ascii_lowercase + string.digits
        display_name = "".join(random.choices(chars, k=6))
        if not email or not password:
            return render_template("register.html", error="이메일과 비밀번호를 입력하세요.")
        try:
            conn = db.get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (email, password_hash, display_name) VALUES (%s, %s, %s)",
                    (email, generate_password_hash(password), display_name),
                )
            conn.close()
            return redirect(url_for("login"))
        except Exception as e:
            if "Duplicate" in str(e) or "1062" in str(e):
                return render_template("register.html", error="이미 사용 중인 이메일입니다.")
            raise
    return render_template("register.html")


@app.route("/auth/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""
        conn = db.get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, display_name, password_hash FROM users WHERE email = %s",
                (email,),
            )
            row = cur.fetchone()
        conn.close()
        if row and check_password_hash(row["password_hash"], password):
            user = User(row["id"], row["email"], row.get("display_name"))
            login_user(user)
            next_url = request.args.get("next") or url_for("index")
            return redirect(next_url)
        return render_template("login.html", error="이메일 또는 비밀번호가 올바르지 않습니다.")
    return render_template("login.html")


@app.route("/auth/logout")
def logout():
    logout_user()
    return redirect(url_for("index"))


@app.route("/auth/profile")
@login_required
def profile():
    return render_template("profile.html", user=current_user)


# ---------- Posts ----------
@app.route("/post/share", methods=["POST"])
@login_required
def post_share():
    image_filename = (request.form.get("image_filename") or "").strip()
    try:
        ai_score = float(request.form.get("ai_score", 0))
    except (TypeError, ValueError):
        ai_score = 0.0
    mode = (request.form.get("mode") or "room").strip().lower()
    if mode not in ("room", "desk"):
        mode = "room"
    if not image_filename:
        return "이미지 정보가 없습니다.", 400
    src_path = os.path.join(UPLOAD_FOLDER, image_filename)
    if not os.path.isfile(src_path):
        return "이미지를 찾을 수 없습니다.", 400
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO posts (user_id, image_path, ai_score, mode) VALUES (%s, %s, %s, %s)",
                (current_user.id, image_filename, ai_score, mode),
            )
            post_id = cur.lastrowid
        ext = os.path.splitext(image_filename)[1] or ".jpg"
        dest_name = f"{post_id}{ext}"
        dest_path = os.path.join(POSTS_IMAGE_FOLDER, dest_name)
        shutil.copy2(src_path, dest_path)
        stored_path = os.path.join("posts", dest_name)
        with conn.cursor() as cur:
            cur.execute("UPDATE posts SET image_path = %s WHERE id = %s", (stored_path, post_id))
        conn.close()
    except Exception:
        conn.close()
        raise
    return redirect(url_for("post_detail", post_id=post_id))


@app.route("/posts")
def posts_feed():
    page = max(1, int(request.args.get("page", 1)))
    per_page = 20
    offset = (page - 1) * per_page
    conn = db.get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT p.id, p.image_path, p.ai_score, p.mode, p.created_at, u.email, u.display_name "
            "FROM posts p LEFT JOIN users u ON p.user_id = u.id "
            "ORDER BY p.created_at DESC LIMIT %s OFFSET %s",
            (per_page + 1, offset),
        )
        rows = cur.fetchall()
    post_ids = [r["id"] for r in rows[:per_page]]
    score_agg = {}
    if post_ids:
        placeholders = ",".join(["%s"] * len(post_ids))
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT post_id, AVG(score) AS avg_score, COUNT(*) AS cnt FROM user_scores WHERE post_id IN ({placeholders}) GROUP BY post_id",
                post_ids,
            )
            for r in cur.fetchall():
                score_agg[r["post_id"]] = {"avg": float(r["avg_score"]), "cnt": int(r["cnt"])}
    conn.close()
    has_next = len(rows) > per_page
    posts = rows[:per_page]
    for p in posts:
        p["author"] = p.get("display_name") or p.get("email") or "알 수 없음"
        p["image_url"] = url_for("uploaded_file", filename=p["image_path"])
        agg = score_agg.get(p["id"]) or {"avg": None, "cnt": 0}
        p["user_score_avg"] = agg["avg"]
        p["user_score_count"] = agg["cnt"]
    return render_template(
        "posts_feed.html",
        posts=posts,
        page=page,
        has_next=has_next,
    )


@app.route("/post/<int:post_id>", methods=["GET"])
def post_detail(post_id):
    conn = db.get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT p.id, p.image_path, p.ai_score, p.mode, p.created_at, u.email, u.display_name "
            "FROM posts p LEFT JOIN users u ON p.user_id = u.id WHERE p.id = %s",
            (post_id,),
        )
        row = cur.fetchone()
    if not row:
        conn.close()
        abort(404)
    author = row.get("display_name") or row.get("email") or "알 수 없음"
    user_score_avg = None
    user_score_count = 0
    current_user_score = None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(AVG(score), 0) AS avg_score, COUNT(*) AS cnt FROM user_scores WHERE post_id = %s",
            (post_id,),
        )
        agg = cur.fetchone()
        if agg:
            user_score_avg = float(agg["avg_score"])
            user_score_count = int(agg["cnt"] or 0)
        if current_user.is_authenticated:
            cur.execute(
                "SELECT score FROM user_scores WHERE post_id = %s AND user_id = %s",
                (post_id, current_user.id),
            )
            us = cur.fetchone()
            if us:
                current_user_score = float(us["score"])
    conn.close()
    return render_template(
        "post_detail.html",
        post_id=row["id"],
        image_url=url_for("uploaded_file", filename=row["image_path"]),
        ai_score=row["ai_score"],
        mode=row["mode"],
        created_at=row["created_at"],
        author=author,
        user_score_avg=user_score_avg,
        user_score_count=user_score_count,
        current_user_score=current_user_score,
    )


@app.route("/post/<int:post_id>/score", methods=["POST"])
@login_required
def post_score(post_id):
    try:
        cleanliness_pct = float(request.form.get("score", 0))
    except (TypeError, ValueError):
        return redirect(url_for("post_detail", post_id=post_id))
    cleanliness_pct = max(0.0, min(100.0, cleanliness_pct))
    score = 1.0 - (cleanliness_pct / 100.0)
    conn = db.get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO user_scores (user_id, post_id, score) VALUES (%s, %s, %s) "
            "ON DUPLICATE KEY UPDATE score = VALUES(score), updated_at = CURRENT_TIMESTAMP",
            (current_user.id, post_id, score),
        )
    conn.close()
    return redirect(url_for("post_detail", post_id=post_id))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            return "파일이 없습니다."

        mode = (request.form.get("mode") or "room").strip().lower()
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        result = run_all_analyses(file_path, mode, UPLOAD_FOLDER, file.filename)
        keras_probability = result["keras_probability"]
        yolo_result = result["yolo_result"]
        clip_result = result["clip_result"]
        error_keras = result["error_keras"]
        error_yolo = result["error_yolo"]
        error_clip = result["error_clip"]

        # Terminal debug
        print("\n" + "=" * 50)
        print(f"[POST] image: {file.filename}")
        print("  Keras:", f"prob={keras_probability:.4f}" if keras_probability is not None else f"error={error_keras}")
        if yolo_result is not None:
            print("  YOLO:", f"score={yolo_result.get('yolo_score'):.4f}", yolo_result.get("detected_objects", [])[:3])
        else:
            print("  YOLO:", f"error={error_yolo}")
        if clip_result is not None:
            print("  CLIP:", clip_result.get("status"))
            for label, conf in clip_result.get("all_labels", []):
                print(f"      {label}: {conf*100:.1f}%")
        else:
            print("  CLIP:", f"error={error_clip}")
        print("=" * 50 + "\n")

        if keras_probability is None and yolo_result is None:
            return (
                f"이미지 처리 중 에러가 발생했습니다. "
                f"Keras: {error_keras or 'unknown'} | YOLO: {error_yolo or 'unknown'}"
            )

        heatmap_only_url = (
            url_for("uploaded_file", filename=result["heatmap_only_filename"])
            if result.get("heatmap_only_filename") else None
        )
        yolo_boxes_url = (
            url_for("uploaded_file", filename=result["yolo_boxes_filename"])
            if result.get("yolo_boxes_filename") else None
        )
        overlay_url = None
        if yolo_result and yolo_result.get("overlay_url"):
            overlay_url = url_for("uploaded_file", filename=yolo_result.get("overlay_url"))

        return render_template(
            "result.html",
            probability=result["prob_for_legacy"],
            heatmap_only_url=heatmap_only_url,
            keras_heatmap_guide=result.get("keras_heatmap_guide"),
            yolo_boxes_url=yolo_boxes_url,
            total_score=result["total_score"],
            keras_probability=keras_probability,
            yolo_result=yolo_result,
            yolo_error=error_yolo,
            keras_error=error_keras,
            image_url=url_for("uploaded_file", filename=file.filename),
            image_filename=file.filename,
            mode=mode,
            detected_objects=yolo_result.get("detected_objects", []) if yolo_result else [],
            overlay_url=overlay_url,
            messy_count=yolo_result.get("messy_count", 0) if yolo_result else 0,
            spread=yolo_result.get("spread", 0.0) if yolo_result else 0.0,
            messy_objects=yolo_result.get("messy_objects", []) if yolo_result else [],
            yolo_features=yolo_result.get("yolo_features") if yolo_result else None,
            yolo_feature_names=YOLO_FEATURE_NAMES,
            clip_result=clip_result,
            clip_error=error_clip,
        )

    return render_template("index.html")


if __name__ == "__main__":
    # Verify DB connection before starting
    try:
        conn = db.get_connection()
        conn.close()
    except Exception as e:
        print("Warning: MySQL connection failed. Set MYSQL_* env or run: python init_db.py")
        print(e)
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)
