# app.py — MetroVision · Space-ready (robust loader)

import os, json, time, tempfile, subprocess
import torch, gradio as gr
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from gradio.themes import Monochrome

# ---------- model selection ----------
# 优先：读取环境变量 MODEL_ID；否则用一个公开可用的演示模型，保证 Space 可直接运行
MODEL_ID = os.getenv("MODEL_ID", "google/vit-base-patch16-224")

APP_DIR = os.path.dirname(__file__)
LOG     = os.path.join(APP_DIR, "preds.jsonl")

ACCENT = "#ff3b3b"
THEME = Monochrome(primary_hue="red", neutral_hue="slate").set(
    body_background_fill="#000000",
    body_text_color="#f2f2f2",
    block_background_fill="#0a0a0a",
    block_border_color="#151515",
    button_primary_background_fill=ACCENT,
    button_primary_text_color="#ffffff",
)

CSS = f"""
:root, html, body, .gradio-container, .app, .main, .wrap, .block, .tabs, .tabitem {{ background:#000 !important; }}
#root{{ max-width:1120px; margin:0 auto; }}
.gradio-container{{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
.neon{{ color:{ACCENT}; text-shadow:0 0 14px rgba(255,59,59,.75); font-weight:800; letter-spacing:.6px; }}
.tabs .tab-nav button.selected{{ color:#fff; border-color:{ACCENT}; }}
.tabs .tab-nav{{ border-bottom:1px solid #1a1a1a; }}
.card{{ background:linear-gradient(180deg,#0d0d0d 0%, #060606 100%); border:1px solid #141414; border-radius:18px; padding:16px; box-shadow:0 10px 30px rgba(0,0,0,.45); }}
.gr-box{{ border-radius:16px !important; }}
.label{{ font-weight:700; color:#e8e8e8; }}
.top3{{ display:grid; gap:8px; }}
.row{{ display:flex; justify-content:space-between; font-size:12px; color:#d0d0d0; }}
.bar{{ height:10px; border-radius:8px; background:#ffffff33; position:relative; overflow:hidden; }}
.bar>span{{ position:absolute; left:0; top:0; bottom:0; background:linear-gradient(90deg,{ACCENT},#ff6b6b,#ffffff); box-shadow:0 0 10px rgba(255,59,59,.35); }}
footer, [data-testid="block-portal"] footer{{ display:none !important; }}
#img_up .file-drop .upload-text, #img_up .file-drop .upload-text *, #img_up .upload-zone .upload-text, #img_up .upload-zone .upload-text * {{ display:none !important; visibility:hidden !important; }}
#vid_up .file-drop .upload-text, #vid_up .file-drop .upload-text *, #vid_up .upload-zone .upload-text, #vid_up .upload-zone .upload-text * {{ display:none !important; visibility:hidden !important; }}
#img_up .file-drop svg, #vid_up .file-drop svg{{ width:48px; height:48px; opacity:.85; color:#e6e6e6; }}
#img_up .file-drop, #vid_up .file-drop{{ color:transparent !important; }}
"""

# ---------- robust loader ----------
def load_model_and_classes():
    try:
        token = os.getenv("HF_TOKEN", None)
        kw = {"use_fast": True}
        if token: kw["token"] = token
        processor = AutoImageProcessor.from_pretrained(MODEL_ID, **kw)
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID, **({"token": token} if token else {}))
        model.eval()
        classes = [model.config.id2label[i] for i in range(model.config.num_labels)]
        return processor, model, classes
    except Exception as e:
        msg = (
            f"模型加载失败：{e}\n\n"
            f"当前 MODEL_ID={MODEL_ID}\n"
            f"如果你要加载私有模型，请在 Space 的 Settings → Variables 添加 HF_TOKEN；\n"
            f"或把模型设为 Public；\n"
            f"也可把本地权重放到 runs/metrovision 并改代码从本地加载。"
        )
        raise RuntimeError(msg)

processor, model, CLASSES = load_model_and_classes()

# ---------- narration ----------
def narrate(label, scores):
    tip = {
        "weird": "Maybe I'm learning humans’ fascination with the ‘abnormal’.",
        "performance": "Crowd and rhythm often look like ‘intentional art’.",
        "danger": "Motion tension suggests ‘danger’—but I can be wrong.",
        "empty": "No people and static scene suggest ‘empty’.",
        "normal": "Looks like an ordinary commute."
    }.get(label, "This is what the model thinks based on its training data.")
    p = scores.get(label, 0.0)
    return f"I think this is **{label}** ({p*100:.1f}%).\n\n> {tip}\n\n*AI learned this definition from human uploads.*"

# ---------- inference ----------
def _predict_tensor(img):
    x = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        probs = model(**x).logits.softmax(-1)[0]
    scores = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    top3   = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3])
    label  = next(iter(top3))
    return top3, scores, label

def predict_pil(img, source="image"):
    top3, scores, label = _predict_tensor(img)
    try:
        with open(LOG, "a") as f:
            f.write(json.dumps({"ts": time.time(), "source": source, "top3": top3}) + "\n")
    except Exception:
        pass
    return top3, narrate(label, scores)

def predict_image(image):
    if image is None: return {}, ""
    return predict_pil(image.convert("RGB"), "image")

def predict_video(video):
    if video is None: return {}, ""
    tmp   = tempfile.mkdtemp()
    frame = os.path.join(tmp, "mid.jpg")
    # 取中间帧（在 Spaces 通常有 ffmpeg）
    subprocess.run(
        ["ffmpeg","-y","-i",video,"-vf","select='eq(n,round(n/2))',scale='min(768,iw)':-2","-vframes","1",frame],
        check=True
    )
    return predict_pil(Image.open(frame).convert("RGB"), "video")

def top3_to_html(d: dict):
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:3]
    rows = []
    for k, v in items:
        pct = f"{v*100:.1f}%"
        rows.append(
            f"<div class='row'><span>{k}</span><span>{pct}</span></div>"
            f"<div class='bar'><span style='width:{v*100:.1f}%'></span></div>"
        )
    return f"<div class='top3'>{''.join(rows)}</div>"

# ---------- UI ----------
with gr.Blocks(theme=THEME, css=CSS, title="MetroVision · Underground Vision") as demo:
    gr.HTML(f"""
      <div style="display:flex;align-items:center;gap:12px;margin:14px 2px 6px">
        <div style="width:10px;height:10px;border-radius:50%;background:{ACCENT};box-shadow:0 0 12px {ACCENT};"></div>
        <div class="neon">METROVISION · HOW AI SEES THE UNDERGROUND</div>
      </div>
    """)

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="card">')
                gr.Markdown("**Image**", elem_classes=["label"])
                img = gr.Image(type="pil", height=360, show_label=False, elem_id="img_up",
                               sources=["upload","clipboard","webcam"])
                btn_img = gr.Button("Classify Image", variant="primary")
                gr.HTML('</div>')
            with gr.Column(scale=1):
                gr.HTML('<div class="card">')
                top3_html_img = gr.HTML("<div class='top3'></div>")
                txt_img = gr.Markdown("")
                gr.HTML('</div>')

    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="card">')
                gr.Markdown("**Video**", elem_classes=["label"])
                vid = gr.Video(height=360, show_label=False, elem_id="vid_up", sources=["upload","webcam"])
                btn_vid = gr.Button("Classify Video")
                gr.HTML('</div>')
            with gr.Column(scale=1):
                gr.HTML('<div class="card">')
                top3_html_vid = gr.HTML("<div class='top3'></div>")
                txt_vid = gr.Markdown("")
                gr.HTML('</div>')

    def _img_wrap(image):
        top3, narration = predict_image(image)
        return top3_to_html(top3), narration

    def _vid_wrap(video):
        top3, narration = predict_video(video)
        return top3_to_html(top3), narration

    btn_img.click(_img_wrap, inputs=img, outputs=[top3_html_img, txt_img])
    btn_vid.click(_vid_wrap, inputs=vid, outputs=[top3_html_vid, txt_vid])

if __name__ == "__main__":
    is_space = any(k in os.environ for k in ("SPACE_ID", "SYSTEM", "HF_SPACE"))
    demo.queue().launch(
        server_name="0.0.0.0" if is_space else "127.0.0.1",
        server_port=7860,
        inbrowser=not is_space,
        show_error=True,
        debug=True,
        show_api=False,
        prevent_thread_lock=False
    )
