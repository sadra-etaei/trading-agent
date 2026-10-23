"""Create a Persian review packet. Does not create audio or video."""
from pathlib import Path
import base64
import html
import json
import shutil

OUT = Path(__file__).resolve().parent
PREVIEW = Path(r"C:\Users\User\AppData\Local\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Roaming\Codex\web\Codex\artifact-sessions\exports\latest\515630b387edc37accabe29402c9d9419a9ea8b32ca199f3469362ff45dd409f")
content = json.loads((OUT / "fa-demo-content.json").read_text(encoding="utf-8"))
content["slides"][4]["script"] = content["slides"][4]["script"].replace("با قیمت‌های هم‌زمان", "بر اساس قیمت‌های ذخیره‌شده")
(OUT / "fa-demo-content.json").write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
images = OUT / "fa-slides"
images.mkdir(exist_ok=True)
sections, markdown = [], ["# متن فارسی گویندگی دمو", "", "نسخهٔ بررسی برای تأیید کاربر؛ هنوز صدا یا ویدئو ساخته نشده است.", "", "مدت پیشنهادی: حدود ۴ تا ۵ دقیقه. زمان‌های زیر، برنامهٔ اولیهٔ تدوین‌اند و با اجرای گویندگی تنظیم می‌شوند.", ""]
clock = 0
for i, slide in enumerate(content["slides"], 1):
    image = PREVIEW / f"slide-{i}.png"
    assert image.exists(), image
    shutil.copyfile(image, images / image.name)
    encoded = base64.b64encode(image.read_bytes()).decode()
    t0 = f"{clock//60:02}:{clock%60:02}"
    clock += slide["duration"]
    t1 = f"{clock//60:02}:{clock%60:02}"
    sections.append(f'''<section id="slide-{i}"><h2>اسلاید {i}: {html.escape(slide['title'])}</h2><div class="timing" dir="ltr">{t0} – {t1}</div><div class="review"><img src="data:image/png;base64,{encoded}" alt="اسلاید {i}"><article><h3>متن گویندگی</h3><p>{html.escape(slide['script'])}</p></article></div></section>''')
    markdown += [f"## اسلاید {i}: {slide['title']}", "", f"زمان پیشنهادی: {t0} تا {t1}", "", slide["script"], ""]
shutil.copyfile(PREVIEW / "fa-deck-overview.png", OUT / "fa-deck-overview.png")
(OUT / "fa-narration.md").write_text("\n".join(markdown), encoding="utf-8")
page = '''<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>اسلایدها و متن گویندگی دمو والکس</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#edf3fc;font-family:Tahoma,Arial,sans-serif}header{padding:35px 4vw;border-bottom:1px solid #293a50}h1{font-size:30px;line-height:1.7;margin:0 0 12px}header p{color:#a9bbd2;line-height:2;font-size:18px;max-width:1100px}nav{display:flex;gap:12px;flex-wrap:wrap;margin-top:22px}nav a{color:#72b7ff;text-decoration:none;border-bottom:1px solid #40536c;padding:5px 0;font-size:15px}main{padding:0 4vw 50px}section{padding-top:30px;margin-bottom:35px;border-bottom:1px solid #293a50;padding-bottom:30px}h2{font-size:25px;line-height:1.8;margin:0}h3{font-size:20px;color:#43ddb2;margin:0 0 12px}.timing{color:#a9bbd2;font-size:15px;margin:8px 0 18px;text-align:right}.review{display:grid;grid-template-columns:1.6fr 1fr;gap:30px;align-items:start}.review img{width:100%;height:auto}.review article{font-size:20px;line-height:2.3}.review p{margin:0}footer{padding:30px 4vw;color:#a9bbd2;font-size:16px;line-height:2}.approval{color:#43ddb2;font-weight:bold}@media(max-width:1050px){.review{grid-template-columns:1fr}.review article{font-size:19px}}@media print{header nav{display:none}.review{display:block}section{break-after:page}.review img{max-height:45vh;object-fit:contain}body{background:white;color:black}.review article{font-size:14px}h3,.timing{color:#333}}
</style><header><h1>اسلایدها و متن فارسی گویندگی</h1><p>نسخهٔ قابل‌بررسی برای تأیید شما. ارائه شامل ۹ اسلاید است و با مکث‌های معمول حدود ۴ تا ۵ دقیقه طول می‌کشد. زمان‌های کنار اسلایدها، برنامهٔ اولیهٔ تدوین هستند.</p><p class="approval">این مرحله فقط برای بررسی اسلایدها و متن است. ساخت صدا و ویدئو پس از تأیید شما انجام می‌شود.</p><nav>'''
page += "".join(f'<a href="#slide-{i}">اسلاید {i}</a>' for i in range(1, 10))
page += "</nav></header><main>" + "".join(sections) + "</main><footer>بازده تومانی بک‌تست از نتیجهٔ سایه جداست. نمودارها و جدول‌ها بر اساس فایل‌های ذخیره‌شدهٔ پروژه‌اند. نمودار دلار از تتر به‌عنوان نمایندهٔ دلار استفاده می‌کند و هزینهٔ تبدیل اضافی را شامل نمی‌شود.</footer></html>"
(OUT / "fa-demo-review.html").write_text(page, encoding="utf-8")
print(json.dumps({"slides":len(content["slides"]),"planned_seconds":clock,"script_words":sum(len(s["script"].split()) for s in content["slides"]),"review":str(OUT / "fa-demo-review.html"),"video_created":False}, ensure_ascii=False, indent=2))
