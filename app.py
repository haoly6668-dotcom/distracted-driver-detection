#1.基础库导入区 忽略警告 网络端口管理 浏览器自动打开 Gradio网页开发 PyTorch推理 图像处理
import warnings
warnings.filterwarnings("ignore")
import socket
from threading import Timer
import webbrowser
# Gradio 库，用于快速搭建 Web 交互界面
import gradio as gr
# PyTorch 深度学习框架
import torch
import torch.nn as nn
# 预训练模型和图像预处理工具
from torchvision import models, transforms
# PIL 图像处理库
from PIL import Image

#2.驾驶行为类设计
class DistractedDriverDetector:
    def __init__(self, model_path, device):
        self.device = device
        self.ckpt = torch.load(model_path, map_location=device)
        self.class_names = self.ckpt["class_names"]
        self.model = models.mobilenet_v3_small(weights=None)
        in_features = self.model.classifier[3].in_features

        self.model.classifier[3] = nn.Sequential(
            nn.Linear(in_features, 256),   
            nn.Hardswish(),            
            nn.Dropout(0.3),            
            nn.Linear(256, len(self.class_names)) 
        )
        self.model.load_state_dict(self.ckpt["model_state_dict"])
        self.model.to(device)
        self.model.eval()
 #实例化       
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = DistractedDriverDetector("best_model.pth", device)
model = detector.model
class_names = detector.class_names

#3.行为类别映射区
BEHAVIOR_DESCRIPTIONS = [
    "安全驾驶",
    "右手发短信",
    "右手打电话",
    "左手发短信",
    "左手打电话",
    "操作收音机",
    "喝水",
    "伸手拿东西",
    "整理头发或化妆",
    "与乘客交谈"
]



def get_friendly_name(class_name):

    if class_name.startswith('c') and class_name[1:].isdigit():
        idx = int(class_name[1:])
    else:
        idx = class_names.index(class_name)

    if 0 <= idx < len(BEHAVIOR_DESCRIPTIONS):
        return f"c{idx} {BEHAVIOR_DESCRIPTIONS[idx]}"

    return class_name
# 判断危险驾驶
def is_dangerous_class(class_name):

    if class_name.startswith('c') and class_name[1:].isdigit():

        return int(class_name[1:]) != 0

    return class_names.index(class_name) != 0



#4.图像预处理区 保证输入与训练时一致
transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),   

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#5.核心预测区
"""
上传图片
      ↓
预处理
      ↓
MobileNetV3
      ↓
Softmax
      ↓
得到10类概率
      ↓
排序
      ↓
输出前三名
      ↓
判断安全/危险
      ↓
返回网页显示
"""
def predict(image):

    if image is None:

        placeholder = Image.new("RGB", (224, 224), color=(10, 15, 25))

        empty_probs = """
        <div class="card">
            <div class="card-title">最可能的3种驾驶行为</div>
            <div class="empty-text">请上传图像以查看前3种行为</div>
        </div>
        """
        empty_status = '<div class="card"><div class="status-safe">等待输入</div></div>'
        return placeholder, empty_probs, empty_status


    #图片处理
    img_tensor = transform(image).unsqueeze(0).to(device)
    #推理
    with torch.no_grad():
        output = model(img_tensor)                  
        probs = torch.softmax(output, dim=1)[0].cpu().numpy() 

    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    #排序
    sorted_items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    #获取Top1
    top1_class, top1_conf = sorted_items[0]
    confidence = top1_conf * 100  
    #判断危险驾驶
    danger = is_dangerous_class(top1_class)  

    annotated = image.copy()

#前3概率展示模块
    bars = ""
    for cls, prob in sorted_items[:3]:          
        idx = class_names.index(cls)
        friendly = get_friendly_name(cls)     
        percent = prob * 100
        color = "#ef4444" if (danger and cls == top1_class) else "#0ea5e9"
        bars += f"""
        <div class="prob-item">
            <div class="prob-header">
                <span>{friendly}</span>
                <span>{percent:.1f}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{percent}%; background:{color};"></div>
            </div>
        </div>"""

    probs_html = f"""
    <div class="card">
        <div class="card-title">最可能的3种驾驶行为</div>
        {bars}
    </div>"""
#安全状态判断模块
    if danger:
        status_html = f"""
        <div class="danger-card">
            <div class="status-title">⚠️ 检测到危险驾驶</div>
            <div class="status-label">驾驶员行为：{get_friendly_name(top1_class)}</div>
            <div class="status-confidence">置信度 {confidence:.1f}%</div>
        </div>"""
    else:
        status_html = f"""
        <div class="safe-card">
            <div class="status-title">✅ 安全驾驶状态</div>
            <div class="status-label">驾驶员行为：{get_friendly_name(top1_class)}</div>
            <div class="status-confidence">置信度 {confidence:.1f}%</div>
        </div>"""

    return annotated, probs_html, status_html

#6.CSS样式区
css = """
/* 全局背景色和字体 */
body, .gradio-container { background:#050816 !important; color:white; font-family: Inter, sans-serif; }
/* 隐藏 Gradio 默认的页脚 */
footer { display:none !important; }
/* 去除组件默认的边框和背景 */
.gr-box, .gr-panel, .block, .gr-form { border:none !important; background:transparent !important; }
/* 标题样式 */
.title { text-align:center; font-size:42px; font-weight:700; letter-spacing:2px; margin-bottom:30px; }
/* 侧边面板：半透明卡片效果 */
.side-panel { border:2px solid rgba(255,255,255,0.2); border-radius:20px; background:rgba(0,0,0,0.2); padding:16px; }
/* 隐藏图片组件上的下载和分享按钮 */
.gr-image button, button[aria-label="Download"], button[aria-label="Share"] { display:none !important; }
/* 全局隐藏标签文字（用我们的自定义卡片替代） */
label { display:none !important; }
/* 卡片通用样式 */
.card, .safe-card, .danger-card { background:rgba(15,23,42,.72); border:1px solid rgba(255,255,255,.06); border-radius:18px; padding:20px; box-shadow:0 8px 24px rgba(0,0,0,.25); }
/* 卡片标题 */
.card-title { font-size:20px; font-weight:600; margin-bottom:18px; }
/* 图片组件圆角 */
.gr-image { border-radius:18px !important; overflow:hidden; border:1px solid rgba(255,255,255,.08); }
/* 分析按钮样式 */
.gr-button-primary { background:#0ea5e9 !important; border:none !important; border-radius:14px !important; height:52px !important; font-size:16px !important; font-weight:600 !important; }
/* 概率条目 */
.prob-item { margin-bottom:16px; }
/* 概率条头部（标签+百分比） */
.prob-header { display:flex; justify-content:space-between; margin-bottom:8px; font-size:14px; }
/* 进度条背景 */
.bar-bg { background:rgba(255,255,255,.08); height:10px; border-radius:20px; overflow:hidden; }
/* 进度条填充部分，带过渡动画 */
.bar-fill { height:100%; border-radius:20px; transition:0.4s; }
/* 安全状态卡片，左侧绿色边框 */
.safe-card { border-left:4px solid #10b981; }
/* 危险状态卡片，左侧红色边框 */
.danger-card { border-left:4px solid #ef4444; }
/* 状态标题 */
.status-title { font-size:22px; font-weight:700; margin-bottom:16px; }
/* 状态标签（具体行为） */
.status-label { font-size:18px; margin-bottom:12px; color:#cbd5e1; }
/* 置信度文字 */            
.status-confidence { font-size:15px; color:#94a3b8; }
/* 空状态提示文字 */
.empty-text { color:#64748b; text-align:center; padding:30px 0; }
/* 实时时钟样式 */
.live-clock { background:rgba(0,0,0,0.6); border-radius:16px; padding:12px 20px; text-align:center; font-size:32px; font-weight:600; font-family:monospace; letter-spacing:2px; color:#0ea5e9; margin-top:16px; border:1px solid rgba(255,255,255,0.1); min-height:60px; line-height:36px; }
"""



#7.JavaScript实时时钟区
clock_js = """
<script>
(function() {
    // 更新所有时钟的函数
    function updateClocks() {
        // 获取页面上所有类名为 'live-clock' 的元素
        var clocks = document.querySelectorAll('.live-clock');
        var now = new Date();
        // 格式化为 HH:MM:SS
        var timeStr = String(now.getHours()).padStart(2,'0') + ':' +
                      String(now.getMinutes()).padStart(2,'0') + ':' +
                      String(now.getSeconds()).padStart(2,'0');
        // 将时间写入每个时钟元素
        clocks.forEach(function(el) { el.textContent = timeStr; });
    }
    // 页面加载完成后启动定时器
    window.addEventListener('load', function() {
        updateClocks();
        setInterval(updateClocks, 1000);
    });
    // 如果页面已经加载完成，直接启动
    if (document.readyState === 'complete') {
        updateClocks();
        setInterval(updateClocks, 1000);
    }
})();
</script>
"""

#8.Gradio界面构建区
with gr.Blocks(css=css, theme=gr.themes.Soft(), head=clock_js) as demo:
    gr.HTML('<div class="title">驾驶行为检测系统</div>')

    with gr.Row():
        with gr.Column(scale=1, elem_classes="side-panel"):
            img_input = gr.Image(type="pil", label="", height=360, show_label=False) 
            gr.HTML('<div class="live-clock">--:--:--</div>')  
            btn = gr.Button("分析", variant="primary")         

        with gr.Column(scale=1, elem_classes="side-panel"):
            annotated_output = gr.Image(type="pil", label="", height=360, show_label=False) 
            probs_output = gr.HTML()
            status_output = gr.HTML()

    btn.click(fn=predict, inputs=img_input, outputs=[annotated_output, probs_output, status_output])


#9.自动端口管理区 自动寻找未被占用的端口
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0)) 
        return s.getsockname()[1] 
#程序入口
if __name__ == "__main__":
    port = find_free_port()
    Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    demo.launch(server_port=port, share=False)