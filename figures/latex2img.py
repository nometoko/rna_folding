import os
import urllib.parse
import urllib.request
from PIL import Image
import re

# 数式とファイル名のセット（数式はLaTeX形式）
formulas = [
    (
        r"\begin{align*}f(x) &= x \cdot \tanh(\mathrm{softplus}(x)) \\ \mathrm{softplus}(x) &= \log(1 + e^x) \end{align*}",
        "mish",
    ),
    (
        r"\mathbf{e} = \left[ \left( \frac{1}{|\Omega|} \sum_{u \in \Omega} x^p_{cu} \right)^\frac{1}{p} \right]_{c = 1, \cdots, C}",
        "GeM",
    ),
    (
        r"\begin{align*} \mathrm{Output}(Q, K, V) &= \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\ \end{align*}",
        "Scaled_DotProduct_Attention",
    ),
]

output_dir = "math_figures"
os.makedirs(output_dir, exist_ok=True)


def latex_to_png_url(latex, dpi=900):
    base = "https://latex.codecogs.com/png.image?"
    encoded = urllib.parse.quote(latex)
    return f"{base}\\dpi{{{dpi}}}&space;{encoded}"


def add_white_background(input_path, output_path):
    img = Image.open(input_path)

    # 白い背景の画像を作成（元の画像と同じサイズ）
    bg = Image.new("RGBA", img.size, (255, 255, 255))  # 白背景
    bg.paste(img, (0, 0), img.convert("RGBA").split()[3])  # 透明部分を白で埋める

    # PNGとして保存
    bg.save(output_path, "PNG")
    print(f"Saved with white background: {output_path}")


def sanitize_filename(filename):
    # ファイル名として使用できない文字を取り除く
    return re.sub(r'[\\/*?:"<>|]', "_", filename)


for formula, filename in formulas:
    # 数式をファイル名に適用（ファイル名をサニタイズ）
    safe_filename = sanitize_filename(filename)

    # 高解像度PNGのURLを取得
    png_url = latex_to_png_url(formula)

    # 保存先のファイルパス
    png_path = os.path.join(output_dir, f"{safe_filename}.png")

    # 画像をダウンロードして保存
    urllib.request.urlretrieve(png_url, png_path)
    print(f"Downloaded PNG: {png_path}")

    # 白背景を追加して保存
    white_bg_png_path = os.path.join(output_dir, f"{safe_filename}.png")
    add_white_background(png_path, white_bg_png_path)
