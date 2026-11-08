# 图表源文件

`docs/images/` 中 P02、P03、P05 和 `GPU-measurement.png` 的源文件（P01、P04 的源文件没有保留）。每张图是 `src/` 下的一段 HTML，共用 `style.css`。

重新生成（需要本机安装 Google Chrome，并能访问 Google Fonts）：

```bash
python3 docs/images/src/render.py docs/images P02-mining-coverage P03-useful-vs-pure P05-result-accuracy GPU-measurement
```

脚本用无头 Chrome 以 2 倍分辨率截图，宽 3200 像素，高度按内容裁剪。
