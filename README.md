# PKparmas Web

这是一个基于 Streamlit 的轻量网页工具：上传本地 `.xlsx`，选择房室模型与初值，在线拟合并输出参数与拟合曲线。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署

### Streamlit Community Cloud
- 在 Streamlit Cloud 连接该仓库并部署, 获得可分享链接。


## 数据格式
- 第一列为时间，后续列为浓度（至少一列数值型）

可在页面左侧直接下载模板。
