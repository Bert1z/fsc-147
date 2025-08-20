#!/usr/bin/env python
# coding: utf-8

"""
Flask 图片服务器
兼容 API 通过 image_url 拉取图片
"""

from flask import Flask, send_from_directory, abort
import argparse
import os

app = Flask(__name__)

@app.route('/<path:filename>')
def serve_image(filename):
    # 确认文件存在
    if os.path.exists(os.path.join(app.config['IMAGE_DIR'], filename)):
        return send_from_directory(app.config['IMAGE_DIR'], filename)
    else:
        abort(404)

def main():
    parser = argparse.ArgumentParser(description="启动 Flask 图片服务器")
    parser.add_argument("--image_dir", type=str, default="/root/paddlejob/workspace/lyb/LearningToCountEverything/data/images_384_VarV2/",
                        help="图片目录路径")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="监听主机（0.0.0.0 对外开放）")
    parser.add_argument("--port", type=int, default=8000,
                        help="监听端口")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"❌ 图片目录不存在: {args.image_dir}")
        return

    app.config['IMAGE_DIR'] = args.image_dir
    print(f"✅ Flask 图片服务器启动中，目录: {args.image_dir}")
    print(f"可通过 http://{args.host}:{args.port}/<图片名> 访问")
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()
