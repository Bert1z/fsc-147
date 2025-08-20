#!/usr/bin/env python
# coding: utf-8

"""
测试代码 - 用于评估多模态大语言模型API在少样本计数任务上的性能
原作者: Viresh Ranjan
最后修改: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
修改日期: 2021/04/19
API适配修改: 2025
"""

import copy
import requests
import io
from PIL import Image
import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists
import re
import socket
from datetime import datetime
import sys

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="少样本计数模型评估代码")
parser.add_argument("-dp", "--data_path", type=str, default='/root/paddlejob/workspace/lyb/LearningToCountEverything/data/', 
                   help="FSC147数据集的路径")
parser.add_argument("-ts", "--test_split", type=str, default='val', 
                   choices=["val_PartA","val_PartB","test_PartA","test_PartB","test", "val"], 
                   help="选择要评估的数据分割")
# parser.add_argument("-u", "--api_url", type=str, 
#                    default="http://10.213.78.139:8180/v1/chat/completions", 
#                    help="API服务器URL")
parser.add_argument("-u", "--api_url", type=str, 
                   default="http://10.78.119.144:8801/v1/chat/completions", 
                   help="API服务器URL")
parser.add_argument("-m", "--model_name", type=str, 
                   default="Qwen2_5-VL-3B-Instruct", 
                   help="API模型名称")
parser.add_argument("-mt", "--max_tokens", type=int, default=128, 
                   help="API响应的最大token数")
parser.add_argument("-s", "--save_results", action='store_true', default=True,
                   help="保存详细结果到文件")
parser.add_argument("-o", "--output_file", type=str, default="", 
                   help="详细结果的输出文件名（可选，默认自动生成）")
parser.add_argument("--http_host", type=str, default="10.46.247.23", 
                   help="本机HTTP服务IP, 默认为本机内网IP自动获取")
parser.add_argument("--http_port", type=int, default=8000, 
                   help="本机HTTP服务端口")
args = parser.parse_args()

# 自动生成结果文件名
def generate_output_filename():
    """根据脚本名称和当前时间生成输出文件名"""
    # 获取脚本文件名（不含扩展名）
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 组合文件名
    filename = f"{script_name}_results_{timestamp}.json"
    
    return filename

# 如果用户没有指定输出文件名，则自动生成
if not args.output_file:
    args.output_file = generate_output_filename()

# 设置数据路径
data_path = args.data_path
anno_file = data_path + 'annotation_FSC147_384.json'
data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
im_dir = data_path + 'images_384_VarV2'

if not exists(anno_file) or not exists(im_dir):
    print("Make sure you set up the --data-path correctly.")
    exit(-1)

# 自动获取本机内网IP（如果未指定）
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# 如果用户没有指定http_host，则自动获取本机IP
http_host = args.http_host if args.http_host else get_local_ip()
http_port = args.http_port

print(f"使用HTTP服务地址: {http_host}:{http_port}")
print(f"API服务器地址: {args.api_url}")
print(f"模型名称: {args.model_name}")
print(f"测试分割: {args.test_split}")
print(f"数据路径: {args.data_path}")
print(f"结果保存: {args.output_file}")
print("-" * 60)

def call_counting_api(image_path, bboxes, api_url, model_name, max_tokens, class_name):
    """调用计数API（使用HTTP URL传图像）"""
    
    image_filename = os.path.basename(image_path)
    image_url = f"http://{http_host}:{http_port}/{image_filename}"
    
    bbox_descriptions = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        bbox_descriptions.append(f"示例{i+1}: 左上角({x1},{y1}), 右下角({x2},{y2})")
    bbox_text = "\n".join(bbox_descriptions)
    
    prompt = f"""请仔细观察这张图片，并根据提供的示例边界框和要识别的物品类别来计数图片中相同类型的物体数量。

图片中要识别并计数物品的类别: {class_name}

📝 任务说明：
这是一张包含{class_name}的图片，需要准确计数所有{class_name}的数量。

🔍 计数步骤：
1. 仔细分析示例边界框中的{class_name}特征
2. 在整张图片中系统性地寻找相同特征的{class_name}
3. 对于密集区域，采用网格扫描方式，确保不遗漏
4. 注意处理重叠、遮挡、边界模糊的情况

⚠️ 特别提醒：
- 密集排列的{class_name}需要逐个识别
- 不要因为密集而忽略任何{class_name}
- 保持耐心，仔细区分每个独立的{class_name}

示例边界框信息：
{bbox_text}

请按照上述步骤，准确数出图片中所有{class_name}的数量。
请只回答一个数字，表示{class_name}的总数量，不要包含任何其他文字或解释。"""
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        numbers = re.findall(r'\d+', content)
        if numbers:
            return float(numbers[0])
        else:
            print(f"警告: 无法从API响应中提取数字: {content}")
            return 0.0
    except requests.exceptions.RequestException as e:
        print(f"API调用失败: {e}")
        return 0.0
    except (KeyError, IndexError) as e:
        print(f"解析API响应失败: {e}")
        return 0.0

# 加载标注数据
with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)

# 加载图像类别信息
image_classes = {}
with open(data_path + 'ImageClasses_FSC147.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split('\t')
            if len(parts) == 2:
                image_filename = parts[0]
                class_name = parts[1]
                image_classes[image_filename] = class_name

print(f"已加载 {len(image_classes)} 张图像的类别信息")

# 检查数据一致性
print("正在检查数据一致性...")
missing_class_count = 0
missing_annotation_count = 0

for im_id in data_split[args.test_split]:
    # 检查是否有类别信息
    if im_id not in image_classes:
        missing_class_count += 1
        print(f"警告: 图像 {im_id} 在类别文件中缺失")
    
    # 检查是否有标注信息
    if im_id not in annotations:
        missing_annotation_count += 1
        print(f"警告: 图像 {im_id} 在标注文件中缺失")

print(f"数据一致性检查完成:")
print(f"  - 缺失类别信息的图像: {missing_class_count}")
print(f"  - 缺失标注信息的图像: {missing_annotation_count}")

if missing_class_count > 0 or missing_annotation_count > 0:
    print("⚠️  警告: 发现数据不一致，可能影响评估结果")
    user_input = input("是否继续执行? (y/n): ")
    if user_input.lower() != 'y':
        print("终止执行")
        exit(1)
else:
    print("✅ 数据一致性检查通过")

# 初始化评估指标 - 与test.py完全一致
cnt = 0  # 图像计数器
SAE = 0  # 绝对误差总和 (Sum of Absolute Errors)
SSE = 0  # 平方误差总和 (Sum of Square Errors)
detailed_results = []

print("Evaluation on {} data using API: {}".format(args.test_split, args.api_url))
im_ids = data_split[args.test_split]

# 使用 tqdm 进度条
pbar = tqdm(im_ids)

for im_id in pbar:
    # 数据完整性检查
    if im_id not in annotations:
        print(f"错误: 图像 {im_id} 在标注文件中缺失，跳过")
        continue
        
    if im_id not in image_classes:
        print(f"错误: 图像 {im_id} 在类别文件中缺失，跳过")
        continue
    
    anno = annotations[im_id]
    bboxes = anno['box_examples_coordinates']
    dots = np.array(anno['points'])
    
    # 获取当前图像的类别信息
    class_name = image_classes[im_id]  # 移除默认值，确保数据存在
    
    # 处理边界框坐标格式转换 - 修复为与test.py一致的格式
    rects = []
    for bbox in bboxes:
        x1, y1 = bbox[0][0], bbox[0][1]  # 左上角坐标
        x2, y2 = bbox[2][0], bbox[2][1]  # 右下角坐标
        # 注意：这里保持[x1, y1, x2, y2]格式，因为API可能期望这种格式
        # 如果API需要[y1, x1, y2, x2]格式，请修改为：rects.append([y1, x1, y2, x2])
        rects.append([x1, y1, x2, y2])
    
    image_path = os.path.join(im_dir, im_id)
    
    pred_cnt = call_counting_api(
        image_path,
        rects,
        args.api_url,
        args.model_name,
        args.max_tokens,
        class_name  # 传递类别信息
    )
    
    # 计算真实计数和预测计数 - 与test.py完全一致
    gt_cnt = dots.shape[0]  # 真实计数（点的数量）
    
    # 更新统计信息 - 与test.py完全一致
    cnt = cnt + 1
    err = abs(gt_cnt - pred_cnt)  # 绝对误差
    SAE += err  # 累积绝对误差
    SSE += err**2  # 累积平方误差
    
    detailed_results.append({
        "image_id": im_id,
        "class_name": class_name,  # 添加类别信息到结果中
        "ground_truth": int(gt_cnt),
        "predicted": pred_cnt,
        "absolute_error": err,
        "bboxes": rects
    })
    
    # 更新进度条描述，显示当前图像的结果和累计指标 - 与test.py格式保持一致
    description = '{:<8}: 类别: {:<15} 真实-预测: {:6d}, {:6.1f}, 误差: {:6.1f}. 当前MAE: {:5.2f}, RMSE: {:5.2f}'.format(
        im_id, class_name[:15], gt_cnt, pred_cnt, err, SAE/cnt, (SSE/cnt)**0.5
    )
    pbar.set_description(description)
    print("")  # 添加空行，与test.py保持一致

# 输出最终评估结果 - 与test.py完全一致
final_mae = SAE/cnt  # MAE = 累积绝对误差 / 图像数量
final_rmse = (SSE/cnt)**0.5  # RMSE = sqrt(累积平方误差 / 图像数量)
print('在 {} 数据上，MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.test_split, final_mae, final_rmse))

if args.save_results:
    final_results = {
        "test_split": args.test_split,
        "api_url": args.api_url,
        "model_name": args.model_name,
        "total_samples": cnt,
        "mae": final_mae,
        "rmse": final_rmse,
        "detailed_results": detailed_results
    }
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"详细结果已保存到: {args.output_file}")