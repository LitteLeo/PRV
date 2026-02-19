#!/usr/bin/env python3
"""
启动服务、运行推理、自动释放服务
"""
import argparse
import subprocess
import time
import signal
import sys
import os
import atexit
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 全局变量存储进程
services = {
    'e5': None,
    'vllm': None
}

def cleanup_services():
    """清理所有服务"""
    print("\nCleaning up services...")
    for name, proc in services.items():
        if proc and proc.poll() is None:
            print(f"Stopping {name} service (PID: {proc.pid})...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"{name} service stopped")

def start_e5_service():
    """启动 E5 检索服务"""
    print("Starting E5 search service...")
    proc = subprocess.Popen(
        ['python', '-m', 'uvicorn', 'search.start_e5_server_main:app',
         '--host', '0.0.0.0', '--port', '8090'],
        cwd='/home/lfy/projects/REAP',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    services['e5'] = proc
    time.sleep(5)  # 等待服务启动
    if proc.poll() is None:
        print(f"E5 service started (PID: {proc.pid})")
        return True
    else:
        print("E5 service failed to start")
        return False

def start_vllm_service(model_path):
    """启动 vLLM 服务"""
    print(f"Starting vLLM service for {model_path}...")
    proc = subprocess.Popen(
        ['vllm', 'serve', model_path,
         '--host', '0.0.0.0', '--port', '8000',
         '--tensor-parallel-size', '1', '--dtype', 'bfloat16'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    services['vllm'] = proc
    time.sleep(30)  # 等待模型加载
    if proc.poll() is None:
        print(f"vLLM service started (PID: {proc.pid})")
        return True
    else:
        print("vLLM service failed to start")
        return False

def run_inference(dataset_file, output_file, max_workers=2):
    """运行推理"""
    print(f"Running inference on {dataset_file}...")
    cmd = [
        'python', 'evaluation/generate_predictions_from_multistep.py',
        dataset_file, output_file,
        '--max_workers', str(max_workers)
    ]
    
    # 如果有已存在的输出文件，使用 resume 模式
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        cmd.append('--resume')
        print("Resuming from existing predictions...")
    
    result = subprocess.run(cmd, cwd='/home/lfy/projects/REAP')
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--vllm_model', required=True)
    parser.add_argument('--max_workers', type=int, default=2)
    parser.add_argument('--timeout_minutes', type=int, default=5)
    args = parser.parse_args()
    
    # 注册清理函数
    atexit.register(cleanup_services)
    signal.signal(signal.SIGINT, lambda s, f: (cleanup_services(), sys.exit(1)))
    signal.signal(signal.SIGTERM, lambda s, f: (cleanup_services(), sys.exit(1)))
    
    try:
        # 启动服务
        if not start_e5_service():
            print("Failed to start E5 service")
            return 1
        
        if not start_vllm_service(args.vllm_model):
            print("Failed to start vLLM service")
            return 1
        
        # 运行推理
        success = run_inference(args.dataset_file, args.output_file, args.max_workers)
        
        if success:
            print(f"\nInference completed. Waiting {args.timeout_minutes} minutes before cleanup...")
            time.sleep(args.timeout_minutes * 60)
        else:
            print("Inference failed")
            return 1
        
    finally:
        cleanup_services()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
