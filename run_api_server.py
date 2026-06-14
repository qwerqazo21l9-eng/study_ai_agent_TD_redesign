#!/usr/bin/env python
"""
FastAPI 服务器启动脚本

用法:
    python run_api_server.py              # 开发模式
    python run_api_server.py --prod       # 生产模式（多进程）
    python run_api_server.py --host 0.0.0.0 --port 8000
"""

import sys
import argparse
from src.api.server import create_app


def main():
    parser = argparse.ArgumentParser(description="AI Agent FastAPI 服务器")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址")
    parser.add_argument("--port", type=int, default=8000, help="绑定端口")
    parser.add_argument("--prod", action="store_true", help="生产模式（多进程）")
    parser.add_argument("--workers", type=int, default=4, help="工作进程数")
    parser.add_argument("--reload", action="store_true", help="热重载（开发模式）")
    parser.add_argument("--log-level", default="info", help="日志级别")
    
    args = parser.parse_args()
    
    import uvicorn
    
    app = create_app()
    
    # 配置
    config = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "loop": "uvloop",  # 高性能事件循环
    }
    
    if args.prod:
        # 生产模式
        config.update({
            "workers": args.workers,
            "access_log": True,
        })
        print(f"🚀 启动生产模式服务器 ({args.workers} 进程)")
    else:
        # 开发模式
        config.update({
            "reload": args.reload or True,
            "access_log": True,
        })
        print(f"🚀 启动开发模式服务器 (热重载: {args.reload or True})")
    
    print(f"📍 监听: http://{args.host}:{args.port}")
    print(f"📚 API 文档: http://{args.host}:{args.port}/docs")
    print(f"📊 OpenAPI: http://{args.host}:{args.port}/openapi.json")
    
    uvicorn.run(**config)


if __name__ == "__main__":
    main()
