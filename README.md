基于[开源项目](https://github.com/facefusion/facefusion)

1. 实现基于api方式进行功能调用，实现异步调用。[参考项目](https://github.com/kyolabnet/facefusion-api)
   1. 更改 facefusion.ini 文件：app = True
   2. port可选，默认 8000
     ```shell
     python run.py
      ```
2. Docker容器化
   1. .env配置文件 `cp .env.template .env`
   2. 启动
      ```shell
      # 构建镜像
      docker compose build

      # 启动
      docker compose up -d
	  ```
   3. 注：
      1. 默认使用API模式，切换为UI模式，请删除`docker-compose.yml`中的`--api`，且切换`TARGET_PORT`为7860
      2. 默认引擎为`face_swapper`，若需要切换至其他，则更改`FRAME_P`
