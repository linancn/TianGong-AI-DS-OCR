python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn==2.7.3 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple

modelscope download --model deepseek-ai/DeepSeek-OCR --local_dir ./models
