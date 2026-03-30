这个目录包含 `download_resources.sh`，用于帮助下载数据集与模型权重。

使用说明：

1. 准备环境变量（可选）：
   - `HF_TOKEN`：Hugging Face token（若需从 HF 下载模型）
   - `SCANNET_URL` / `SCANREFER_URL` / `SCANQA_URL` / `SPAR7M_URL`：分别指向数据集 zip 文件的下载链接（可选）

2. 先做 dry-run（不会下载大型模型）：

   ```bash
   ./scripts/download_resources.sh
   ```

   这会尝试根据上面环境变量下载数据集 zip（若提供），但不会自动从 Hugging Face 下载大型模型。

3. 确认并下载模型权重：

   ```bash
   HF_TOKEN=REDACTED ./scripts/download_resources.sh --confirm
   ```

   注意：下载大型模型会占用大量磁盘与带宽，请确保你有足够的资源。

4. 下载路径：
   - 数据会解压到 `data/raw/<dataset>`
   - Hugging Face 模型会缓存到 `models_cache/`（位于仓库根目录）

安全提示：
- 如果模型仓库是 gated（受限），请使用有权限的 HF_TOKEN。
- 如果你不确定某个数据集的下载 URL，建议先手动从官方网站或 GitHub 下载再放到 `data/raw/` 下。
