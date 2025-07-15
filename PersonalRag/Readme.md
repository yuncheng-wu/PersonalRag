# PersonalRag

PersonalRag 是一個基於 Streamlit 的應用程式，用於與 AI 代理進行互動。
此專案可以透過url爬取網站資料存入 Supabase 資料庫，並使用 Azure OpenAI 來處理 AI 互動並顯示訊息。

## 專案結構

- `streamlit_ui.py`：提供 Streamlit 使用者介面，讓使用者可以輸入問題並接收 AI 代理的回應。
- `pydantic_ai_expert.py`：定義了 AI 代理及其工具，用於檢索相關文檔、列出文檔頁面以及獲取特定文檔頁面的內容。
- `crawl_url.py`：爬取 URL 的腳本。
- `utils.py`：包含輔助函數，例如 `get_azure_model`。
- `requirements.txt`：列出專案所需的所有依賴項。

## 環境設置

1. 安裝所需的 Python 套件：

    ```bash
    pip install -r requirements.txt
    ```

2. 設置環境變數。創建一個 `.env` 文件，並添加以下內容：

    ```env
    AZURE_KEY=your_azure_key
    AZURE_ENDPOINT=your_azure_endpoint
    AZURE_LLM=your_azure_llm
    AZURE_VERSION=your_azure_version
    SUPABASE_URL=your_supabase_url
    SUPABASE_SERVICE_KEY=your_supabase_service_key
    LOGFIRE_TOKEN=your_logfire_token
    ```

## 使用方法

1. 運行 Streamlit 應用程式：

    ```bash
    streamlit run streamlit_ui.py
    ```

2. 在瀏覽器中打開應用程式，輸入您的問題並接收 AI 代理的回應。

## 工具

### retrieve_relevant_documentation

檢索與查詢相關的文檔片段。

### list_documentation_pages

列出所有可用的文檔頁面。

### get_page_content

獲取特定文檔頁面的完整內容。

