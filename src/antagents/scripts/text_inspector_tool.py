from antagents import Tool
from antagents.models import Model

from urllib.parse import urlparse  # URL校验
from PyPDF2 import PdfReader       # 新增专业PDF解析库（需安装：pip install PyPDF2）

import xml.etree.ElementTree as ET
import re
import requests  # 用于HTTP请求
import tempfile  # 创建临时文件
import html2text
import time
import threading


def html_to_markdown(html_content):
    """将HTML内容高质量转换为Markdown格式"""
    converter = html2text.HTML2Text()
    # 配置转换规则（优化效果）
    converter.ignore_links = False          # 保留链接
    converter.ignore_images = True          # 忽略图片（避免乱码）
    converter.ignore_emphasis = False       # 保留加粗/斜体
    converter.body_width = 0                # 不限制行宽
    converter.skip_internal_links = True    # 跳过内部锚点链接
    converter.bypass_tables = False         # 转换表格为Markdown格式
    return converter.handle(html_content).strip()


def is_valid_xml(xml_str):
    """简单校验是否为XML格式字符串"""
    # 检查是否包含XML声明或根标签
    return (re.match(r'^\s*<\?xml', xml_str.strip(), re.IGNORECASE) is not None or
            re.match(r'^\s*<[^>]+>', xml_str.strip()) is not None)


def xml_content_to_markdown(xml_str):
    # 先校验是否为XML格式
    if not is_valid_xml(xml_str):
        raise ValueError("输入内容不是有效的XML格式")

    # 解析XML内容
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        raise ValueError("XML格式解析失败，内容可能存在语法错误")

    # 命名空间映射
    namespaces = {
        'w': 'http://schemas.microsoft.com/office/word/2003/wordml',
        'wx': 'http://schemas.microsoft.com/office/word/2003/auxHint'
    }

    # 提取类别内容
    paragraphs = root.findall('.//w:body/wx:sect/w:p/w:r/w:t', namespaces)
    text_list = [p.text.strip() for p in paragraphs if p.text and p.text.strip()]
    categories = [item.strip('",') for item in text_list[2:-1]]

    # 生成纯内容Markdown列表
    return '\n'.join([f"- {cat}" for cat in categories])


def is_url(path):
    """判断输入是否为有效的HTTP/HTTPS URL"""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except Exception as e:
        print(f"【{path} 解析错误】{str(e)}")
        return False


def extract_pdf_text_with_titles(pdf_bytes):
    """
    用PyPDF2提取PDF文本（解决乱码），并识别标题层级生成MD格式
    """
    try:
        # 将二进制内容转为文件对象，供PyPDF2解析
        from io import BytesIO
        pdf_file = BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)

        # 提取所有页面文本
        all_text_blocks = []
        for page_num, page in enumerate(reader.pages):
            # 提取页面文本（PyPDF2会自动处理编码，解决乱码）
            page_text = page.extract_text()
            if not page_text:
                continue

            # 按行分割，过滤空行
            lines = [line.strip() for line in page_text.split('\n') if line.strip()]
            for line in lines:
                # 记录文本行+长度（用于识别标题）
                all_text_blocks.append({
                    'text': line,
                    'length': len(line),
                    'page': page_num + 1
                })

        if not all_text_blocks:
            return "【提示】未从PDF中提取到文本内容（可能是扫描件PDF）"

        # 识别标题并生成MD格式
        md_text = "# PDF提取内容\n\n"  # 总标题
        # 标题判定规则：
        # 1. 长度<60字符 且 首字符为中文/英文大写/数字 → 优先判定为标题
        # 2. 长度<30字符 → 一级/二级标题；30-60字符 → 三级标题；>60 → 正文
        for block in all_text_blocks:
            text = block['text']
            length = block['length']

            # 标题特征判定
            is_title = False
            if length < 60:
                first_char = text[0]
                # 首字符为中文、大写字母、数字 → 判定为标题
                if first_char.isupper() or first_char.isdigit() or '\u4e00' <= first_char <= '\u9fff':
                    is_title = True

            if is_title:
                if length < 30:
                    md_text += f"## {text}\n\n"  # 二级标题
                else:
                    md_text += f"### {text}\n\n"  # 三级标题
            else:
                md_text += f"{text}\n\n"  # 正文

        # 清理多余空行，格式化
        md_text = '\n'.join([line for line in md_text.split('\n') if line.strip()])
        return md_text

    except Exception as e:
        return f"【PDF解析错误】{str(e)}"


# 用于记录每个网站的最后下载时间，线程安全
_site_last_download = {}
_site_lock = threading.Lock()


def download_web_page(url):
    """下载网页内容并保存为临时HTML文件
    特性：
    1. 失败自动重试3次
    2. 同一网站下载间隔≥5秒，不足则sleep
    3. 支持多线程安全调用
    """
    # 解析网站域名（提取url的netloc部分，如www.baidu.com）
    parsed_url = urlparse(url)
    site_key = parsed_url.netloc or parsed_url.hostname or url  # 兼容各种url格式

    max_retries = 3  # 最大重试次数
    retry_count = 0  # 当前重试计数

    while retry_count < max_retries:
        try:
            # ========== 1. 控制下载间隔（线程安全） ==========
            with _site_lock:
                last_download_time = _site_last_download.get(site_key, 0)
                current_time = time.time()
                time_diff = current_time - last_download_time

                # 不足5秒则sleep
                if time_diff < 5:
                    sleep_time = 5 - time_diff
                    print(f"[TextInspectorTool] {site_key} 下载间隔不足5秒，sleep {sleep_time:.2f}秒")
                    time.sleep(sleep_time)

                # 更新最后下载时间为当前时间（确保下次计算准确）
                _site_last_download[site_key] = time.time()

            # ========== 2. 执行网页下载 ==========
            # 设置请求头，模拟浏览器访问（避免被网站拦截）
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()  # 抛出HTTP错误（4xx/5xx）

            # 判断响应是否为PDF文件
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' in content_type:
                # 1. 用PyPDF2解析PDF，提取清晰文本（解决乱码）
                pdf_text_with_titles = extract_pdf_text_with_titles(response.content)

                # 2. 将提取的带标题文本保存为MD文件
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.md',
                    delete=False,
                    encoding='utf-8'
                )
                temp_file.write(pdf_text_with_titles)
                temp_file.close()

                # 计算文件大小
                file_size_bytes = len(pdf_text_with_titles.encode('utf-8'))
                file_size = (
                    f"{file_size_bytes / 1024:.2f} KB"
                    if file_size_bytes < 1024*1024
                    else f"{file_size_bytes / (1024*1024):.2f} MB"
                )

                print(
                    f"[TextInspectorTool] Extract clean PDF text (with titles) from {url} "
                    f"to MD file: '{temp_file.name}'"
                )
                print(f"[TextInspectorTool] MD file size: {file_size} ({file_size_bytes} bytes)")
                return temp_file.name
            else:
                # 非PDF内容：转为Markdown保存（保留原有逻辑）
                markdown_content = html_to_markdown(response.text)
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.md',
                    delete=False,
                    encoding='utf-8'
                )
                temp_file.write(markdown_content)
                temp_file.close()
                print(f"[TextInspectorTool] Download {url} to file: '{temp_file.name}'")
                return temp_file.name

        except requests.exceptions.RequestException as e:
            retry_count += 1
            error_msg = f"[TextInspectorTool] Downloading {url} failed (retry {retry_count}/{max_retries}): {str(e)}"
            print(error_msg)

            # 最后一次重试失败则抛出异常
            if retry_count >= max_retries:
                raise Exception(f"下载网页失败（已重试{max_retries}次）: {str(e)}")

            # 重试前短暂休眠（避免高频重试被拦截）
            time.sleep(2)

    # 理论上不会走到这里，防止逻辑漏洞
    raise Exception(f"下载网页失败：达到最大重试次数{max_retries}次")


class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = (
        'You cannot load files yourself: instead call this tool to read a file as markdown text and '
        'ask questions about it.\n'
        'This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", '
        '".mp3", ".m4a", ".flac", ".pdf", ".docx"], and all other types of text files.\n'
        'IT DOES NOT HANDLE IMAGES (like .png, .jpg).'
    )

    inputs = {
        "file_path": {
            "description": (
                "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. "
                "It can be a valid URL starting with http:// or https://. If it is an image, "
                "use the visualizer tool instead!"
            ),
            "type": "string",
        },
        "question": {
            "description": (
                "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. "
                "Do not pass this parameter if you just want to directly return the content of the file."
            ),
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model = None, text_limit: int = 100000):
        super().__init__()
        self.model = model
        self.text_limit = text_limit
        from .mdconvert import MarkdownConverter

        self.md_converter = MarkdownConverter()

    def forward_initial_exam_mode(self, file_path, question):
        from antagents.models import MessageRole

        result = self.md_converter.convert(file_path)

        if is_valid_xml(result.text_content):
            result.text_content = xml_content_to_markdown(result.text_content)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        if len(result.text_content) < 4000:
            return "Document content: " + result.text_content

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Now please write a short, 5 sentence caption for this document, "
                            "that could help someone asking this question: "
                            + question
                            + "\n\nDon't answer the question yourself! Just provide useful notes on the document"
                        ),
                    }
                ],
            },
        ]
        return self.model(messages).content

    def forward(self, file_path, question: str | None = None) -> str:
        from antagents.models import MessageRole

        # 标记是否为网页来源
        is_web_source = False
        # 判断是否为URL，若是则下载为临时文件
        temp_file_path = None
        if is_url(file_path):
            temp_file_path = download_web_page(file_path)
            file_path = temp_file_path  # 替换为临时文件路径
            is_web_source = True  # 标记是网页来源

        result = self.md_converter.convert(file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        content = result.text_content[:2000] + ("..." if len(result.text_content) > 2000 else "")
        print(f"[TextInspectorTool] {file_path} Text Content:\n'{content}'")

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": (    # 标注来源类型
                            "You will have to write a short caption for this web page, "
                            "then answer this question:" + question
                            if is_web_source
                            else "You will have to write a short caption for this file, "
                            "then answer this question:" + question,
                        )
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": (   # 显式标注是网页内容
                            "Here is the complete web page:\n### " + str(result.title) + "\n\n"
                            + result.text_content[: self.text_limit]
                            if is_web_source
                            else "Here is the complete file:\n### " + str(result.title) + "\n\n"
                            + result.text_content[: self.text_limit]
                        )
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Now answer the question below. Use these three headings: '1. Short answer', "
                            "'2. Extremely detailed answer', '3. Additional Context on the document and "
                            "question asked'." + question
                        ),
                    }
                ],
            },
        ]
        return self.model(messages).content
