import os
from markitdown import MarkItDown

def convert_to_markdown(input_dir: str, output_dir: str):
    """Конвертирует DOCX/PDF в Markdown с сохранением структуры"""
    md_converter = MarkItDown()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if filename.endswith(('.docx', '.pdf', '.pptx')):
            print(f"Processing {filename}...")
            try:
                result = md_converter.convert(file_path)
                
                meta_header = f"# METADATA\nSource: {filename}\nType: Document\n\n"
                full_content = meta_header + result.text_content
                
                out_name = os.path.splitext(filename)[0] + ".md"
                with open(os.path.join(output_dir, out_name), 'w', encoding='utf-8') as f:
                    f.write(full_content)
            except Exception as e:
                print(f"Error parsing {filename}: {e}")

if __name__ == "__main__":
    convert_to_markdown("data/raw", "data/processed")

