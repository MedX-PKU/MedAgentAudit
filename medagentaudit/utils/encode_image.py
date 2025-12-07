import base64
import os

def encode_image(image_path: str) -> str:
    """
    # .read() 读取图片文件的原始二进制数据
    # base64.b64encode() 将二进制数据编码为Base64格式
    # .decode("utf-8") 将Base64字节串转换为UTF-8字符串
    Encode an image file as a base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If there's an error reading the image file
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    try:
        with open(image_path, "rb") as image_file:

            return base64.b64encode(image_file.read()).decode("utf-8")
    except IOError as e:
        raise IOError(f"Error reading image file: {e}")