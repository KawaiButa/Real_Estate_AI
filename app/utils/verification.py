import hashlib
from uuid import UUID
from litestar import Request
import requests
from io import BytesIO
from PIL import Image


def get_device_fingerprint(request: Request) -> str:
    """Generate a device fingerprint based on request attributes"""
    components = [
        request.headers.get("User-Agent"),
        request.headers.get("Accept-Language"),
        request.client.host,
        request.headers.get("Accept"),
        request.headers.get("Accept-Encoding"),
    ]
    fingerprint_string = "|".join(filter(None, components))
    return hashlib.sha256(fingerprint_string.encode()).hexdigest()


def generate_qr_code(property__id: UUID, code: str) -> Image.Image:
    """
    Generate a QR code image using the QR Server API.

    Args:
        data (str): The data to encode in the QR code.
        size (int, optional): The size of the QR code in pixels (both width and height). Defaults to 150.

    Returns:
        Image.Image: The generated QR code as a PIL Image object.

    Raises:
        ValueError: If the API request fails or returns an invalid response.
    """
    base_url = "https://api.qrserver.com/v1/create-qr-code/"
    params = {
        "data": data,
        "size": f"{128}x{128}"
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise ValueError(f"Failed to generate QR code: {response.status_code} {response.reason}")