"""
Fetch a single public-domain sample dermoscopy image from the ISIC Archive, so python -m melanoma_ia.cli.predict --image samples/lesao.jpg works out of the box on a fresh clone, the original repo's README referenced a lesao.jpg that was never included, so a reviewer following it hit FileNotFoundError on step one.

Usage:
    python scripts/download_sample_image.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

ISIC_API_LIST_URL = "https://api.isic-archive.com/api/v2/images/?limit=1"
OUTPUT_PATH = Path("samples/lesao.jpg")


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Consultando {ISIC_API_LIST_URL} ...")
    resp = requests.get(ISIC_API_LIST_URL, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        print("Nenhuma imagem retornada pela API do ISIC Archive.", file=sys.stderr)
        return 1

    image_meta = results[0]
    image_url = image_meta["files"]["full"]["url"]
    isic_id = image_meta.get("isic_id", "unknown")

    print(f"Baixando imagem de amostra ({isic_id}) de {image_url} ...")
    img_resp = requests.get(image_url, timeout=30)
    img_resp.raise_for_status()
    OUTPUT_PATH.write_bytes(img_resp.content)

    print(f"Salvo em {OUTPUT_PATH} ({len(img_resp.content) / 1024:.1f} KB)")
    print(
        "Teste rápido:\n"
        f"  python -m melanoma_ia.cli.predict --image {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
