import imagehash
from PIL import Image
from typing import List, Tuple

def match_hash(query_image: Image.Image, db_images: List[Image.Image], threshold: int = 5) -> List[Tuple[str, float]]:
    """Match input image hash against database hashes.
    
    Args:
        query_image: PIL image to hash
        db_images: List of (PIL image, identifier)
        threshold: Max Hamming distance for match
    
    Returns:
        List of (identifier, similarity score)
    """
    query_hash = imagehash.phash(query_image)
    results = []
    for img, label in db_images:
        target_hash = imagehash.phash(img)
        similarity = 1 - (query_hash - target_hash) / len(query_hash.hash)**2
        if similarity >= (1 - threshold / 64):
            results.append((label, similarity))
    return sorted(results, key=lambda x: -x[1])
